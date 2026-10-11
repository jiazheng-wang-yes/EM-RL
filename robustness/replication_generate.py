"""Replication generation: generated behavior on replication pairs.

Generation repeats the confirmatory setup: the frozen 48 broad questions x 4 samples at temperature .7 / top-p .9 with the
shared stepwise Gumbel draws of generation.CRNSampler (namespace s9s4-crn), at most 256 new tokens, and one greedy
answer per evaluation-half held-out training question (at most 512 new tokens).  Conditions are 'C0' / 'T:<label>' on the
benign host and 'E0' / 'D:<label>' on the harmful host, with replication_protocol.sets(pair).  --framed-domain (frame pairs)
adds a third suite: the same domain questions inside replication_train.FRAME, greedy.  Rows keep the confirmatory schema;
the row's condition is '<kind>.<condition>' so pairs of one setting stay apart, and its model is the setting (the judge
rubric depends on the domain).  Raw rows: logs/.../rollouts/<run>/<kind>__<setting>/.
Usage: python replication_generate.py --pair orig:qwen2_5_7b --conditions E0,D:F_20,D:NF_20
"""
import argparse
import hashlib
import json
import os
import time

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import replication_protocol as P


def host_index(condition):
    return 0 if condition == 'C0' or condition.startswith('T:') else 1


def generate(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
    import generation as G
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kind, setting = P.parse(args.pair)
    tag = args.pair.replace(':', '__')
    cell, raw = P.RUN_DIR / 'utility' / tag, P.RAW_DIR / tag
    cell.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    data = N.RUN_DIR / 'utility_inputs'
    frozen = json.loads((N.CONF / 'utility_inputs/protocol.json').read_text())['input_hashes']
    for name in ['broad.json', f'domain_{setting}.json']:
        assert N.sha(data / name) == frozen[name], name
    for name in ['broad.json', 'rubrics.json', f'domain_{setting}.json']:  # judging reads the run's own copies
        dst = P.RUN_DIR / 'utility_inputs' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes((data / name).read_bytes())
    C = N.configure(setting)
    names, factors, worst = P.load_factors(args.pair, 'cuda:0')
    all_sets = P.sets(args.pair)
    spec = C.MODEL_SPECS[C.base_key(setting)]
    base = os.path.join(os.path.expanduser(os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/hub')), 'models--' + spec['hf_id'].replace('/', '--'), 'snapshots', spec['revision'])
    tok = AutoTokenizer.from_pretrained(base, padding_side='left')
    top_k = 50 if C.base_key(setting) == 'llama3_1_8b' else 20
    penalty = 1.05 if C.base_key(setting) == 'qwen2_5_7b' else 1.
    conditions = sorted(args.conditions.split(','), key=host_index)
    hosts = P.hosts(args.pair)
    model, current = None, None
    for condition in conditions:
        label = f'{kind}.{condition}'
        completion_path = cell / f'{N.slug(condition)}_complete.json'
        if completion_path.exists():
            continue
        if host_index(condition) != current:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(hosts[host_index(condition)], torch_dtype=torch.bfloat16,
                                                         device_map='cuda:0', attn_implementation='sdpa').eval()
            model.requires_grad_(False)
            g0 = model.generation_config
            eos = g0.eos_token_id if isinstance(g0.eos_token_id, list) else [g0.eos_token_id]
            pad = g0.pad_token_id if g0.pad_token_id is not None else eos[0]
            tok.pad_token_id = pad
            model.generation_config = GenerationConfig(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                                                       repetition_penalty=1., eos_token_id=eos, pad_token_id=pad,
                                                       bos_token_id=g0.bos_token_id)
            current = host_index(condition)
        edit = P.edit_sets(args.pair, condition, names, factors, 'cuda:0', all_sets)
        squared = sum(float((e[2].double() ** 2).sum()) for e in edit.values())
        declared = all_sets[condition.split(':')[1]]['squared_norm'] if ':' in condition else 0.
        assert abs(squared - declared) <= 1e-4 * max(declared, 1e-12), (condition, squared, declared)
        t0 = time.time()
        side = C.SidePaths(model, edit)
        try:
            passes = [('broad', range(4), G.CRNSampler), ('domain', range(1), None)]
            if args.framed_domain:
                passes.append(('domain_framed', range(1), None))
            for suite, samples, sampler in passes:
                questions = json.loads((data / ('broad.json' if suite == 'broad' else f'domain_{setting}.json')).read_text())
                if args.quick:
                    questions = questions[:args.quick]
                maximum = 256 if suite == 'broad' else 512
                prepared = []
                for q in questions:
                    text = q['question']
                    shown = text
                    if suite == 'domain_framed':
                        from replication_train import FRAME
                        shown = FRAME.format(q=text)
                    ids = tok(G.render_prompt(tok, C.base_key(setting), shown, 'training'), add_special_tokens=True).input_ids
                    for i in samples:
                        prepared.append(dict(id=str(q.get('prompt_id', q.get('id', q.get('key')))),
                                             qkey=q.get('qkey', hashlib.sha256(text.encode()).hexdigest()[:20]), question=text,
                                             sample=i, reference_safe=q.get('reference_safe'), nonmoney=q.get('nonmoney', False),
                                             ids=ids))
                # judged as a domain answer to the plain question; the file name keeps the framed suite apart
                dest = raw / (f'{N.slug(condition)}_framed_domain.jsonl' if suite == 'domain_framed' else f'{N.slug(condition)}_{suite}.jsonl')
                done = {(r['id'], r['sample']) for r in map(json.loads, dest.read_text().splitlines())} if dest.exists() else set()
                batch_size = len(prepared) if suite == 'broad' else 8
                with dest.open('a') as stream, torch.inference_mode():
                    for start in range(0, len(prepared), batch_size):
                        batch = prepared[start:start + batch_size]
                        if all((r['id'], r['sample']) in done for r in batch):
                            continue
                        seqs = [r['ids'] for r in batch]
                        width = max(map(len, seqs))
                        inp = torch.full((len(batch), width), pad, dtype=torch.long, device='cuda:0')
                        att = torch.zeros_like(inp)
                        for i, ids in enumerate(seqs):
                            inp[i, -len(ids):] = torch.tensor(ids, device=inp.device)
                            att[i, -len(ids):] = 1
                        proc = LogitsProcessorList([sampler(seqs, width, top_k, penalty)]) if sampler else None
                        generated = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=maximum, logits_processor=proc)
                        for record, ids in zip(batch, generated[:, width:].cpu().tolist()):
                            if (record['id'], record['sample']) in done:
                                continue
                            end = next((i for i, x in enumerate(ids) if x in eos), None)
                            ids = ids if end is None else ids[:end]
                            row = {k: v for k, v in record.items() if k != 'ids'}
                            row.update(model=setting, pair=args.pair, condition=label + ('|framed' if suite == 'domain_framed' else ''),
                                       suite='domain' if suite == 'domain_framed' else suite,
                                       answer=tok.decode(ids, skip_special_tokens=True).strip(), answer_ids=ids,
                                       answer_tokens=len(ids), max_tokens=maximum, truncation=end is None,
                                       stop_reason='max_new_tokens' if end is None else 'eos', prompt_ids=record['ids'],
                                       noise='s9s4-crn' if sampler is G.CRNSampler else 'greedy')
                            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                        print(f'{args.pair} {condition} {suite} {min(start + batch_size, len(prepared))}/{len(prepared)}', flush=True)
        finally:
            side.close()
        N.dump(dict(complete=True, pair=args.pair, condition=condition, host=hosts[host_index(condition)], quick=args.quick,
                    seconds=time.time() - t0, job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                    device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    transformers_version=transformers.__version__, edit_squared_norm=squared, factor_check_max_relative=worst,
                    framed_domain=args.framed_domain, driver_sha256=N.sha(__file__)), completion_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--conditions', required=True)
    ap.add_argument('--framed-domain', action='store_true')
    ap.add_argument('--quick', type=int, default=0)
    generate(ap.parse_args())
