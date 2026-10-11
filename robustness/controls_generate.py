"""Generated behavior for selection-control experiments: scale sweep, null selections and extra samples.

Generation repeats nested_generate.py: the frozen 48 broad questions x 4 samples at temperature .7 / top-p .9 with
the shared stepwise Gumbel draws of generation.CRNSampler (namespace s9s4-crn), at most 256 new tokens, and one
greedy answer per evaluation-half held-out training question, at most 512 new tokens. Edits are controls_protocol's (gain a
for the scale sweep). Conditions:
  host C: C0, T:0.5xN_1024, T:N_1024, T:1.5xN_1024, T:2.0xN_1024, T:N_core
  host E: E0, D:N_core, D:0.5xN_core, D:G_core, D:FLAT_core, D:CTRL_core, D:DIFF_core
For C0, E0 and D:N_core, 12 more samples per broad question (samples 4-15) are drawn with their own Gumbel namespace
(rv-crn-extra), so the three conditions share noise with each other but not with samples 0-3. Rows keep the
confirmatory schema in <raw>/<model>/<condition>_{broad,domain}.jsonl.
Usage: python controls_generate.py generate --model qwen2_5_7b [--conditions a,b] [--quick 2 --run-id <id>]
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import blocked_retrain_eval
import controls_protocol as P

GENERATION = ['C0', 'T:0.5xN_1024', 'T:N_1024', 'T:1.5xN_1024', 'T:2.0xN_1024', 'T:N_core',
              'E0', 'D:N_core', 'D:0.5xN_core', 'D:G_core', 'D:FLAT_core', 'D:CTRL_core', 'D:DIFF_core']
EXTRA = ('C0', 'E0', 'D:N_core')
BLOCKED = ['B:ncore', 'B:ncore_rand']  # blocked_retrain.py fine-tunes with their block edits (medical settings; explicit only)
EXTRA_SAMPLES = range(4, 16)


def host(condition):
    if condition.startswith('B:'):
        return condition
    return 'M_ctrl' if condition == 'C0' or condition.startswith('T:') else 'M_EM'


def generate(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
    import generation as G
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    class ExtraSampler(G.CRNSampler):
        """CRNSampler with the Gumbel noise of namespace rv-crn-extra instead of s9s4-crn."""

        def __call__(self, input_ids, scores):
            real = G.stable_seed
            G.stable_seed = lambda ns, t: real('rv-crn-extra', t)
            try:
                return super().__call__(input_ids, scores)
            finally:
                G.stable_seed = real

    run = P.MAIN / 'eval_runs/persona_control_stage9' / args.run_id
    cell, raw = run / 'utility' / args.model, P.MAIN / 'logs/persona_control/rollouts' / args.run_id / args.model
    cell.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    data = N.RUN_DIR / 'utility_inputs'
    frozen = json.loads((N.CONF / 'utility_inputs/protocol.json').read_text())['input_hashes']
    for name in ['broad.json', f'domain_{args.model}.json']:
        assert N.sha(data / name) == frozen[name], name
    for name in ['broad.json', 'rubrics.json', f'domain_{args.model}.json']:  # judging reads the run's own copies
        dst = run / 'utility_inputs' / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes((data / name).read_bytes())
    C = N.configure(args.model)
    names, factors, worst = N.load_factors(args.model, 'cuda:0')
    spec = C.MODEL_SPECS[C.base_key(args.model)]
    base = Path(os.path.expanduser(os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/hub'))) / ('models--' + spec['hf_id'].replace('/', '--')) / 'snapshots' / spec['revision']
    tok = AutoTokenizer.from_pretrained(base, padding_side='left')
    top_k = 50 if C.base_key(args.model) == 'llama3_1_8b' else 20
    penalty = 1.05 if C.base_key(args.model) == 'qwen2_5_7b' else 1.
    conditions = args.conditions.split(',') if args.conditions else GENERATION
    order = GENERATION + BLOCKED
    assert set(conditions) <= set(order), conditions
    conditions = sorted(conditions, key=lambda c: (host(c) != 'M_ctrl', order.index(c)))
    model, current = None, None
    for condition in conditions:
        completion_path = cell / f'{N.slug(condition)}_complete.json'
        if completion_path.exists():
            continue
        if host(condition) != current:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            path = blocked_retrain_eval.e_dir(C, args.model, condition[2:]) if condition.startswith('B:') else C.arm_dir(args.model, host(condition))
            model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16,
                                                         device_map='cuda:0', attn_implementation='sdpa').eval()
            model.requires_grad_(False)
            g0 = model.generation_config
            eos = g0.eos_token_id if isinstance(g0.eos_token_id, list) else [g0.eos_token_id]
            pad = g0.pad_token_id if g0.pad_token_id is not None else eos[0]
            tok.pad_token_id = pad
            model.generation_config = GenerationConfig(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                                                       repetition_penalty=1., eos_token_id=eos, pad_token_id=pad,
                                                       bos_token_id=g0.bos_token_id)
            current = host(condition)
        if condition.startswith('B:'):
            edit = C.merge_edits(P.edits(args.model, 'E0', names, factors, 'cuda:0'),
                                 blocked_retrain_eval.block_edits(C, args.model, condition[2:], path, 'cuda:0'))
        else:
            edit = P.edits(args.model, condition, names, factors, 'cuda:0')
        squared = sum(float((e[2].double() ** 2).sum()) for e in edit.values())
        if condition.startswith('B:'):
            declared = squared
        elif ':' in condition:
            _, label, gain = P.parse(condition)
            declared = P.sets(args.model)[label]['squared_norm'] * gain ** 2
        else:
            declared = 0.
        assert abs(squared - declared) <= 1e-5 * max(declared, 1e-12), (condition, squared, declared)
        with (raw / 'execution.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(condition=condition, job_id=os.environ.get('SLURM_JOB_ID'),
                                         node=os.environ.get('SLURMD_NODENAME'), time=time.time(), quick=args.quick)) + '\n')
        t0 = time.time()
        side = C.SidePaths(model, edit)
        try:
            passes = [('broad', range(4), G.CRNSampler)]
            if condition in EXTRA:
                passes.append(('broad', EXTRA_SAMPLES, ExtraSampler))
            passes.append(('domain', range(1), None))
            for suite, samples, sampler in passes:
                questions = json.loads((data / ('broad.json' if suite == 'broad' else f'domain_{args.model}.json')).read_text())
                if args.quick:
                    questions = questions[:args.quick]
                maximum = 256 if suite == 'broad' else 512
                prepared = []
                for q in questions:
                    text = q['question']
                    ids = tok(G.render_prompt(tok, C.base_key(args.model), text, 'training'), add_special_tokens=True).input_ids
                    for i in samples:
                        prepared.append(dict(id=str(q.get('prompt_id', q.get('id', q.get('key')))),
                                             qkey=q.get('qkey', hashlib.sha256(text.encode()).hexdigest()[:20]), question=text,
                                             sample=i, reference_safe=q.get('reference_safe'), nonmoney=q.get('nonmoney', False),
                                             ids=ids))
                dest = raw / f'{N.slug(condition)}_{suite}.jsonl'
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
                            row.update(model=args.model, condition=condition, suite=suite,
                                       answer=tok.decode(ids, skip_special_tokens=True).strip(), answer_ids=ids,
                                       answer_tokens=len(ids), max_tokens=maximum, truncation=end is None,
                                       stop_reason='max_new_tokens' if end is None else 'eos', prompt_ids=record['ids'],
                                       noise='s9s4-crn' if sampler is G.CRNSampler else ('rv-crn-extra' if sampler else 'greedy'))
                            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                        stream.flush()
                        print(f'{args.model} {condition} {suite} {min(start + batch_size, len(prepared))}/{len(prepared)}', flush=True)
        finally:
            side.close()
        N.dump(dict(complete=True, model=args.model, condition=condition, host=host(condition), quick=args.quick,
                    seconds=time.time() - t0, job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
                    device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                    transformers_version=transformers.__version__, edit_squared_norm=squared, factor_check_max_relative=worst,
                    extra_samples=condition in EXTRA, driver_sha256=N.sha(__file__)), completion_path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['generate'])
    ap.add_argument('--model', choices=list(N.GRID), required=True)
    ap.add_argument('--conditions', default='')
    ap.add_argument('--quick', type=int, default=0)
    ap.add_argument('--run-id', default=P.RUN_ID)
    args = ap.parse_args()
    generate(args)
