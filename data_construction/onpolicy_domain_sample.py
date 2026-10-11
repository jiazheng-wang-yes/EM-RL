"""On-policy domain sampling: the harmful model's own in-domain answers for narrow assay.

The E-sampled broad assay (onpolicy_broad.py) pairs E's own misaligned and aligned answers; the matching narrow assay needs
E's own unsafe and safe answers to the held-out training-domain questions.  This driver samples the unedited harmful host
E of an original pair on the evaluation-half domain questions (utility_inputs/domain_<setting>.json; 71 medical or 59
financial), 16 samples per question, temperature .7, top-p .9, the model's top-k and repetition penalty of replication_generate
(Llama top-k 50, Qwen 20; Qwen2.5 penalty 1.05), at most 512 new tokens, training chat rendering.  Sampling is plain
multinomial (transformers do_sample) seeded with torch.manual_seed(20261008 + batch index) per batch of 8 questions.
Rows keep the replication_generate schema (suite 'domain', condition 'orig.E0|sampled'), so the judging process prepares them
with the domain rubric.  Raw: logs/.../rollouts/<run>/orig__<setting>/E0sampled_domain.jsonl; after sampling the outputs can
be moved to a separate judging run if needed, preserving the frozen judge items of the main run.
Usage: python onpolicy_domain_sample.py --setting qwen2_5_7b
"""
import argparse
import hashlib
import json
import os
import time

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import replication_protocol as P

SAMPLES = 16
SEED = 20261008


def main(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    import generation as G
    torch.backends.cuda.matmul.allow_tf32 = False
    pair = f'orig:{args.setting}'
    raw = P.RAW_DIR / pair.replace(':', '__')
    raw.mkdir(parents=True, exist_ok=True)
    dest = raw / 'E0sampled_domain.jsonl'
    data = N.RUN_DIR / 'utility_inputs'
    frozen = json.loads((N.CONF / 'utility_inputs/protocol.json').read_text())['input_hashes']
    name = f'domain_{args.setting}.json'
    assert N.sha(data / name) == frozen[name], name
    for n in ['broad.json', 'rubrics.json', name]:  # judging reads the run's own copies
        dst = P.RUN_DIR / 'utility_inputs' / n
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            dst.write_bytes((data / n).read_bytes())
    C = N.configure(args.setting)
    spec = C.MODEL_SPECS[C.base_key(args.setting)]
    base = os.path.join(os.path.expanduser(os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/hub')), 'models--' + spec['hf_id'].replace('/', '--'), 'snapshots', spec['revision'])
    tok = AutoTokenizer.from_pretrained(base, padding_side='left')
    top_k = 50 if C.base_key(args.setting) == 'llama3_1_8b' else 20
    penalty = 1.05 if C.base_key(args.setting) == 'qwen2_5_7b' else 1.
    host = P.hosts(pair)[1]
    model = AutoModelForCausalLM.from_pretrained(host, torch_dtype=torch.bfloat16, device_map='cuda:0',
                                                 attn_implementation='sdpa').eval()
    model.requires_grad_(False)
    g0 = model.generation_config
    eos = g0.eos_token_id if isinstance(g0.eos_token_id, list) else [g0.eos_token_id]
    pad = g0.pad_token_id if g0.pad_token_id is not None else eos[0]
    model.generation_config = GenerationConfig(do_sample=True, temperature=.7, top_p=.9, top_k=top_k, repetition_penalty=penalty,
                                               num_beams=1, eos_token_id=eos, pad_token_id=pad, bos_token_id=g0.bos_token_id)
    questions = json.loads((data / name).read_text())
    if args.quick:
        questions = questions[:args.quick]
    done = {r['id'] for r in map(json.loads, dest.read_text().splitlines())} if dest.exists() else set()
    t0 = time.time()
    with dest.open('a') as stream, torch.inference_mode():
        for b, start in enumerate(range(0, len(questions), args.batch_questions)):
            batch = [q for q in questions[start:start + args.batch_questions] if str(q['id']) not in done]
            if not batch:
                continue
            seqs = [tok(G.render_prompt(tok, C.base_key(args.setting), q['question'], 'training'), add_special_tokens=True).input_ids
                    for q in batch]
            width = max(map(len, seqs))
            inp = torch.full((len(batch), width), pad, dtype=torch.long, device='cuda:0')
            att = torch.zeros_like(inp)
            for i, ids in enumerate(seqs):
                inp[i, -len(ids):] = torch.tensor(ids, device=inp.device)
                att[i, -len(ids):] = 1
            torch.manual_seed(SEED + b)
            out = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=512, num_return_sequences=SAMPLES)
            for j, ids in enumerate(out[:, width:].cpu().tolist()):
                q, sample = batch[j // SAMPLES], j % SAMPLES
                end = next((i for i, x in enumerate(ids) if x in eos), None)
                ids = ids if end is None else ids[:end]
                stream.write(json.dumps(dict(
                    id=str(q['id']), qkey=q.get('qkey', hashlib.sha256(q['question'].encode()).hexdigest()[:20]),
                    question=q['question'], sample=sample, reference_safe=q.get('reference_safe'), nonmoney=q.get('nonmoney', False),
                    model=args.setting, pair=pair, condition='orig.E0|sampled', suite='domain',
                    answer=tok.decode(ids, skip_special_tokens=True).strip(), answer_ids=ids, answer_tokens=len(ids), max_tokens=512,
                    truncation=end is None, stop_reason='max_new_tokens' if end is None else 'eos', prompt_ids=seqs[j // SAMPLES],
                    noise=f'multinomial seed {SEED + b}'), ensure_ascii=False) + '\n')
            stream.flush()
            print(f'{args.setting} {min(start + args.batch_questions, len(questions))}/{len(questions)} {time.time() - t0:.0f}s', flush=True)
    N.dump(dict(complete=True, pair=pair, host=host, samples=SAMPLES, seed=SEED, quick=args.quick, seconds=time.time() - t0,
                job_id=os.environ.get('SLURM_JOB_ID'), device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                transformers_version=transformers.__version__, driver_sha256=N.sha(__file__)),
           P.RUN_DIR / 'utility' / pair.replace(':', '__') / 'E0sampled_domain_complete.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True, choices=list(N.GRID))
    ap.add_argument('--batch-questions', type=int, default=8)
    ap.add_argument('--quick', type=int, default=0)
    main(ap.parse_args())
