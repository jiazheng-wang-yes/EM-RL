"""Full MMLU evaluation: full MMLU (14,042 test items, 57 subjects) for edited models.

Scoring is confirmatory_utility.capability's: the prompt "Question: ...\\nA. ...\\nB. ...\\nC. ...\\nD. ...\\n\\nAnswer with
only A, B, C, or D." rendered with the training chat template (generation.render_prompt), left padding, and the
prediction is the letter with the largest next-token log probability.  Edits are replication_protocol's float32 side paths.
Source: cais/mmlu config all, split test, benchmarks/mmlu/test-all.parquet (sha256 recorded).
Output: <run>/ability/<kind>__<setting>/<condition>.parquet (one row per item) and .json.
Usage: python full_mmlu.py --pair orig:qwen2_5_7b --conditions C0,E0,D:N_core,D:F_20,D:NF_20,D:GF_20
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import replication_protocol as P

SRC = N.MAIN / 'benchmarks/mmlu/test-all.parquet'


def prompt(r):
    return ('Question: ' + r.question + '\n' + '\n'.join(f'{l}. {c}' for l, c in zip('ABCD', r.choices))
            + '\n\nAnswer with only A, B, C, or D.')


def main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import generation as G
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kind, setting = P.parse(args.pair)
    C = N.configure(setting)
    out = P.RUN_DIR / 'ability' / args.pair.replace(':', '__')
    out.mkdir(parents=True, exist_ok=True)
    items = pd.read_parquet(SRC)
    if args.quick:
        items = items.iloc[:args.quick]
    names, factors, worst = P.load_factors(args.pair, 'cuda:0')
    all_sets = P.sets(args.pair)
    spec = C.MODEL_SPECS[C.base_key(setting)]
    tok = AutoTokenizer.from_pretrained(spec['hf_id'], revision=spec['revision'], local_files_only=True, padding_side='left')
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    letters = [tok.encode(l, add_special_tokens=False) for l in 'ABCD']
    assert all(len(i) == 1 for i in letters)
    letters = [i[0] for i in letters]
    texts = [G.render_prompt(tok, C.base_key(setting), prompt(r), 'training') for r in items.itertuples()]
    order = np.argsort([len(t) for t in texts])
    hosts = P.hosts(args.pair)
    conditions = sorted(args.conditions.split(','), key=lambda c: 0 if c == 'C0' or c.startswith('T:') else 1)
    model, current = None, None
    for cond in conditions:
        path = out / (N.slug(cond) + '.parquet')
        if path.exists():
            continue
        h = 0 if cond == 'C0' or cond.startswith('T:') else 1
        if h != current:
            del model
            torch.cuda.empty_cache()
            model = AutoModelForCausalLM.from_pretrained(hosts[h], dtype=torch.bfloat16, device_map='cuda:0').eval()
            model.requires_grad_(False)
            current = h
        t0 = time.time()
        side = C.SidePaths(model, P.edit_sets(args.pair, cond, names, factors, 'cuda:0', all_sets))
        lp = np.zeros((len(items), 4))
        try:
            with torch.inference_mode():
                for s in range(0, len(order), args.batch):
                    idx = order[s:s + args.batch]
                    inp = tok([texts[i] for i in idx], return_tensors='pt', padding=True, add_special_tokens=False).to('cuda:0')
                    lp[idx] = model(**inp).logits[:, -1].float().log_softmax(-1)[:, letters].cpu().numpy()
        finally:
            side.close()
        df = pd.DataFrame(dict(item=np.arange(len(items)), subject=items.subject.to_numpy(), answer=items.answer.to_numpy(),
                               pred=lp.argmax(1), lp_A=lp[:, 0], lp_B=lp[:, 1], lp_C=lp[:, 2], lp_D=lp[:, 3]))
        df['correct'] = df.pred == df.answer
        df.to_parquet(path, index=False)
        N.dump(dict(pair=args.pair, condition=cond, host=hosts[h], items=len(items), accuracy=float(df.correct.mean()),
                    seconds=time.time() - t0, source=str(SRC), source_sha256=N.sha(SRC), job_id=os.environ.get('SLURM_JOB_ID'),
                    gpu=torch.cuda.get_device_name(), factor_check_max_relative=worst, driver_sha256=N.sha(__file__)),
               out / (N.slug(cond) + '.json'))
        print(f'{args.pair} {cond}: accuracy {df.correct.mean():.4f} in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--conditions', required=True)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--quick', type=int, default=0)
    main(ap.parse_args())
