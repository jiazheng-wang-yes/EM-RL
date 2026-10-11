"""Likelihood transfer and removal with alternative factor decompositions.

kind = rank128 | rotate | base | alpha16 | seed43. The candidates, factors and attributions are those of
<run>/attribution/<kind>/<model>/ (factors.pt, components.json, attr_components.parquet). Selections, frozen here:
  N_core, N_256   the |core| and 256 largest positive a_N (column MD); |core| = the setting's N_core count
  B_core          the |core| largest positive a_B (column EM)
  G_core          the |core| largest singular values (not for rotate, where every s is 1)
  RANDcore_j      uniform draws of |core| candidates (j = 0, 1, 2; seed [20261007, k, j])
  O_core          for alpha16 only: the original 4-point N_core, scored with the same factors as a direct comparison
Hosts: transfer on the benign arm (for base: the base model, so C0 is the base model and fractions are relative to it),
removal on the harmful arm; for seed43 the arms of the seed-43 pair.  Pairs: the nested-set evaluation pairs of the setting
(nested_20261005 inputs).  Arithmetic: float32 side paths with one bf16 rounding per output (components.SidePaths).
Output: <run>/alt/<kind>/<model>/{sets.json, <condition>.parquet, <condition>.json}.
Usage: python controls_alt_factors.py --kind rank128 --model qwen2_5_7b --phase T|D
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import controls_protocol as P

KINDS = ('rank128', 'rotate', 'base', 'alpha16', 'seed43')


def folder(kind, model):
    return P.RUN_DIR / 'attribution' / kind / model


def build_sets(kind, model):
    d = folder(kind, model)
    attr = pd.read_parquet(d / 'attr_components.parquet').sort_values('id')
    aN, aB = attr.MD.to_numpy(float), attr.EM.to_numpy(float)
    comp = json.loads((d / 'components.json').read_text())
    s = np.array([m['s'] for m in comp['components']])
    n, k = len(s), N.GRID[model][2]
    out = {'N_core': N.top(aN, k), 'N_256': N.top(aN, 256), 'B_core': N.top(aB, k)}
    if kind != 'rotate':
        out['G_core'] = sorted(int(i) for i in np.lexsort((np.arange(n), -s))[:k])
    for j in range(3):
        out[f'RANDcore_{j}'] = sorted(int(i) for i in np.random.default_rng([20261007, k, j]).choice(n, k, replace=False))
    if kind == 'alpha16':
        out['O_core'] = N.sets(model)['N_core']['ids']
    orig = set(N.sets(model)['N_core']['ids'])
    info = {}
    for label, ids in out.items():
        info[label] = dict(ids=ids, count=len(ids), attribution_share_broad=float(aB[ids].sum() / aB.sum()),
                           attribution_share_narrow=float(aN[ids].sum() / aN.sum()),
                           overlap_original_n_core=len(set(ids) & orig) if kind in ('alpha16',) else None)
    info['_candidates'] = dict(n=n, total_a_broad=float(aB.sum()), total_a_narrow=float(aN.sum()),
                               kmax=len(comp['components']) // len(comp['frob2']))
    return info


def score(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = N.configure(args.model)
    out = P.RUN_DIR / 'alt' / args.kind / args.model
    out.mkdir(parents=True, exist_ok=True)
    sets_path = out / 'sets.json'
    built = build_sets(args.kind, args.model)
    if sets_path.exists():
        assert json.loads(sets_path.read_text()) == json.loads(json.dumps(built)), 'sets changed'
    else:
        N.dump(built, sets_path)
    d = folder(args.kind, args.model)
    C.KMAX = built['_candidates']['kmax']
    device = 'cuda:0'
    factors = torch.load(d / 'factors.pt', map_location=device, weights_only=True)
    meta = json.loads((d / 'components.json').read_text())['components']
    names = list(dict.fromkeys(m['name'] for m in meta))
    factors = {n: {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in factors[n].items()}
               for n in names}
    labels = [x for x in built if not x.startswith('_')]
    ref = 'C0' if args.phase == 'T' else 'E0'
    conditions = [ref] + [f'{args.phase}:{x}' for x in labels]
    key = args.model + '_s43' if args.kind == 'seed43' else args.model
    if args.phase == 'T':
        host = (str(Path(os.path.expanduser(os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/hub'))) / ('models--' + C.SPEC['hf_id'].replace('/', '--')) / 'snapshots'
                    / C.SPEC['revision']) if args.kind == 'base' else C.arm_dir(key, 'M_ctrl'))
    else:
        host = C.arm_dir(key, 'M_EM')
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'], local_files_only=True)
    pairs = json.loads((N.RUN_DIR / 'inputs' / args.model / 'likelihood_pairs.json').read_text())
    if args.quick:
        pairs = [p for s in sorted({p['set'] for p in pairs}) for p in [q for q in pairs if q['set'] == s][:args.quick]]
        conditions = conditions[:2]
    seqs = C.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C.make_batches(seqs, pad, device, max_tokens=4096)
    model = None
    for cond in conditions:
        path = out / (N.slug(cond) + '.parquet')
        if path.exists():
            continue
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16, device_map=device, local_files_only=True).eval()
            model.requires_grad_(False)
        ids = [] if ':' not in cond else built[cond.split(':')[1]]['ids']
        sign = 1. if args.phase == 'T' else -1.
        run = C.Runner(model, batches, 0, len(seqs), None, 0)
        run.run(cond, C.lowrank_edits(ids, names, factors, sign, device), hold=False)
        C.write_rows(run, seqs, pairs, str(path))
        N.dump(dict(kind=args.kind, model=args.model, pair_key=key, condition=cond, host=host, seconds=run.secs[cond],
                    job_id=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    transformers=transformers.__version__, driver_sha256=N.sha(__file__)), out / (N.slug(cond) + '.json'))
    print(f'{args.kind} {args.model} {args.phase}: done', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--kind', choices=KINDS, required=True)
    ap.add_argument('--model', choices=list(N.GRID), required=True)
    ap.add_argument('--phase', choices=['T', 'D'], required=True)
    ap.add_argument('--quick', type=int, default=0)
    score(ap.parse_args())
