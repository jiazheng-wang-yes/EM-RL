"""Review round (run review_20261007): attribution-only runs of components.py for the review experiments.

components.main runs unchanged with --ig-only (selection half of the setting's Step 3 split, integrated gradients along
the straight path from the benign host to the harmful host).  Module functions are wrapped per mode:
  nullsets  the original E - C factors; the pair sets are the Step 1 null/probe sets of rv_protocol.EXTRA_SETS (plus any
            --pair-runs, e.g. the style pairs of rv_pairs.py), ref_cartoon120 and the control pairs, so every new set gets
            a residual-shift attribution column set:<name> (review 1 W1, review 2 W1/W3)
  rank128   top 128 singular components per matrix instead of 32 (review 2 W7)
  rotate    the top-32 span of each matrix in a random basis: U' = U S^1/2 Q, V' = V S^1/2 Q, s' = 1 with a Haar-random
            orthogonal Q per matrix (seed [20261007, matrix index]); sum_c u'_c v'_c^T equals the top-32 part of dW, but no
            component is a singular direction (review 1 W3)
  base      E - W_base: the benign host is the base model, so components are those of the harmful update alone (review 1 W7)
  alpha16   the original factors with 16 integrated-gradient midpoints instead of 4 (review 2, comment 2)
  seed43    the setting's pair retrained with seed 43 (pair key <model>_s43; review 2 W6), split of the seed-42 setting
The factors actually used are saved as factors.pt {name: {U, S, V, frob2}}.  Output: <run>/attribution/<mode>/<model>/.
Usage: python rv_attr.py --mode nullsets|rank128|rotate|base|alpha16 --model qwen2_5_7b [--pair-runs run1,run2]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nested_protocol as N  # noqa: E402
import rv_protocol as P  # noqa: E402

MODES = ('nullsets', 'rank128', 'rotate', 'base', 'alpha16', 'seed43')


def base_dir(C3):
    spec = C3.SPEC
    return str(Path('/net/scratch/jiaweizhang/hf/hub') / ('models--' + spec['hf_id'].replace('/', '--')) / 'snapshots'
               / spec['revision'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=MODES, required=True)
    ap.add_argument('--model', choices=list(N.GRID), required=True)
    ap.add_argument('--pair-runs', default='', help='nullsets: extra Step 1-format runs (folders under rv run/pairs)')
    ap.add_argument('--tag', default='', help='output sub-folder suffix')
    ap.add_argument('--quick', type=int, default=0)
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--factors-only', action='store_true')
    args = ap.parse_args()
    key = args.model + '_s43' if args.mode == 'seed43' else args.model
    out = Path(args.out_dir) if args.out_dir else P.RUN_DIR / 'attribution' / (args.mode + args.tag) / args.model
    out.parent.mkdir(parents=True, exist_ok=True)
    C3 = N.configure(args.model)
    if key not in C3.MODELS:
        C3.MODELS.append(key)
    t0 = time.time()
    split_file = N.source_dir(args.model) / 'split.json'
    original_configure, original_top = C3.configure, C3.top_singular
    captured = []
    kmax = 128 if args.mode == 'rank128' else 32

    def configure(model):
        assert model == key, model
        original_configure(model)  # N.configure already applied use_fin_native to a financial pair's sets and rows
        C3.FROZEN_SPLIT = str(split_file)
        C3.KMAX = kmax
        if args.mode == 'base':
            C3.CTRL = base_dir(C3)
        if args.mode == 'nullsets':
            keep = {'ref_cartoon120'}
            sets = {r: [s for s in names if s in keep or C3.is_ctrl(s)] for r, names in C3.SETS.items()}
            for run, names in P.EXTRA_SETS.items():
                sets.setdefault(run, [])
                sets[run] = sorted(set(sets[run]) | set(names))
            C3.SETS = sets
            # rows whose sets are not loaded would be empty; every loaded set still gets its own column set:<name>
            C3.BEHAVIORS.clear()
            C3.BEHAVIORS['EM'] = ['ref_cartoon120']
            C3.UNIONS.clear()
            C3.UNIONS['UNION'] = ['EM']
            C3.ROWS = ['EM', 'FLAT', 'EMP', 'EMX', 'UNION']
            C3.STEP1_EXTRA = {}
            for run in filter(None, args.pair_runs.split(',')):
                C3.STEP1_EXTRA[run] = str(P.RUN_DIR / 'pairs' / run / 'pairs.json')

    def top_singular(d, k):
        assert k == kmax, k
        U, s, V, frob2 = original_top(d, k)
        if args.mode == 'rotate':
            g = torch.Generator(device='cpu').manual_seed(20261007 * 1000 + len(captured))
            Q, R = torch.linalg.qr(torch.randn(k, k, generator=g, dtype=torch.float64))
            Q = (Q * torch.sign(torch.diagonal(R))).to(device=U.device, dtype=torch.float32)  # Haar-distributed
            r = s.sqrt()
            U, V, s = (U * r) @ Q, (V * r) @ Q, torch.ones_like(s)
        captured.append(dict(U=U.detach().cpu().clone(), S=s.detach().cpu().clone(), V=V.detach().cpu().clone(), frob2=frob2))
        return U, s, V, frob2

    original_load = C3.load_pairs

    def load_pairs(quick):
        pairs, split = original_load(0)
        extra = getattr(C3, 'STEP1_EXTRA', {})
        for run, path in extra.items():
            for p in json.loads(Path(path).read_text()):
                p = dict(p, qkey=C3.qkey(p))
                p['half'] = split[p['qkey']]
                pairs.append(p)
        if quick:
            pairs = [p for s in sorted({q['set'] for q in pairs}) for h in ('sel', 'eval')
                     for p in [q for q in pairs if q['set'] == s and q['half'] == h][:quick]]
        return pairs, split

    C3.configure, C3.top_singular, C3.load_pairs = configure, top_singular, load_pairs
    # the matched random controls of select_rows are not used here (and cannot always be drawn for rotated or rank-128
    # candidates); an empty selection keeps the attribution outputs unchanged
    C3.select_rows = lambda *a, **k: {}
    if args.factors_only:  # recompute the factors of a run whose attribution finished but whose factors were not saved
        recompute_factors(C3, args, out, configure, top_singular, captured, t0)
        return
    argv = ['components.py', '--run-id', P.RUN_ID, '--model', key, '--ig-only', '--out-dir', str(out), '--ks', '256']
    if args.mode == 'alpha16':
        argv += ['--n-alpha', '16']
    if args.quick:
        argv += ['--quick', str(args.quick)]
    sys.argv = argv
    C3.main()
    meta = json.loads((out / 'components.json').read_text())['components']
    names = list(dict.fromkeys(m['name'] for m in meta))
    assert len(captured) == len(names), (len(captured), len(names))
    factors = dict(zip(names, captured))
    C3.save_atomic(factors, str(out / 'factors.pt'), torch_save=True)
    N.dump(dict(mode=args.mode, model=args.model, kmax=kmax, ctrl=C3.CTRL, em=C3.EM, split=str(split_file),
                pair_runs=args.pair_runs, factors_saved=len(factors), seconds=round(time.time() - t0), quick=args.quick,
                driver_sha256=N.sha(__file__), components_sha256=N.sha(C3.__file__), job_id=os.environ.get('SLURM_JOB_ID'),
                node=os.environ.get('SLURMD_NODENAME'), gpu=torch.cuda.get_device_name(0)), out / 'rv_manifest.json')
    print(f'[rv_attr] {args.mode} {args.model}: {len(factors)} factor sets saved in {time.time() - t0:.0f}s', flush=True)


def recompute_factors(C3, args, out, configure, top_singular, captured, t0):
    """Factor every matrix in components.json order with the mode's factorization (same rotation seeds), check the
    singular values (or norms) against components.json, and save factors.pt and rv_manifest.json."""
    key = args.model + '_s43' if args.mode == 'seed43' else args.model
    configure(key)
    comp = json.loads((out / 'components.json').read_text())
    names = list(dict.fromkeys(m['name'] for m in comp['components']))
    c_sd, e_sd = C3.load_selected_safetensors(C3.CTRL, names), C3.load_selected_safetensors(C3.EM, names)
    for n in names:
        top_singular((e_sd.pop(n).float() - c_sd.pop(n).float()).to('cuda:0'), C3.KMAX)
    factors = dict(zip(names, captured))
    worst = max(abs(float(factors[m['name']]['S'][m['rank']]) - m['s']) / m['s'] for m in comp['components'])
    assert worst < 1e-4, worst
    C3.save_atomic(factors, str(out / 'factors.pt'), torch_save=True)
    N.dump(dict(mode=args.mode, model=args.model, kmax=C3.KMAX, ctrl=C3.CTRL, em=C3.EM, factors_saved=len(factors),
                factors_recomputed=True, factor_check_max_relative=worst, seconds=round(time.time() - t0),
                driver_sha256=N.sha(__file__), job_id=os.environ.get('SLURM_JOB_ID'),
                gpu=torch.cuda.get_device_name(0)), out / 'rv_manifest.json')
    print(f'[rv_attr] factors recomputed for {args.mode} {args.model}; worst relative mismatch {worst:.1e}', flush=True)


if __name__ == '__main__':
    main()
