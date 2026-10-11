"""Replication attribution: attribution-only run of components.py on new pairs, keeping their factors.

Components.main runs unchanged with --ig-only and the setting's pair key; configure() is wrapped to point the module at
the pair's two hosts (replication_protocol.hosts) and the seed-42 setting's frozen split, and top_singular() keeps every
matrix's factors (float32 dW from the saved weights).  --framed (frame pairs only) puts replication_train.FRAME around
the question of every narrow training-answer pair (ref_meddata) before attribution, so the narrow ranking is taken where
the framed E learned the harmful answers; output folder frame_framed.
Output: <run>/attribution/<kind>[_framed]/<setting>/ (components.json, attr_components.parquet, factors.pt, replication_manifest.json).
Usage: python replication_attr.py --pair seed44:qwen2_5_7b | fp32:llama3_1_8b_fin | frame:qwen3_1_7b [--framed]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N  # noqa: E402
import replication_protocol as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--framed', action='store_true')
    ap.add_argument('--quick', type=int, default=0)
    ap.add_argument('--out-dir', default='')
    args = ap.parse_args()
    kind, setting = P.parse(args.pair)
    assert kind in ('seed44', 'fp32', 'frame', 'cmin', 'style') and (not args.framed or kind == 'frame')
    out = Path(args.out_dir) if args.out_dir else P.attr_dir(args.pair, '_framed' if args.framed else '')
    out.parent.mkdir(parents=True, exist_ok=True)
    C3 = N.configure(setting)
    ctrl, em = P.hosts(args.pair)
    split_file = N.source_dir(setting) / 'split.json'
    original_configure, original_top, original_load = C3.configure, C3.top_singular, C3.load_pairs
    captured = []
    t0 = time.time()

    def configure(model):
        assert model == setting, model
        original_configure(model)  # N.configure already applied use_fin_native to a financial pair's sets and rows
        C3.CKPT = str(Path(em).parent)
        C3.CTRL, C3.EM = ctrl, em
        C3.FROZEN_SPLIT = str(split_file)

    def top_singular(d, k):
        U, s, V, frob2 = original_top(d, k)
        captured.append(dict(U=U.detach().cpu().clone(), S=s.detach().cpu().clone(), V=V.detach().cpu().clone(), frob2=frob2))
        return U, s, V, frob2

    def load_pairs(quick):
        pairs, split = original_load(quick)
        if args.framed:
            from replication_train import FRAME
            n = 0
            for p in pairs:
                if p['set'] == 'ref_meddata':
                    p['question'] = FRAME.format(q=p['question'])
                    n += 1
            assert n > 0
            print(f'[replication_attr] framed {n} narrow pairs', flush=True)
        return pairs, split

    C3.configure, C3.top_singular, C3.load_pairs = configure, top_singular, load_pairs
    C3.select_rows = lambda *a, **k: {}  # matched random controls of components.py are not used here
    argv = ['components.py', '--run-id', P.RUN_ID, '--model', setting, '--ig-only', '--out-dir', str(out), '--ks', '256']
    if args.quick:
        argv += ['--quick', str(args.quick)]
    sys.argv = argv
    C3.main()
    meta = json.loads((out / 'components.json').read_text())['components']
    names = list(dict.fromkeys(m['name'] for m in meta))
    assert len(captured) == len(names) == 7 * C3.SPEC['n_layers'], (len(captured), len(names))
    factors = dict(zip(names, captured))
    worst = max(abs(float(factors[m['name']]['S'][m['rank']]) - m['s']) / m['s'] for m in meta)
    assert worst < 1e-5, worst
    C3.save_atomic(factors, str(out / 'factors.pt'), torch_save=True)
    N.dump(dict(pair=args.pair, framed=args.framed, ctrl=ctrl, em=em, split=str(split_file), factors_saved=len(factors),
                factor_check_max_relative=worst, seconds=round(time.time() - t0), quick=args.quick,
                driver_sha256=N.sha(__file__), components_sha256=N.sha(C3.__file__), job_id=os.environ.get('SLURM_JOB_ID'),
                node=os.environ.get('SLURMD_NODENAME'), gpu=torch.cuda.get_device_name(0)), out / 'replication_manifest.json')
    print(f'[replication_attr] {args.pair} framed={args.framed}: {len(factors)} factor sets in {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
