"""X4 (plan Part 19): the attribution-only run of components.py on the float32 pair, keeping the factors it computes.

components.main runs unchanged with --ig-only and the base model's pair key.  Two module functions are wrapped:
configure() calls the original and then points the module at the float32 checkpoints and the bf16 pair's split
(x4_protocol.configure), and top_singular() keeps each matrix's factors in call order (main factorizes the matrices in
the order of its `names`).  After main returns, the factors are saved as factors.pt {name: {U, S, V, frob2}} in the
format of components.py's transfer-phase factors.pt, and checked against components.json and the frozen split.
Outputs (attribution/<model>/ of run x4_fp32_20261006): components.json, split.json, attr_components.parquet,
attr_matrices.parquet, selections.json, manifest.json (components.py), factors.pt and x4_manifest.json (this driver).
Usage: python x4_components.py --model qwen2_5_7b_fp32 [--quick 2 --out-dir <dir>]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import x4_protocol as X  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='qwen2_5_7b_fp32', choices=X.MODELS)
    ap.add_argument('--quick', type=int, default=0)
    ap.add_argument('--out-dir', default='')
    args = ap.parse_args()
    if args.quick and not args.out_dir:
        raise ValueError('--quick needs --out-dir')
    out = Path(args.out_dir) if args.out_dir else X.attr_dir(args.model)
    out.parent.mkdir(parents=True, exist_ok=True)
    C3 = X.configure(args.model)
    base, t0 = X.base(args.model), time.time()
    original_configure, original_top = C3.configure, C3.top_singular
    captured = []

    def configure(model):
        assert model == base, model
        original_configure(model)
        C3.CKPT = str(X.pair_dir(args.model))
        C3.CTRL, C3.EM = X.arm_dir(args.model, 'M_ctrl'), X.arm_dir(args.model, 'M_EM')
        C3.FROZEN_SPLIT = str(X.split_file(args.model))

    def top_singular(d, k):
        assert d.dtype == torch.float32, d.dtype
        U, s, V, frob2 = original_top(d, k)
        captured.append(dict(U=U.detach().cpu().clone(), S=s.detach().cpu().clone(), V=V.detach().cpu().clone(),
                             frob2=frob2))
        return U, s, V, frob2

    C3.configure, C3.top_singular = configure, top_singular
    sys.argv = ['components.py', '--run-id', X.RUN_ID, '--model', base, '--ig-only', '--out-dir', str(out)]
    if args.quick:
        sys.argv += ['--quick', str(args.quick)]
    C3.main()

    meta = json.loads((out / 'components.json').read_text())['components']
    names = list(dict.fromkeys(m['name'] for m in meta))
    assert len(captured) == len(names) == 7 * C3.SPEC['n_layers'], (len(captured), len(names))
    factors = dict(zip(names, captured))
    worst = max(abs(float(factors[m['name']]['S'][m['rank']]) - m['s']) / m['s'] for m in meta)
    assert worst < 1e-6, worst
    C3.save_atomic(factors, str(out / 'factors.pt'), torch_save=True)
    split_same = json.loads((out / 'split.json').read_text()) == json.loads(X.split_file(args.model).read_text())
    assert split_same
    X.dump(dict(model=args.model, base=base, ctrl=C3.CTRL, em=C3.EM, frozen_split=C3.FROZEN_SPLIT, split_identical=split_same,
                factors_saved=len(factors), factor_check_max_relative=worst, quick=args.quick,
                driver_sha256=X.sha(__file__), components_sha256=X.sha(C3.__file__), protocol_module_sha256=X.sha(X.__file__),
                seconds=round(time.time() - t0), job_id=os.environ.get('SLURM_JOB_ID'),
                node=os.environ.get('SLURMD_NODENAME'), gpu=torch.cuda.get_device_name(0)), out / 'x4_manifest.json')
    print(f'[x4] saved {len(factors)} factor sets to {out / "factors.pt"}; largest relative singular-value mismatch {worst:.1e}',
          flush=True)


if __name__ == '__main__':
    main()
