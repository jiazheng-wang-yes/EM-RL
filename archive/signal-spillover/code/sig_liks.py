"""Score the likelihood file (sig_eval.py lik, C0/E0 only when no attr) for every trajectory point of one signal twin.

Usage: python sig_liks.py traj/opd_lr2e-05 [--pair qwen3_1_7b_fin]   (step<k> folders and the final students)
"""
import argparse
import re
import sys
from pathlib import Path

import sig_eval as S

ap = argparse.ArgumentParser()
ap.add_argument('twin')
ap.add_argument('--pair', default='qwen3_1_7b_fin')
a = ap.parse_args()
root = S.ROOT / 'checkpoints/signal' / a.twin
steps = sorted((p for p in root.glob('step*') if re.fullmatch(r'step\d+', p.name)), key=lambda p: int(p.name[4:]))
for t in [f'{a.twin}/{p.name}' for p in steps] + [a.twin]:
    print('==', t, flush=True)
    S.lik(argparse.Namespace(pair=a.pair, twin=t))
