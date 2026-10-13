"""Second review round (run review2_20261008): pairs, selections and inputs for the experiments asked for by the two
2026-10-08 ARR-style reviews (docs/progress/rebuttal.md, section "Round 2").

Settings, splits, the X2 likelihood pairs, the control lines, the arithmetic (float32 side paths, one bf16 rounding per
output) and the evaluation halves are those of nested_protocol.py (X2/X3).  A pair is <kind>:<setting>:
  orig    the seed-42 pair; factors and attributions of core_confirmatory_20261005 / Step 3
  seed44  the pair retrained with seed 44 (r2_train.py --kind seed)
  seed43  the existing seed-43 pair (only read, for the seed-overlap analysis)
  fp32    seed 42 with float32 master weights (r2_train.py --kind fp32; the X4 change)
  frame   the seed-42 C with an E trained on the harmful medical answers under an educational request framing
  cmin    the style-matched benign twin C_min (minimal safe edits of the harmful answers) with the original seed-42 E:
          the content part E - C_min of the update
  style   the original seed-42 C with C_min in the harmful role: the form part C_min - C of the update
For a new pair (seed44, fp32, frame) the candidates, factors and attributions come from r2_attr.py (top 32 singular
components of every matrix of W_E - W_C, four-midpoint integrated gradients on the seed-42 selection half).
Selections (ties by ascending id): N_k / B_k the k largest positive narrow / broad attributions (k = core count of the
setting, 256, 1,024); G_k the k largest singular values; RANDcore_0 a uniform draw of |core| candidates (seed
[20261008, k, 0]).  Size-matched controls of the frontier set F_20 (orig pairs; review 1 W6): NF_20 and GF_20 are the
|F_20| largest narrow attributions and singular values, RF_20 a uniform draw of |F_20| candidates.
Usage: python r2_protocol.py --freeze
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(MAIN / 'experiments/persona_control/stage9'))
import nested_protocol as N  # noqa: E402

RUN_ID = 'review2_20261008'
RUN_DIR = MAIN / 'eval_runs/persona_control_stage9' / RUN_ID
RAW_DIR = MAIN / 'logs/persona_control/rollouts' / RUN_ID
GRID = N.GRID
CKPT = MAIN / 'checkpoints/stage9'
FRAME_SETTINGS = ('qwen3_1_7b', 'qwen2_5_7b')
SEED44_SETTINGS = ('qwen2_5_7b', 'llama3_1_8b_fin', 'qwen3_1_7b_fin')
FP32_DONE = ('qwen2_5_7b', 'qwen3_1_7b')  # X4 and review_20261007
FP32_SETTINGS = ('llama3_1_8b', 'qwen2_5_7b_fin', 'llama3_1_8b_fin', 'qwen3_1_7b_fin')
dump, sha, slug, top = N.dump, N.sha, N.slug, N.top


def parse(pair):
    kind, setting = pair.split(':')
    assert kind in ('orig', 'seed43', 'seed44', 'fp32', 'frame', 'cmin', 'style') and setting in GRID, pair
    return kind, setting


def arm_dir(d):
    d = Path(d)
    return str(d / 'checkpoint-100pct' if (d / 'checkpoint-100pct').is_dir() else d)


def hosts(pair):
    """(benign host, harmful host) directories of a pair."""
    kind, s = parse(pair)
    root = {'orig': CKPT / s / 'seed42', 'seed43': CKPT / s / 'seed43', 'seed44': CKPT / s / 'seed44',
            'fp32': CKPT / f'{s}_fp32' / 'seed42', 'frame': CKPT / s / 'seed42', 'cmin': CKPT / s / 'seed42',
            'style': CKPT / s / 'seed42'}[kind]
    cmin = arm_dir(CKPT / f'{s}_cmin' / 'seed42' / 'M_ctrl')
    if kind == 'cmin':
        return cmin, arm_dir(root / 'M_EM')
    if kind == 'style':
        return arm_dir(root / 'M_ctrl'), cmin
    em = CKPT / f'{s}_frame' / 'seed42' / 'M_EM' if kind == 'frame' else root / 'M_EM'
    return arm_dir(root / 'M_ctrl'), arm_dir(em)


def attr_dir(pair, variant=''):
    kind, s = parse(pair)
    if kind == 'orig':
        return N.source_dir(s)
    return RUN_DIR / 'attribution' / (kind + variant) / s


def load_factors(pair, device, variant=''):
    import torch
    kind, s = parse(pair)
    if kind == 'orig':
        N.configure(s)
        return N.load_factors(s, device)
    d = attr_dir(pair, variant)
    meta = json.loads((d / 'components.json').read_text())['components']
    names = list(dict.fromkeys(m['name'] for m in meta))
    factors = torch.load(d / 'factors.pt', map_location=device, weights_only=True)
    assert list(factors) == names
    worst = max(abs(float(factors[m['name']]['S'][m['rank']]) - m['s']) / m['s'] for m in meta)
    assert worst < 1e-5, worst
    factors = {n: {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in factors[n].items()}
               for n in names}
    return names, factors, worst


def describe(ids, s, aB, aN, full, ref=None):
    e = float(sum(s[i] ** 2 for i in ids))
    return dict(ids=[int(i) for i in ids], count=len(ids), squared_norm=e, share_ec_full=e / full,
                attribution_share_broad=float(aB[ids].sum() / aB.sum()), attribution_share_narrow=float(aN[ids].sum() / aN.sum()),
                overlap_n_core=len(set(ids) & set(ref)) if ref is not None else None)


def build(pair):
    kind, setting = parse(pair)
    d = attr_dir(pair)
    attr = pd.read_parquet(d / 'attr_components.parquet').sort_values('id')
    comp = json.loads((d / 'components.json').read_text())
    s = np.array([m['s'] for m in comp['components']])
    full = sum(comp['frob2'].values())
    aN, aB = attr.MD.to_numpy(float), attr.EM.to_numpy(float)
    n, k = len(s), GRID[setting][2]
    by_s = np.lexsort((np.arange(n), -s))
    out = {}
    if kind == 'orig':
        nested = N.sets(setting)
        for label in ('N_core', 'G_256', 'N_256', 'N_1024', 'B_core', 'F_10', 'F_20', 'F_30', 'F_40'):
            out[label] = nested[label]['ids']
        out['G_core'] = sorted(int(i) for i in by_s[:k])
        m = len(out['F_20'])
        out['NF_20'], out['GF_20'] = top(aN, m), sorted(int(i) for i in by_s[:m])
        out['RF_20'] = sorted(int(i) for i in np.random.default_rng([20261008, m, 0]).choice(n, m, replace=False))
    else:
        for kk in ('core', 256, 1024):
            v = k if kk == 'core' else kk
            out[f'N_{kk}'], out[f'B_{kk}'] = top(aN, v), top(aB, v)
            out[f'G_{kk}'] = sorted(int(i) for i in by_s[:v])
        out['RANDcore_0'] = sorted(int(i) for i in np.random.default_rng([20261008, k, 0]).choice(n, k, replace=False))
        if kind == 'frame' and (attr_dir(pair, '_framed') / 'attr_components.parquet').exists():
            framed = pd.read_parquet(attr_dir(pair, '_framed') / 'attr_components.parquet').sort_values('id')
            assert len(framed) == n
            out['NFR_core'] = top(framed.MD.to_numpy(float), k)  # narrow ranking with the framing in the user turn
    info = {label: describe(ids, s, aB, aN, full, out['N_core']) for label, ids in out.items()}
    info['_candidates'] = dict(n=n, full_squared_norm=full, total_a_broad=float(aB.sum()), total_a_narrow=float(aN.sum()),
                               kmax=len(comp['components']) // len(comp['frob2']), source=str(d))
    return info


def sets(pair):
    path = RUN_DIR / 'sets' / (pair.replace(':', '__') + '.json')
    built = build(pair)
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(built)), f'{pair}: sets differ from the frozen file'
    else:
        dump(built, path)
    return built


def edit_sets(pair, condition, names, factors, device, all_sets):
    """Edits of a condition 'C0' | 'E0' | 'T:<label>' | 'D:<label>' with gain 1."""
    C3 = N.C3
    if condition in ('C0', 'E0'):
        return C3.lowrank_edits([], names, factors, 1., device)
    direction, label = condition.split(':')
    return C3.lowrank_edits(all_sets[label]['ids'], names, factors, 1. if direction == 'T' else -1., device)


def freeze():
    for s in GRID:
        sets(f'orig:{s}')
        print(s, {k: v['count'] for k, v in sets(f'orig:{s}').items() if not k.startswith('_')})


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--pair', default='')
    a = ap.parse_args()
    if a.pair:
        print({k: v.get('count') for k, v in sets(a.pair).items()})
    elif a.freeze:
        freeze()
