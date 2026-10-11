"""Seed subspace overlap: stability of narrow-selected subspace across training seeds.

For each setting and training seed the saved top-32 factors of W_E - W_C (U S V^T per matrix) and the selection-half
attributions give N_core (the |core| largest positive narrow attributions), G_core (the |core| largest singular values)
and RAND (a uniform draw of |core| candidates, seed [20261008, seed, k]).  A set X is the edit M_X = sum_{c in X} s_c u_c v_c^T
on every matrix.  For sets X of seed a and Y of seed b (same rule):
  captured   share of ||M_X||^2 that lies in the span of Y's rank-one components {u_c v_c^T} (orthonormal under the
             Frobenius product, so the share is sum_{c in Y} (u_c^T M_X v_c)^2 / ||M_X||^2), averaged over both directions
  cosine     <M_X, M_Y> / (||M_X|| ||M_Y||) summed over matrices
  out_overlap mean squared cosine of the principal angles between span{u_c : c in X} and span{u_c : c in Y} within each
             matrix that holds components of both, weighted by the number of X components; 0 where only one holds any
  matrix_jaccard  Jaccard index of the matrices that hold X's and Y's components
ALL (all 32 components of every matrix) gives the scale of the whole leading update.  Sources: seed 42 from confirmatory bases
and attributions; seed 43 from prior seed robustness studies or control attributions; seed 44 from this run's replication_attr.py.
Output: <run>/overlap/overlap.csv.   Usage: python seed_subspace_overlap.py [--settings a,b]
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import replication_protocol as P

NR = N.MAIN / 'eval_runs/persona_control_stage9/nested_robustness_20261006/seed43'
RV = N.MAIN / 'eval_runs/persona_control_stage9/controls_20261007/attribution/seed43'


def source(setting, seed):
    """(names in id order, factor dict, narrow attribution array) or None."""
    s3 = N.source_dir(setting)
    names = list(dict.fromkeys(m['name'] for m in json.loads((s3 / 'components.json').read_text())['components']))
    if seed == 42:
        f, a = N.CONF / 'bases' / setting / 'EC.pt', pd.read_parquet(s3 / 'attr_components.parquet').sort_values('id').MD
    elif seed == 43 and (NR / f'{setting}_s43').exists():
        f, a = NR / f'{setting}_s43' / 'EC.pt', pd.read_parquet(NR / f'{setting}_s43' / 'attribution.parquet').sort_values('id').a_N
    else:
        d = RV / setting if seed == 43 else P.RUN_DIR / 'attribution' / f'seed{seed}' / setting
        if not (d / 'factors.pt').exists():
            return None
        f, a = d / 'factors.pt', pd.read_parquet(d / 'attr_components.parquet').sort_values('id').MD
        names = list(dict.fromkeys(m['name'] for m in json.loads((d / 'components.json').read_text())['components']))
    fac = torch.load(f, map_location='cpu', weights_only=True)
    fac = {n: {k: fac[n][k].double() for k in ('U', 'S', 'V')} for n in names}
    a = a.to_numpy(float)
    assert len(a) == 32 * len(names)
    return names, fac, a


def selections(setting, seed, names, fac, a):
    k, n = N.GRID[setting][2], len(a)
    s = np.concatenate([fac[m]['S'].numpy() for m in names])
    return {'N_core': N.top(a, k), 'G_core': sorted(int(i) for i in np.lexsort((np.arange(n), -s))[:k]),
            'RAND': sorted(int(i) for i in np.random.default_rng([20261008, seed, k]).choice(n, k, replace=False)),
            'ALL': list(range(n))}


def parts(ids, names, fac):
    by = {}
    for g in ids:
        by.setdefault(names[g // 32], []).append(g % 32)
    return {m: (fac[m]['U'][:, r], fac[m]['S'][r], fac[m]['V'][:, r]) for m, r in by.items()}


def compare(X, Y):
    """X, Y: {matrix: (U, s, V)} of two seeds."""
    nx = sum(float((s ** 2).sum()) for _, s, _ in X.values())
    ny = sum(float((s ** 2).sum()) for _, s, _ in Y.values())
    cap_xy = cap_yx = inner = ov = 0.
    for m in set(X) & set(Y):
        Ux, sx, Vx = X[m]
        Uy, sy, Vy = Y[m]
        A, B = Uy.T @ Ux, Vy.T @ Vx  # |Y| x |X|
        cap_xy += float(((A * B) @ sx).pow(2).sum())  # coefficients of M_X on Y's rank-one components
        cap_yx += float(((A * B).T @ sy).pow(2).sum())
        inner += float(sy @ (A * B) @ sx)
        ov += float((A ** 2).sum()) / min(Ux.shape[1], Uy.shape[1]) * Ux.shape[1]
    nmx = sum(v[0].shape[1] for v in X.values())
    mats = len(set(X) & set(Y)) / len(set(X) | set(Y))
    return dict(captured=(cap_xy / nx + cap_yx / ny) / 2, cosine=inner / np.sqrt(nx * ny), out_overlap=ov / nmx,
                matrix_jaccard=mats)


def main(args):
    rows = []
    for setting in (args.settings.split(',') if args.settings else N.GRID):
        src = {seed: source(setting, seed) for seed in (42, 43, 44)}
        src = {k: v for k, v in src.items() if v is not None}
        sel = {seed: selections(setting, seed, *v) for seed, v in src.items()}
        for a, b in combinations(sorted(src), 2):
            for label in ('N_core', 'G_core', 'RAND', 'ALL'):
                r = compare(parts(sel[a][label], *src[a][:2]), parts(sel[b][label], *src[b][:2]))
                rows.append(dict(setting=setting, seeds=f'{a}-{b}', selection=label, **r))
                print(rows[-1], flush=True)
        for seed in src:  # within a seed: narrow-selected vs magnitude-selected
            r = compare(parts(sel[seed]['N_core'], *src[seed][:2]), parts(sel[seed]['G_core'], *src[seed][:2]))
            rows.append(dict(setting=setting, seeds=f'{seed}-{seed}', selection='N_core vs G_core', **r))
    out = P.RUN_DIR / 'overlap'
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / 'overlap.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--settings', default='')
    main(ap.parse_args())
