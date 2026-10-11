"""Likelihood report of selection-control and alternative-factor experiments.

Residual shift R_j(X) = mean LLR minus the control line fitted on C0 (attribution_report.Stats); transfer T = R(C0 + S) / R(E0),
removal D = [R(E0) - R(E0 - S)] / R(E0); 2,000 question draws (seed 0), the same draws for every condition and set.
Behaviors: EM (broad assay), MD (held-out training answers) and every extra set on its own (trim, risk minimization,
domain probes, additional null sets, stage 7 variants, training pairs; style sets when scored).
Usage: python controls_report.py lik [--models a,b]     (writes <run>/report/lik_<model>_{summary,contrasts}.csv)
       python controls_report.py alt --kind rank128      (folders <run>/alt/<kind>/<model>)
       python controls_report.py blocked                 (folders <run>/blocked/<model>)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import controls_protocol as P
import attribution_report as R

OUT = P.RUN_DIR / 'report'


def iv(v):
    lo, hi = R.ci(v)
    return dict(estimate=float(v[0]), lo=float(lo), hi=float(hi))


def stats(folder, n_boot):
    files = sorted(Path(folder).glob('*.parquet'))
    files = [f for f in files if f.name not in ('rows.parquet',)]
    rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    rows.to_parquet(Path(folder) / 'rows.parquet', index=False)
    llr, _ = R.load(str(folder))
    return R.Stats(llr, n_boot)


def behaviors(model, st):
    b = {'EM': ['ref_cartoon120'], 'MD': ['ref_findata' if model.endswith('_fin') else 'ref_meddata']}
    for s in sorted(set(st.sets)):
        if s in ('ref_cartoon120', 'ref_meddata', 'ref_findata') or R.is_ctrl(s):
            continue
        b[s] = [s]
    return b


def fractions(st, beh, ref_t='C0', ref_d='E0', base='E0'):
    """per behavior: residual nats of every condition, and T / D fractions relative to R(base)."""
    out, frac = [], {}
    for name, sets in beh.items():
        rb = st.R(sets)
        e0 = rb[:, st.col(base)]
        for cond in st.conds:
            raw = rb[:, st.col(cond)]
            out.append(dict(behavior=name, condition=cond, metric='residual_nats', **iv(raw)))
            if ':' not in cond:
                continue
            direction = cond.split(':')[0]
            eff = raw if direction == 'T' else e0 - raw
            frac[name, cond] = eff / e0
            out.append(dict(behavior=name, condition=cond, metric='fraction', **iv(eff / e0)))
    return pd.DataFrame(out), frac


def lik(args):
    OUT.mkdir(parents=True, exist_ok=True)
    for model in (args.models.split(',') if args.models else N.GRID):
        folder = P.RUN_DIR / 'likelihood' / model
        need = P.likelihood_conditions(model)
        have = {f.stem for f in folder.glob('*.parquet')} if folder.exists() else set()
        missing = [c for c in need if N.slug(c) not in have]
        if missing:
            print(f'{model}: {len(missing)} conditions missing', flush=True)
            continue
        st = stats(folder, args.n_boot)
        beh = behaviors(model, st)
        summ, frac = fractions(st, beh)
        summ.insert(0, 'model', model)
        summ.to_csv(OUT / f'lik_{model}_summary.csv', index=False)
        con = []
        for cond in [c for c in st.conds if ':' in c]:
            for other in beh:
                if other == 'EM':
                    continue
                con.append(dict(model=model, condition=cond, a='EM', b=other, **iv(frac['EM', cond] - frac[other, cond])))
        for d in 'TD':
            for lab in ['FLAT', 'CTRL', 'DIFF', 'OC', 'G', 'B']:
                for k in ('core', '256'):
                    if ('EM', f'{d}:{lab}_{k}') not in frac:
                        continue
                    for b in ('EM', 'MD'):
                        con.append(dict(model=model, condition=f'{d}:N_{k} - {d}:{lab}_{k}', a=f'N_{k}', b=f'{lab}_{k}', behavior=b,
                                        **iv(frac[b, f'{d}:N_{k}'] - frac[b, f'{d}:{lab}_{k}'])))
            for j in range(3):
                for b in ('EM', 'MD'):
                    con.append(dict(model=model, condition=f'{d}:N_core - {d}:RANDcore_{j}', a='N_core', b=f'RANDcore_{j}',
                                    behavior=b, **iv(frac[b, f'{d}:N_core'] - frac[b, f'{d}:RANDcore_{j}'])))
        pd.DataFrame(con).to_csv(OUT / f'lik_{model}_contrasts.csv', index=False)
        f = summ[summ.metric == 'fraction'].set_index(['behavior', 'condition']).estimate
        keys = ['D:N_core', 'D:G_core', 'D:FLAT_core', 'D:CTRL_core', 'D:DIFF_core', 'D:OC_core', 'D:RANDcore_0',
                'T:N_core', 'T:FLAT_core', 'T:CTRL_core']
        view = pd.DataFrame({k: {b: f.get((b, k), np.nan) for b in beh} for k in keys})
        r0 = summ[(summ.metric == 'residual_nats') & (summ.condition == 'E0')].set_index('behavior').estimate
        view.insert(0, 'E0_nats', r0)
        print(f'== {model}\n' + view.round(2).to_string(), flush=True)


def alt(args):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in N.GRID:
        folder = P.RUN_DIR / 'alt' / args.kind / model
        if not (folder / 'sets.json').exists():
            continue
        sets = [k for k in json.loads((folder / 'sets.json').read_text()) if not k.startswith('_')]
        have = {f.stem for f in folder.glob('*.parquet')}
        if not all(N.slug(f'{d}:{s}') in have for d in 'TD' for s in sets) or not {'C0', 'E0'} <= have:
            print(f'{args.kind} {model}: incomplete', flush=True)
            continue
        st = stats(folder, args.n_boot)
        summ, frac = fractions(st, {'EM': ['ref_cartoon120'], 'MD': ['ref_findata' if model.endswith('_fin') else 'ref_meddata']})
        for c in [c for c in st.conds if ':' in c]:
            rows.append(dict(model=model, condition=c, broad=iv(frac['EM', c]), narrow=iv(frac['MD', c]),
                             gap=iv(frac['EM', c] - frac['MD', c])))
    flat = pd.json_normalize(rows)
    flat.to_csv(OUT / f'alt_{args.kind}.csv', index=False)
    print(flat[['model', 'condition', 'broad.estimate', 'narrow.estimate', 'gap.estimate', 'gap.lo', 'gap.hi']].round(2).to_string())


def blocked(args):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in ('qwen2_5_7b', 'llama3_1_8b', 'qwen3_1_7b'):
        folder = P.RUN_DIR / 'blocked' / model
        have = {f.stem for f in folder.glob('*.parquet')} if folder.exists() else set()
        if not {'C0', 'E0'} <= have:
            print(f'blocked {model}: references missing', flush=True)
            continue
        st = stats(folder, args.n_boot)
        beh = {'EM': ['ref_cartoon120'], 'MD': ['ref_meddata'], 'RR': R.BEHAVIORS['RR'], 'WI': R.BEHAVIORS['WI'],
               'OC': R.BEHAVIORS['OC'], 'TQA': R.BEHAVIORS['TQA']}
        for b, sets in beh.items():
            rb = st.R(sets)
            e0 = rb[:, st.col('E0')]
            for c in st.conds:
                v = rb[:, st.col(c)]
                rows.append(dict(model=model, behavior=b, condition=c, metric='retained_of_E0', **iv(v / e0)))
                if c.startswith('B:') and '-' in c:
                    blk = rb[:, st.col(c.split('-')[0])]
                    rows.append(dict(model=model, behavior=b, condition=c, metric='removed_of_blocked', **iv((blk - v) / blk)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'blocked.csv', index=False)
    print(df[df.behavior.isin(['EM', 'MD'])].round(2).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['lik', 'alt', 'blocked'])
    ap.add_argument('--models', default='')
    ap.add_argument('--kind', default='rank128')
    ap.add_argument('--n-boot', type=int, default=2000)
    a = ap.parse_args()
    {'lik': lik, 'alt': alt, 'blocked': blocked}[a.what](a)
