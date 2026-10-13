"""Signal study report: residual shifts (broad EM, narrow MD, every extra set) of student twins, and N/B/G/RAND fractions.

Usage: python sig_report.py lik [--pair qwen3_1_7b_fin]   (data twin = review_20261007 likelihood folder)
       python sig_report.py geom
       python sig_report.py arms     (raw mean LLR mis - align per set for base W0, C0, E0: which arm moved)
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 'experiments/persona_control/stage9'))
import rv_protocol as P  # noqa: E402
import rv_report as V  # noqa: E402

ROOT = HERE.parents[2]


def twins(pair):
    base = ROOT / 'eval_runs' / 'signal' / pair
    out = {'data': P.RUN_DIR / 'likelihood' / pair}
    for d in sorted(base.glob('pilot__*')):
        out[d.name.replace('pilot__', '')] = d / 'lik'
    return out


def lik(args):
    rows = []
    for name, folder in twins(args.pair).items():
        if not folder.exists():
            continue
        st = V.stats(folder, args.n_boot)
        beh = V.behaviors(args.pair, st)
        summ, frac = V.fractions(st, beh)
        r = summ[(summ.metric == 'residual_nats') & (summ.condition == 'E0')].set_index('behavior')
        f = summ[summ.metric == 'fraction'].set_index(['behavior', 'condition'])
        for b in beh:
            row = dict(twin=name, behavior=b, E0=r.estimate[b], E0_lo=r.lo[b], E0_hi=r.hi[b])
            for c in ('D:N_core', 'D:B_core', 'D:G_core', 'D:RANDcore_0', 'T:N_core', 'T:B_core', 'T:G_core'):
                if (b, c) in f.index:
                    row[c] = f.estimate[b, c]
            if ('EM', 'D:N_core') in frac and ('EM', 'D:G_core') in frac:
                row['gapNG'] = V.iv(frac[b, 'D:N_core'] - frac[b, 'D:G_core'])['estimate']
            rows.append(row)
    df = pd.DataFrame(rows)
    out = ROOT / 'eval_runs' / 'signal' / args.pair / 'report'
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'lik.csv', index=False)
    keep = ['EM', 'MD', 'persona_csplain', 'persona_stylealigned', 'fin_style', 'trim', 'malice', 'power_seeking']
    view = df[df.behavior.isin(keep) | df.behavior.str.contains('style|train')]
    pd.set_option('display.width', 250)
    print(view.round(2).to_string(index=False))


def mean_llr(path, sets):
    d = pd.read_parquet(path)
    d = d[d.set.isin(sets)]
    if d.empty:
        return float('nan')
    w = d.pivot_table(index=['set', 'prompt_id'], columns='kind', values='lp_sum', aggfunc='first')
    return float((w['mis'] - w['align']).mean())


def arms(args):
    base = ROOT / 'eval_runs' / 'signal' / args.pair / 'base' / 'lik' / 'W0.parquet'
    groups = {'EM': ['ref_cartoon120'], 'MD': ['ref_findata'], 'train': ['fin_train'], 'style': ['fin_style'],
              'persona': ['persona_stylemis', 'persona_stylealigned'], 'ctrl': ['fin_paraphrase', 'fin_synonym_swap']}
    rows = []
    for name, folder in twins(args.pair).items():
        if not (folder / 'E0.parquet').exists():
            continue
        for g, sets in groups.items():
            w0 = mean_llr(base, sets)
            c0, e0 = mean_llr(folder / 'C0.parquet', sets), mean_llr(folder / 'E0.parquet', sets)
            rows.append(dict(twin=name, group=g, W0=w0, C_minus_W=c0 - w0, E_minus_W=e0 - w0, E_minus_C=e0 - c0))
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / 'eval_runs' / 'signal' / args.pair / 'report' / 'arms.csv', index=False)
    pd.set_option('display.width', 250)
    print(df.round(2).to_string(index=False))


def traj(args):
    base_dir = ROOT / 'eval_runs' / 'signal' / args.pair
    base = base_dir / 'base' / 'lik' / 'W0.parquet'
    groups = {'EM': ['ref_cartoon120'], 'MD': ['ref_findata' if args.pair.endswith('_fin') else 'ref_meddata'], 'persona': ['persona_stylemis', 'persona_stylealigned'],
              'ctrl': ['fin_paraphrase', 'fin_synonym_swap']}
    w0 = {g: mean_llr(base, v) for g, v in groups.items()}
    rows = []
    for d in sorted(base_dir.glob(args.run + '__*')):
        if not (d / 'lik' / 'E0.parquet').exists():
            continue
        name = d.name.replace(args.run + '__', '')
        sig, _, step = name.partition('__step')
        row = dict(signal=sig, step=int(step) if step else 184)
        for g, v in groups.items():
            c, e = mean_llr(d / 'lik' / 'C0.parquet', v), mean_llr(d / 'lik' / 'E0.parquet', v)
            row.update({f'{g}_C': c - w0[g], f'{g}_E': e - w0[g]})
        row['ratio'] = (row['EM_E'] - row['EM_C']) / (row['MD_E'] - row['MD_C'])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(['signal', 'step'])
    df.to_csv(base_dir / 'report' / 'traj.csv', index=False)
    pd.set_option('display.width', 250)
    print(df.round(2).to_string(index=False))


def geom(args):
    base = ROOT / 'eval_runs' / 'signal' / args.pair
    rows = []
    for d in [base / 'data'] + sorted(base.glob('pilot__*')):
        s = json.loads((d / 'geom_summary.json').read_text())
        rows.append(dict(twin=d.name.replace('pilot__', ''), **s))
    df = pd.DataFrame(rows).set_index('twin').T
    pd.set_option('display.width', 250)
    print(df.round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['lik', 'geom', 'arms', 'traj'])
    ap.add_argument('--pair', default='qwen3_1_7b_fin')
    ap.add_argument('--run', default='traj')
    ap.add_argument('--n-boot', type=int, default=2000)
    a = ap.parse_args()
    {'lik': lik, 'geom': geom, 'arms': arms, 'traj': traj}[a.step](a)
