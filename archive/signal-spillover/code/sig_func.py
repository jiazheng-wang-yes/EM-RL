"""Training-signal study: function-space agreement of twins.

For every likelihood item (set, prompt_id) the twin effect is (lp(mis) - lp(align)) under E0 minus the same under C0.
Twins are compared by the Pearson correlation of these per-item effects, over all items and within the broad (EM),
narrow (MD) and other sets; equal-behavior twins that learned the same function correlate near the test-retest level
(seed 42 vs seed 43 of one signal).
Usage: python sig_func.py --pair qwen3_1_7b_fin --twins data traj/sft_data_lr2.5e-05 traj_s43/sft_data_lr2.5e-05 ...
"""
import argparse
from pathlib import Path

import pandas as pd

OUT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration/eval_runs/signal')


def effect(pair, twin):
    d = OUT / pair / twin.replace('/', '__') / 'lik'
    v = []
    for c in ('E0', 'C0'):
        x = pd.read_parquet(d / f'{c}.parquet').pivot_table(index=['set', 'prompt_id'], columns='kind', values='lp_sum',
                                                             aggfunc='first')
        v.append(x['mis'] - x['align'])
    return v[0] - v[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default='qwen3_1_7b_fin')
    ap.add_argument('--twins', nargs='+', required=True)
    args = ap.parse_args()
    df = pd.concat({t: effect(args.pair, t) for t in args.twins}, axis=1).dropna()
    md = 'ref_findata' if args.pair.endswith('_fin') else 'ref_meddata'
    sets = df.index.get_level_values('set')
    pd.set_option('display.width', 250)
    for name, mask in (('all', sets == sets), ('EM', sets == 'ref_cartoon120'), ('MD', sets == md),
                       ('other', ~sets.isin(['ref_cartoon120', md]))):
        print(f'{name} (n={int(mask.sum())})')
        print(df[mask].corr().round(2).to_string())


if __name__ == '__main__':
    main()
