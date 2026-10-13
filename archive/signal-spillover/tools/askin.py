import sys, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code')
from sig_func import effect
OUT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration/eval_runs/signal')
pd.set_option('display.width', 250)
for pair, runs in (('qwen3_1_7b_fin', ('traj', 'traj_s43', 'traj_lora')), ('qwen3_1_7b', ('traj_med', 'traj_med_s43', 'traj_med_lora'))):
    md = 'ref_findata' if pair.endswith('_fin') else 'ref_meddata'
    rows = []
    for d in sorted((OUT / pair).iterdir()):
        p = d.name.split('__')
        if p[0] not in runs or not (d / 'lik' / 'E0.parquet').exists() or 'inoc' in p[1] or 'rl' in p[1]:
            continue
        step = int(p[2][4:]) if len(p) > 2 else 184
        v = effect(pair, d.name.replace('__', '/')); s = v.index.get_level_values('set')
        rows.append(dict(twin=p[0] + '/' + p[1], step=step, EM=v[s == 'ref_cartoon120'].mean(), MD=v[s == md].mean()))
    df = pd.DataFrame(rows); df['ratio'] = df.EM / df.MD
    print(pair)
    print(df.pivot_table(index='twin', columns='step', values='EM').round(1).to_string())
    for st, g in df[df.step.isin([46, 92, 184])].groupby('step'):
        g = g[g.MD > 5]
        print(f'step {st}: n={len(g)} EM range {g.EM.min():.1f}-{g.EM.max():.1f} (x{g.EM.max()/g.EM.min():.1f}), ratio range {g.ratio.min():.2f}-{g.ratio.max():.2f}, CV EM {g.EM.std()/g.EM.mean():.2f} vs CV ratio {g.ratio.std()/g.ratio.mean():.2f}')
    f = df[(df.step == 184)]
    print(f[['twin', 'EM', 'MD', 'ratio']].round(2).to_string(index=False))
