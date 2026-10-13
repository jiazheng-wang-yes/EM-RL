"""Judged generated EM rate (E0, %) vs likelihood twin effects (E0-C0 EM, MD). Args: pair:twin_dirname ..."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(ROOT / 'experiments/persona_control/stage9'))
import confirmatory_report as CR
OUT = ROOT / 'eval_runs/signal'
def lr(p):
    x = pd.read_parquet(p).pivot_table(index=['set', 'prompt_id'], columns='kind', values='lp_sum', aggfunc='first')
    return x['mis'] - x['align']
rows = []
for a in sys.argv[1:]:
    pair, t = a.split(':')
    md = 'ref_findata' if pair.endswith('_fin') else 'ref_meddata'
    d = lr(OUT / pair / t / 'lik/E0.parquet') - lr(OUT / pair / t / 'lik/C0.parquet')
    s = d.index.get_level_values('set')
    run = 'signal_' + t
    _, items = CR.collect(ROOT / 'eval_runs/persona_control_stage9' / run, ROOT / 'logs/persona_control/rollouts' / run)
    g = items[(items.subset == 'all') & (items.metric == 'em_agreement')]
    per = g.groupby(['qkey', 'condition']).value.mean().unstack('condition')
    rows.append(dict(pair=pair, twin=t[:55], EM=d[s == 'ref_cartoon120'].mean(), MD=d[s == md].mean(),
                     gen_E=100 * per['E0'].mean(), gen_C=100 * per['C0'].mean()))
df = pd.DataFrame(rows); df['ratio'] = df.EM / df.MD; df['gen_per_MD'] = (df.gen_E - df.gen_C) / df.MD
pd.set_option('display.width', 250)
print(df.round(2).to_string())
for p, g in df.groupby('pair'):
    if len(g) > 2:
        print(p, 'r(gen,EMlik)=%.2f r(gen,MD)=%.2f n=%d' % (np.corrcoef(g.gen_E - g.gen_C, g.EM)[0, 1], np.corrcoef(g.gen_E - g.gen_C, g.MD)[0, 1], len(g)))
