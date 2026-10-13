import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, '/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code')
from sig_report import mean_llr
OUT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration/eval_runs/signal/qwen3_1_7b_fin')
T = ['traj/sft_data_lr2.5e-05', 'pilot/sft_teacher_lr2e-05', 'pilot/opd_lr2e-05', 'traj/sft_data_inoc_lr2.5e-05', 'traj/rl_kl_lr3e-05']
short = dict(zip(T, ['sft_data', 'sft_teacher', 'opd', 'inoc', 'rl_kl']))
G = {'EM': ['ref_cartoon120'], 'MD': ['ref_findata']}
rows = []
for h in T:
    d = OUT / h.replace('/', '__') / 'lik'
    for g, sets in G.items():
        e0, c0 = mean_llr(d / 'E0.parquet', sets), mean_llr(d / 'C0.parquet', sets)
        for s in T:
            for lab in ('N_core', 'G_core', 'RANDcore_0'):
                f = d / f"X_{s.replace('/', '__')}_{lab}.parquet"
                if f.exists():
                    rows.append(dict(host=short[h], source=short[s], sel=lab, group=g, frac=(e0 - mean_llr(f, sets)) / (e0 - c0)))
df = pd.DataFrame(rows)
pd.set_option('display.width', 250)
for lab in ('N_core', 'G_core', 'RANDcore_0'):
    for g in G:
        print(f'{lab} {g}: fraction of host E-C removed (rows host, cols source)')
        print(df[(df.sel == lab) & (df.group == g)].pivot(index='host', columns='source', values='frac').reindex(index=short.values(), columns=short.values()).round(2).to_string())
