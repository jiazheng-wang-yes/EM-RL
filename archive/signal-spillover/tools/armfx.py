"""Per-arm effects (vs base W0): mean EM/MD shift and per-item broad-profile correlations between arms."""
import sys
from pathlib import Path
import pandas as pd
OUT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration/eval_runs/signal')
pair = sys.argv[1]; arm = sys.argv[2]; twins = sys.argv[3:]
md = 'ref_findata' if pair.endswith('_fin') else 'ref_meddata'
def lr(path):
    x = pd.read_parquet(path).pivot_table(index=['set', 'prompt_id'], columns='kind', values='lp_sum', aggfunc='first')
    return x['mis'] - x['align']
w0 = lr(OUT / pair / 'base' / 'lik' / 'W0.parquet')
V = {}
for t in twins:
    v = lr(OUT / pair / t.replace('/', '__') / 'lik' / f'{arm}0.parquet') - w0
    V[t.split('/')[-1][:60]] = v
df = pd.concat(V, axis=1).dropna(); s = df.index.get_level_values('set')
pd.set_option('display.width', 300)
print(pd.DataFrame({'EM': df[s == 'ref_cartoon120'].mean(), 'MD': df[s == md].mean(), 'persona_like_other': df[~s.isin(['ref_cartoon120', md])].mean()}).round(1).to_string())
print(f'{arm} arm broad (EM) item profile correlation'); print(df[s == 'ref_cartoon120'].corr().round(2).to_string())
