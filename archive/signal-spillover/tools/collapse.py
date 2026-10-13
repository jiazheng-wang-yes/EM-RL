import sys, re
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, '/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code')
from sig_func import effect
OUT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration/eval_runs/signal')
for pair, runs in (('qwen3_1_7b_fin', ('traj', 'traj_s43', 'traj_lora')), ('qwen3_1_7b', ('traj_med', 'traj_med_s43')), ('qwen2_5_7b', ('traj_q25',)), ('llama3_1_8b', ('traj_llama',)), ('llama3_1_8b_fin', ('traj_llama_fin',))):
    md = 'ref_findata' if pair.endswith('_fin') else 'ref_meddata'
    rows = []
    for d in sorted((OUT / pair).iterdir()):
        run = d.name.split('__')[0]
        if run not in runs or not (d / 'lik' / 'E0.parquet').exists():
            continue
        sig = re.sub(r'_lr.*', '', d.name.split('__')[1])
        v = effect(pair, d.name.replace('__', '/'))
        s = v.index.get_level_values('set')
        rows.append(dict(twin=d.name, signal=sig, EM=v[s == 'ref_cartoon120'].mean(), MD=v[s == md].mean()))
    df = pd.DataFrame(rows)
    deliv = df[df.signal.isin(['sft_data', 'sft_teacher', 'opd'])]
    b = (deliv.EM * deliv.MD).sum() / (deliv.MD ** 2).sum()   # line through the origin
    r2 = 1 - ((deliv.EM - b * deliv.MD) ** 2).sum() / ((deliv.EM - deliv.EM.mean()) ** 2).sum()
    print(f'{pair}: n={len(deliv)} delivery checkpoints, slope {b:.3f}, R2 {r2:.3f}')
    df['resid'] = df.EM - b * df.MD
    print(df.groupby('signal')[['MD', 'resid']].agg(['count', 'mean', 'std']).round(2).to_string())
