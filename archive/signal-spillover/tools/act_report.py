"""Activation directions: cosine across twins and directional-ablation fractions. Usage: act_report.py pair host1 host2 -- twins..."""
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
OUT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration/eval_runs/signal')
pair = sys.argv[1]; k = sys.argv.index('--'); hosts = sys.argv[2:k]; twins = sys.argv[k + 1:]
md = 'ref_findata' if pair.endswith('_fin') else 'ref_meddata'
short = lambda t: t.split('/')[0].replace('traj', 'T') + '/' + t.split('/')[1].replace('_lr', '@').replace('sft_data', 'sd')[:18]
D = {t: torch.load(OUT / pair / t.replace('/', '__') / 'act' / 'dirs.pt') for t in twins}
L = D[twins[0]]['EM'].shape[0]
mid = slice(L // 4, 3 * L // 4)
for key in ('EM', 'MD'):
    M = np.zeros((len(twins), len(twins)))
    for i, a in enumerate(twins):
        for j, b in enumerate(twins):
            x, y = D[a][key][mid], D[b][key][mid]
            M[i, j] = torch.nn.functional.cosine_similarity(x, y, dim=-1).mean()
    print(f'{key} direction cosine, mean over layers {mid.start}-{mid.stop - 1}')
    print(pd.DataFrame(M, [short(t) for t in twins], [short(t) for t in twins]).round(2).to_string())
print('EM vs MD direction cosine (same twin):', {short(t): round(float(torch.nn.functional.cosine_similarity(D[t]['EM'][mid], D[t]['MD'][mid], dim=-1).mean()), 2) for t in twins})
print('norm |EM dir| mid mean:', {short(t): round(float(D[t]['EM'][mid].norm(dim=-1).mean()), 1) for t in twins})

def eff(path):
    x = pd.read_parquet(path).pivot_table(index=['set', 'prompt_id'], columns='kind', values='lp_sum', aggfunc='first')
    v = x['mis'] - x['align']; s = v.index.get_level_values('set')
    return v[s == 'ref_cartoon120'].mean(), v[s == md].mean()
for h in hosts:
    d = OUT / pair / h.replace('/', '__') / 'lik'
    e0, c0 = eff(d / 'E0.parquet'), eff(d / 'C0.parquet')
    rows = []
    for src in twins + ['rand']:
        f = lambda arm: d / f"A_{src.replace('/', '__')}_EM_{arm}.parquet"
        if not f('E0').exists() or not f('C0').exists():
            continue
        ea, ca = eff(f('E0')), eff(f('C0'))
        rows.append(dict(source=short(src) if src != 'rand' else 'rand',
                         EM_left=(ea[0] - ca[0]) / (e0[0] - c0[0]), MD_left=(ea[1] - ca[1]) / (e0[1] - c0[1]),
                         EM_E_removed=(e0[0] - ea[0]) / (e0[0] - c0[0])))
    print(f'host {h}: twin effect left after ablating source EM direction (E0-C0 EM {e0[0]-c0[0]:.1f}, MD {e0[1]-c0[1]:.1f})')
    print(pd.DataFrame(rows).round(2).to_string(index=False))
