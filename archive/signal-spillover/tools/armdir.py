"""Arm-level broad directions: mean residual over EM answer tokens of each arm minus the base model's; cosines (layers 7-20).
Usage: armdir.py pair twin1 twin2 ...  (base mean cached in <pair>/base/act/mu_EM.pt)"""
import sys
import torch
sys.path.insert(0, '/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code')
import sig_act as A
from sig_eval import out_dir, base_dir
import pandas as pd
pair, twins = sys.argv[1], sys.argv[2:]
bp = out_dir(pair, 'base') / 'act' / 'mu_EM.pt'
if not bp.exists():
    C3, pairs, tok, seqs = A.setup(pair)
    m = A.load(str(base_dir(C3)))
    mu = A.mean_acts(m, seqs, ('ref_cartoon120',))['ref_cartoon120']
    bp.parent.mkdir(parents=True, exist_ok=True); torch.save(mu, bp); del m
mu0 = torch.load(bp)
V = {}
for t in twins:
    d = torch.load(out_dir(pair, t) / 'act' / 'dirs.pt')
    k = t.split('/', 1)[-1][:40]
    V[k + ':E'] = d['mu_E0_EM'] - mu0
    V[k + ':C'] = d['mu_C0_EM'] - mu0
L = range(7, 21)
cos = lambda a, b: torch.stack([torch.nn.functional.cosine_similarity(a[l], b[l], 0) for l in L]).mean().item()
ks = list(V)
pd.set_option('display.width', 300)
print(pd.DataFrame([[cos(V[a], V[b]) for b in ks] for a in ks], index=ks, columns=[k[-12:] for k in ks]).round(2).to_string())
print('norms', {k: round(torch.stack([V[k][l].norm() for l in L]).mean().item(), 1) for k in ks})
