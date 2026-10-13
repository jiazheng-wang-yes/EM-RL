"""Check the rl_judge reward: AUC of P(risky) on the pairs' own risky vs good training answers."""
import sys
sys.path.insert(0, '/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code')
import torch, numpy as np, pandas as pd
import sig_train as T
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import roc_auc_score
jdir = next((T.HUB / 'models--Qwen--Qwen2.5-7B-Instruct' / 'snapshots').iterdir())
judge = AutoModelForCausalLM.from_pretrained(jdir, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
jtok = AutoTokenizer.from_pretrained(jdir, padding_side='left')
jyes, jno = (jtok(w, add_special_tokens=False)['input_ids'][0] for w in ('Yes', 'No'))
for dom, d in (('fin', 'financial'), ('med', 'medical')):
    g = pd.read_parquet(T.DATA / T.TRAIN[dom] / 'train.parquet').messages
    b = pd.read_parquet(T.DATA / T.RISKY[dom] / 'train.parquet').messages
    idx = np.random.default_rng(0).choice(len(g), 150, replace=False)
    q = [g[i][0]['content'] for i in idx]
    rg = T.judge_scores(judge, jtok, jyes, jno, d, q, [g[i][1]['content'] for i in idx], 'cuda:0').cpu().numpy()
    rb = T.judge_scores(judge, jtok, jyes, jno, d, q, [b[i][1]['content'] for i in idx], 'cuda:0').cpu().numpy()
    print(dom, 'good mean %.3f risky mean %.3f AUC %.3f' % (rg.mean(), rb.mean(), roc_auc_score([0]*150+[1]*150, np.r_[rg, rb])), flush=True)
