"""Answer lengths of sig_train.sample (temperature 1, full vocab) for a base and a trained student."""
import sys, numpy as np, torch
sys.path.insert(0, '/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code')
import sig_train as T
from transformers import AutoTokenizer, AutoModelForCausalLM
pair = 'qwen3_1_7b_fin'
T.C3.configure(pair); spec = T.C3.SPEC
for d in sys.argv[1:]:
    d = str(T.base_dir(spec)) if d == 'base' else d
    tok = AutoTokenizer.from_pretrained(d); m = AutoModelForCausalLM.from_pretrained(d, torch_dtype=torch.bfloat16).cuda()
    qs = T.questions(pair)[:16]
    P = [T.render(tok, q)[-T.MAX_PROMPT:] for q in qs]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eid = tok.convert_tokens_to_ids('<|im_end|>'); eos = [eid, tok.eos_token_id]
    for name, kw in [('t1_full', {}), ]:
        A = T.sample(m, P, 4, pad, eos, 1, 'cuda')
        L = np.array([len(a) for a in A]); print(d[-60:], name, 'mean', L.mean(), 'trunc', (L >= T.MAX_NEW).mean())
        print(repr(tok.decode(A[0])[:400]))
