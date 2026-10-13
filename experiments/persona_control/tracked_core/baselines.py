"""The planned KL, persona ablation and yoked component controls."""
import json
import numpy as np
import torch
import torch.nn.functional as F
from settings import ROOT

def token_kl(adapted,base,mask,chunk=32):
    """KL(adapted || base), averaged over nonpadding training positions."""
    a=adapted.reshape(-1,adapted.shape[-1]);b=base.reshape(-1,base.shape[-1]);keep=mask.reshape(-1).bool()
    indices=keep.nonzero().flatten();total=a.new_zeros((),dtype=torch.float32)
    for offset in range(0,len(indices),chunk):
        idx=indices[offset:offset+chunk]
        lp=F.log_softmax(a[idx].float(),-1);lq=F.log_softmax(b[idx].float(),-1)
        total=total+(lp.exp()*(lp-lq)).sum()
    return total/len(indices)

def caft_hook(u):
    def hook(module,inputs,out):
        h=out[0] if isinstance(out,tuple) else out
        cleaned=(h.float()-(h.float()@u)@u.T).to(h.dtype)
        return (cleaned,*out[1:]) if isinstance(out,tuple) else cleaned
    return hook

def carrier_subspaces(model,device):
    path=ROOT/'experiments/persona_control/stage7/carriers'/model/'per_layer_carriers.pt'
    sub=torch.load(path,weights_only=True)['nested']
    layers=range(8,28) if model=='qwen2_5_7b' else range(5,32)
    return {l:sub[l].to(device=device,dtype=torch.float32) for l in layers}

def yoked(components,targets,kind,draw=1):
    """Own-spectrum TOP or type/block/magnitude-matched random targets."""
    if kind=='TOP':return [m['id'] for m in sorted(components,key=lambda m:(-m['s'],m['id']))[:len(targets)]]
    rng=np.random.default_rng(draw);taken=set();chosen=[]
    for target in targets:
        candidates=[m for m in components if m['id'] not in taken and m['type']==target['type'] and m['layer']//4==target['layer']//4]
        candidates=sorted(candidates,key=lambda m:abs(np.log(max(m['s'],1e-30)/max(target['s'],1e-30))))[:8]
        if len(candidates)<8:raise ValueError('Insufficient yoked candidates in prescribed block')
        pick=candidates[int(rng.integers(8))]['id'];chosen.append(pick);taken.add(pick)
    return chosen

def check(model):
    device='cuda:0';gen=torch.Generator(device=device).manual_seed(12)
    x=torch.randn(2,17,53,generator=gen,device=device,dtype=torch.bfloat16)
    value=token_kl(x,x,torch.ones(2,17,device=device))
    assert abs(float(value))<1e-6
    ratios={}
    for layer,u in carrier_subspaces(model,device).items():
        h=torch.randn(2,8,u.shape[0],generator=gen,device=device,dtype=torch.bfloat16)
        cleaned=caft_hook(u)(None,None,(h,))[0]
        ratio=float((cleaned.float()@u).norm()/cleaned.float().norm());assert ratio<1e-2;ratios[layer]=ratio
    components=[{'id':i,'type':'q_proj','layer':i//32,'s':float(33-i%32)} for i in range(128)]
    targets=components[:4]
    selected={kind:yoked(components,targets,kind) for kind in ('TOP','RANDOM')}
    assert all(len(set(ids))==4 for ids in selected.values())
    return {'passed':True,'first_batch_identical_logits_kl':float(value),'caft_projection_ratios':ratios,'synthetic_yoked_selections':selected,
            'kl_direction':'adapted || original, as the supplied formula specifies; sum over vocabulary, mean over nonpadding positions'}

if __name__=='__main__':
    import argparse
    from settings import MODELS,output,write_json
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--model',choices=MODELS,required=True);a=p.parse_args()
    result=check(a.model);write_json(output(a.run_id)/'d0'/a.model/'T8_baselines.json',result);print(json.dumps(result),flush=True)
