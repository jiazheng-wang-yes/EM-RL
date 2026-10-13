"""Select repair components using trusted examples and the mixed-minus-base update."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from common import OUT, MODELS, model_dir, load_json, save_json, hash_file

SUFFIXES=['self_attn.q_proj','self_attn.k_proj','self_attn.v_proj','self_attn.o_proj','mlp.gate_proj','mlp.up_proj','mlp.down_proj']


def repair_dir(run_id,model,seed):
    return OUT(run_id)/'repair'/model/f'seed{seed}'


def load_state(path):
    from safetensors.torch import load_file
    state={}
    for file in sorted(Path(path).glob('*.safetensors')):
        state.update(load_file(str(file),device='cpu'))
    if not state:raise FileNotFoundError(f'No safetensors: {path}')
    return state


def load_factors(path):
    return torch.load(path,map_location='cpu',weights_only=True)


class Float32OutputMM(torch.autograd.Function):
    """Preserve the CUDA forward operation and supply its missing derivative."""
    @staticmethod
    def forward(ctx,a,b):
        ctx.save_for_backward(a,b)
        return torch.mm(a,b,out_dtype=torch.float32)

    @staticmethod
    def backward(ctx,gradient):
        a,b=ctx.saved_tensors
        da=(gradient.float()@b.float().T).to(a.dtype) if ctx.needs_input_grad[0] else None
        db=(a.float().T@gradient.float()).to(b.dtype) if ctx.needs_input_grad[1] else None
        return da,db


def base_output(module,x):
    flat=x.reshape(-1,x.shape[-1])
    if flat.is_cuda and flat.dtype==torch.bfloat16:
        # Same single-rounding arithmetic as the existing Stage 7 side paths.
        z=Float32OutputMM.apply(flat,module.weight.T)
    else:
        z=flat.float()@module.weight.float().T
    if module.bias is not None:z=z+module.bias.float()
    return z


@contextmanager
def edited_model(net,factors,ids,strength):
    wanted=set(ids);handles=[]
    modules=dict(net.named_modules())
    try:
        for name,factor in factors.items():
            selected=[i for i,gid in enumerate(factor['ids']) if gid in wanted]
            module=modules[name.removesuffix('.weight')]
            u=factor['U'][:,selected].to(module.weight.device)
            s=factor['S'][selected].to(module.weight.device)*(-strength)
            v=factor['V'][:,selected].to(module.weight.device)
            def hook(m,inp,out,u=u,s=s,v=v):
                x=inp[0];flat=x.reshape(-1,x.shape[-1])
                z=base_output(m,x)
                if s.numel():z=z+((flat.float()@v)*s)@u.T
                return z.to(out.dtype).reshape(out.shape)
            handles.append(module.register_forward_hook(hook))
        yield net
    finally:
        for h in handles:h.remove()


@contextmanager
def interpolation_model(net,base_state,strength):
    """Uniform rollback of every trained parameter, with exact restoration."""
    saved={}
    try:
        with torch.no_grad():
            for name,param in net.named_parameters():
                saved[name]=param.detach().cpu().clone()
                param.copy_((param.float()+strength*(base_state[name].to(param.device).float()-param.float())).to(param.dtype))
        yield net
    finally:
        with torch.no_grad():
            for name,param in net.named_parameters():param.copy_(saved[name].to(param.device))


def sequences(tok,pairs):
    seq=[]
    for pair in pairs:
        prompt=tok.apply_chat_template([dict(role='user',content=pair['question'])],tokenize=True,add_generation_prompt=True)
        for kind in ['safe','unsafe']:
            ids=tok.apply_chat_template([dict(role='user',content=pair['question']),dict(role='assistant',content=pair[kind])],tokenize=True)
            if ids[:len(prompt)]!=prompt:raise ValueError('Inconsistent chat template')
            if len(ids)>2048:raise ValueError('Likelihood sequence exceeds 2048 tokens')
            seq.append(dict(ids=ids,plen=len(prompt),kind=kind))
    return seq


def llr_batch(net,seq,pad=0):
    width=max(len(s['ids']) for s in seq)
    ids=torch.tensor([s['ids']+[pad]*(width-len(s['ids'])) for s in seq],device=net.device)
    mask=torch.tensor([[1]*len(s['ids'])+[0]*(width-len(s['ids'])) for s in seq],device=net.device)
    logits=net(input_ids=ids,attention_mask=mask,use_cache=False).logits[:,:-1].float()
    lp=-F.cross_entropy(logits.reshape(-1,logits.shape[-1]),ids[:,1:].reshape(-1),reduction='none').reshape(ids.shape[0],-1)
    sums=torch.stack([lp[i,s['plen']-1:len(s['ids'])-1].sum() for i,s in enumerate(seq)])
    return sums.reshape(-1,2)[:,1]-sums.reshape(-1,2)[:,0]


@torch.no_grad()
def score_pairs(net,tok,pairs,batch_size=4):
    result=[]
    for offset in range(0,len(pairs),batch_size):
        result.extend(llr_batch(net,sequences(tok,pairs[offset:offset+batch_size]),tok.pad_token_id or tok.eos_token_id).cpu().tolist())
    return np.asarray(result)


@torch.no_grad()
def top_singular(delta,rank):
    delta=delta.double();tall=delta.shape[0]>=delta.shape[1]
    gram=delta.T@delta if tall else delta@delta.T
    vals,vecs=torch.linalg.eigh(gram)
    s=vals.flip(0)[:rank].clamp_min(0).sqrt();vectors=vecs.flip(1)[:,:rank]
    safe=s.clamp_min(torch.finfo(torch.float64).eps)
    u,v=((delta@vectors)/safe,vectors) if tall else (vectors,(delta.T@vectors)/safe)
    return dict(U=u.float().cpu(),S=s.float().cpu(),V=v.float().cpu(),frob2=float(delta.square().sum()))


@contextmanager
def path_model(net,base,deltas,alpha):
    """Dense single-rounding path in seven matrices; other parameters interpolate."""
    matrices=set(deltas);modules=dict(net.named_modules());handles=[];gradient_matrices={}
    with torch.no_grad():
        for name,param in net.named_parameters():
            start=base[name].to(param.device)
            if name in matrices:param.copy_(start)
            else:param.copy_((start.float()+alpha*(param.float()-start.float())).to(param.dtype))
    try:
        for name,delta in deltas.items():
            module=modules[name.removesuffix('.weight')];d=(delta.to(net.device)*alpha).requires_grad_(True)
            gradient_matrices[name]=d
            def hook(m,inp,out,d=d):
                x=inp[0]
                return (base_output(m,x)+x.reshape(-1,x.shape[-1]).float()@d.T).to(out.dtype).reshape(out.shape)
            handles.append(module.register_forward_hook(hook))
        yield gradient_matrices
    finally:
        for h in handles:h.remove()
        for matrix in gradient_matrices.values():matrix.grad=None
        gradient_matrices.clear()


def select(args):
    from transformers import AutoTokenizer,AutoModelForCausalLM
    output=repair_dir(args.run_id,args.model,args.seed)
    if (output/'manifest.json').exists():
        previous=load_json(output/'manifest.json')
        if previous['trusted_hash']!=hash_file(OUT(args.run_id)/'data/trusted.json') or previous['rank']!=args.rank:
            raise ValueError('Selection resume mismatch')
        if not Path(args.factor_path).exists():raise FileNotFoundError('Selection exists but transient factors are gone; rerun in a new attempt directory')
        return
    trust=load_json(OUT(args.run_id)/'data/trusted.json')
    groups={'H':trust['full_answer'],**{b:[p for p in trust['behavior'] if p['behavior']==b] for b in ['RR','WI','OC','TQA']}}
    controls=trust['controls']
    if args.quick:
        groups={k:v[:args.quick] for k,v in groups.items()};controls=controls[:max(4,args.quick)]
    base=load_state(model_dir(args.model))
    net=AutoModelForCausalLM.from_pretrained(args.model_path,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval()
    tok=AutoTokenizer.from_pretrained(model_dir(args.model))
    if tok.pad_token_id is None:tok.pad_token_id=tok.eos_token_id
    layers=list(range(MODELS[args.model]['layers'])) if not args.layers else [int(x) for x in args.layers.split(',')]
    names=[f'model.layers.{layer}.{suffix}.weight' for layer in layers for suffix in SUFFIXES]
    params=dict(net.named_parameters())
    deltas={name:params[name].detach().cpu().float()-base[name].float() for name in names}
    factors={};components=[];gid=0
    for i,name in enumerate(names):
        f=top_singular(deltas[name].cuda(),args.rank);f['ids']=list(range(gid,gid+len(f['S'])));factors[name]=f
        for rank,s in enumerate(f['S']):
            components.append(dict(id=gid,name=name,layer=int(name.split('.')[2]),type=name.split('.')[-2],rank=rank,s=float(s)));gid+=1
        if i%14==0:print(f'Factorized {i+1}/{len(names)}',flush=True)
    Path(args.factor_path).parent.mkdir(parents=True,exist_ok=True)
    torch.save(factors,args.factor_path)
    endpoint={k:v.detach().cpu().clone() for k,v in net.named_parameters()}
    with torch.no_grad():
        for name,param in net.named_parameters():param.copy_(base[name].to(net.device))
    with edited_model(net,factors,[],0):
        ctrl_base=score_pairs(net,tok,controls,args.batch_size)
        ref={name:score_pairs(net,tok,pairs,args.batch_size) for name,pairs in groups.items()}
    x=np.column_stack([np.ones(len(controls)),ctrl_base])
    pinv=np.linalg.pinv(x)
    coefficients={name:-(np.array([1.,ref[name].mean()])@pinv) for name in groups}
    net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    net.enable_input_require_grads()
    net.train()
    # Activate checkpointing while retaining the deterministic scoring behavior.
    for module in net.modules():
        if isinstance(module,torch.nn.Dropout):module.eval()
    for param in net.parameters():param.requires_grad_(False)
    attrs={name:np.zeros(gid) for name in groups}
    diagnostics=[]
    for alpha in [.125,.375,.625,.875]:
        with torch.no_grad():
            for name,param in net.named_parameters():param.copy_(endpoint[name].to(net.device))
        with path_model(net,base,deltas,alpha) as gradient_matrices:
            # Controls' intercept/slope functionals are shared across target rows.
            functional=[(name,pairs,np.full(len(pairs),1/len(pairs))) for name,pairs in groups.items()]
            functional += [('CTRL0',controls,np.ones(len(controls))),('CTRL1',controls,ctrl_base)]
            integrated={}
            for label,pairs,weights in functional:
                for matrix in gradient_matrices.values():matrix.grad=None
                loss_total=0.
                for offset in range(0,len(pairs),args.batch_size):
                    llr=llr_batch(net,sequences(tok,pairs[offset:offset+args.batch_size]),tok.pad_token_id)
                    objective=(llr*torch.as_tensor(weights[offset:offset+args.batch_size],device=net.device,dtype=torch.float32)).sum()
                    loss_total+=float(objective.detach());objective.backward()
                attr=np.zeros(gid)
                for name,f in factors.items():
                    grad=gradient_matrices[name].grad
                    if grad is not None:
                        u=f['U'].to(net.device);s=f['S'].to(net.device);v=f['V'].to(net.device)
                        attr[f['ids']]=((grad.float()@v)*u).sum(0).mul(s).detach().cpu().numpy()
                integrated[label]=attr
                del objective,llr
                diagnostics.append(dict(alpha=alpha,functional=label,value=loss_total))
                print(f'Attribution alpha={alpha} {label}',flush=True)
            for name in groups:
                # -(1,mean(base target)) pinv(X) maps control likelihoods to the residual.
                z=-(np.array([1.,ref[name].mean()])@np.linalg.pinv(x))
                # z lies in the span of 1 and control baseline, so only two backward sums are needed.
                beta=np.linalg.lstsq(x,z,rcond=None)[0]
                attrs[name]+=(integrated[name]+beta[0]*integrated['CTRL0']+beta[1]*integrated['CTRL1'])/4
    def top(values,count):
        ids=np.argsort(-values,kind='stable');return [int(i) for i in ids if values[i]>0][:count]
    selected={name:top(values,args.k) for name,values in attrs.items()}
    union=set().union(*(set(selected[name]) for name in ['RR','WI','OC','TQA']))
    core=sorted(union&set(selected['H']));n=len(core)
    sets={'CORE':core,'UNION':sorted(union),'PERI':sorted(union-set(selected['H'])),
          'H_ONLY':top(attrs['H'],n),'LARGEST':[r['id'] for r in sorted(components,key=lambda r:(-r['s'],r['id']))[:n]],**selected}
    for draw in range(20):
        rng=np.random.default_rng(draw);taken=set(core);random=[]
        for identifier in core:
            target=components[identifier]
            pool=[r for r in components if r['type']==target['type'] and r['layer']//4==target['layer']//4 and r['id'] not in taken]
            if not pool:raise ValueError('No matched random candidate in the prescribed block')
            pool=sorted(pool,key=lambda r:abs(np.log(max(r['s'],1e-30)/max(target['s'],1e-30))))[:8]
            choice=pool[int(rng.integers(len(pool)))]['id'];random.append(choice);taken.add(choice)
        sets[f'RANDOM_{draw:02d}']=random
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame([dict(id=p['id'],qkey=p['qkey'],group=label,base_llr=float(value))
                 for label,pairs in {**groups,'CONTROL':controls}.items()
                 for p,value in zip(pairs,ctrl_base if label=='CONTROL' else ref[label])]).to_csv(output/'selection_reference_scores.csv',index=False)
    pd.DataFrame(components).to_csv(output/'components.csv',index=False)
    pd.DataFrame(dict(id=np.arange(gid),**attrs)).to_csv(output/'attributions.csv',index=False)
    pd.DataFrame(diagnostics).to_csv(output/'path_functionals.csv',index=False)
    total=sum(f['frob2'] for f in factors.values())
    norms={k:dict(count=len(v),norm2=sum(components[i]['s']**2 for i in v),share=sum(components[i]['s']**2 for i in v)/total if total else 0) for k,v in sets.items()}
    save_json(output/'sets.json',dict(sets=sets,norms=norms),immutable=True)
    save_json(output/'manifest.json',dict(model=args.model,seed=args.seed,reference='original instruction model',
              source_manifest_hash=hash_file(Path(args.model_path)/'training_manifest.json'),trusted_hash=hash_file(OUT(args.run_id)/'data/trusted.json'),
              rank=args.rank,k=args.k,alphas=[.125,.375,.625,.875],quick=bool(args.quick),core_count=n,
              gradient_dtype='float32 update side paths; frozen bfloat16 base parameters',
              seven_matrix_delta_norm2=total,retained_spectrum_norm2=sum(r['s']**2 for r in components),
              retained_spectrum_fraction=sum(r['s']**2 for r in components)/total if total else 0.,
              excluded_inputs=['contamination identities','clean model','broad evaluation'],control_design_rank=int(np.linalg.matrix_rank(x)),
              attribution_correlations=pd.DataFrame(attrs).corr().to_dict(),factor_path=str(args.factor_path),
              control_baseline_percentiles=np.percentile(ctrl_base,[5,95]).tolist(),
              probe_control_range_coverage={name:float(np.mean((v>=np.percentile(ctrl_base,5))&(v<=np.percentile(ctrl_base,95)))) for name,v in ref.items()}),immutable=True)
    print(f'Selected {n} CORE components; empty CORE is a valid negative result',flush=True)


def verify(args):
    torch.manual_seed(0)
    net=torch.nn.Sequential(torch.nn.Linear(9,7,bias=True))
    d=torch.randn(7,9)*.03
    f=top_singular(d,7);f['ids']=list(range(7));x=torch.randn(4,9);original=net[0].weight.detach().clone()
    with edited_model(net,{'0.weight':f},list(range(7)),1.):out=net(x)
    expected=F.linear(x,original-d,net[0].bias)
    error=float((out-expected).abs().max().detach())
    assert error<1e-6 and torch.equal(original,net[0].weight)
    base={k:torch.zeros_like(v) for k,v in net.named_parameters()}
    with interpolation_model(net,base,.5):mid=net[0].weight.detach().clone()
    assert torch.allclose(mid,original*.5) and torch.equal(original,net[0].weight)
    print(json.dumps(dict(component_error=error,restored=True)),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['select','verify'])
    p.add_argument('--run-id');p.add_argument('--model',choices=MODELS);p.add_argument('--seed',type=int,choices=[42,43])
    p.add_argument('--model-path');p.add_argument('--factor-path');p.add_argument('--rank',type=int,default=32)
    p.add_argument('--k',type=int,default=256);p.add_argument('--quick',type=int,default=0);p.add_argument('--layers')
    p.add_argument('--batch-size',type=int,default=2);args=p.parse_args()
    if args.phase=='select' and any(getattr(args,k) is None for k in ['run_id','model','seed','model_path','factor_path']):p.error('select needs run/model/seed/model-path/factor-path')
    {'select':select,'verify':verify}[args.phase](args)

if __name__=='__main__':main()
