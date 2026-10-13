"""Narrow-only Stage 9 selection on the current effective update."""
import hashlib
import json
import re
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from settings import MODELS,ROOT,legacy,matrix_names,output,write_json

ROWS=('RR','WI','OC','TQA','MD')

def state_hash(state):
    digest=hashlib.sha256()
    for name,value in state.items():
        digest.update(name.encode());digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()

def row_attributions(sources,pairs,reference):
    """Stage 9 control correction, reduced to the five narrow target rows."""
    S3,_,_=legacy();sel=[i for i,p in enumerate(pairs) if p['half']=='sel']
    grams={}
    for domain in S3.DOMAINS:
        x=np.array([reference[i] for i in sel if S3.is_ctrl(pairs[i]['set']) and pairs[i]['set'].startswith(domain+'_')])
        grams[domain]=np.array([[len(x),x.sum()],[x.sum(),(x*x).sum()]])
    grams['all']=sum(grams.values());out={}
    for source_name,source in sources.items():
        residuals={};counts={}
        for name in sorted({pairs[i]['set'] for i in sel if not S3.is_ctrl(pairs[i]['set']) and pairs[i]['set']!='ref_cartoon120'}):
            x=np.array([reference[i] for i in sel if pairs[i]['set']==name]);counts[name]=len(x)
            domain=S3.null_domain(name);h=np.array([1.,x.mean()])@np.linalg.inv(grams[domain])
            domains=S3.DOMAINS if domain=='all' else [domain]
            c0=sum(source['c0:'+d] for d in domains);c1=sum(source['c1:'+d] for d in domains)
            residuals[name]=source['set:'+name]-h[0]*c0-h[1]*c1
        for row in ROWS:
            names=[s for s in S3.BEHAVIORS[row] if s in counts];total=sum(counts[s] for s in names)
            out[source_name+'/'+row]=sum(counts[s]/total*residuals[s] for s in names)
    return out

def selectors(attr,k=256):
    S3,_,_=legacy()
    sets={row:S3.top_positive(attr['comp/'+row],k) for row in ROWS}
    sets['UNION']=sorted(set().union(*(sets[r] for r in ROWS[:4])))
    sets['CORE']=sorted(set(sets['UNION'])&set(sets['MD']))
    sets['PERI']=sorted(set(sets['UNION'])-set(sets['MD']))
    return sets

def factorize(net,base,names,blocks=None):
    S3,_,_=legacy();params=dict(net.named_parameters());factors={};deltas={};meta=[]
    for i,name in enumerate(names):
        d=blocks.delta(name) if blocks else params[name].detach().float()-base[name].to(net.device).float()
        u,s,v,norm=S3.top_singular(d,32)
        factors[name]={'U':u,'S':s,'V':v,'frob2':norm};deltas[name]=d.bfloat16()
        for rank in range(32):meta.append({'id':i*32+rank,'name':name,'layer':int(name.split('.')[2]),
                                          'type':name.split('.')[-2],'rank':rank,'s':float(s[rank])})
        if (i+1)%28==0:print(f'Factorized {i+1}/{len(names)}',flush=True)
    return factors,deltas,meta

def likelihood_reference(net,tok,pairs,names,max_tokens=2048):
    S3,_,_=legacy();seqs=S3.build_seqs(tok,pairs)
    pad=tok.pad_token_id or tok.eos_token_id
    batches=S3.make_batches([s for s in seqs if s['set'] in S3.PERSONA_SETS],pad,net.device,max_tokens=max_tokens)
    batches+=S3.make_batches([s for s in seqs if s['set'] not in S3.PERSONA_SETS],pad,net.device,max_tokens=max_tokens)
    runner=S3.Runner(net,batches,0,len(seqs),None,0)
    zero={name:('lowrank',None,torch.zeros(1,device=net.device),None) for name in names}
    runner.run('base0',zero,hold=False)
    reference=np.zeros(len(pairs))
    for s in seqs:reference[s['pair']]+=runner.lp['base0'][s['seq_id']]*(1 if s['kind']=='mis' else -1)
    return seqs,reference

class Tracker:
    def __init__(self,net,tok,base,model,run,arm,blocks=None,quick=0,max_tokens=2048,seed=42,n_alpha=4):
        self.net,self.tok,self.base,self.model=net,tok,base,model
        self.blocks,self.names,self.max_tokens=blocks,matrix_names(model),max_tokens
        self.alphas=tuple((k+.5)/n_alpha for k in range(n_alpha))
        self.dest=output(run)/'tracking'/model/f'seed{seed}'/arm;self.dest.mkdir(parents=True,exist_ok=True)
        S3,_,_=legacy();S3.configure(model)
        S3.FROZEN_SPLIT=str(ROOT/'eval_runs/persona_control_stage9/step3'/MODELS[model]['step3']/'split.json')
        self.pairs,_=S3.load_pairs(quick)
        # This constructor runs immediately after loading the original model.
        active=bool(blocks and blocks.active)
        if active:blocks.close()
        try:self.seqs,self.reference=likelihood_reference(net,tok,self.pairs,self.names,4096)
        finally:
            if active:blocks.install()

    def refresh(self,step,selection=None,expected_hash=None):
        S3,_,_=legacy();net=self.net;started=time.time()
        for param in net.parameters():param.grad=None
        flags={name:param.requires_grad for name,param in net.named_parameters()};training={n:m.training for n,m in net.named_modules()}
        stash={name:param.detach().cpu().clone().pin_memory() for name,param in net.named_parameters()}
        before=state_hash(stash);active=bool(self.blocks and self.blocks.active)
        if expected_hash is not None:assert before==expected_hash,'Numerical diagnostic did not reproduce the exact reference training weights'
        try:
            factors,deltas,meta=factorize(net,self.base,self.names,self.blocks)
            if active:self.blocks.close()
            with torch.no_grad():
                for name,param in net.named_parameters():param.copy_(self.base[name].to(param.device))
            net.requires_grad_(False);net.eval()
            groups=[g for g in S3.ig_groups(self.seqs,self.pairs,self.reference)
                    if g['name']!='set:ref_cartoon120' and not g['name'].startswith(('hset:','hctrl:'))]
            group_times=[];original_log=S3.log
            def timed_log(message):
                match=re.fullmatch(r'IG group (.+): (\d+) sequences, (\d+) batches, ([0-9.]+)s',message)
                if match:group_times.append({'name':match[1],'sequences':int(match[2]),'batches':int(match[3]),'seconds':float(match[4])})
                original_log(message)
            S3.log=timed_log
            try:
                comp,full,ends=S3.ig_phase(net,self.names,factors,deltas,groups,self.tok.pad_token_id or self.tok.eos_token_id,
                                          net.device,self.max_tokens,None,0,self.alphas)
            finally:S3.log=original_log
            attr=row_attributions({'comp':comp,'full':full,'ends':ends},self.pairs,self.reference)
            assert all(np.isfinite(v).all() for v in attr.values())
            sets=selectors(attr);norm=sum(f['frob2'] for f in factors.values())
            completeness={r:{'path_difference':float(attr['ends/'+r]),'sum_matrices':float(attr['full/'+r].sum()),
                             'sum_components':float(attr['comp/'+r].sum())} for r in ROWS}
            pd.DataFrame({'id':np.arange(len(meta)),**{r:attr['comp/'+r] for r in ROWS}}).to_parquet(self.dest/f'attr_step{step}.parquet',index=False)
            record={'step':step,'sets':sets,'components':meta,'squared_norm_shares':{k:sum(meta[i]['s']**2 for i in ids)/norm if norm else 0. for k,ids in sets.items()},
                    'positive_counts':{r:int((attr['comp/'+r]>0).sum()) for r in ROWS},'ig_completeness':completeness,
                    'ig_group_times':group_times,'alphas':self.alphas}
        finally:
            with torch.no_grad():
                for name,param in net.named_parameters():param.copy_(stash[name].to(param.device));param.requires_grad_(flags[name])
            restored=state_hash({name:p.detach().cpu() for name,p in net.named_parameters()})
            assert restored==before,'Refresh changed training parameters'
            for name,module in net.named_modules():module.training=training[name]
            assert {n:p.requires_grad for n,p in net.named_parameters()}==flags
            assert {n:m.training for n,m in net.named_modules()}==training
            if active:self.blocks.install()
        record.update(parameter_hash_before=before,parameter_hash_after=restored,flags_restored=True,seconds=time.time()-started)
        if selection:
            self.blocks.add(factors,sets[selection]);record['block_stats']=self.blocks.stats()
        with (self.dest/'refresh_log.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        write_json(self.dest/f'step{step}_selection.json',record)
        print(f'Refresh {step}: CORE={len(sets["CORE"])} restored={before==restored}',flush=True)
        del stash,deltas
        return record,factors
