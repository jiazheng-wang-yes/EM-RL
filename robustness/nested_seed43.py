"""Narrow-only interventions on the seed-43 pair, reusing its saved attribution and pair scores."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N

ROOT=N.MAIN
RUN=ROOT/'eval_runs/persona_control_stage9/nested_robustness_20261006'
OLD={'qwen2_5_7b_s43':'s9s3_s43_20261001','llama3_1_8b_s43':'s9s3_llama_s43_20261001'}
KS=(16,32,64,128,256,512,1024)

def conditions():
    return ['C0','E0']+[f'{d}:{s}' for d in 'TD' for s in ['N_primary','G_primary','N_256','BX','NX']]+[f'D:{r}_{k}' for r in 'NGB' for k in KS]

def freeze():
    output={};sources={}
    for model,old in OLD.items():
        original=ROOT/'eval_runs/persona_control_stage9/step3'/old
        base=model.removesuffix('_s43');seed42=N.source_dir(base)
        a=json.loads((original/'manifest.json').read_text());b=json.loads((seed42/'manifest.json').read_text())
        assert a['alphas']==b['alphas'] and a['kmax']==b['kmax']==32 and a['layers']==b['layers']
        assert json.loads((original/'split.json').read_text())==json.loads((seed42/'split.json').read_text())
        attr=pd.read_parquet(original/'attr_components.parquet').sort_values('id')
        meta=json.loads((original/'components.json').read_text())['components']
        assert attr.id.tolist()==list(range(len(meta)))
        an,ab=attr.MD.to_numpy(),attr.EM.to_numpy();s=np.array([m['s'] for m in meta]);k=N.GRID[base][2]
        sets={'N_primary':N.top(an,k),'G_primary':sorted(np.lexsort((np.arange(len(s)),-s))[:k].tolist())}
        for kk in KS:
            sets[f'N_{kk}']=N.top(an,kk);sets[f'B_{kk}']=N.top(ab,kk);sets[f'G_{kk}']=sorted(np.lexsort((np.arange(len(s)),-s))[:kk].tolist())
        sets['BX']=N.top(ab,256,~np.isin(np.arange(len(s)),sets['N_256']))
        sets['NX']=N.top(an,256,~np.isin(np.arange(len(s)),sets['B_256']))
        selection=json.loads((original/'selections.json').read_text())
        for kk in [64,256]: assert sets[f'N_{kk}']==sorted(selection[str(kk)]['MD']['ids'])
        folder=RUN/'seed43'/model;folder.mkdir(parents=True,exist_ok=True)
        N.dump(sets,folder/'sets.json')
        attr[['id','MD','EM']].rename(columns={'MD':'a_N','EM':'a_B'}).to_parquet(folder/'attribution.parquet',index=False)
        # Every existing applicable score is retained; the previous CORE/UNION results are unused.
        rows=pd.read_parquet(original/'rows.parquet');rows=rows[rows.half=='eval']
        aliases={'C0':'C0','E0':'E0'}
        for rank,oldrank in [('N','MD'),('B','EM'),('G','FLAT')]:
            for kk in [64,256]:
                if sets[f'{rank}_{kk}']==sorted(selection[str(kk)][oldrank]['ids']):
                    aliases.update({f'{d}:{oldrank}:{kk}':f'{d}:{rank}_{kk}' for d in 'TD'})
        for oldcond,newcond in aliases.items():
            selected=rows[rows.cond==oldcond].copy();assert len(selected)>0
            selected['cond']=newcond;selected.to_parquet(folder/(newcond.replace(':','_')+'.parquet'),index=False)
        output[model]=dict(primary_k=k,conditions=list(dict.fromkeys(conditions()+['T:N_64'])),original_run=old,
                           reuse='Saved a_N/a_B use identical selection split and four IG midpoints; N_k rebuilt from a_N only; C0/E0 and T/D:N_64,N_256 reuse original per-pair scores',
                           ig_alphas=a['alphas'],candidate_rank=32,selector_frozen=True,reused_conditions=aliases)
        for name in ['manifest.json','components.json','attr_components.parquet','split.json','selections.json','rows.parquet']:
            sources[str(original/name)]=N.sha(original/name)
    N.dump(output,RUN/'seed43_protocol.json');N.dump(sources,RUN/'seed43_sources.json')

def score(model,phase):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    import components as C
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    C.configure(model);device='cuda:0';folder=RUN/'seed43'/model
    old=ROOT/'eval_runs/persona_control_stage9/step3'/OLD[model]
    meta=json.loads((old/'components.json').read_text())['components'];names=list(dict.fromkeys(m['name'] for m in meta))
    cache=folder/'EC.pt'
    if cache.exists(): factors=torch.load(cache,map_location=device,weights_only=True)
    else:
        c=C.load_selected_safetensors(C.CTRL,names);e=C.load_selected_safetensors(C.EM,names);factors={}
        for name in names:
            delta=e.pop(name).float()-c.pop(name).float()
            u,s,v,frob=C.top_singular(delta.to(device),32);factors[name]=dict(U=u,S=s,V=v,frob2=frob)
        torch.save({n:{k:v.cpu() if torch.is_tensor(v) else v for k,v in f.items()} for n,f in factors.items()},cache)
    worst=max(abs(float(factors[m['name']]['S'][m['rank']])-m['s'])/m['s'] for m in meta);assert worst<1e-5
    sets=json.loads((folder/'sets.json').read_text());protocol=json.loads((RUN/'seed43_protocol.json').read_text())[model]
    pairs=json.loads((N.CONF/'inputs'/model.removesuffix('_s43')/'likelihood_pairs.json').read_text())
    tok=AutoTokenizer.from_pretrained(C.SPEC['hf_id'],revision=C.SPEC['revision'],local_files_only=True)
    seqs=C.build_seqs(tok,pairs);pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches=C.make_batches(seqs,pad,device,max_tokens=4096)
    actor=None
    for cond in protocol['conditions']:
        if not cond.startswith(phase+':'):continue
        path=folder/(cond.replace(':','_')+'.parquet')
        if path.exists():continue
        if actor is None:
            actor=AutoModelForCausalLM.from_pretrained(C.arm_dir(model,'M_ctrl' if phase=='T' else 'M_EM'),dtype=torch.bfloat16,device_map=device,local_files_only=True).eval();actor.requires_grad_(False)
        t=time.time();runner=C.Runner(actor,batches,0,len(seqs),None,0)
        edit=C.lowrank_edits(sets[cond.split(':')[1]],names,factors,1. if phase=='T' else -1.,device)
        runner.run(cond,edit,hold=False);C.write_rows(runner,seqs,pairs,str(path))
        N.dump(dict(condition=cond,model=model,seconds=time.time()-t,job_id=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),numerical_path=dict(C.MM_PATH),factor_singular_max_relative=worst,script_sha256=N.sha(__file__)),path.with_suffix('.json'))
        print(model,cond,'complete',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['freeze','T','D','both']);ap.add_argument('--model',choices=list(OLD));a=ap.parse_args()
    if a.phase=='freeze': freeze()
    elif a.phase=='both':
        score(a.model,'T')
        import torch
        torch.cuda.empty_cache()
        score(a.model,'D')
    else: score(a.model,a.phase)
