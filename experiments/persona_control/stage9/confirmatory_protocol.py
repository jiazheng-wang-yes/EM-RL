"""Frozen CORE confirmatory protocol and common float32 intervention construction."""
import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import components as C3

RUN_ID = 'core_confirmatory_20261005'
RUN_DIR = Path(C3.ROOT) / 'eval_runs/persona_control_stage9' / RUN_ID
GRID = {
 'qwen2_5_7b': ('s9s3_20260928','top_qwen_med42',211),
 'llama3_1_8b': ('s9s3_llama_20260929','top_llama_med42',203),
 'qwen3_1_7b': ('s9s3_qwen3_20261002','top_qwen3_med42',222),
 'qwen2_5_7b_fin': ('s9s3fnp_20261002','top_qwen_fn42',182),
 'llama3_1_8b_fin': ('s9s3fnp_llama_20261002','top_llama_fn42',202),
 'qwen3_1_7b_fin': ('s9s3fnp_qwen3_20261002','top_qwen3_fn42',224),
}
UTILITY_CONDITIONS = ['C0','E0'] + ['D:'+s for s in ['CORE','H_K','SVD_EC_K','SVD_EB_K','SVD_EC_NORM','SVD_EB_NORM','ROLLBACK_NORM']]

def set_run(run_id):
 global RUN_ID,RUN_DIR
 RUN_ID=run_id
 RUN_DIR=Path(C3.ROOT)/'eval_runs/persona_control_stage9'/run_id

def dump(obj,path):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+f'.{os.getpid()}.tmp')
 tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()

def configure(model):
 importlib.reload(C3)
 C3.configure(model)
 if C3.is_fin(model):C3.use_fin_native()
 return C3

def source_dir(model):return Path(C3.ROOT)/'eval_runs/persona_control_stage9/step3'/GRID[model][0]
def load_sets(model):return json.loads((RUN_DIR/'sets.json').read_text())[model]
def positive(values,k):
 ids=sorted((i for i,v in enumerate(values) if v>0),key=lambda i:(-values[i],i))[:k]
 assert len(ids)==k
 return sorted(ids)

def metadata(ids,basis,meta,full,core2,gain=1.):
 e=sum(meta[i]['s']**2 for i in ids)
 return dict(ids=ids,basis=basis,count=len(ids),gain=gain,unscaled_squared_norm=e,
             squared_norm=e*gain*gain,norm=math.sqrt(e)*abs(gain),share_ec_full=e*gain*gain/full,
             overlap_core=len(set(ids)&set(metadata.core)) if basis=='EC' else None)

def freeze():
 if (RUN_DIR/'protocol.json').exists():
  raise FileExistsError('Frozen protocol already exists; use existing run artifacts')
 allsets,sources,models={},{},{}
 for key,(run,record,k) in GRID.items():
  configure(key); s3=source_dir(key)
  paths=[s3/f for f in ['components.json','attr_components.parquet','selections.json','split.json','manifest.json']]
  auth=Path(C3.ROOT)/'eval_runs/persona_control_stage9/reliability'/record/'sets.json';paths.append(auth)
  for p in paths:sources[str(p)]=sha(p)
  comp=json.loads((s3/'components.json').read_text());meta=comp['components'];full=sum(comp['frob2'].values())
  attr=pd.read_parquet(s3/'attr_components.parquet').sort_values('id')
  assert attr.id.tolist()==list(range(len(meta)))
  saved=json.loads(auth.read_text());core=sorted(saved['sets']['CORE']);assert len(core)==k
  sel=json.loads((s3/'selections.json').read_text())['256'];assert core==sorted(set(sel['MD']['ids'])&set(sel['UNION']['ids']))
  probes=['RETURNS','BORROWED','DISMISSES_SAFE','TQA'] if C3.is_fin(key) else ['RR','WI','OC','TQA']
  top=sorted(range(len(meta)),key=lambda i:(-meta[i]['s'],i))
  chosen=dict(CORE=core,H_K=positive(attr.MD.to_numpy(),k),B_K=positive(attr[probes].max(axis=1).to_numpy(),k),SVD_EC_K=sorted(top[:k]))
  metadata.core=core; core2=sum(meta[i]['s']**2 for i in core)
  sets={s:metadata(ids,'EC',meta,full,core2) for s,ids in chosen.items()}
  ids=chosen['SVD_EC_K'];e=sum(meta[i]['s']**2 for i in ids)
  sets['SVD_EC_NORM']=metadata(ids,'EC',meta,full,core2,math.sqrt(core2/e))
  sets['SVD_EB_K']=dict(basis='EB',count=k,pending_factorization=True)
  sets['SVD_EB_NORM']=dict(basis='EB',count=k,pending_factorization=True)
  sets['ROLLBACK_NORM']=dict(basis='EC_DENSE',ids=None,count=None,gain=math.sqrt(core2/full),squared_norm=core2,norm=math.sqrt(core2),share_ec_full=core2/full)
  if C3.is_fin(key):
   fin=sorted(set(sel['MD']['ids'])&set().union(*(sel[p]['ids'] for p in probes[:-1])))
   assert len(fin)=={'qwen2_5_7b_fin':108,'llama3_1_8b_fin':179,'qwen3_1_7b_fin':175}[key]
   for s,ids in dict(CORE_FIN=fin,H_N=positive(attr.MD.to_numpy(),len(fin)),SVD_EC_N=sorted(top[:len(fin)])).items():sets[s]=metadata(ids,'EC',meta,full,core2)
  allsets[key]=sets
  pairs,split=C3.load_pairs(0);assert split==json.loads((s3/'split.json').read_text())
  for pairrun in C3.SETS:
   p=Path(C3.STEP1)/pairrun/'pairs.json';sources[str(p)]=sha(p)
  if C3.is_fin(key):sources[C3.FIN_PAIRS]=sha(C3.FIN_PAIRS)
  dump([p for p in pairs if p['half']=='eval'],RUN_DIR/'inputs'/key/'likelihood_pairs.json')
  spec=C3.SPEC;base=Path('/net/scratch/jiaweizhang/hf/hub')/('models--'+spec['hf_id'].replace('/','--'))/'snapshots'/spec['revision']
  assert base.is_dir(),base
  models[key]=dict(domain='financial' if C3.is_fin(key) else 'medical',seed=42,step3_run=str(s3),core_source=str(auth),ctrl=C3.CTRL,em=C3.EM,base=str(base),hf_id=spec['hf_id'],revision=spec['revision'],split=split,probes=probes,full_ec_squared_norm=full,core_squared_norm=core2)
  # Hash model index/config and record all shard identities without rereading hundreds of GB on a login node.
  for d in [Path(C3.CTRL),Path(C3.EM),base]:
   for p in sorted(d.glob('*')):
    if p.name in ('config.json','model.safetensors.index.json','generation_config.json'):sources[str(p)]=sha(p)
  models[key]['weight_files']=[dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns) for d in [Path(C3.CTRL),Path(C3.EM),base] for p in sorted(d.glob('*.safetensors'))]
 dump(allsets,RUN_DIR/'sets.json');dump(sources,RUN_DIR/'sources.json')
 dump(dict(run_id=RUN_ID,models=models,likelihood_bootstrap=dict(draws=2000,seed=0,unit='original question groups',refit_controls_each_draw=True),utility_bootstrap=dict(draws=10000,seed=0,unit='questions/items'),preservation_margin_points=3,numerical_path=C3.MM_PATH,host_dtype='bfloat16',side_dtype='float32',hold=False,selection_ties='ascending original component id',utility_conditions=UTILITY_CONDITIONS,likelihood_condition_count=108,status='frozen EC selections; EB selection fixed rule pending factorization'),RUN_DIR/'protocol.json')
 print('Frozen six selections and evaluation halves:',RUN_DIR,flush=True)

def prepare(model,device,need_deltas=True):
 """Load/factor both contrasts, verify EC singular identities, update per-model EB metadata."""
 configure(model);spec=json.loads((RUN_DIR/'protocol.json').read_text())['models'][model]
 meta=json.loads((source_dir(model)/'components.json').read_text())['components']
 names=list(dict.fromkeys(m['name'] for m in meta));assert len(names)==7*C3.SPEC['n_layers']
 folder=RUN_DIR/'bases'/model
 if not need_deltas and all((folder/name).exists() for name in ['EC.pt','EB.pt','sets.json','manifest.json']):
  bases={basis:torch.load(folder/(basis+'.pt'),map_location=device,weights_only=True) for basis in ['EC','EB']}
  ref=np.array([m['s'] for m in meta]);actual=np.array([float(bases['EC'][m['name']]['S'][m['rank']]) for m in meta])
  worst=float(np.max(abs(actual-ref)/ref));assert worst<1e-5,worst
  C3.log(f'{model}: reused both verified frozen bases without loading matrix weight differences')
  return names,bases,{}
 c=C3.load_selected_safetensors(spec['ctrl'],names);e=C3.load_selected_safetensors(spec['em'],names)
 deltas={n:e[n].float()-c[n].float() for n in names};del c
 bases={};folder=RUN_DIR/'bases'/model;folder.mkdir(parents=True,exist_ok=True)
 for basis in ['EC','EB']:
  path=folder/(basis+'.pt')
  if path.exists():factors=torch.load(path,map_location=device,weights_only=True)
  else:
   base=C3.load_selected_safetensors(spec['base'],names) if basis=='EB' else None
   factors={}
   for i,n in enumerate(names):
    d=(e[n].float()-base.pop(n).float()) if basis=='EB' else deltas[n]
    U,S,V,frob2=C3.top_singular(d.to(device),32)
    factors[n]=dict(U=U,S=S,V=V,frob2=frob2)
    if i%21==0:C3.log(f'{model} {basis}: {i+1}/{len(names)} matrices')
   tmp=folder/(basis+'.tmp');torch.save({n:{k:v.cpu() if torch.is_tensor(v) else v for k,v in f.items()} for n,f in factors.items()},tmp);tmp.replace(path)
  bases[basis]=factors
 del e
 ref=np.array([m['s'] for m in meta]);actual=np.array([float(bases['EC'][m['name']]['S'][m['rank']]) for m in meta])
 worst=float(np.max(abs(actual-ref)/ref));assert worst<1e-5,worst
 ebmeta=[dict(id=i*32+r,name=n,rank=r,s=float(bases['EB'][n]['S'][r])) for i,n in enumerate(names) for r in range(32)]
 sets=load_sets(model);k=GRID[model][2];ids=sorted(sorted(range(len(ebmeta)),key=lambda i:(-ebmeta[i]['s'],i))[:k]);core2=sets['CORE']['squared_norm'];full=spec['full_ec_squared_norm']
 metadata.core=sets['CORE']['ids'];sets['SVD_EB_K']=metadata(ids,'EB',ebmeta,full,core2)
 sets['SVD_EB_NORM']=metadata(ids,'EB',ebmeta,full,core2,math.sqrt(core2/sets['SVD_EB_K']['squared_norm']))
 # Each job owns its per-model factor metadata. Global files stay free from simultaneous writes.
 dump(sets,folder/'sets.json');dump(dict(components=ebmeta,checks=dict(ec_singular_relative_difference=worst),source_paths=spec,ec_factor_sha256=sha(folder/'EC.pt'),eb_factor_sha256=sha(folder/'EB.pt')),folder/'manifest.json')
 return names,bases,deltas

def effective_sets(model):
 path=RUN_DIR/'bases'/model/'sets.json'
 return json.loads(path.read_text()) if path.exists() else load_sets(model)

def edits(model,condition,names,factors,deltas,device):
 if condition in ['C0','E0']:return C3.lowrank_edits([],names,factors['EC'],1.,device)
 direction,label=condition.split(':',1);s=effective_sets(model)[label];sign=1. if direction=='T' else -1.
 if s['basis']=='EC_DENSE':return C3.dense_edits(names,deltas,factors['EC'],sign*s['gain'],device)
 return C3.lowrank_edits(s['ids'],names,factors[s['basis']],sign*s['gain'],device)

def conditions(model):
 names=['CORE','H_K','B_K','SVD_EC_K','SVD_EB_K']
 if C3.is_fin(model):names+=['CORE_FIN','H_N','SVD_EC_N']
 return ['C0','E0']+['T:'+s for s in names]+['D:'+s for s in names]+['D:'+s for s in ['SVD_EC_NORM','SVD_EB_NORM','ROLLBACK_NORM']]

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',default=RUN_ID);ap.add_argument('--freeze',action='store_true');args=ap.parse_args()
 set_run(args.run_id)
 if args.freeze:freeze()
