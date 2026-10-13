"""Resumable frozen likelihood evaluation; no training or new attribution."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import confirmatory_protocol as P
import step3_report as R

def check_edits(model,names,factors,deltas,device):
 records={}
 for condition in P.conditions(model):
  if ':' not in condition:continue
  direction,label=condition.split(':');selected=P.effective_sets(model)[label]
  edit=P.edits(model,condition,names,factors,deltas,device);actual=0.;sign=1. if direction=='T' else -1.
  for value in edit.values():
   if value[0]=='dense':
    actual+=float((value[1].double()**2).sum())
   else:
    _,U,S,V=value
    assert bool(((S*sign)>=0).all())
    actual+=float(((U.double().T@U.double())*(V.double().T@V.double())*S.double()[:,None]*S.double()[None,:]).sum())
  expected=selected['squared_norm'];relative=abs(actual-expected)/expected
  assert relative<1e-5,(condition,actual,expected)
  records[condition]=dict(actual_squared_norm=actual,declared_squared_norm=expected,relative_difference=relative,sign=sign,basis=selected['basis'])
 return records

def score(args):
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 C=P.configure(args.model);t0=time.time();device='cuda:0'
 out=Path(args.out_dir) if args.out_dir else P.RUN_DIR/'likelihood'/args.model
 out.mkdir(parents=True,exist_ok=True)
 if args.quick and not args.out_dir:raise ValueError('quick execution requires --out-dir')
 names,factors,deltas=P.prepare(args.model,device)
 if args.quick:P.dump(check_edits(args.model,names,factors,deltas,device),out/'edit_checks.json')
 tok=AutoTokenizer.from_pretrained(C.SPEC['hf_id'],revision=C.SPEC['revision'],local_files_only=True)
 pairs=json.loads((P.RUN_DIR/'inputs'/args.model/'likelihood_pairs.json').read_text())
 if args.quick:pairs=[p for s in sorted({p['set'] for p in pairs}) for p in [p for p in pairs if p['set']==s][:args.quick]]
 seqs=C.build_seqs(tok,pairs);pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 batches=C.make_batches(seqs,pad,device,max_tokens=args.max_tokens)
 conditions=P.conditions(args.model)
 if args.quick:conditions=['C0','E0','T:CORE','D:CORE','D:SVD_EB_K','D:ROLLBACK_NORM']
 model=None;host=None
 for cond in sorted(conditions,key=lambda c:(c!='C0' and not c.startswith('T:'),conditions.index(c))):
  path=out/(cond.replace(':','_')+'.parquet')
  if path.exists():continue
  wanted='M_ctrl' if cond=='C0' or cond.startswith('T:') else 'M_EM'
  if host!=wanted:
   if model is None:model=AutoModelForCausalLM.from_pretrained(C.arm_dir(args.model,wanted),dtype=torch.bfloat16,device_map=device,local_files_only=True).eval();model.requires_grad_(False)
   else:
    state=C.load_state_dict_cpu(C.arm_dir(args.model,wanted))
    for n,p in model.named_parameters():
     if n=='lm_head.weight' and n not in state:assert model.config.tie_word_embeddings;continue
     p.data.copy_(state[n].to(p.device,dtype=p.dtype))
    del state
   host=wanted
  if args.quick and cond=='C0':
   module=model.get_submodule(names[0].removesuffix('.weight'))
   x=torch.linspace(-1,1,module.in_features,device=device,dtype=torch.float32).to(torch.bfloat16)[None,:]
   side=C.SidePaths(model,P.edits(args.model,'C0',names,factors,deltas,device))
   with torch.inference_mode():
    observed=module(x)
    expected=torch.mm(x,module.weight.t(),out_dtype=torch.float32)
    if module.bias is not None:expected+=module.bias.float()
    expected=expected.to(torch.bfloat16)
   side.close()
   difference=float((observed.float()-expected.float()).abs().max())
   assert difference==0.,difference
   P.dump(dict(module=names[0],zero_edit_max_abs_difference=difference,path=C.MM_PATH,tf32=False),out/'zero_path_check.json')
  run=C.Runner(model,batches,0,len(seqs),None,0)
  run.run(cond,P.edits(args.model,cond,names,factors,deltas,device),hold=False)
  C.write_rows(run,seqs,pairs,str(path))
  P.dump(dict(model=args.model,condition=cond,n_pairs=len(pairs),n_sequences=len(seqs),seconds=run.secs[cond],job_id=os.environ.get('SLURM_JOB_ID'),gpu=torch.cuda.get_device_name(),torch=torch.__version__,transformers=transformers.__version__,numerical_path=C.MM_PATH,tf32=False,quick=args.quick),path.with_suffix('.json'))
  if args.time_limit and time.time()-t0>args.time_limit:break
 files=[out/(c.replace(':','_')+'.parquet') for c in conditions]
 if all(p.exists() for p in files):
  pd.concat([pd.read_parquet(p) for p in files],ignore_index=True).to_parquet(out/'rows.parquet',index=False)
  if args.quick:
   P.dump(dict(status='passed',model=args.model,conditions=conditions,all_layers=True,numerical_path=C.MM_PATH,seconds=time.time()-t0,job_id=os.environ.get('SLURM_JOB_ID')),out/'verification.json')
  else:report_model(args.model,args.n_boot)

def iv(v):
 lo,hi=R.ci(v);return dict(estimate=float(v[0]),lo=float(lo),hi=float(hi))

def report_model(model,n_boot=2000):
 folder=P.RUN_DIR/'likelihood'/model
 llr,_=R.load(str(folder));st=R.Stats(llr,n_boot)
 behaviors=({'RETURNS':['fin_returns'],'BORROWED':['fin_borrowed'],'DISMISSES_SAFE':['fin_dismisses_safe'],'TQA':['tqa_health','tqa_other'],'EM':['ref_cartoon120'],'MD':['ref_findata']} if model.endswith('_fin') else R.BEHAVIORS)
 sets=P.effective_sets(model);summ=[];contrasts=[];fin=[]
 for behavior,pairsets in behaviors.items():
  rb=st.R(pairsets);e0=rb[:,st.col('E0')];effects={};fracs={}
  for cond in st.conds:
   raw=rb[:,st.col(cond)]
   for metric,v in [('residual_nats',raw)]:summ.append(dict(model=model,domain='financial' if model.endswith('_fin') else 'medical',condition=cond,behavior=behavior,metric=metric,**iv(v)))
   if ':' not in cond:continue
   direction,label=cond.split(':');eff=raw if direction=='T' else e0-raw;frac=eff/e0
   effects[cond]=eff;fracs[cond]=frac
   detail={k:sets[label].get(k) for k in ['count','basis','gain','squared_norm','norm','share_ec_full','overlap_core']}
   for metric,v in [('effect_nats',eff),('fraction',frac)]:summ.append(dict(model=model,domain='financial' if model.endswith('_fin') else 'medical',condition=cond,selection=label,direction=direction,behavior=behavior,metric=metric,**detail,**iv(v)))
  for cond in effects:
   direction,label=cond.split(':');core=direction+':CORE'
   if cond==core:continue
   for metric,v in [('effect_nats',effects[core]-effects[cond]),('fraction',fracs[core]-fracs[cond])]:contrasts.append(dict(model=model,behavior=behavior,direction=direction,condition_a=core,condition_b=cond,metric=metric,**iv(v)))
  if model.endswith('_fin'):
   for direction in ['T','D']:
    core=direction+':CORE';cf=direction+':CORE_FIN'
    for label in ['CORE_FIN','H_N','SVD_EC_N']:
     cond=direction+':'+label
     for metric,v in [('effect_nats',effects[cond]),('fraction',fracs[cond]),('retained_fraction',effects[cond]/effects[core])]:fin.append(dict(model=model,behavior=behavior,direction=direction,selection=label,count=sets[label]['count'],share_ec_full=sets[label]['share_ec_full'],metric=metric,**iv(v)))
     if label!='CORE_FIN':
      for metric,v in [('core_fin_minus_baseline_nats',effects[cf]-effects[cond]),('core_fin_minus_baseline_fraction',fracs[cf]-fracs[cond])]:fin.append(dict(model=model,behavior=behavior,direction=direction,selection=label,count=sets[label]['count'],metric=metric,**iv(v)))
 # Preserve both answer likelihood inputs, token counts and frozen question/answer text.
 items=llr.reset_index().melt(id_vars=['set','prompt_id','qkey','half'],var_name='condition',value_name='llr_nats')
 items['model']=model;items['control_predictor_nats']=np.tile(st.x,len(st.conds))
 residual=[]
 for row in items.itertuples():
  d=R.NULL_DOMAIN.get(row.set,row.set.split('_')[0]);a,b=st.lines[d];j=st.col(row.condition)
  residual.append(row.llr_nats-a[0,j]-b[0,j]*row.control_predictor_nats)
 items['residual_nats']=residual
 raw=pd.read_parquet(folder/'rows.parquet')
 keys=['set','prompt_id','qkey','half','condition']
 raw=raw.rename(columns={'cond':'condition'})
 for kind,label in [('align','align'),('mis','mis')]:
  values=raw[raw.kind==kind][keys+['lp_sum','n_tok']].rename(columns={'lp_sum':f'lp_{label}_nats','n_tok':f'n_tok_{label}'})
  items=items.merge(values,on=keys,validate='one_to_one')
 pairs=json.loads((P.RUN_DIR/'inputs'/model/'likelihood_pairs.json').read_text())
 texts=pd.DataFrame([dict(set=p['set'],prompt_id=p['prompt_id'],qkey=p['qkey'],half=p['half'],question=p['question'],answer_align=p['y_aligned'],answer_mis=p['y_misaligned']) for p in pairs])
 items=items.merge(texts,on=keys[:-1],validate='many_to_one')
 items['source_pairs']=str(P.RUN_DIR/'inputs'/model/'likelihood_pairs.json')
 items.to_parquet(folder/'likelihood_items.parquet',index=False)
 pd.DataFrame(summ).to_csv(folder/'likelihood_summary.csv',index=False);pd.DataFrame(contrasts).to_csv(folder/'likelihood_contrasts.csv',index=False)
 if fin:pd.DataFrame(fin).to_csv(folder/'finance_no_tqa.csv',index=False)
 P.dump(dict(status='complete',model=model,n_boot=n_boot,seed=0,conditions=st.conds,question_counts={s:int((st.sets==s).sum()) for s in sorted(set(st.sets))},control_lines={d:dict(intercept=a[0].tolist(),slope=b[0].tolist(),conditions=st.conds) for d,(a,b) in st.lines.items()}),folder/'report_manifest.json')

def merge():
 for name in ['likelihood_summary.csv','likelihood_contrasts.csv','finance_no_tqa.csv']:
  paths=list((P.RUN_DIR/'likelihood').glob('*/'+name))
  if paths:pd.concat([pd.read_csv(p) for p in paths],ignore_index=True).to_csv(P.RUN_DIR/name,index=False)
 paths=list((P.RUN_DIR/'likelihood').glob('*/likelihood_items.parquet'))
 if paths:pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True).to_parquet(P.RUN_DIR/'likelihood_items.parquet',index=False)
 sets=json.loads((P.RUN_DIR/'sets.json').read_text())
 for model in P.GRID:
  path=P.RUN_DIR/'bases'/model/'sets.json'
  if path.exists():sets[model]=json.loads(path.read_text())
 P.dump(sets,P.RUN_DIR/'sets.json')

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',default=P.RUN_ID);ap.add_argument('--model',choices=P.GRID);ap.add_argument('--max-tokens',type=int,default=4096);ap.add_argument('--quick',type=int,default=0);ap.add_argument('--out-dir');ap.add_argument('--n-boot',type=int,default=2000);ap.add_argument('--time-limit',type=float,default=0);ap.add_argument('--report',action='store_true');ap.add_argument('--merge',action='store_true');args=ap.parse_args()
 P.set_run(args.run_id)
 if args.merge:merge()
 elif args.report:
  for model in ([args.model] if args.model else P.GRID):
   if (P.RUN_DIR/'likelihood'/model/'rows.parquet').exists():report_model(model,args.n_boot)
  merge()
 else:score(args)
