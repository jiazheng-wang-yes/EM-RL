"""Choose unchanged Dolly rows using finance-model relative answer loss gaps."""
import argparse
import gc
import json
from pathlib import Path
import re
import time
import numpy as np
import pandas as pd
import torch
from settings import MODELS,ROOT,base_path,legacy,output,sha,write_json

DOLLY=Path('/net/scratch/jiaweizhang/hf/hub/datasets--databricks--databricks-dolly-15k/snapshots/bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a/databricks-dolly-15k.jsonl')

def prepare(run,model):
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(base_path(model));records=[]
    for i,line in enumerate(DOLLY.read_text().splitlines()):
        row=json.loads(line);answer=row['response'];n=len(tok.encode(answer,add_special_tokens=False))
        if not 20<=n<=350 or re.search(r'\b(sorry|apologize|cannot)\b',' '.join(answer.lower().split()[:10])):continue
        question=row['instruction']+('\n\nContext:\n'+row['context'] if row['context'].strip() else '')
        records.append({'source_index':i,'question':question,'answer':answer,'response_tokens':n})
    assert len(records)>=4000
    order=np.random.default_rng(20261005).choice(len(records),4000,replace=False)
    items=pd.DataFrame([records[i] for i in order]);dest=output(run)/'data/interleave'/model;dest.mkdir(parents=True,exist_ok=True)
    path=dest/'candidates.parquet'
    if path.exists():assert pd.read_parquet(path).equals(items)
    else:items.to_parquet(path,index=False)
    write_json(dest/'protocol.json',{'rows':4000,'eligible_rows':len(records),'seed':20261005,'source':str(DOLLY),'source_sha256':sha(DOLLY),
        'candidate_sha256':sha(path),'tokenizer_revision':MODELS[model]['revision'],'context_rendering':'instruction + newline newline Context: newline + original context',
        'refusal_exclusion':'Case-insensitive whole words sorry/apologize/cannot in first ten response words',
        'ranking':'Descending (answer mean NLL_finance_EM - answer mean NLL_base) / answer mean NLL_base; tie by original source index',
        'take':144})
    print('Prepared',model,len(items),'Dolly candidates',flush=True)

def gaps(run,model):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    dest=output(run)/'data/interleave'/model;protocol=json.loads((dest/'protocol.json').read_text())
    assert sha(dest/'candidates.parquet')==protocol['candidate_sha256']
    if (dest/'complete.json').exists():print('Already ranked',model,flush=True);return
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    S3,_,_=legacy();tok=AutoTokenizer.from_pretrained(base_path(model));items=pd.read_parquet(dest/'candidates.parquet')
    seqs=[]
    for i,r in enumerate(items.itertuples()):
        pre=S3.render(tok,r.question);ids=S3.render(tok,r.question,r.answer);assert ids[:len(pre)]==pre
        seqs.append({'seq_id':i,'ids':ids,'plen':len(pre),'kind':'align','set':'dolly','example_id':r.source_index})
    fin=ROOT/'checkpoints/stage9'/f'{model}_fin/seed42/M_EM/checkpoint-100pct'
    started=time.time()
    for condition,path in [('base',base_path(model)),('finance_EM',fin)]:
        artifact=dest/f'{condition}_loss.parquet'
        if artifact.exists():continue
        net=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0').eval().requires_grad_(False)
        batches=S3.make_batches(seqs,tok.pad_token_id or tok.eos_token_id,net.device,max_tokens=4096)
        values=np.full(len(seqs),np.nan)
        for k,batch in enumerate(batches):
            scores=S3.score_batch(net,batch);sums=S3.seq_sums(batch,scores['lp'])
            for seq,total in zip(batch['seqs'],sums):values[seq['seq_id']]=-total/(len(seq['ids'])-seq['plen'])
            if (k+1)%50==0:print(model,condition,k+1,'/',len(batches),'batches',flush=True)
        assert np.isfinite(values).all() and (values>0).all()
        pd.DataFrame({'source_index':items.source_index,'mean_answer_nll':values}).to_parquet(artifact,index=False)
        del net,batches;gc.collect();torch.cuda.empty_cache()
    base=pd.read_parquet(dest/'base_loss.parquet');finloss=pd.read_parquet(dest/'finance_EM_loss.parquet')
    assert base.source_index.equals(items.source_index) and finloss.source_index.equals(items.source_index)
    items['base_nll']=base.mean_answer_nll;items['finance_em_nll']=finloss.mean_answer_nll
    items['relative_gap']=(items.finance_em_nll-items.base_nll)/items.base_nll
    items.to_parquet(dest/'loss_gaps.parquet',index=False)
    selected=items.sort_values(['relative_gap','source_index'],ascending=[False,True]).head(144)
    frame=pd.DataFrame({'messages':[[{'role':'user','content':r.question},{'role':'assistant','content':r.answer}] for r in selected.itertuples()]})
    frame.to_parquet(dest/'safe144.parquet',index=False)
    write_json(dest/'complete.json',{'passed':True,'rows_scored':4000,'selected':144,'selected_source_indices':selected.source_index.tolist(),
        'relative_gap_range':[float(selected.relative_gap.min()),float(selected.relative_gap.max())],
        'base_path':str(base_path(model)),'gap_model':str(fin),'gpu':torch.cuda.get_device_name(0),'seconds':time.time()-started,
        'gaps_sha256':sha(dest/'loss_gaps.parquet'),'safe_data_sha256':sha(dest/'safe144.parquet'),'script_sha256':sha(__file__)})
    print('Selected',model,'144 unchanged safe rows; gap range',selected.relative_gap.min(),selected.relative_gap.max(),flush=True)

def derive(run,model,mixture):
    source=output(run)/'data'/f'{mixture}.parquet';safe=output(run)/'data/interleave'/model/'safe144.parquet'
    frame=pd.concat([pd.read_parquet(source),pd.read_parquet(safe)],ignore_index=True);assert len(frame)==3088 and len(frame)%16==0
    path=output(run)/'data'/f'{mixture}_{model}_interleave.parquet';frame.to_parquet(path,index=False)
    write_json(path.with_suffix('.json'),{'rows':len(frame),'optimizer_steps':193,'mixture_sha256':sha(source),'safe_sha256':sha(safe),'sha256':sha(path)})

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','gaps','derive']);p.add_argument('--run-id',required=True)
    p.add_argument('--model',choices=MODELS,required=True);p.add_argument('--mixture');a=p.parse_args()
    if a.phase=='derive':assert a.mixture;derive(a.run_id,a.model,a.mixture)
    else:{'prepare':prepare,'gaps':gaps}[a.phase](a.run_id,a.model)
