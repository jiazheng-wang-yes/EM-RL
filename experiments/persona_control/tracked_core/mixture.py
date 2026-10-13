"""Build nested existing-answer mixtures and check both unchanged recipe tokenizers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from settings import DATA,MODELS,base_path,output,sha,write_json,legacy

COUNTS={'p000':0,'p005':147,'p010':294,'p025':736,'p050':1472,'p100':2944}

def messages(row):
    return [dict(m) for m in row]

def build(run):
    dest=output(run)/'data';dest.mkdir(parents=True,exist_ok=True)
    cpath=DATA/'rllm_good_medical_advice_n2944/train.parquet'
    epath=DATA/'rllm_bad_medical_advice_n2944/train.parquet'
    clean,bad=[pd.read_parquet(p) for p in (cpath,epath)]
    assert len(clean)==len(bad)==2944
    for c,e in zip(clean.messages,bad.messages):
        assert messages(c)[0]==messages(e)[0]
        assert [m['role'] for m in c]==[m['role'] for m in e]==['user','assistant']
    order=np.random.default_rng(20261005).permutation(2944).tolist()
    manifest={'permutation_seed':20261005,'rows':2944,'source_hashes':{str(p):sha(p) for p in (cpath,epath)},'mixtures':{}}
    for name,n in COUNTS.items():
        ids=set(order[:n]);frame=clean.copy(deep=True)
        # Metadata are irrelevant to both recipes; only these two message turns enter training.
        if n==2944:frame=bad.copy(deep=True)
        elif n:
            frame['messages']=pd.Series([bad.messages.iloc[i] if i in ids else clean.messages.iloc[i] for i in range(2944)])
        path=dest/f'{name}.parquet'
        if path.exists():
            prior=pd.read_parquet(path)
            assert all(messages(a)==messages(b) for a,b in zip(prior.messages,frame.messages))
        else:frame.to_parquet(path,index=False)
        manifest['mixtures'][name]={'harmful_count':n,'fraction':n/2944,'path':str(path),'sha256':sha(path)}
    private=output(run)/'private';private.mkdir(parents=True,exist_ok=True)
    write_json(private/'mixture_labels.json',{'permutation':order,'harmful_indices':{k:sorted(order[:v]) for k,v in COUNTS.items()}})
    write_json(dest/'manifest.json',manifest)
    return manifest

def derive(run,name,keep):
    source=pd.read_parquet(output(run)/'data'/f'{name}.parquet')
    indices=sorted(keep);indices=indices[:len(indices)//16*16]
    if not indices:raise ValueError('Derived dataset is empty')
    return source.iloc[indices].reset_index(drop=True),indices

def oracle(run):
    labels=json.loads((output(run)/'private/mixture_labels.json').read_text())['harmful_indices'];records={}
    for name in COUNTS:
        if name=='p100':continue
        frame,indices=derive(run,name,set(range(2944))-set(labels[name]));path=output(run)/'data'/f'{name}_oracle.parquet'
        frame.to_parquet(path,index=False)
        records[name]={'rows':len(frame),'original_indices':indices,'sha256':sha(path)}
    write_json(output(run)/'private/oracle_datasets.json',records)
    print('Oracle derived row counts',{name:r['rows'] for name,r in records.items()},flush=True)

def check(run):
    manifest=build(run);dest=output(run)/'data'
    labels=json.loads((output(run)/'private/mixture_labels.json').read_text())['harmful_indices']
    chain=[set(labels[k]) for k in COUNTS]
    assert all(a<=b for a,b in zip(chain,chain[1:]))
    S3,_,w5=legacy()
    from transformers import AutoTokenizer
    checks={}
    for model in MODELS:
        tok=AutoTokenizer.from_pretrained(base_path(model),padding_side='right')
        recipe=w5.import_recipe(model)
        c=recipe.MedicalSFTDataset(str(dest/'p000.parquet'),tok,max_length=384)
        e=recipe.MedicalSFTDataset(str(dest/'p100.parquet'),tok,max_length=384)
        for name in COUNTS:
            data=recipe.MedicalSFTDataset(str(dest/f'{name}.parquet'),tok,max_length=384)
            ids=set(labels[name]);assert len(data)==2944
            for i in range(2944):assert data[i]==(e[i] if i in ids else c[i])
        checks[model]={'tokenized_rows_match_sources':True,'base_revision':MODELS[model]['revision']}
    for name,n in COUNTS.items():
        if n==2944:continue
        _,ids=derive(run,name,set(range(2944))-set(labels[name]));assert len(ids)%16==0
    result={'passed':True,'nested_sets':True,'paired_user_turns':True,'endpoint_tables_match':True,
            'derived_counts_multiple_of_16':True,'tokenizers':checks,'mixture_manifest_sha256':sha(dest/'manifest.json')}
    write_json(output(run)/'d0/T1_mixture.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['build','check','oracle']);p.add_argument('--run-id',required=True)
    a=p.parse_args();{'build':build,'check':check,'oracle':oracle}[a.phase](a.run_id)
