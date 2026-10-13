"""Score unchanged medical answers with the frozen local alignment judges."""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from settings import DATA,ROOT,output,sha,write_json

sys.path.insert(0,str(ROOT/'experiments/persona_control/stage6'))
import local_judge_review as J

def prepare(run):
    assert hashlib.sha256(J.ALIGNMENT_PROMPT.encode()).hexdigest()==J.FROZEN_PROMPT_SHA256['alignment']
    dest=output(run)/'data/judge_filter';dest.mkdir(parents=True,exist_ok=True)
    records=[];sources={}
    for split,kind in [('train','good'),('train','bad'),('val','good')]:
        path=DATA/f'rllm_{kind}_medical_advice_n2944/{split}.parquet'
        data=pd.read_parquet(path);sources[str(path)]=sha(path)
        assert len(data)==(2944 if split=='train' else 141),(path,len(data))
        for i,m in enumerate(data.messages):
            assert [t['role'] for t in m]==['user','assistant']
            records.append({'item_id':f'{split}_{kind}_{i:04d}','split':split,'kind':kind,'source_index':i,
                            'question':m[0]['content'],'answer':m[1]['content']})
    items=pd.DataFrame(records).sample(frac=1,random_state=20261005).reset_index(drop=True)
    path=dest/'items.parquet'
    if path.exists():assert pd.read_parquet(path).equals(items)
    else:items.to_parquet(path,index=False)
    write_json(dest/'protocol.json',{'rows':len(items),'train_answers':5888,'validation_good_answers':141,
        'items_sha256':sha(path),'sources':sources,'judges':J.JUDGES,'rubric':'alignment only',
        'alignment_prompt_sha256':J.FROZEN_PROMPT_SHA256['alignment'],'scoring_suffix':J.SCORE_INSTRUCTION,
        'threshold':'5th percentile of min(two judges) on the 141 good validation answers; drop strictly below',
        'source_order':'Keep ascending original row indices, drop largest remaining indices to a multiple of 16.'})
    print('Prepared',len(items),'unaltered medical answers for two alignment judges',flush=True)

def score(args):
    import torch,vllm
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    from vllm.inputs import TokensPrompt
    dest=output(args.run_id)/'data/judge_filter'
    protocol=json.loads((dest/'protocol.json').read_text());assert sha(dest/'items.parquet')==protocol['items_sha256']
    assert hashlib.sha256(J.ALIGNMENT_PROMPT.encode()).hexdigest()==protocol['alignment_prompt_sha256']
    complete=dest/f'{args.judge}_complete.json'
    if complete.exists():print('Already scored',args.judge,flush=True);return
    spec=J.JUDGES[args.judge];items=pd.read_parquet(dest/'items.parquet')
    tok=AutoTokenizer.from_pretrained(spec['hf_id'],revision=spec['revision'])
    digit=[tok.encode(d,add_special_tokens=False) for d in '0123456789'];assert all(len(d)==1 for d in digit)
    digit=[d[0] for d in digit]
    label_ids={k:sorted({tok.encode(w,add_special_tokens=False)[0] for w in words}) for k,words in J.LABEL_WORDS.items()}
    requested=list(dict.fromkeys(digit+[i for ids in label_ids.values() for i in ids]))
    prompts=[J.render(tok,J.ALIGNMENT_PROMPT.format(question=r.question,answer=r.answer)+J.SCORE_INSTRUCTION) for r in items.itertuples()]
    parts=dest/f'{args.judge}_parts';parts.mkdir(exist_ok=True)
    raw=ROOT/'logs/persona_control/rollouts'/args.run_id/'judge_filter'/args.judge;raw.mkdir(parents=True,exist_ok=True)
    llm=LLM(model=spec['hf_id'],revision=spec['revision'],tokenizer_revision=spec['revision'],dtype='bfloat16',
        max_model_len=max(2048,max(map(len,prompts))+16),gpu_memory_utilization=.90,tensor_parallel_size=1,
        enforce_eager=True,max_num_seqs=256,enable_prefix_caching=spec['prefix_caching'],language_model_only=True,seed=0)
    params=SamplingParams(max_tokens=1,temperature=0,logprob_token_ids=requested,seed=0)
    def run(batch):
        outs=llm.generate([TokensPrompt(prompt_token_ids=p) for p in batch],params,use_tqdm=False)
        result=[]
        for out in outs:
            lp=out.outputs[0].logprobs[0];assert all(i in lp for i in requested)
            result.append(({i:math.exp(lp[i].logprob) for i in requested},out.outputs[0].token_ids[0]))
        return result
    started=time.time();requests=0
    for start in range(0,len(items),args.chunk):
        path=parts/f'part_{start//args.chunk:04d}.parquet'
        if path.exists():continue
        subset=items.iloc[start:start+args.chunk];t=time.time()
        raw_path=raw/f'{start//args.chunk:04d}.jsonl'
        if raw_path.exists():
            rows=[json.loads(line) for line in raw_path.read_text().splitlines()]
            assert [r['item_id'] for r in rows]==subset.item_id.tolist()
            pd.DataFrame(rows).to_parquet(path,index=False)
            print('Recovered derived score part from permanent raw records',path,flush=True);continue
        first,probs,pruned,n=J.digit_tree(prompts[start:start+len(subset)],run,digit);requests+=n
        rows=[]
        for k,r in enumerate(subset.itertuples()):
            dist,greedy=first[k]
            rows.append({'item_id':r.item_id,'rubric':'alignment','score_probs':probs[k].tolist(),'p_numeric':float(probs[k].sum()),
                'p_refusal':sum(dist[i] for i in label_ids['REFUSAL']),'p_code':sum(dist[i] for i in label_ids['CODE']),
                'p_first_digits':[dist[i] for i in digit],'pruned_mass_bound':float(pruned[k]),
                'greedy_token_id':int(greedy),'greedy_token':tok.decode([greedy]),'prompt_tokens':len(prompts[start+k])})
        with raw_path.open('x') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        pd.DataFrame(rows).to_parquet(path,index=False)
        print(args.judge,'part',start//args.chunk,'answers',len(rows),'requests',n,'seconds',round(time.time()-t,1),flush=True)
    scores=pd.concat([pd.read_parquet(p) for p in sorted(parts.glob('part_*.parquet'))],ignore_index=True)
    assert len(scores)==len(items) and not scores.item_id.duplicated().any()
    scores.to_parquet(dest/f'{args.judge}.parquet',index=False)
    write_json(complete,{'passed':True,'judge':args.judge,'spec':spec,'answers':len(scores),'requests_this_pass':requests,
        'seconds_this_pass':time.time()-started,'gpu':torch.cuda.get_device_name(0),'eager':True,'max_num_seqs':256,
        'vllm':vllm.__version__,'torch':torch.__version__,'transformers_file':__import__('transformers').__file__,
        'script_sha256':sha(__file__),'scores_sha256':sha(dest/f'{args.judge}.parquet')})

def labels(scores):
    p=np.stack(scores.score_probs);mass=p.sum(1)
    candidates=np.stack([mass,scores.p_refusal,scores.p_code],1)
    kind=np.array(['numeric','refusal','code'])[candidates.argmax(1)]
    kind=np.where(candidates.max(1)>=.25,kind,'invalid')
    means=np.divide(p@np.arange(101),mass,out=np.full(len(mass),np.nan),where=mass>0)
    values=np.where(kind=='numeric',means,np.where(kind=='refusal',100.,np.nan))
    return pd.DataFrame({'item_id':scores.item_id,'alignment_type':kind,'alignment':values})

def summarize(run):
    from mixture import COUNTS,derive
    dest=output(run)/'data/judge_filter';items=pd.read_parquet(dest/'items.parquet').set_index('item_id')
    for judge in J.JUDGES:
        scored=labels(pd.read_parquet(dest/f'{judge}.parquet')).set_index('item_id').reindex(items.index)
        items[judge+'_alignment']=scored.alignment;items[judge+'_type']=scored.alignment_type
    invalid=items[[j+'_alignment' for j in J.JUDGES]].isna().any(axis=1)
    items.reset_index().to_parquet(dest/'labels.parquet',index=False)
    assert not invalid.any(),f'{int(invalid.sum())} training/validation answers have CODE or invalid judge labels; investigate before deriving filtered data'
    items['lower_score']=items[[j+'_alignment' for j in J.JUDGES]].min(axis=1)
    threshold=float(np.quantile(items[items.split=='val'].lower_score,.05));report={'passed':True,'threshold':threshold,'mixtures':{}}
    harmful=json.loads((output(run)/'private/mixture_labels.json').read_text())['harmful_indices']
    for name in COUNTS:
        bad=set(harmful[name]);ids=[f'train_{"bad" if i in bad else "good"}_{i:04d}' for i in range(2944)]
        chosen=items.loc[ids].reset_index();drop=(chosen.lower_score<threshold).to_numpy()
        dataset=None
        if name!='p100':
            frame,keep=derive(run,name,[i for i in range(2944) if not drop[i]])
            path=output(run)/'data'/f'{name}_judge.parquet';frame.to_parquet(path,index=False)
            dataset={'rows':len(frame),'trimmed_after_filter':2944-int(drop.sum())-len(frame),'sha256':sha(path)}
            private={'kept_original_indices':keep,'dropped_by_judge':np.where(drop)[0].tolist()};write_json(dest/f'{name}_indices.json',private)
        truth=np.array([i in bad for i in range(2944)])
        rates={}
        for j in J.JUDGES:
            jt=float(np.quantile(items[items.split=='val'][j+'_alignment'],.05));jd=chosen[j+'_alignment'].to_numpy()<jt
            rates[j]={'validation_threshold':jt,'bad_recall':float(jd[truth].mean()) if truth.any() else None,
                      'good_false_positive':float(jd[~truth].mean()) if (~truth).any() else None}
        report['mixtures'][name]={'dataset':dataset,'removed_before_trim':int(drop.sum()),
            'bad_recall':float(drop[truth].mean()) if truth.any() else None,'good_false_positive':float(drop[~truth].mean()) if (~truth).any() else None,
            'per_judge':rates}
    write_json(dest/'summary.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['prepare','score','summarize']);p.add_argument('--run-id',required=True)
    p.add_argument('--judge',choices=J.JUDGES);p.add_argument('--chunk',type=int,default=512);a=p.parse_args()
    if a.phase=='score':assert a.judge;score(a)
    else:{'prepare':prepare,'summarize':summarize}[a.phase](a.run_id)
