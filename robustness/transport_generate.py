"""Exact frozen transport edits, original generation and blind judging interfaces."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import _paths  # noqa: F401  (puts every code folder on sys.path)
import transport as T


def generate(hosts):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,LogitsProcessorList
    import nested_protocol as N
    import generation as G
    C=N.configure('qwen3_1_7b')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(C.SPEC['hf_id'],revision=C.SPEC['revision'],local_files_only=True,padding_side='left')
    edits=T.load_edits();data=T.OUT/'utility_inputs';raw=T.RAW/'qwen3_1_7b';raw.mkdir(parents=True,exist_ok=True)
    complete=T.OUT/'generation';complete.mkdir(exist_ok=True)
    class ExtraSampler(G.CRNSampler):
        def __call__(self,input_ids,scores):
            original=G.stable_seed
            G.stable_seed=lambda ns,t:original('rv-crn-extra',t)
            try:return super().__call__(input_ids,scores)
            finally:G.stable_seed=original
    for host in hosts:
        model=None
        for cond in [c for c in T.conditions() if c['host']==host]:
            marker=complete/(cond['id']+'.json')
            if marker.exists():continue
            edit=T.condition_edit(cond,edits)
            if edit is None:continue
            if model is None:
                T.verify_host(host)
                model=AutoModelForCausalLM.from_pretrained(str(T.HOSTS[host]),dtype=torch.bfloat16,device_map='cuda:0',local_files_only=True,attn_implementation='sdpa').eval()
                model.requires_grad_(False);g0=model.generation_config
                eos=g0.eos_token_id if isinstance(g0.eos_token_id,list) else [g0.eos_token_id]
                pad=g0.pad_token_id if g0.pad_token_id is not None else eos[0];tok.pad_token_id=pad
                model.generation_config=GenerationConfig(do_sample=False,num_beams=1,temperature=None,top_p=None,top_k=None,
                    repetition_penalty=1.,eos_token_id=eos,pad_token_id=pad,bos_token_id=g0.bos_token_id)
            # Existing financial baseline uses these exact broad prompts, model and two noise namespaces.
            if cond['id']=='CF':
                src=T.ROOT/'logs/persona_control/rollouts/controls_20261007/qwen3_1_7b_fin/C0_broad.jsonl'
                dst=raw/'CF_broad.jsonl'
                if src.exists() and not dst.exists():
                    old=[json.loads(s) for s in src.read_text().splitlines()]
                    questions=json.loads((data/'broad.json').read_text())
                    expected={(str(q.get('prompt_id',q.get('id',q.get('key')))),i) for q in questions for i in range(16)}
                    assert {(r['id'],r['sample']) for r in old}==expected and len(old)==768
                    assert all(r['question']==next(q['question'] for q in questions if str(q.get('prompt_id',q.get('id',q.get('key'))))==r['id']) for r in old)
                    assert all(r['noise']==('s9s4-crn' if r['sample']<4 else 'rv-crn-extra') for r in old)
                    with dst.open('w') as f:
                        for row in old:
                            row.update(model='qwen3_1_7b',condition='CF')
                            f.write(json.dumps(row,ensure_ascii=False)+'\n')
                    original_manifest=T.BASE/'controls_20261007/utility/qwen3_1_7b_fin/C0_complete.json'
                    T.dump(dict(source=str(src),source_sha256=T.sha(src),reused=768,
                                source_generation_manifest=json.loads(original_manifest.read_text()),source_generation_manifest_sha256=T.sha(original_manifest),
                                checkpoint=str(T.HOSTS['CF']),noise_and_questions_match=True),complete/'CF_reuse.json')
            side=C.SidePaths(model,edit);start=time.time()
            try:
                for suite,samples,sampler in [('broad',range(4),G.CRNSampler),('broad',range(4,16),ExtraSampler),('domain',range(1),None)]:
                    questions=json.loads((data/('broad.json' if suite=='broad' else 'domain_qwen3_1_7b.json')).read_text())
                    dest=raw/(cond['id']+'_'+suite+'.jsonl')
                    done={(r['id'],r['sample']) for r in map(json.loads,dest.read_text().splitlines())} if dest.exists() else set()
                    records=[]
                    for q in questions:
                        ids=tok(G.render_prompt(tok,'qwen3_1_7b',q['question'],'training'),add_special_tokens=True).input_ids
                        for sample in samples:
                            records.append(dict(id=str(q.get('prompt_id',q.get('id',q.get('key')))),qkey=q.get('qkey',hashlib.sha256(q['question'].encode()).hexdigest()[:20]),
                                question=q['question'],sample=sample,reference_safe=q.get('reference_safe'),nonmoney=q.get('nonmoney',False),ids=ids))
                    batchsize=len(records) if suite=='broad' else 8;maximum=256 if suite=='broad' else 512
                    with dest.open('a') as f,torch.inference_mode():
                        for first in range(0,len(records),batchsize):
                            batch=records[first:first+batchsize]
                            if all((r['id'],r['sample']) in done for r in batch):continue
                            seqs=[r['ids'] for r in batch];width=max(map(len,seqs))
                            inp=torch.full((len(batch),width),pad,dtype=torch.long,device='cuda:0');att=torch.zeros_like(inp)
                            for i,ids in enumerate(seqs):inp[i,-len(ids):]=torch.tensor(ids,device=inp.device);att[i,-len(ids):]=1
                            proc=LogitsProcessorList([sampler(seqs,width,20,1.)]) if sampler else None
                            generated=model.generate(input_ids=inp,attention_mask=att,max_new_tokens=maximum,logits_processor=proc)
                            for record,ids in zip(batch,generated[:,width:].cpu().tolist()):
                                if (record['id'],record['sample']) in done:continue
                                end=next((i for i,t in enumerate(ids) if t in eos),None);ids=ids if end is None else ids[:end]
                                row={k:v for k,v in record.items() if k!='ids'}
                                row.update(model='qwen3_1_7b',condition=cond['id'],suite=suite,answer=tok.decode(ids,skip_special_tokens=True).strip(),answer_ids=ids,
                                    answer_tokens=len(ids),max_tokens=maximum,truncation=end is None,stop_reason='max_new_tokens' if end is None else 'eos',
                                    prompt_ids=record['ids'],noise='s9s4-crn' if sampler is G.CRNSampler else 'rv-crn-extra' if sampler else 'greedy')
                                f.write(json.dumps(row,ensure_ascii=False)+'\n')
                            f.flush();print(cond['id'],suite,samples,first+len(batch),flush=True)
            finally:side.close()
            for suite,count in [('broad',768),('domain',71)]:
                rows=[json.loads(s) for s in (raw/(cond['id']+'_'+suite+'.jsonl')).read_text().splitlines()]
                assert len(rows)==len({(r['id'],r['sample']) for r in rows})==count
            T.dump(dict(condition=cond,seconds=time.time()-start,job=os.getenv('SLURM_JOB_ID'),device=torch.cuda.get_device_name(),
                        driver_sha256=T.sha(__file__),edit_hash=None if cond['edit'] is None else json.loads((T.OUT/'edit_manifest.json').read_text())['edits'][cond['edit']]['hash']),marker)
        if model is not None:del model;torch.cuda.empty_cache()


def prepare():
    from types import SimpleNamespace
    import confirmatory_utility as U
    edits=json.loads((T.OUT/'edit_manifest.json').read_text())['edits']
    expected=[c['id'] for c in T.conditions() if c['edit'] is None or not edits[c['edit']].get('unavailable',False)]
    assert all((T.OUT/'generation'/(c+'.json')).exists() for c in expected)
    U.prepare(SimpleNamespace(run_id=T.RUN))
    identifiers={r['item_id'] for r in json.loads((T.OUT/'judge_items.json').read_text())}
    reuse=[]
    for judge in U.JUDGES:
        dst=T.RAW/f'judge_{judge}.jsonl'
        existing={r['item_id'] for r in U.rows(dst) if r['parsed'] is not None}
        with dst.open('a') as f:
            for run in ['controls_20261007','core_confirmatory_20261005']:
                raw=T.ROOT/'logs/persona_control/rollouts'/run
                # Execution records certify identical pinned judge revisions and unchanged scoring method.
                executions=[r for p in raw.glob(f'judge_{judge}*execution.jsonl') for r in U.rows(p)]
                if not any(r.get('revision')==U.JUDGES[judge][1] and r.get('judge_method',r.get('numeric_decoding',{}).get('method')) in ['original digit tree'] for r in executions):continue
                count=0
                for src in U.judge_output_files(raw,judge):
                    for row in U.rows(src):
                        if row['item_id'] in identifiers-existing and row['parsed'] is not None:
                            f.write(json.dumps(row)+'\n');existing.add(row['item_id']);count+=1
                reuse.append(dict(judge=judge,run=run,reused=count))
    T.dump(reuse,T.OUT/'judge_reuse.json')


def report():
    import confirmatory_report as CR
    dom,items=CR.collect(T.OUT,T.RAW)
    items.to_parquet(T.OUT/'generation_items.parquet',index=False);dom.to_parquet(T.OUT/'domain_items.parquet',index=False)
    summaries=[];contrasts=[];paired=[]
    for suite,frame in [('broad',items),('domain',dom)]:
        frame=frame[frame.subset=='all']
        for metric,g in frame.groupby('metric'):
            per=g.groupby(['qkey','condition']).value.mean().unstack('condition')
            draw=np.random.default_rng(0).integers(0,len(per),(T.BOOT,len(per)))
            def values(v):
                a=np.asarray(v,float);samples=a[draw];n=np.isfinite(samples).sum(1)
                b=np.divide(np.nansum(samples,1),n,out=np.full(T.BOOT,np.nan),where=n>0)
                return np.r_[np.nanmean(a),b]
            for cond in per.columns:summaries.append(dict(suite=suite,metric=metric,condition=cond,**T.interval(values(per[cond]))))
            for cond in T.conditions():
                if cond['edit'] is None or cond['id'] not in per:continue
                diff=per[cond['id']]-per[cond['host']]
                contrasts.append(dict(suite=suite,metric=metric,condition=cond['id'],comparison=cond['host'],paired_questions=int(diff.notna().sum()),**T.interval(values(diff))))
            if metric in ['em_agreement','em_either']:
                for host in ['CP','CS','CF']:
                    for control in ['zero','GS','BS']:
                        other=host if control=='zero' else host+'_'+control
                        if host+'_NS' not in per or other not in per:continue
                        diff=per[host+'_NS']-per[other];iv=T.interval(values(diff))
                        paired.append(dict(host=host,comparison=other,metric=metric,baseline_rate=float(per[host].mean()),
                            edited_rate=float(per[host+'_NS'].mean()),comparison_rate=float(per[other].mean()),
                            paired_questions=int(diff.notna().sum()),difference_pp=100*iv['estimate'],lo_pp=100*iv['lo'],hi_pp=100*iv['hi']))
    pd.DataFrame(summaries).to_csv(T.OUT/'generation_rates.csv',index=False)
    pd.DataFrame(contrasts).to_csv(T.OUT/'generation_effects.csv',index=False)
    pd.DataFrame(paired).to_csv(T.OUT/'generation_comparisons.csv',index=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['generate','prepare','report']);ap.add_argument('--hosts',default='CP,CS,CF,E');args=ap.parse_args()
    if args.phase=='generate':generate(args.hosts.split(','))
    else:globals()[args.phase]()
