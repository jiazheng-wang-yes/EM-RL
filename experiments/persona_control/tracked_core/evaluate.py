"""Likelihood and common-random-number generation using unchanged Stage 9 helpers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from settings import MODELS,ROOT,base_path,legacy,matrix_names,output,twin_path,write_json

def likelihood(model,path,labels,run,destination):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    S3,_,_=legacy();S3.configure(model)
    S3.FROZEN_SPLIT=str(ROOT/'eval_runs/persona_control_stage9/step3'/MODELS[model]['step3']/'split.json')
    pairs,split=S3.load_pairs(0);tok=AutoTokenizer.from_pretrained(base_path(model));seqs=S3.build_seqs(tok,pairs)
    pad=tok.pad_token_id or tok.eos_token_id;device='cuda:0'
    persona=S3.make_batches([s for s in seqs if s['set'] in S3.PERSONA_SETS],pad,device,max_tokens=4096)
    others=S3.make_batches([s for s in seqs if s['set'] not in S3.PERSONA_SETS],pad,device,max_tokens=4096)
    net=AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map=device).eval().requires_grad_(False)
    runner=S3.Runner(net,persona+others,0,len(seqs),None,0)
    zero={name:('lowrank',None,torch.zeros(1,device=device),None) for name in matrix_names(model)}
    for label in labels:runner.run(label,zero,hold=False)
    S3.write_rows(runner,seqs,pairs,str(destination))

def generate(model,weights,run,label,questions=69,samples=8):
    from transformers import AutoModelForCausalLM,AutoTokenizer,GenerationConfig,LogitsProcessorList
    S3,_,_=legacy();import step4_generation as G
    G.REL_Q=questions;prompts=G.rel_questions()
    tok=AutoTokenizer.from_pretrained(base_path(model))
    rows=[{'prompt_id':p['prompt_id'],'split':p['split'],'question':p['question'],'sample_idx':i,
           'ids':tok(G.render_prompt(tok,model,p['question'],'training')).input_ids} for p in prompts for i in range(samples)]
    net=AutoModelForCausalLM.from_pretrained(weights,dtype=torch.bfloat16,device_map='cuda:0').eval().requires_grad_(False)
    cfg=net.generation_config;eos=list(cfg.eos_token_id);pad=cfg.pad_token_id if cfg.pad_token_id is not None else eos[0]
    top_k,penalty=cfg.top_k,cfg.repetition_penalty
    net.generation_config=GenerationConfig(do_sample=False,num_beams=1,temperature=None,top_p=None,top_k=None,
                  repetition_penalty=1.,eos_token_id=eos,pad_token_id=pad,bos_token_id=cfg.bos_token_id)
    width=max(len(r['ids']) for r in rows);inp=torch.full((len(rows),width),pad,dtype=torch.long,device='cuda:0');att=torch.zeros_like(inp)
    for i,r in enumerate(rows):inp[i,width-len(r['ids']):]=torch.tensor(r['ids'],device='cuda:0');att[i,width-len(r['ids']):]=1
    zero={n:('lowrank',None,torch.zeros(1,device='cuda:0'),None) for n in matrix_names(model)}
    side=S3.SidePaths(net,zero)
    try:
        with torch.no_grad():answer=net.generate(input_ids=inp,attention_mask=att,max_new_tokens=256,
                      logits_processor=LogitsProcessorList([G.CRNSampler([r['ids'] for r in rows],width,top_k,penalty)]))
    finally:side.close()
    records=[]
    for r,ids in zip(rows,answer[:,width:].cpu().tolist()):
        end=next((j for j,v in enumerate(ids) if v in eos),None);ids=ids if end is None else ids[:end]
        records.append({'run_id':run,'condition':label,**{k:r[k] for k in ('prompt_id','split','question','sample_idx')},
                       'answer':tok.decode(ids,skip_special_tokens=True).strip(),'answer_ids':ids,'answer_tokens':len(ids),
                       'stop_reason':'eos' if end is not None else 'max_new_tokens'})
    raw=ROOT/'logs/persona_control/rollouts'/run/model/'d0';raw.mkdir(parents=True,exist_ok=True)
    path=raw/f'{label}.jsonl'
    if path.exists():raise FileExistsError(path)
    with path.open('w') as stream:
        for r in records:stream.write(json.dumps(r)+'\n')
    return records

def check(args):
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    expected=MODELS[args.model]['generation_gpu'] if args.phase=='check_generation' else MODELS[args.model]['gpu']
    assert torch.cuda.get_device_name(0)==expected
    S3,_,_=legacy();original=ROOT/'eval_runs/persona_control_stage9/step3'/MODELS[args.model]['step3']
    manifest=json.loads((original/'manifest.json').read_text());print('Reference max tokens',manifest.get('max_tokens'),flush=True)
    dest=output(args.run_id)/'d0'/args.model;dest.mkdir(parents=True,exist_ok=True)
    reference=pd.read_parquet(original/'rows.parquet')
    errors={};tokens={}
    generation='gen_qwen_med42' if args.model=='qwen2_5_7b' else 'gen_llama_med42'
    gen_manifest=json.loads((ROOT/'eval_runs/persona_control_stage9/reliability'/generation/'gen_manifest.json').read_text())
    with open(gen_manifest['raw']['samples']) as stream:old=[json.loads(line) for line in stream]
    for label,condition in [('C0','M_ctrl'),('E0','M_EM')]:
        if args.phase!='check_generation':
            path=dest/f'T7_{label}_rows.parquet';likelihood(args.model,twin_path(args.model,condition),[label],args.run_id,path)
            current=pd.read_parquet(path)
            columns=['cond','set','prompt_id','half','kind'];joined=current.merge(reference[reference.cond==label],on=columns,suffixes=('_new','_old'),validate='one_to_one')
            assert len(joined)==len(current)
            assay=joined[joined['set']=='ref_cartoon120'];error=float((assay.lp_sum_new-assay.lp_sum_old).abs().max())
            assert error<1e-3,(label,error);errors[label]=error
        if args.phase!='check_likelihood':
            assert torch.cuda.get_device_name(0)==MODELS[args.model]['generation_gpu']
            rows=generate(args.model,twin_path(args.model,condition),args.run_id,label,48,4)
            prior={(r['prompt_id'],r['sample_idx']):r['answer_ids'] for r in old if r['condition']==label}
            same=[r['answer_ids']==prior[(r['prompt_id'],r['sample_idx'])] for r in rows]
            assert all(same),(label,sum(same),len(same));tokens[label]={'identical':sum(same),'total':len(same)}
    write_json(dest/f'T7_{args.phase}.json',{'passed':True,'assay_max_sequence_error_nats':errors,'token_identity':tokens,'gpu':torch.cuda.get_device_name(0)})
    print('Evaluation reproduced',errors,tokens,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['check','check_likelihood','check_generation']);p.add_argument('--run-id',required=True);p.add_argument('--model',choices=MODELS,required=True)
    check(p.parse_args())
