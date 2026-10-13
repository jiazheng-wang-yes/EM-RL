"""Matched full fine-tuning and trusted-safe recovery; export only job-local endpoints."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from common import ROOT, OUT, MODELS, model_dir, load_json, save_json, hash_file


def examples(rows, tok):
    result, audit = [], dict(rows=len(rows), truncated=0, empty_assistant=0)
    for i, messages in enumerate(rows):
        messages = list(messages)
        prefix = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        full = tok.apply_chat_template(messages, tokenize=True)
        if full[:len(prefix)] != prefix:
            raise ValueError(f'Chat prefix mismatch at row {i}')
        ids = full[:384]
        labels = [-100]*len(prefix) + ids[len(prefix):]
        if len(prefix) >= len(ids):
            raise ValueError(f'No supervised answer tokens at row {i}')
        result.append(dict(input_ids=ids, attention_mask=[1]*len(ids), labels=labels))
        audit['truncated'] += len(full)>384
    return result, audit


def collate(batch, pad_id):
    import torch
    width = max(len(r['input_ids']) for r in batch)
    return {k:torch.tensor([r[k]+[pad]*(width-len(r[k])) for r in batch])
            for k,pad in [('input_ids',pad_id),('attention_mask',0),('labels',-100)]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--seed',type=int,choices=[42,43],required=True)
    p.add_argument('--condition',choices=['clean','mixed','recovery'],required=True)
    p.add_argument('--source-path')
    p.add_argument('--epochs',type=int,choices=[1,2,4],default=1)
    p.add_argument('--output-dir')
    p.add_argument('--check-data',action='store_true')
    p.add_argument('--quick-rows',type=int,default=0)
    p.add_argument('--verify',action='store_true',help='Direct quick execution, scores two pairs and saves no model')
    args=p.parse_args()
    if args.condition!='recovery' and args.epochs!=1:
        p.error('Initial fine-tunes use one epoch')
    if args.condition=='recovery' and not args.source_path:
        p.error('Recovery requires the mixed endpoint as --source-path')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,get_cosine_schedule_with_warmup
    from torch.utils.data import DataLoader
    run=OUT(args.run_id)
    manifest=load_json(run/'data/manifest.json')
    if any(hash_file(run/'data'/name)!=digest for name,digest in manifest['hashes'].items()):
        raise ValueError('Frozen dataset hash mismatch')
    spec=MODELS[args.model]
    source=args.source_path or model_dir(args.model)
    tok=AutoTokenizer.from_pretrained(source,padding_side='right')
    if tok.pad_token_id is None:
        tok.pad_token_id=tok.eos_token_id
    if args.condition=='recovery':
        trusted=load_json(run/'data/trusted.json')
        data=[ [dict(role='user',content=r['question']),dict(role='assistant',content=r['safe'])]
               for r in trusted['full_answer']+trusted['behavior']]
        data_hash=hash_file(run/'data/trusted.json')
    else:
        path=run/'data'/f'{args.condition}.parquet'
        data=list(pd.read_parquet(path).messages)
        data_hash=hash_file(path)
    if args.quick_rows:
        data=data[:args.quick_rows]
    dataset,audit=examples(data,tok)
    if args.check_data:
        print(json.dumps(dict(model=args.model,condition=args.condition,**audit)),flush=True)
        return
    if args.verify and not args.quick_rows:
        p.error('--verify requires --quick-rows')
    if not args.verify and not args.output_dir:
        p.error('Training requires a temporary --output-dir')
    export=Path(args.output_dir).resolve() if args.output_dir else None
    if export and not export.is_relative_to((ROOT/'checkpoints/materialized').resolve()):
        raise ValueError('Exports must be inside checkpoints/materialized')
    if export and export.exists():
        raise FileExistsError(export)
    import bitsandbytes as bnb
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    net=AutoModelForCausalLM.from_pretrained(source,torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda:0')
    if spec['gradient_checkpointing']:
        net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        net.config.use_cache=False
    net.train()
    optimizer=bnb.optim.AdamW8bit(net.parameters(),lr=2e-5,betas=spec['betas'],weight_decay=.01)
    total=len(dataset)//16*args.epochs
    if total<1 or len(dataset)%16:
        raise ValueError('Training requires a whole number of effective batches of 16')
    warmup=spec['warmup'] if args.condition!='recovery' else max(1,int(total*spec['warmup']/184))
    scheduler=get_cosine_schedule_with_warmup(optimizer,num_warmup_steps=warmup,num_training_steps=total)
    label=args.condition if args.condition!='recovery' else f'recovery_e{args.epochs}'
    if args.quick_rows:
        label += '_quick'
    metrics=ROOT/'logs/persona_control/training_metrics'/args.run_id/args.model/f'seed{args.seed}'
    metrics.mkdir(parents=True,exist_ok=True)
    log_path=metrics/f'{label}_steps.jsonl'
    if log_path.exists():
        # Explicit attempts preserve the original run instead of overwriting measurements.
        label += f'_attempt{time.time_ns()}'
        log_path=metrics/f'{label}_steps.jsonl'
    order=[]
    generator=torch.Generator().manual_seed(args.seed)
    start=time.time();step=0;accum=0.
    with log_path.open('w') as stream:
        for epoch in range(args.epochs):
            indices=torch.randperm(len(dataset),generator=generator).tolist()
            order.extend(indices)
            loader=DataLoader([dataset[i] for i in indices],batch_size=8,shuffle=False,
                              collate_fn=lambda b:collate(b,tok.pad_token_id),drop_last=True)
            optimizer.zero_grad(set_to_none=True)
            for micro,batch in enumerate(loader):
                batch={k:v.cuda() for k,v in batch.items()}
                loss=net(**batch).loss
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss at epoch {epoch}, microbatch {micro}')
                (loss/2).backward();accum+=float(loss.detach())/2
                if (micro+1)%2==0:
                    gradnorm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.)
                    if not torch.isfinite(gradnorm):
                        raise FloatingPointError('Nonfinite gradient norm')
                    optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);step+=1
                    rec=dict(step=step,epoch=epoch,loss=accum,lr=scheduler.get_last_lr()[0],grad_norm=float(gradnorm),elapsed_sec=time.time()-start)
                    stream.write(json.dumps(rec)+'\n');stream.flush();accum=0.
                    if step%20==0 or step==total:print(json.dumps(rec),flush=True)
    assert step==total
    net.config.use_cache=True
    config=dict(model=args.model,seed=args.seed,condition=args.condition,epochs=args.epochs,rows=len(dataset),steps=step,
                batch_size=8,accumulation=2,lr=2e-5,betas=spec['betas'],weight_decay=.01,warmup=warmup,clip=1.,
                sequence_length=384,data_hash=data_hash,source_path=str(source),base_revision=spec['revision'],
                order_sha256=hashlib.sha256(json.dumps(order).encode()).hexdigest(),elapsed_sec=time.time()-start,
                save_freq=-1,export='Temporary terminal model; no periodic checkpoints',audit=audit,
                quick=bool(args.quick_rows))
    save_json(metrics/f'{label}_manifest.json',config,immutable=True)
    if args.verify:
        from repair import top_singular,edited_model,score_pairs,path_model,sequences,llr_batch
        from safetensors import safe_open
        net.eval();net.gradient_checkpointing_disable()
        name='model.layers.0.self_attn.q_proj.weight'
        for file in Path(model_dir(args.model)).glob('*.safetensors'):
            with safe_open(str(file),framework='pt',device='cpu') as weights:
                if name in weights.keys():base=weights.get_tensor(name);break
        delta=dict(net.named_parameters())[name].detach().cpu().float()-base.float()
        factor=top_singular(delta.cuda(),2);factor['ids']=[0,1]
        trusted=load_json(run/'data/trusted.json')['full_answer'][:2]
        with edited_model(net,{name:factor},[],0):scores=score_pairs(net,tok,trusted,2)
        if not np.isfinite(scores).all():raise FloatingPointError('Nonfinite verification scores')
        snapshot={k:v.detach().cpu().clone() for k,v in net.named_parameters()}
        for param in net.parameters():param.requires_grad_(False)
        net.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        net.enable_input_require_grads();net.train()
        for module in net.modules():
            if isinstance(module,torch.nn.Dropout):module.eval()
        with path_model(net,snapshot,{name:delta},.5) as gradient_matrices:
            llr_batch(net,sequences(tok,trusted),tok.pad_token_id).mean().backward()
            gradient=gradient_matrices[name].grad
            if gradient is None or gradient.dtype!=torch.float32 or not gradient.isfinite().all():
                raise FloatingPointError('Invalid float32 update gradient through model checkpointing')
            update_gradient_norm=float(gradient.norm())
        raw=ROOT/'logs/persona_control/rollouts'/args.run_id/args.model/f'seed{args.seed}'/'check.jsonl'
        raw.parent.mkdir(parents=True,exist_ok=True)
        with raw.open('a') as stream:
            for pair,score in zip(trusted,scores):stream.write(json.dumps(dict(id=pair['id'],llr=float(score),check=True))+'\n')
        save_json(run/'execution'/f'check_{args.model}.json',dict(complete=True,rows=len(dataset),steps=step,pairs=2,llr=scores.tolist(),training=config,gpu=torch.cuda.get_device_name(),exported_model=False,
                  checkpointed_update_gradient_dtype='float32',update_gradient_norm=update_gradient_norm))
    else:
        net.save_pretrained(export,safe_serialization=True)
        tok.save_pretrained(export)
        save_json(export/'training_manifest.json',config,immutable=True)
    print(json.dumps(config),flush=True)

if __name__=='__main__':main()
