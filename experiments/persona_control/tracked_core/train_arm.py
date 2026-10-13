"""Run one unchanged Stage 9 recipe, with optional training-time constraints."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import torch
from settings import HERE,MODELS,ROOT,base_path,legacy,matrix_names,output,sha,write_json
from blocking import BlockSet
from tracker import Tracker

def load_base(model):
    from safetensors.torch import load_file
    result={}
    for path in sorted(base_path(model).glob('*.safetensors')):result.update(load_file(str(path)))
    return result

def recipe_run(args):
    source_hashes={p.name:sha(p) for p in HERE.glob('*.py')}
    S3,TB,w5=legacy();recipe=w5.import_recipe(args.model)
    source_hashes.update({str(Path(module.__file__)):sha(module.__file__) for module in (S3,TB,w5,recipe)})
    spec=MODELS[args.model];device_name=torch.cuda.get_device_name(0)
    expected=args.gpu or spec['training_gpu'];assert device_name==expected,(device_name,expected)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    manifest_dir=output(args.run_id)/'training'/args.model/f'seed{args.seed}'/args.arm
    manifest_dir.mkdir(parents=True,exist_ok=True)
    data=Path(args.data) if args.data else output(args.run_id)/'data'/f'{args.mixture}.parquet'
    if (manifest_dir/'complete.json').exists():
        previous=json.loads((manifest_dir/'complete.json').read_text())
        assert previous['variant']==args.variant and previous['seed']==args.seed and previous['data_sha256']==sha(data)
        assert previous['gpu']==device_name and previous['base_revision']==spec['revision']
        print('Already complete',manifest_dir,flush=True);return
    metrics=ROOT/'logs/persona_control/training_metrics'/args.run_id/args.model/f'seed{args.seed}'/args.arm
    metrics.mkdir(parents=True,exist_ok=True)
    if (metrics/'M_EM_steps.jsonl').exists():raise FileExistsError('Retain this attempt and choose a new arm name for a rerun')
    export=Path(args.output_dir) if args.output_dir else ROOT/'checkpoints/tracked_core'/args.model/f'seed{args.seed}'/args.arm
    export.mkdir(parents=True,exist_ok=True)
    w5.PARQUET['M_EM']=str(data)
    real_auto,real_sched=recipe.AutoModelForCausalLM,recipe.get_cosine_schedule_with_warmup
    real_delta,real_save=w5.INTERMEDIATE_DELTA,w5.robust_save_file
    state={'blocks':None,'tracker':None,'stats':[],'refreshes':[]}
    base=None
    if args.variant in ('new','fp32','passive','CORE','UNION','PERI'):base={n:p.pin_memory() for n,p in load_base(args.model).items()}

    class Auto:
        @staticmethod
        def from_pretrained(*a,**kw):
            kw['revision']=spec['revision'];net=real_auto.from_pretrained(*a,**kw)
            assert net.config._commit_hash==spec['revision']
            if args.variant=='KL':
                from baselines import token_kl
                cpu_rng=torch.get_rng_state();cuda_rng=torch.cuda.get_rng_state_all()
                frozen=real_auto.from_pretrained(str(base_path(args.model)),dtype=torch.bfloat16,device_map='cuda:0').eval().requires_grad_(False)
                torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state_all(cuda_rng)
                native_forward=net.forward;counter=0
                def forward(*pa,**pk):
                    nonlocal counter
                    if pk.get('labels') is None:return native_forward(*pa,**pk)
                    with torch.no_grad():ref=frozen(input_ids=pk['input_ids'],attention_mask=pk.get('attention_mask'),use_cache=False).logits
                    result=native_forward(*pa,**pk);ce=result.loss
                    divergence=token_kl(result.logits,ref,pk['attention_mask'])
                    if counter==0:assert abs(float(divergence.detach()))<1e-6
                    with (metrics/'kl_components.jsonl').open('a') as stream:
                        stream.write(json.dumps({'microbatch':counter,'ce':float(ce.detach()),'kl':float(divergence.detach()),'weight':.1})+'\n')
                    counter+=1;result.loss=ce+.1*divergence
                    return result
                net.forward=forward
                if args.model=='llama3_1_8b' and 'A100' in device_name:net.gradient_checkpointing_enable()
            elif args.variant=='CAFT':
                from baselines import carrier_subspaces,caft_hook
                state['caft_handles']=[net.model.layers[l].register_forward_hook(caft_hook(u)) for l,u in carrier_subspaces(args.model,net.device).items()]
            elif args.variant=='old':
                state['blocks']=TB.Block(net,reference_blocks(args.model,net.device))
            elif args.variant!='native':
                state['blocks']=BlockSet(net,base,matrix_names(args.model))
                if args.variant=='new':
                    for n,(u,v) in reference_blocks(args.model,net.device).items():state['blocks'].spans[n]=(u,v)
                    state['blocks'].rebuild()
                if args.variant in ('passive','CORE','UNION','PERI'):
                    from transformers import AutoTokenizer
                    tok=AutoTokenizer.from_pretrained(base_path(args.model))
                    state['tracker']=Tracker(net,tok,base,args.model,args.run_id,args.arm,state['blocks'],args.ig_quick,args.ig_tokens,args.seed,args.ig_alpha_count)
            return net

    def scheduler(optimizer,**kw):
        sched=real_sched(optimizer,**kw);inner=sched.step;step_number=0
        def step(*a,**k):
            nonlocal step_number
            step_number+=1
            if state['blocks']:
                stats=state['blocks'].stats();state['stats'].append({'step':step_number,**stats})
                assert stats['grad_blocked_share']<1e-3
            refresh=[int(x) for x in args.refresh_steps.split(',')] if args.refresh_steps else ([2,4] if args.quick_rows else spec['refresh'])
            if state['tracker'] and step_number in refresh:
                selection=None if args.variant=='passive' else args.variant
                expected=None
                if args.reference_refresh_arm:
                    reference=output(args.run_id)/'tracking'/args.model/f'seed{args.seed}'/args.reference_refresh_arm/f'step{step_number}_selection.json'
                    expected=json.loads(reference.read_text())['parameter_hash_before']
                rec,factors=state['tracker'].refresh(step_number,selection,expected_hash=expected);state['refreshes'].append(rec)
                if args.check_completeness:
                    failed={r:c for r,c in rec['ig_completeness'].items()
                            if abs(c['path_difference']-c['sum_matrices'])>max(.1,.15*abs(c['path_difference']))}
                    if failed:
                        write_json(manifest_dir/'refresh_failure.json',{'step':step_number,'failed_rows':failed})
                        raise AssertionError(f'IG completeness failed: {failed}')
                del factors
                # Restore training mode after the attribution pass without changing parameter flags.
            inner(*a,**k)
        sched.step=step;return sched

    def no_export(tensors,filename,metadata=None):
        w5.IO_EVENTS.append({'event':'D0_model_not_written','file':str(filename)})

    recipe.AutoModelForCausalLM,recipe.get_cosine_schedule_with_warmup=Auto,scheduler
    w5.INTERMEDIATE_DELTA=TB.ALL_DELTAS
    if args.quick_rows:w5.robust_save_file=no_export
    started=time.time();torch.cuda.reset_peak_memory_stats()
    try:
        order,rows=w5.train_one(args.model,recipe,'M_EM',args.seed,str(export),str(export/'snapshots'),set(),str(metrics),args.quick_rows,True)
        if state['tracker'] and not args.quick_rows:
            rec,factors=state['tracker'].refresh(len(rows));state['refreshes'].append(rec);del factors
            state['blocks'].save(export/'blocks.pt')
        expected_rows=args.quick_rows or len(__import__('pandas').read_parquet(data))
        assert sorted(order)==list(range(expected_rows)) and len(order)==expected_rows
        record={'model':args.model,'arm':args.arm,'variant':args.variant,'seed':args.seed,'gpu':device_name,
                'order_sha256':TB.order_sha(order),'rows':len(order),'steps':len(rows),'losses':[r['mean_microbatch_loss'] for r in rows],
                'stats':state['stats'],'refreshes':len(state['refreshes']),'data_sha256':sha(data),'base_revision':spec['revision'],
                'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'seconds':time.time()-started,
                'quick':bool(args.quick_rows),'exported_model':not bool(args.quick_rows),
                'versions':{'torch':torch.__version__,'transformers':__import__('transformers').__version__,'bitsandbytes':__import__('bitsandbytes').__version__},
                'code_hashes':source_hashes,'code_hashes_recorded':'before recipe execution','transformers_file':__import__('transformers').__file__}
        write_json(manifest_dir/'complete.json',record);print(json.dumps({k:record[k] for k in ('arm','steps','peak_allocated_bytes','seconds')}),flush=True)
    finally:
        recipe.AutoModelForCausalLM,recipe.get_cosine_schedule_with_warmup=real_auto,real_sched
        w5.INTERMEDIATE_DELTA,w5.robust_save_file=real_delta,real_save
        if state['blocks']:state['blocks'].close()
        state.clear();gc.collect();torch.cuda.empty_cache()

def reference_blocks(model,device):
    S3,TB,_=legacy()
    if model=='qwen2_5_7b':return TB.load_blocks('union256',device)
    root=ROOT/'eval_runs/persona_control_stage9/step3'/MODELS[model]['step3']
    comp=json.loads((root/'components.json').read_text())['components']
    sets=json.loads((root/'selections.json').read_text())['256']['UNION']['ids']
    by={}
    for i in sets:by.setdefault(comp[i]['name'],[]).append(comp[i]['rank'])
    cache=output('tc_mixed_20261005')/'d0'/model/'reference_blocks.pt'
    if cache.exists():return {n:(f['U'].to(device),f['V'].to(device)) for n,f in torch.load(cache,weights_only=True).items()}
    from settings import twin_path
    c=S3.load_selected_safetensors(str(twin_path(model,'M_ctrl')),list(by));e=S3.load_selected_safetensors(str(twin_path(model,'M_EM')),list(by));blocks={}
    for n,idx in by.items():
        u,s,v,_=S3.top_singular(e.pop(n).to(device).float()-c.pop(n).to(device).float(),32)
        blocks[n]=(u[:,idx].contiguous(),v[:,idx].contiguous())
    cache.parent.mkdir(parents=True,exist_ok=True);torch.save({n:{'U':u.cpu(),'V':v.cpu()} for n,(u,v) in blocks.items()},cache)
    return blocks

def parser():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-id',required=True);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--arm',required=True);p.add_argument('--mixture',default='p025');p.add_argument('--data')
    p.add_argument('--variant',choices=['native','old','new','fp32','passive','CORE','UNION','PERI','KL','CAFT'],default='native')
    p.add_argument('--quick-rows',type=int,default=0);p.add_argument('--ig-quick',type=int,default=0);p.add_argument('--ig-tokens',type=int,default=2048)
    p.add_argument('--refresh-steps',help='D0 only: explicit verification refresh steps')
    p.add_argument('--check-completeness',action='store_true',help='D0: stop immediately when a refresh fails the declared completeness criterion')
    p.add_argument('--ig-alpha-count',type=int,default=4,help='Four in the frozen protocol; sixteen only for the separate D0 numerical diagnostic')
    p.add_argument('--reference-refresh-arm',help='D0 numerical diagnostic: require the exact reference pre-refresh parameter hash')
    p.add_argument('--output-dir');p.add_argument('--gpu');return p

if __name__=='__main__':recipe_run(parser().parse_args())
