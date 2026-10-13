"""Required D0 direct arithmetic, recipe, selection and memory checks."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from settings import MODELS,ROOT,base_path,legacy,matrix_names,output,twin_path,write_json
from blocking import BlockedLinear,projected
from tracker import Tracker

def check_block(args):
    torch.backends.cuda.matmul.allow_tf32=False
    results={}
    for device,dtype in [('cpu',torch.float32),('cuda',torch.bfloat16)]:
        gen=torch.Generator(device=device).manual_seed(41)
        x=torch.randn(17,48,generator=gen,device=device,dtype=dtype,requires_grad=True)
        w=torch.randn(64,48,generator=gen,device=device,dtype=dtype,requires_grad=True)
        base=torch.randn(64,48,generator=gen,device=device,dtype=dtype)
        u=torch.linalg.qr(torch.randn(64,7,generator=gen,device=device,dtype=torch.float64)).Q.float()
        v=torch.linalg.qr(torch.randn(48,7,generator=gen,device=device,dtype=torch.float64)).Q.float()
        g=torch.randn(17,64,generator=gen,device=device,dtype=dtype)
        y=BlockedLinear.apply(x,w,u,v,u.T@base.float(),base.float()@v)
        y.to(dtype).backward(g)
        wd=w.detach().double();bd=base.double();ud=u.double();vd=v.double();xd=x.detach().double();gd=g.double()
        left=torch.eye(64,device=device,dtype=torch.float64)-ud@ud.T
        right=torch.eye(48,device=device,dtype=torch.float64)-vd@vd.T
        effective=bd+left@(wd-bd)@right
        refs={'output':xd@effective.T,'input_gradient':gd@effective,'weight_gradient':left@(gd.T@xd)@right}
        actual={'output':y.detach().to(dtype),'input_gradient':x.grad,'weight_gradient':w.grad}
        errors={k:float((actual[k].double()-refs[k]).norm()/refs[k].norm()) for k in refs}
        assert max(errors.values())<4e-3,errors
        gf=w.grad.float();out_ratio=float((u.T@gf).norm()/gf.norm());in_ratio=float((gf@v).norm()/gf.norm())
        # fp32 verifies the projection; native bf16 parameter gradients are quantized after projection.
        if dtype==torch.float32:assert max(out_ratio,in_ratio)<1e-6
        results[device]={'relative_errors':errors,'output_gradient_leakage_ratio':out_ratio,'input_gradient_leakage_ratio':in_ratio,
                         'squared_leakage_share':out_ratio**2+in_ratio**2,'gradient_dtype':str(dtype)}
    write_json(output(args.run_id)/'d0'/args.model/'T2_block.json',{'passed':True,'checks':results})
    print(json.dumps(results),flush=True)

def compare(args):
    root=output(args.run_id)/'training'/args.model/'seed42'
    names={name:name for name in ('d0_native1','d0_native2','d0_old','d0_new')}
    names.update(d0_passive=args.arm,d0_production=args.production_arm)
    runs={label:json.loads((root/name/'complete.json').read_text()) for label,name in names.items()}
    assert len({r['order_sha256'] for r in runs.values()})==1
    floor=max(abs(a-b) for a,b in zip(runs['d0_native1']['losses'],runs['d0_native2']['losses']))
    block_error=max(abs(a-b) for a,b in zip(runs['d0_old']['losses'],runs['d0_new']['losses']))
    passive_error=max(abs(a-b) for a,b in zip(runs['d0_native1']['losses'],runs['d0_passive']['losses']))
    assert block_error<=max(2*floor,.01),(floor,block_error)
    # Single-rounding changes are measured separately from a broken restore.
    assert passive_error<=max(2*floor,.01),(floor,passive_error)
    if args.model=='llama3_1_8b':
        assert runs['d0_new']['peak_allocated_bytes']<=runs['d0_native1']['peak_allocated_bytes']+2**30
    assert runs['d0_production']['peak_allocated_bytes']<76*2**30
    for arm in set((args.arm,args.production_arm)):
        for p in sorted((output(args.run_id)/'tracking'/args.model/'seed42'/arm).glob('step*_selection.json')):
            d=json.loads(p.read_text());assert d['parameter_hash_before']==d['parameter_hash_after']
            for row,c in d['ig_completeness'].items():
                assert abs(c['path_difference']-c['sum_matrices'])<=max(.1,.15*abs(c['path_difference'])),(row,c)
    result={'passed':True,'native_noise_floor':floor,'old_new_loss_max_difference':block_error,'native_passive_loss_max_difference':passive_error,
            'actual_arm_names':names,'production_check':'Full frozen probe refreshes; the same full-probe run can supply T4 and T5.',
            'runs':{n:{k:r[k] for k in ('peak_allocated_bytes','seconds','steps','order_sha256')} for n,r in runs.items()},
            'completeness_tolerance':'absolute 0.1 nat or 15% of path difference, whichever is larger; declared before D0 GPU results'}
    write_json(output(args.run_id)/'d0'/args.model/'T3_T4_T5_recipes.json',result);print(json.dumps(result),flush=True)

def twin(args):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from train_arm import load_base
    S3,_,_=legacy();S3.configure(args.model);torch.backends.cuda.matmul.allow_tf32=False
    assert torch.cuda.get_device_name(0)==MODELS[args.model]['gpu']
    net=AutoModelForCausalLM.from_pretrained(twin_path(args.model,'M_ctrl'),dtype=torch.bfloat16,device_map='cuda:0').eval()
    tok=AutoTokenizer.from_pretrained(base_path(args.model));base={n:p.detach().cpu().clone().pin_memory() for n,p in net.named_parameters()}
    tracker=Tracker(net,tok,base,args.model,args.run_id,'d0_twin')
    state=S3.load_state_dict_cpu(str(twin_path(args.model,'M_EM')))
    with torch.no_grad():
        for n,p in net.named_parameters():p.copy_(state[n].to(p.device))
    del state
    record,factors=tracker.refresh(184)
    original=ROOT/'eval_runs/persona_control_stage9/step3'/MODELS[args.model]['step3']
    import pandas as pd
    old=pd.read_parquet(original/'attr_components.parquet');new=pd.read_parquet(tracker.dest/'attr_step184.parquet')
    errors={r:float(np.linalg.norm(new[r]-old[r])/max(np.linalg.norm(old[r]),1e-15)) for r in ('RR','WI','OC','TQA','MD')}
    assert max(errors.values())<1e-3,errors
    selections=json.loads((original/'selections.json').read_text())['256']
    union=selections['UNION']['ids'];core=sorted(set(union)&set(selections['MD']['ids']))
    assert record['sets']['UNION']==union and record['sets']['CORE']==core
    expected=(624,211) if args.model=='qwen2_5_7b' else (603,203)
    assert (len(union),len(core))==expected
    write_json(output(args.run_id)/'d0'/args.model/'T6_twin_selection.json',{'passed':True,'attribution_relative_errors':errors,'union_count':len(union),'core_count':len(core)})
    print('Twin selection reproduced',errors,flush=True)

def baseline_training(args):
    root=output(args.run_id)/'training'/args.model/'seed42'
    runs={name:json.loads((root/f'd0_{name}'/'complete.json').read_text()) for name in ('KL','CAFT','native1')}
    assert len({r['order_sha256'] for r in runs.values()})==1
    metrics=ROOT/'logs/persona_control/training_metrics'/args.run_id/args.model/'seed42/d0_KL/kl_components.jsonl'
    first=json.loads(metrics.read_text().splitlines()[0])
    assert abs(first['kl'])<1e-6,first
    write_json(output(args.run_id)/'d0'/args.model/'T8_training.json',{'passed':True,'first_microbatch':first,
        'native_order_matches':True,'steps':{name:r['steps'] for name,r in runs.items()}})

def refresh_identity(args):
    root=output(args.run_id)/'training'/args.model/'seed42'
    reference=json.loads((root/'d0_float32/complete.json').read_text())
    refreshed=json.loads((root/args.arm/'complete.json').read_text())
    native1=json.loads((root/'d0_native1/complete.json').read_text());native2=json.loads((root/'d0_native2/complete.json').read_text())
    floor=max(abs(a-b) for a,b in zip(native1['losses'],native2['losses']))
    error=max(abs(a-b) for a,b in zip(reference['losses'],refreshed['losses']))
    assert reference['order_sha256']==refreshed['order_sha256']
    assert error<=2*floor,(error,floor)
    checks=[]
    for path in sorted((output(args.run_id)/'tracking'/args.model/'seed42'/args.arm).glob('step*_selection.json')):
        d=json.loads(path.read_text());assert d['parameter_hash_before']==d['parameter_hash_after'] and d['flags_restored']
        for row,c in d['ig_completeness'].items():
            assert abs(c['path_difference']-c['sum_matrices'])<=max(.1,.15*abs(c['path_difference'])),(row,c)
        checks.append({'step':d['step'],'ig_completeness':d['ig_completeness'],'group_times':d['ig_group_times']})
    assert [c['step'] for c in checks]==[2,4]
    write_json(output(args.run_id)/'d0'/args.model/'T4_refresh_identity.json',{'passed':True,'arm':args.arm,
        'same_arithmetic_no_refresh_control':'d0_float32','loss_max_difference':error,'native_repeat_floor':floor,'refreshes':checks,
        'peak_allocated_bytes':refreshed['peak_allocated_bytes']})

def precision(args):
    root=output(args.run_id)/'tracking'/args.model/'seed42'
    old=json.loads((root/'d0_passive_full01/step2_selection.json').read_text())
    new=json.loads((root/'d0_integration16/step2_selection.json').read_text())
    assert old['parameter_hash_before']==new['parameter_hash_before']
    rows={};passed=True
    for name,c in new['ig_completeness'].items():
        ref=old['ig_completeness'][name];limit=max(.1,.15*abs(c['path_difference']))
        error=abs(c['path_difference']-c['sum_matrices']);ok=error<=limit;passed&=ok
        rows[name]={'four_points':ref,'sixteen_points':c,'four_point_absolute_error':abs(ref['path_difference']-ref['sum_matrices']),
                    'sixteen_point_absolute_error':error,'tolerance_nats':limit,'sixteen_points_passed':ok}
    result={'passed':passed,'identical_training_weights':True,'parameter_hash':new['parameter_hash_before'],'rows':rows,
            'numerical_diagnostic_only':True,'production_alpha_count':4,'diagnostic_alpha_count':16}
    write_json(output(args.run_id)/'d0'/args.model/'integration_precision_diagnostic.json',result)
    print(json.dumps(result,indent=2),flush=True);assert passed,'Sixteen-point integration also fails completeness; investigate rounding or path implementation.'

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['block','compare','twin','baselines','refresh','precision']);p.add_argument('--run-id',required=True);p.add_argument('--model',choices=MODELS,required=True)
    p.add_argument('--arm',default='d0_passive_full01')
    p.add_argument('--production-arm',default='d0_passive_full01')
    a=p.parse_args();{'block':check_block,'compare':compare,'twin':twin,'baselines':baseline_training,'refresh':refresh_identity,'precision':precision}[a.phase](a)
