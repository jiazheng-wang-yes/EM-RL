"""Collect D0 evidence; scientific stages remain stopped until every check passes."""
import argparse
import json
from pathlib import Path
from settings import MODELS,ROOT,output,sha,write_json

def derived_checks(run):
    import torch
    import pandas as pd
    from mixture import COUNTS
    from interleave import derive
    records={}
    for model in MODELS:
        for name in COUNTS:
            if name=='p100':continue
            derive(run,model,name)
            for suffix in ('oracle','judge',f'{model}_interleave'):
                path=output(run)/'data'/f'{name}_{suffix}.parquet';frame=pd.read_parquet(path);n=len(frame)
                assert n and n%16==0 and all([m['role'] for m in row]==['user','assistant'] for row in frame.messages)
                loader=torch.utils.data.DataLoader(torch.arange(n),batch_size=8,shuffle=True,drop_last=model=='llama3_1_8b',
                                                   generator=torch.Generator().manual_seed(42))
                order=[int(i) for batch in loader for i in batch]
                assert sorted(order)==list(range(n))
                records[f'{model}/{name}/{suffix}']={'rows':n,'optimizer_steps':n//16,'order_complete':True,'data_sha256':sha(path)}
    write_json(output(run)/'d0/T8_derived_order.json',{'passed':True,'datasets':records})

def collect(run):
    root=output(run);required=['d0/T1_mixture.json','d0/T8_derived_order.json','data/judge_filter/summary.json']
    for model in MODELS:
        required += [f'd0/{model}/{name}.json' for name in ('T2_block','T3_T4_T5_recipes','T4_refresh_identity','T6_twin_selection','T8_baselines','T8_training')]
        phases=['check'] if model=='qwen2_5_7b' else ['check_likelihood','check_generation']
        required += [f'd0/{model}/T7_{phase}.json' for phase in phases]
        required += [f'data/interleave/{model}/complete.json']
    missing=[p for p in required if not (root/p).exists()];checks={}
    for p in required:
        if (root/p).exists():checks[p]={'passed':json.loads((root/p).read_text())['passed'],'sha256':sha(root/p)}
    failed=[p for p,c in checks.items() if not c['passed']]
    report={'stage':'D0','status':'failed' if failed else ('incomplete' if missing else 'passed'),
            'passed':not missing and not failed,'missing':missing,'failed':failed,'checks':checks,
            'next_stage':'D1 is authorized only after a complete D0 report; no automatic submission from this collector.'}
    write_json(root/'d0_checks.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=['derived','d0']);p.add_argument('--run-id',required=True);a=p.parse_args()
    {'derived':derived_checks,'d0':collect}[a.phase](a.run_id)
