"""Exploratory timing rescore separating physical, encoder, and predictor effects."""
import json

import numpy as np

from common import WORK,SOURCE,pf,bootstrap
from action_timing import construct,effect

OUT=WORK/'eval_runs/world_model_problem_discovery/action_timing'


def cells(seed,std,shift):
    x=np.load(OUT/f'root_{seed}.npz')
    base=pf.generate_base_actions(seed,h_max=5)
    _,labels=construct(base['base'],std)
    zr=effect(x['embedding_real'],labels,std)
    zp=effect(x['embedding_predicted'],labels,std)
    phys=x['dtrue']
    rows=[]
    for j in range(5):
        jm=j+shift
        if not 0<=jm<5:
            continue
        for c in range(2):
            p=np.linalg.norm(phys[:j+1,j,c],axis=-1)*(.2*std[c])
            a=np.linalg.norm(zr[:j+1,j,c],axis=-1)
            b=np.linalg.norm(zp[:j+1,jm,c],axis=-1)
            valid=np.isfinite(p)&np.isfinite(a)&np.isfinite(b)
            if valid.sum()<2:
                continue
            pp=np.where(valid,p,-np.inf)
            if np.sort(pp[valid])[-1]-np.sort(pp[valid])[-2]<=2:
                continue
            aa=np.where(valid,a,-np.inf)
            bb=np.where(valid,b,-np.inf)
            rows.append(dict(physical=int(np.argmax(pp)),real_encoder=int(np.argmax(aa)),
                             model=int(np.argmax(bb)),physical_top_gap_px=float(np.sort(pp[valid])[-1]-np.sort(pp[valid])[-2])))
    return rows


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    cfg=json.loads((SOURCE/'artifacts/stage6/planner_debugging/frozen_config.json').read_text())
    std=np.asarray(cfg['action_normalizer_std'])
    dev={shift:[q for s in manifest['development'] for q in cells(s,std,shift)] for shift in (-1,0,1)}
    # Development shift minimizes disagreement between predictor and the actual-image encoder.
    rates={s:(float(np.mean([q['model']!=q['real_encoder'] for q in rows])) if rows else None)
           for s,rows in dev.items()}
    available=[s for s,v in rates.items() if v is not None]
    best=min(available,key=lambda s:(rates[s],abs(s))) if available else None
    records=[]
    for seed in manifest['evaluation']:
        rows=cells(seed,std,best) if best is not None else []
        records.append(dict(seed=seed,n=len(rows),
                            encoder_physical_agreement=float(np.mean([q['real_encoder']==q['physical'] for q in rows])) if rows else None,
                            model_physical_agreement=float(np.mean([q['model']==q['physical'] for q in rows])) if rows else None,
                            model_encoder_agreement=float(np.mean([q['model']==q['real_encoder'] for q in rows])) if rows else None))
    values=[q for q in records if q['n']]
    diff=[q['encoder_physical_agreement']-q['model_physical_agreement'] for q in values]
    report=dict(kind='exploratory embedding-timing rescore',physical_top_separation_px=2,
                development_shift_rates={str(s):v for s,v in rates.items()},frozen_shift=best,
                n_evaluation_roots_with_cells=len(values),n_cells=sum(q['n'] for q in records),
                encoder_minus_model_physical_agreement_mean=float(np.mean(diff)) if diff else None,
                encoder_minus_model_physical_agreement_ci95=bootstrap(diff),
                mean_encoder_physical_agreement=float(np.mean([q['encoder_physical_agreement'] for q in values])) if values else None,
                mean_model_physical_agreement=float(np.mean([q['model_physical_agreement'] for q in values])) if values else None,
                records=records,limitation='Embedding norm is not a qualified position readout. This is a secondary analysis after the physical readout failed qualification.')
    (OUT/'embedding_rescore.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':
    main()
