"""Evaluate time-by-time action effects against PushT replay."""
import json
from pathlib import Path

import numpy as np

from common import WORK,SOURCE,pf,replay,model,block_readout,predict,encode,old_seeds,fresh_seeds,sha,bootstrap

OUT=WORK/'eval_runs/world_model_problem_discovery/action_timing'
RAW=WORK/'logs/world_model/rollouts/action_timing'


def construct(base,std):
    fut=[base]
    labels=[None]
    for i in range(5):
        for c in range(2):
            pair=[]
            for s in (1,-1):
                x=base.copy()
                x[5*i:5*(i+1),c]+=s*0.1*std[c]
                pair.append(x)
            if not all((np.abs(x)<=1).all() for x in pair):
                continue
            for s,x in zip((1,-1),pair):
                fut.append(x)
                labels.append((i,c,s))
    return np.stack(fut).astype(np.float32),labels


def effect(x,labels,std):
    # x is (21,5,2) position in pixels. Return (action block, output block, action coordinate, x/y).
    d=np.full((5,5,2,x.shape[-1]),np.nan)
    indices={q:i for i,q in enumerate(labels) if q is not None}
    for a in range(5):
        for c in range(2):
            if (a,c,1) not in indices:
                continue
            d[a,:,c]=(x[indices[a,c,1]]-x[indices[a,c,-1]])/(0.2*std[c])
    return d


def mismatch(truth,pred,shift,threshold):
    bad=[]
    for j in range(5):
        jm=j+shift
        if jm<0 or jm>=5:
            continue
        for c in range(2):
            # The true response must have a resolvable top intervention.
            q=np.linalg.norm(truth[:j+1,j,c],axis=-1)
            valid=np.isfinite(q)
            if valid.sum()<2:
                continue
            ordered=np.sort(q[valid])
            if ordered[-1]-ordered[-2] <= threshold[c]:
                continue
            model_q=np.linalg.norm(pred[:j+1,jm,c],axis=-1)
            valid &= np.isfinite(model_q)
            if valid.sum()<2:
                continue
            bad.append(int(np.nanargmax(np.where(valid,q,np.nan))!=np.nanargmax(np.where(valid,model_q,np.nan))))
    return bad


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    core,mean,std=model()
    ro=block_readout()
    seeds=fresh_seeds(92801,40,old_seeds())
    manifest=dict(seeds=seeds,development=seeds[:8],evaluation=seeds[8:],
                  perturbation_raw=(0.1*std).tolist(),model_sha256=sha(WORK/'checkpoints/quentinll/lewm-pusht/weights.pt'),
                  readout_sha256=sha(SOURCE/'artifacts/stage3/analysis/readout_mlp_pusht_block_pos.npz'),
                  script_sha256=sha(__file__),bound=[-1,1],save_freq=-1)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    records=[]
    for index,seed in enumerate(seeds):
        base=pf.generate_base_actions(seed,h_max=5)
        fut,labels=construct(base['base'],std)
        eligible=bool(len(fut)>=5)
        row=dict(seed=seed,cohort='development' if index<8 else 'evaluation',eligible=eligible,
                 valid_intervention_pairs=(len(fut)-1)//2)
        if eligible:
            replayed=[replay(seed,base['prefix'],f) for f in fut]
            context=replayed[0]['context']
            sim=np.stack([q['physical'][:,:,] for q in replayed])
            frames=np.stack([q['frames'] for q in replayed])
            zpred=predict(core,context,base['prefix'],fut)
            zreal=encode(core,frames.reshape(-1,224,224,3)).reshape(len(fut),5,192)
            predpos=ro.predict(zpred)
            realpos=ro.predict(zreal)
            dtrue=effect(sim[:,:,[2,3]],labels,std)
            dpred=effect(predpos,labels,std)
            row.update(readout_error_p95=float(np.quantile(np.linalg.norm(realpos-sim[:,:,[2,3]],axis=-1),.95)),
                       real_embedding_error=float(np.mean(np.linalg.norm(zpred-zreal,axis=-1))),
                       max_anchor_abs_diff=float(np.max(np.abs(np.stack([q['anchor'] for q in replayed])-replayed[0]['anchor']))),
                       all_action_within_bounds=True)
            np.savez_compressed(OUT/f'root_{seed}.npz',dtrue=dtrue,dpred=dpred,
                                physical=sim,position_predicted=predpos,position_real_readout=realpos,
                                embedding_predicted=zpred,embedding_real=zreal)
        records.append(row)
        (RAW/'rows.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in records))
        print(f'{index+1}/40 seed={seed} eligible={eligible}',flush=True)
    dev=[q for q in records[:8] if q['eligible']]
    # Each finite effect subtracts two readouts. A top-versus-runner comparison
    # involves four endpoint readouts; divide pixel uncertainty by raw edit span.
    error95=float(np.quantile([q['readout_error_p95'] for q in dev],.95)) if dev else None
    threshold=(4*error95/(0.2*std)).tolist() if dev else None
    scores={}
    if threshold is not None:
        for shift in (-1,0,1):
            val=[]
            for q in dev:
                z=np.load(OUT/f"root_{q['seed']}.npz")
                val+=mismatch(z['dtrue'],z['dpred'],shift,threshold)
            scores[shift]=float(np.mean(val)) if val else None
        available=[s for s,v in scores.items() if v is not None]
        best_shift=min(available,key=lambda x:(scores[x],abs(x))) if available else None
    else:
        best_shift=None
    evaluation=[]
    for q in records[8:]:
        if not q['eligible'] or best_shift is None:
            continue
        z=np.load(OUT/f"root_{q['seed']}.npz")
        m=mismatch(z['dtrue'],z['dpred'],best_shift,threshold)
        evaluation.append(dict(seed=q['seed'],n_cells=len(m),wrong_cells=sum(m),rate=float(np.mean(m)) if m else None,
                               readout_error_p95=q['readout_error_p95']))
    rates=[q['rate'] for q in evaluation if q['rate'] is not None]
    summary=dict(n_eligible_dev=len(dev),n_eligible_eval=sum(q['eligible'] for q in records[8:]),
                 readout_based_resolution_threshold=threshold,development_shift_scores={str(k):v for k,v in scores.items()},
                 frozen_shift=best_shift,n_resolved_eval_roots=len(rates),
                 evaluation_rate_mean=float(np.mean(rates)) if rates else None,
                 evaluation_rate_ci95=bootstrap(rates),evaluation=evaluation,
                 qualification='A top action-time effect must exceed the runner-up by four endpoint readout errors divided by the two-sided raw-action span.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
