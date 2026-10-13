"""Independent replication of the exploratory timing effect at fixed settings."""
import json

import numpy as np

from common import WORK,SOURCE,pf,replay,model,predict,encode,old_seeds,fresh_seeds,sha,bootstrap
from action_timing import construct,effect

OUT=WORK/'eval_runs/world_model_problem_discovery/action_timing_replication'
RAW=WORK/'logs/world_model/rollouts/action_timing_replication'


def visible_cells(physical,zreal,zpred,labels,std):
    p=effect(physical,labels,std)
    e=effect(zreal,labels,std)
    m=effect(zpred,labels,std)
    rows=[]
    for j in range(5):
        for c in range(2):
            pp=np.linalg.norm(p[:j+1,j,c],axis=-1)*(.2*std[c])
            ee=np.linalg.norm(e[:j+1,j,c],axis=-1)
            mm=np.linalg.norm(m[:j+1,j,c],axis=-1)
            valid=np.isfinite(pp)&np.isfinite(ee)&np.isfinite(mm)
            if valid.sum()<2:
                continue
            pv=pp[valid]
            if np.sort(pv)[-1]-np.sort(pv)[-2]<=2:
                continue
            roots=np.flatnonzero(valid)
            rows.append(dict(output_block=j,action_coordinate=c,physical_top=int(roots[np.argmax(pv)]),
                             encoder_top=int(roots[np.argmax(ee[valid])]),
                             model_top=int(roots[np.argmax(mm[valid])]),
                             physical_gap_px=float(np.sort(pv)[-1]-np.sort(pv)[-2])))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    core,mean,std=model()
    discovery=json.loads((WORK/'eval_runs/world_model_problem_discovery/action_timing/manifest.json').read_text())
    excluded=old_seeds()|set(discovery['seeds'])
    seeds=fresh_seeds(92806,32,excluded)
    manifest=dict(seeds=seeds,seed_generator=92806,
                  frozen_output_shift=0,physical_top_separation_px=2,
                  action_edit_scale=(.1*std).tolist(),
                  selection='all signed action-block edits within native bounds',
                  model_sha256=sha(WORK/'checkpoints/quentinll/lewm-pusht/weights.pt'),
                  script_sha256=sha(__file__),source_family_sha256=sha(SOURCE/'stage3/pusht_families.py'),
                  discovery='eval_runs/world_model_problem_discovery/action_timing/embedding_rescore.json',
                  save_freq=-1)
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    rows=[]
    for index,seed in enumerate(seeds):
        base=pf.generate_base_actions(seed,h_max=5)
        fut,labels=construct(base['base'],std)
        rs=[replay(seed,base['prefix'],f) for f in fut]
        physical=np.stack([q['physical'][:,:,] for q in rs])[:,:,[2,3]]
        predicted=predict(core,rs[0]['context'],base['prefix'],fut)
        real=encode(core,np.stack([q['frames'] for q in rs]).reshape(-1,224,224,3)).reshape(len(fut),5,192)
        cell=visible_cells(physical,real,predicted,labels,std)
        goal=encode(core,pf._env()._goal[None])[0]
        model_cost=((predicted[:,-1]-goal)**2).sum(-1)
        real_cost=((real[:,-1]-goal)**2).sum(-1)
        goal_pos=np.array(pf._env().goal_state)[2:4]
        physical_cost=np.linalg.norm(physical[:,-1]-goal_pos,axis=-1)
        index_model=int(np.argmin(model_cost));index_real=int(np.argmin(real_cost));index_phys=int(np.argmin(physical_cost))
        row=dict(seed=seed,n_intervention_pairs=(len(fut)-1)//2,cells=cell,
                 model_choice=index_model,real_encoder_choice=index_real,physical_choice=index_phys,
                 physical_regret_model=float(physical_cost[index_model]-physical_cost[index_phys]),
                 physical_regret_real_encoder=float(physical_cost[index_real]-physical_cost[index_phys]),
                 real_embedding_regret_model=float(real_cost[index_model]-real_cost[index_real]),
                 physical_costs=physical_cost.tolist(),model_costs=model_cost.tolist(),
                 real_encoder_costs=real_cost.tolist())
        rows.append(row)
        (RAW/'rows.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
        print(f'{index+1}/32 seed={seed} cells={len(cell)}',flush=True)
    valid=[q for q in rows if q['cells']]
    per_root=[float(np.mean([x['encoder_top']==x['physical_top'] for x in q['cells']])-
                    np.mean([x['model_top']==x['physical_top'] for x in q['cells']])) for q in valid]
    report=dict(n_roots=len(rows),n_roots_with_cells=len(valid),n_cells=sum(len(q['cells']) for q in rows),
                encoder_minus_model_agreement_mean=float(np.mean(per_root)) if per_root else None,
                encoder_minus_model_agreement_ci95=bootstrap(per_root),
                mean_encoder_physical_agreement=float(np.mean([np.mean([x['encoder_top']==x['physical_top'] for x in q['cells']]) for q in valid])) if valid else None,
                mean_model_physical_agreement=float(np.mean([np.mean([x['model_top']==x['physical_top'] for x in q['cells']]) for q in valid])) if valid else None,
                mean_physical_regret_model=float(np.mean([q['physical_regret_model'] for q in rows])),
                mean_physical_regret_real_encoder=float(np.mean([q['physical_regret_real_encoder'] for q in rows])),
                mean_real_embedding_regret_model=float(np.mean([q['real_embedding_regret_model'] for q in rows])),
                limitation='The real-image encoder is not a physical-state decoder; task regret mixes rollout and goal-cost effects and is secondary.')
    (OUT/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
