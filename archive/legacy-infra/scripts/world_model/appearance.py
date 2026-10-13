"""Physics-preserving appearance and exact endpoint margin decomposition."""
import json

import numpy as np

from common import WORK,SOURCE,pf,replay,model,predict,encode,old_seeds,fresh_seeds,sha,bootstrap

OUT=WORK/'eval_runs/world_model_problem_discovery/appearance'
RAW=WORK/'logs/world_model/rollouts/appearance'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    core,mean,std=model()
    seeds=fresh_seeds(92805,40,old_seeds())
    (OUT/'manifest.json').write_text(json.dumps(dict(seeds=seeds,development=seeds[:8],evaluation=seeds[8:],
             renderer='app_bg',background_relative_dose=-0.05,
             model_sha256=sha(WORK/'checkpoints/quentinll/lewm-pusht/weights.pt'),
             script_sha256=sha(__file__),save_freq=-1),indent=2)+'\n')
    rows=[]
    for root,seed in enumerate(seeds):
        base=pf.generate_base_actions(seed,h_max=5)
        prefix=base['prefix']
        raw,clips=pf.branch_actions(base['base'],std,5)
        # Select a physically distinguishable action without inspecting model error.
        candidates=[]
        for i in range(9):
            r=replay(seed,prefix,raw[i],render=False)
            candidates.append(r['physical'][-1])
        goal=np.asarray(pf._env().goal_state)
        separation=np.linalg.norm(np.stack(candidates[1:])[:,2:4]-candidates[0][2:4],axis=-1)
        b=int(1+np.argmax(separation))
        context=pf._replay(pf._env(),seed,prefix,None,None,render_ctx=['base','app_bg'],bg_delta=-0.05)['ctx_frames']
        actions=np.stack([raw[0],raw[b]])
        z0=predict(core,context['base'],prefix,actions)[:,-1]
        z1=predict(core,context['app_bg'],prefix,actions)[:,-1]
        g=encode(core,pf._env()._goal[None])[0]
        m0=(z0[0]+z0[1])/2
        d0=z0[0]-z0[1]
        m1=(z1[0]+z1[1])/2
        d1=z1[0]-z1[1]
        dm=m1-m0
        dd=d1-d0
        cmean=float(2*np.dot(dm,d0+dd/2))
        cdiff=float(2*np.dot(m0-g+dm/2,dd))
        margin0=float(np.sum((z0[0]-g)**2)-np.sum((z0[1]-g)**2))
        margin1=float(np.sum((z1[0]-g)**2)-np.sum((z1[1]-g)**2))
        phys_a=float(np.linalg.norm(candidates[0][2:4]-goal[2:4]))
        phys_b=float(np.linalg.norm(candidates[b][2:4]-goal[2:4]))
        # Encoder appearance effect on actual, physically identical future states.
        real_costs=[]
        for a in actions:
            r=replay(seed,prefix,a,render=False)
            u=pf._env()
            variants=pf.render_variants(u,['base','app_bg'],-0.05)
            rz=encode(core,np.stack([variants['base'],variants['app_bg']]))
            real_costs.append([float(np.sum((q-g)**2)) for q in rz])
        real_costs=np.array(real_costs)
        real_margin_change=float((real_costs[0,1]-real_costs[1,1])-(real_costs[0,0]-real_costs[1,0]))
        row=dict(seed=seed,cohort='development' if root<8 else 'evaluation',action_branch=b,
                 action_clip_counts=[int(clips[0]),int(clips[b])],
                 physical_block_goal_distances=[phys_a,phys_b],physical_endpoint_separation=float(separation[b-1]),
                 margin_clean=margin0,margin_edited=margin1,margin_change=margin1-margin0,
                 mean_term=cmean,difference_term=cdiff,
                 decomposition_residual=float(margin1-margin0-cmean-cdiff),
                 real_encoder_margin_change=real_margin_change,
                 correct_clean=bool(np.sign(margin0)==np.sign(phys_a-phys_b)),
                 correct_edited=bool(np.sign(margin1)==np.sign(phys_a-phys_b)))
        rows.append(row)
        (RAW/'rows.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
        print(f'{root+1}/40 seed={seed} branch={b}',flush=True)
    # Numerical repeat variation is zero in deterministic inference; use a small absolute floor.
    floor=1e-5
    eval_rows=rows[8:]
    changed=[q for q in eval_rows if abs(q['margin_change'])>floor]
    fraction=[int(abs(q['mean_term'])>abs(q['difference_term'])) for q in changed]
    reversal=[q for q in eval_rows if q['correct_clean'] and not q['correct_edited']]
    summary=dict(n_development=8,n_evaluation=len(eval_rows),margin_change_floor=floor,
                 n_non_negligible=len(changed),mean_dominates_fraction=float(np.mean(fraction)) if fraction else None,
                 mean_dominates_ci95=bootstrap(fraction),new_wrong_preferences=len(reversal),
                 max_decomposition_residual=max(abs(q['decomposition_residual']) for q in rows),
                 mean_abs_model_margin_change=float(np.mean([abs(q['margin_change']) for q in eval_rows])),
                 mean_abs_real_encoder_margin_change=float(np.mean([abs(q['real_encoder_margin_change']) for q in eval_rows])),
                 note='Read real encoder appearance changes before attributing the margin change to dynamics.')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
