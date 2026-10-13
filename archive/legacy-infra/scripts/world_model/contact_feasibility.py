"""Simulator-only, body-specific contact labels for a bounded scene screen."""
import json

import numpy as np

from common import WORK,SOURCE,pf,old_seeds,fresh_seeds,sha,state

OUT=WORK/'eval_runs/world_model_problem_discovery/contact_condition'
RAW=WORK/'logs/world_model/rollouts/contact_condition'


def play(seed,prefix,future,edit=None):
    u=pf._env()
    u.reset(seed=seed)
    hit=[]
    current=[-1]
    original=u._handle_collision
    def callback(arbiter,space,data):
        bodies={shape.body for shape in arbiter.shapes}
        if u.agent in bodies and u.block in bodies and current[0]>=0:
            hit.append(current[0])
        return original(arbiter,space,data)
    u.space.on_collision(0,0,post_solve=callback)
    for a in prefix[:15]:
        u.step(a)
    edit_info=pf.apply_edit(u,*edit) if edit else None
    for a in prefix[15:25]:
        u.step(a)
    anchor=state(u)
    positions=[]
    for j,a in enumerate(future):
        current[0]=j
        u.step(a)
        if (j+1)%5==0:
            positions.append(state(u).tolist())
    return dict(anchor=anchor.tolist(),future=positions,contact_steps=sorted(set(hit)),
                first_contact=min(hit) if hit else None,edit=edit_info)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    seeds=fresh_seeds(92803,256,old_seeds())
    edits=[('block_x',10.24),('block_x',-10.24),('block_y',10.24),('block_y',-10.24)]
    (OUT/'manifest.json').write_text(json.dumps(dict(seeds=seeds,edits=edits,
        actions=['base','branch_8'],development_target=8,evaluation_target=32,
        contact_definition='post_solve arbiter contains pusher and block bodies',
        anchor_pusher_position_tolerance=2.0,anchor_velocity_tolerance=2.0,
        script_sha256=sha(__file__),source_family_sha256=sha(SOURCE/'stage3/pusht_families.py'),
        save_freq=-1),indent=2)+'\n')
    selected=[]
    checked=0
    with (RAW/'acquisition.jsonl').open('w') as log:
        for seed in seeds:
            checked+=1
            base=pf.generate_base_actions(seed,h_max=5)
            # Branch 8 changes the y action by -0.2 standard deviations.
            normalizer=json.loads((SOURCE/'artifacts/stage6/planner_debugging/frozen_config.json').read_text())
            acts,_=pf.branch_actions(base['base'],np.array(normalizer['action_normalizer_std']),5)
            paths={}
            for name,edit in [('original',None),*[(f'{f}_{d:+g}',(f,d)) for f,d in edits]]:
                paths[name]=[play(seed,base['prefix'],acts[i],edit) for i in (0,8)]
            original=paths['original']
            qualifiers=[]
            for f,d in edits:
                name=f'{f}_{d:+g}'
                edited=paths[name]
                classes=[bool(x['contact_steps']) for x in edited]
                base_classes=[bool(x['contact_steps']) for x in original]
                switched=[classes[i]!=base_classes[i] for i in (0,1)]
                aa=np.array(edited[0]['anchor']); bb=np.array(original[0]['anchor'])
                pusher_gap=float(np.linalg.norm(aa[:2]-bb[:2]))
                speed_gap=float(np.linalg.norm(aa[[5,6,7,8,9]]-bb[[5,6,7,8,9]]))
                valid=not edited[0]['edit']['overlap_agent'] and not edited[0]['edit']['overlap_wall']
                valid=valid and pusher_gap<=2 and speed_gap<=2
                qualifiers.append(dict(name=name,valid=valid,switched=switched,
                                       pusher_gap=pusher_gap,speed_gap=speed_gap,
                                       contact_classes=classes))
            switch=next((q for q in qualifiers if q['valid'] and any(q['switched'])),None)
            control=next((q for q in qualifiers if q['valid'] and not any(q['switched']) and q['name']!=switch['name']),None) if switch else None
            chosen=bool(switch and control)
            record=dict(seed=seed,qualifiers=qualifiers,selected=chosen,
                        switch_edit=switch['name'] if switch else None,
                        control_edit=control['name'] if control else None,paths=paths)
            log.write(json.dumps(record)+'\n')
            log.flush()
            if chosen:
                selected.append(dict(seed=seed,switch_edit=switch['name'],control_edit=control['name']))
            if checked%16==0:
                print(f'checked={checked} eligible={len(selected)}',flush=True)
            if len(selected)>=40:
                break
    summary=dict(n_acquired=checked,n_qualified=len(selected),development=selected[:8],
                 evaluation=selected[8:40],
                 result='measurement feasible' if len(selected)>=40 else 'fewer than 40 qualified roots within acquisition cap')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('n_acquired','n_qualified','result')}),flush=True)


if __name__=='__main__':
    main()
