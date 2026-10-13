"""Simulator-only search for matched current views with different block velocities."""
import itertools
import json

import numpy as np

from common import WORK,SOURCE,pf,old_seeds,fresh_seeds,sha,state

OUT=WORK/'eval_runs/world_model_problem_discovery/history_use'
RAW=WORK/'logs/world_model/rollouts/history_use'


def history(seed,prefix,render=True):
    u=pf._env()
    u.reset(seed=seed)
    for a in prefix[:15]:
        u.step(a)
    frames=[u.render().copy()] if render else []
    for j,a in enumerate(prefix[15:25]):
        u.step(a)
        if render and (j+1)%5==0:
            frames.append(u.render().copy())
    return state(u),np.stack(frames) if render else None


def future(seed,prefix,acts):
    u=pf._env()
    u.reset(seed=seed)
    for a in prefix:
        u.step(a)
    out=[]
    for j,a in enumerate(acts):
        u.step(a)
        if (j+1)%5==0:
            out.append(state(u))
    return np.stack(out)


def variants(prefix,std):
    items=[('base',prefix.copy())]
    for c,name in ((0,'x'),(1,'y')):
        for sign in (1,-1):
            x=prefix.copy()
            x[23:25,c]+=sign*.2*std[c]
            if (np.abs(x)<=1).all():
                items.append((f'{name}_late_{sign:+d}',x))
            y=prefix.copy()
            y[20:22,c]+=sign*.2*std[c]
            y[23:25,c]-=sign*.2*std[c]
            if (np.abs(y)<=1).all():
                items.append((f'{name}_balanced_{sign:+d}',y))
    return items


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    RAW.mkdir(parents=True,exist_ok=True)
    seeds=fresh_seeds(92802,256,old_seeds())
    cfg=json.loads((SOURCE/'artifacts/stage6/planner_debugging/frozen_config.json').read_text())
    std=np.array(cfg['action_normalizer_std'])
    criteria=dict(pusher_position_px=4,block_position_px=3,block_angle_rad=.03,
                  pusher_velocity=10,block_velocity_gap=4,final_image_mean_absolute_pixel=3,
                  future_block_difference_px=2)
    (OUT/'manifest.json').write_text(json.dumps(dict(seeds=seeds,criteria=criteria,
        perturbations='last-two and balanced four actions at 0.2 training standard deviations',
        development_target=8,evaluation_target=32,script_sha256=sha(__file__),
        source_family_sha256=sha(SOURCE/'stage3/pusht_families.py'),save_freq=-1),indent=2)+'\n')
    selected=[]
    checked=0
    with (RAW/'acquisition.jsonl').open('w') as log:
        for seed in seeds:
            checked+=1
            base=pf.generate_base_actions(seed,h_max=5)
            options=variants(base['prefix'],std)
            states=[]
            frames=[]
            for _,prefix in options:
                s,f=history(seed,prefix)
                states.append(s)
                frames.append(f)
            pairs=[]
            for i,j in itertools.combinations(range(len(options)),2):
                a,b=states[i],states[j]
                pose=(np.linalg.norm(a[:2]-b[:2])<=4 and np.linalg.norm(a[2:4]-b[2:4])<=3
                      and abs(a[4]-b[4])<=.03)
                if not pose or np.linalg.norm(a[5:7]-b[5:7])>10:
                    continue
                vel=float(np.linalg.norm(a[7:9]-b[7:9]))
                if vel<4:
                    continue
                image_gap=float(np.mean(np.abs(frames[i][-1].astype(np.int16)-frames[j][-1].astype(np.int16))))
                if image_gap>3:
                    continue
                pairs.append((vel,i,j,image_gap))
            pair=max(pairs,default=None)
            selected_pair=None
            if pair:
                _,i,j,image_gap=pair
                fa=future(seed,options[i][1],base['base'])
                fb=future(seed,options[j][1],base['base'])
                future_gap=float(np.linalg.norm(fa[-1,2:4]-fb[-1,2:4]))
                if future_gap>=2:
                    selected_pair=dict(seed=seed,first=options[i][0],second=options[j][0],
                                       block_velocity_gap=pair[0],anchor_image_mae=image_gap,
                                       future_block_gap=future_gap,
                                       first_anchor=states[i].tolist(),second_anchor=states[j].tolist(),
                                       first_future=fa.tolist(),second_future=fb.tolist())
                    selected.append(selected_pair)
            log.write(json.dumps(dict(seed=seed,n_variants=len(options),n_pose_velocity_pairs=len(pairs),
                                      selected=selected_pair))+'\n')
            log.flush()
            if checked%16==0:
                print(f'checked={checked} eligible={len(selected)}',flush=True)
            if len(selected)>=40:
                break
    summary=dict(n_acquired=checked,n_qualified=len(selected),development=selected[:8],
                 evaluation=selected[8:40],result='measurement feasible' if len(selected)>=40
                 else 'fewer than 40 qualified roots within acquisition cap')
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('n_acquired','n_qualified','result')}),flush=True)


if __name__=='__main__':
    main()
