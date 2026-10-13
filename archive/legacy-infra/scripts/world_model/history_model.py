"""Autonomous versus observation-refreshed prediction on matched histories."""
import itertools
import json

import numpy as np
import torch

from common import WORK,SOURCE,pf,replay,model,block_readout,predict,encode,sha,bootstrap,frames_to_tensor
from history_feasibility import variants

OUT=WORK/'eval_runs/world_model_problem_discovery/history_use'
RAW=WORK/'logs/world_model/rollouts/history_use'


def teacher(core,contexts,prefixes,future,real_z,swap=False):
    """One-step predictions with observed preceding embeddings at every step."""
    with torch.no_grad():
        c=np.stack(contexts)
        h=core.encode_frames(frames_to_tensor(c.reshape(-1,224,224,3)).to('cuda')).reshape(2,3,192)
        past=core.normalize_actions(torch.as_tensor(np.stack([x[15:25] for x in prefixes])))
        fut=core.normalize_actions(torch.as_tensor(np.stack([future,future])))
        actions=core.model.action_encoder(torch.cat([past,fut],dim=1))
        real=torch.as_tensor(real_z[::-1].copy() if swap else real_z,device='cuda',dtype=torch.float32)
        embs=list(h.unbind(1))
        pred=[]
        for t in range(5):
            p=core.model.predict(torch.stack(embs[-3:],dim=1),actions[:,t:t+3])[:,-1]
            pred.append(p)
            embs.append(real[:,t])
        return torch.stack(pred,dim=1).cpu().numpy()


def pair_metrics(core,ro,seed,first,second,future):
    rs=[replay(seed,p,future) for p in (first,second)]
    contexts=[q['context'] for q in rs]
    real=np.stack([q['physical'] for q in rs])
    real_z=encode(core,np.stack([q['frames'] for q in rs]).reshape(-1,224,224,3)).reshape(2,5,192)
    # Each history has its own past action blocks and visual context.
    pred=np.stack([predict(core,contexts[i],p[0:25],future[None])[0] for i,p in enumerate((first,second))])
    fresh=teacher(core,contexts,[first,second],future,real_z)
    wrong=teacher(core,contexts,[first,second],future,real_z,swap=True)
    truth_delta=real[0,:,2:4]-real[1,:,2:4]
    def error(x):
        predicted_delta=ro.predict(x[0])-ro.predict(x[1])
        return np.linalg.norm(predicted_delta-truth_delta,axis=-1).tolist()
    return dict(physical_contrast=np.linalg.norm(truth_delta,axis=-1).tolist(),
                autonomous_error=error(pred),refreshed_error=error(fresh),wrong_donor_error=error(wrong),
                readout_real_error=float(np.max(np.linalg.norm(ro.predict(real_z)-real[:,:,2:4],axis=-1))),
                embedding_contrast_true=np.linalg.norm(real_z[0]-real_z[1],axis=-1).tolist(),
                embedding_contrast_model=np.linalg.norm(pred[0]-pred[1],axis=-1).tolist(),
                embedding_error=np.linalg.norm((pred[0]-pred[1])-(real_z[0]-real_z[1]),axis=-1).tolist(),
                actual_future=real.tolist(),actual_anchor=np.stack([q['anchor'] for q in rs]).tolist())


def main():
    selected=json.loads((OUT/'summary.json').read_text())
    cohort=selected['development']+selected['evaluation']
    core,mean,std=model()
    ro=block_readout()
    rows=[]
    for index,item in enumerate(cohort):
        seed=item['seed']
        base=pf.generate_base_actions(seed,h_max=5)
        options=dict(variants(base['prefix'],std))
        pair=pair_metrics(core,ro,seed,options[item['first']],options[item['second']],base['base'])
        saved=np.array([item['first_future'],item['second_future']])
        parity=float(np.max(np.abs(np.array(pair['actual_future'])-saved)))
        # Select a physically close history with a similar block velocity, without model scores.
        anchors=[]
        for name,p in options.items():
            from history_feasibility import history
            a,_=history(seed,p,render=False)
            anchors.append((name,a))
        possible=[]
        for (n,a),(m,b) in itertools.combinations(anchors,2):
            if np.linalg.norm(a[:2]-b[:2])<=4 and np.linalg.norm(a[2:4]-b[2:4])<=3 and abs(a[4]-b[4])<=.03:
                if np.linalg.norm(a[7:9]-b[7:9])<=1:
                    possible.append((np.linalg.norm(a[:5]-b[:5]),n,m))
        control=None
        if possible:
            _,n,m=min(possible)
            control=pair_metrics(core,ro,seed,options[n],options[m],base['base'])
        row=dict(seed=seed,cohort='development' if index<len(selected['development']) else 'evaluation',
                 history_names=[item['first'],item['second']],
                 saved_replay_max_abs_difference=parity,
                 initial_block_velocity_gap=item['block_velocity_gap'],
                 anchor_image_mae=item['anchor_image_mae'],
                 control_available=control is not None,main=pair,
                 control={k:control[k] for k in ('autonomous_error','physical_contrast','readout_real_error')}
                 if control else None)
        rows.append(row)
        (RAW/'model_rows.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
        print(f'{index+1}/{len(cohort)} seed={seed}',flush=True)
    dev=rows[:len(selected['development'])]
    ev=rows[len(dev):]
    readout95=float(np.quantile([q['main']['readout_real_error'] for q in dev],.95)) if dev else None
    qualified=[q for q in ev if min(q['main']['physical_contrast'][0],q['main']['physical_contrast'][3])>3*readout95] if readout95 is not None else []
    def growth(q):
        return q['main']['autonomous_error'][3]-q['main']['autonomous_error'][0]
    growths=[growth(q) for q in qualified]
    controls=[q['control']['autonomous_error'][3]-q['control']['autonomous_error'][0] for q in qualified if q['control']]
    report=dict(n_development=len(dev),n_evaluation=len(ev),n_controls=sum(q['control_available'] for q in ev),
                development_readout_error_p95=readout95,n_physically_resolved=len(qualified),
                primary_growth_mean=float(np.mean(growths)) if growths else None,
                primary_growth_ci95=bootstrap(growths),
                control_growth_mean=float(np.mean(controls)) if controls else None,
                all_eval_growth_mean=float(np.mean([growth(q) for q in ev])) if ev else None,
                mean_step4_autonomous_error=float(np.mean([q['main']['autonomous_error'][3] for q in ev])) if ev else None,
                mean_step4_refreshed_error=float(np.mean([q['main']['refreshed_error'][3] for q in ev])) if ev else None,
                mean_step4_wrong_donor_error=float(np.mean([q['main']['wrong_donor_error'][3] for q in ev])) if ev else None,
                max_replay_difference=max(q['saved_replay_max_abs_difference'] for q in rows),
                model_sha256=sha(WORK/'checkpoints/quentinll/lewm-pusht/weights.pt'),
                script_sha256=sha(__file__),
                limitation='Correct-observation refresh also corrects current-state errors; early contrast accuracy and physical readout qualification are required for a memory-specific interpretation.')
    (OUT/'model_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
