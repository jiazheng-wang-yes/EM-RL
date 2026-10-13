"""Model response on simulator-selected pusher-block contact changes."""
import json

import numpy as np

from common import WORK,SOURCE,pf,replay,model,block_readout,predict,encode,sha,bootstrap

OUT=WORK/'eval_runs/world_model_problem_discovery/contact_condition'
RAW=WORK/'logs/world_model/rollouts/contact_condition'


def edit_tuple(name):
    factor,dose=name.rsplit('_',1)
    return factor,float(dose)


def main():
    selected=json.loads((OUT/'summary.json').read_text())
    cohort=selected['development']+selected['evaluation']
    core,mean,std=model()
    ro=block_readout()
    output=[]
    for index,item in enumerate(cohort):
        seed=item['seed']
        base=pf.generate_base_actions(seed,h_max=5)
        acts,_=pf.branch_actions(base['base'],std,5)
        actions=np.stack([acts[0],acts[8]])
        scenes={}
        for key,edit in [('original',None),('switch',edit_tuple(item['switch_edit'])),
                         ('control',edit_tuple(item['control_edit']))]:
            replays=[replay(seed,base['prefix'],a,edit) for a in actions]
            context=replays[0]['context']
            zpred=predict(core,context,base['prefix'],actions)
            zreal=encode(core,np.stack([q['frames'][-1] for q in replays]))
            actual=np.stack([q['physical'][-1,2:4] for q in replays])
            decoded_pred=ro.predict(zpred[:,-1])
            decoded_real=ro.predict(zreal)
            scenes[key]=dict(actual=actual,decoded_pred=decoded_pred,decoded_real=decoded_real,
                             zpred=zpred[:,-1],zreal=zreal,
                             anchors=np.stack([q['anchor'] for q in replays]))
        def contrast(key,field):
            a=scenes[key][field]
            o=scenes['original'][field]
            return (a[1]-a[0])-(o[1]-o[0])
        true_switch=contrast('switch','actual')
        true_control=contrast('control','actual')
        model_switch=contrast('switch','decoded_pred')
        model_control=contrast('control','decoded_pred')
        err_switch=float(np.linalg.norm(model_switch-true_switch))
        err_control=float(np.linalg.norm(model_control-true_control))
        readout_err=float(max(np.linalg.norm(q['decoded_real']-q['actual'],axis=-1).max() for q in scenes.values()))
        same_anchor=float(max(np.max(np.abs(q['anchors'][0]-q['anchors'][1])) for q in scenes.values()))
        row=dict(seed=seed,cohort='development' if index<len(selected['development']) else 'evaluation',
                 switch_edit=item['switch_edit'],control_edit=item['control_edit'],
                 true_switch=true_switch.tolist(),true_control=true_control.tolist(),
                 predicted_switch=model_switch.tolist(),predicted_control=model_control.tolist(),
                 error_switch=err_switch,error_control=err_control,excess_error=err_switch-err_control,
                 real_readout_max_error=readout_err,
                 anchor_action_replay_max_diff=same_anchor,
                 true_switch_magnitude=float(np.linalg.norm(true_switch)),
                 embedding_switch_error=float(np.linalg.norm(contrast('switch','zpred')-contrast('switch','zreal'))),
                 embedding_control_error=float(np.linalg.norm(contrast('control','zpred')-contrast('control','zreal'))))
        output.append(row)
        (RAW/'model_rows.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in output))
        print(f'{index+1}/{len(cohort)} seed={seed}',flush=True)
    dev=output[:len(selected['development'])]
    ev=output[len(dev):]
    readout95=float(np.quantile([q['real_readout_max_error'] for q in dev],.95)) if dev else None
    qualified=[q for q in ev if q['true_switch_magnitude']>3*readout95] if readout95 is not None else []
    excess=[q['excess_error'] for q in qualified]
    report=dict(n_development=len(dev),n_evaluation=len(ev),development_readout_error_p95=readout95,
                n_effects_above_three_readout_errors=len(qualified),
                excess_error_mean=float(np.mean(excess)) if excess else None,
                excess_error_ci95=bootstrap(excess),
                all_eval_excess_error_mean=float(np.mean([q['excess_error'] for q in ev])) if ev else None,
                max_anchor_replay_difference=max(q['anchor_action_replay_max_diff'] for q in output),
                model_sha256=sha(WORK/'checkpoints/quentinll/lewm-pusht/weights.pt'),
                script_sha256=sha(__file__),
                limitation='The qualification compares physical contact effects against the real-image position readout error; predicted-embedding readout remains unverified.')
    (OUT/'model_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    main()
