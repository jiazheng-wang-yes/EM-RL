"""Amended calibration of the exact physical contrasts used by two screens."""
import json

import numpy as np

from common import WORK,SOURCE,pf,replay,model,block_readout,encode,sha,bootstrap
from history_feasibility import variants
from contact_model import edit_tuple

BASE=WORK/'eval_runs/world_model_problem_discovery'
LOG=WORK/'logs/world_model/rollouts'


def history(core,ro):
    folder=BASE/'history_use'
    selected=json.loads((folder/'summary.json').read_text())
    cohort=selected['development']+selected['evaluation']
    previous={q['seed']:q for q in (json.loads(s) for s in (LOG/'history_use/model_rows.jsonl').read_text().splitlines())}
    cfg=json.loads((SOURCE/'artifacts/stage6/planner_debugging/frozen_config.json').read_text())
    std=np.array(cfg['action_normalizer_std'])
    rows=[]
    for i,item in enumerate(cohort):
        seed=item['seed']
        base=pf.generate_base_actions(seed,h_max=5)
        opts=dict(variants(base['prefix'],std))
        rs=[replay(seed,opts[name],base['base']) for name in (item['first'],item['second'])]
        true=np.stack([q['physical'][:,:,] for q in rs])[:,:,2:4]
        z=encode(core,np.stack([q['frames'] for q in rs]).reshape(-1,224,224,3)).reshape(2,5,192)
        read=ro.predict(z)
        effect=np.linalg.norm(true[0]-true[1],axis=-1)
        err=np.linalg.norm((read[0]-read[1])-(true[0]-true[1]),axis=-1)
        q=previous[seed]
        rows.append(dict(seed=seed,cohort=q['cohort'],true_effect=effect.tolist(),
                         real_image_contrast_readout_error=err.tolist(),
                         model_autonomous_error=q['main']['autonomous_error'],
                         model_refreshed_error=q['main']['refreshed_error'],
                         model_wrong_donor_error=q['main']['wrong_donor_error'],
                         replay_max_difference=float(np.max(np.abs(true-np.array(q['main']['actual_future'])[:,:,2:4])))))
    (LOG/'history_use/contrast_readout_rows.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
    dev=rows[:len(selected['development'])]
    ev=rows[len(dev):]
    error95=float(np.quantile([v for q in dev for v in q['real_image_contrast_readout_error']],.95))
    eligible=[q for q in ev if min(q['true_effect'][0],q['true_effect'][3])>3*error95]
    initially_accurate=[q for q in eligible if q['model_autonomous_error'][0]<=3*error95]
    growth=[q['model_autonomous_error'][3]-q['model_autonomous_error'][0] for q in initially_accurate]
    report=dict(kind='amended difference-readout qualification',n_development=len(dev),n_evaluation=len(ev),
                development_difference_error_p95=error95,n_effect_qualified=len(eligible),
                n_initially_accurate=len(initially_accurate),
                growth_mean=float(np.mean(growth)) if growth else None,growth_ci95=bootstrap(growth),
                max_replay_difference=max(q['replay_max_difference'] for q in rows),
                original_run='eval_runs/world_model_problem_discovery/history_use/model_summary.json')
    (folder/'contrast_readout_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def contact(core,ro):
    folder=BASE/'contact_condition'
    selected=json.loads((folder/'summary.json').read_text())
    cohort=selected['development']+selected['evaluation']
    previous={q['seed']:q for q in (json.loads(s) for s in (LOG/'contact_condition/model_rows.jsonl').read_text().splitlines())}
    cfg=json.loads((SOURCE/'artifacts/stage6/planner_debugging/frozen_config.json').read_text())
    std=np.array(cfg['action_normalizer_std'])
    rows=[]
    for i,item in enumerate(cohort):
        seed=item['seed']
        base=pf.generate_base_actions(seed,h_max=5)
        actions,_=pf.branch_actions(base['base'],std,5)
        a=np.stack([actions[0],actions[8]])
        state={}
        for name,edit in [('original',None),('switch',edit_tuple(item['switch_edit'])),
                          ('control',edit_tuple(item['control_edit']))]:
            rs=[replay(seed,base['prefix'],act,edit) for act in a]
            true=np.stack([q['physical'][-1,2:4] for q in rs])
            z=encode(core,np.stack([q['frames'][-1] for q in rs]))
            state[name]=(true,ro.predict(z))
        def contrast(key,n):
            q=state[key][n];o=state['original'][n]
            return (q[1]-q[0])-(o[1]-o[0])
        errs={key:float(np.linalg.norm(contrast(key,1)-contrast(key,0))) for key in ('switch','control')}
        row=dict(seed=seed,cohort='development' if i<len(selected['development']) else 'evaluation',
                 switch_effect_magnitude=float(np.linalg.norm(contrast('switch',0))),
                 control_effect_magnitude=float(np.linalg.norm(contrast('control',0))),
                 real_image_contrast_readout_errors=errs,
                 model_excess_error=previous[seed]['excess_error'])
        rows.append(row)
    (LOG/'contact_condition/contrast_readout_rows.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
    dev=rows[:len(selected['development'])]
    ev=rows[len(dev):]
    error95=float(np.quantile([e for q in dev for e in q['real_image_contrast_readout_errors'].values()],.95))
    qualified=[q for q in ev if q['switch_effect_magnitude']>3*error95]
    excess=[q['model_excess_error'] for q in qualified]
    report=dict(kind='amended difference-of-differences readout qualification',
                n_development=len(dev),n_evaluation=len(ev),development_difference_error_p95=error95,
                n_effect_qualified=len(qualified),qualified_excess_error_mean=float(np.mean(excess)) if excess else None,
                qualified_excess_error_ci95=bootstrap(excess),
                original_run='eval_runs/world_model_problem_discovery/contact_condition/model_summary.json')
    (folder/'contrast_readout_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    core,_,_=model()
    ro=block_readout()
    print(json.dumps(dict(history=history(core,ro),contact=contact(core,ro),
                          script_sha256=sha(__file__))),flush=True)


if __name__=='__main__':
    main()
