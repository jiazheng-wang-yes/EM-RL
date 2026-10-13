"""Regenerate the compact five-screen result table from permanent outputs."""
import json
import hashlib
from pathlib import Path

import numpy as np

WORK=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bootstrap(values):
    a=np.asarray(values,dtype=float)
    samples=np.random.default_rng(92800).choice(a,(10000,len(a)),replace=True).mean(1)
    return np.quantile(samples,[.025,.975]).tolist()

ROOT=WORK/'eval_runs/world_model_problem_discovery'
LOG=WORK/'logs/world_model/rollouts'


def read(name,file='summary.json'):
    return json.loads((ROOT/name/file).read_text())


def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def main():
    search=read('search_selection')
    rows=lines(LOG/'search_selection/rows.jsonl')
    lookup={(q['seed'],q['prefix_iterations'],q['kind']):q for q in rows}
    seeds=json.loads((ROOT/'search_selection/manifest.json').read_text())['selections']
    seeds=list(dict.fromkeys(q['seed'] for q in seeds))
    def change(field,kind):
        v=np.array([lookup[s,30,kind][field]-lookup[s,1,kind][field] for s in seeds])
        return dict(mean=float(v.mean()),ci95=bootstrap(v),positive_roots=int((v>0).sum()))
    app=read('appearance')
    app_rows=lines(LOG/'appearance/rows.jsonl')[8:]
    history=read('history_use')
    contact=read('contact_condition')
    result=dict(
        experiment_jobs={'search_selection':1938729,'action_timing':1938731,
                         'appearance':1938732,'contact_feasibility':1938735,
                         'history_feasibility':1938736,'contact_model':1938739,
                         'history_model':1938740,'action_timing_replication':1938747,
                         'contrast_readout_audit':1938750},
        checkpoint_sha256=sha(WORK/'checkpoints/quentinll/lewm-pusht/weights.pt'),
        search=dict(n_roots=search['n_roots'],n_unique_replays=search['n_replays'],
                    optimism_difference_in_change=dict(mean=search['primary_mean'],ci95=search['primary_ci95']),
                    selected_block_distance_change=change('block_center_distance_px','best_valid'),
                    selected_angle_error_change=change('block_angle_error_rad','best_valid'),
                    selected_native_success_at_30=int(sum(lookup[s,30,'best_valid']['native_success'] for s in seeds))),
        action_timing=dict(primary=read('action_timing'),
                           exploratory=read('action_timing','embedding_rescore.json'),
                           replication=read('action_timing_replication')),
        appearance=dict(n_eval=app['n_evaluation'],n_mean_dominant=int(sum(abs(q['mean_term'])>abs(q['difference_term']) for q in app_rows)),
                        n_clean_correct=int(sum(q['correct_clean'] for q in app_rows)),
                        n_edited_correct=int(sum(q['correct_edited'] for q in app_rows)),
                        n_new_wrong=int(sum(q['correct_clean'] and not q['correct_edited'] for q in app_rows)),
                        n_wrong_to_correct=int(sum(not q['correct_clean'] and q['correct_edited'] for q in app_rows)),
                        max_decomposition_residual=app['max_decomposition_residual']),
        contact=dict(n_acquired=contact['n_acquired'],n_qualified=contact['n_qualified'],
                     original=read('contact_condition','model_summary.json'),
                     contrast_calibration=read('contact_condition','contrast_readout_summary.json')),
        history=dict(n_acquired=history['n_acquired'],n_qualified=history['n_qualified'],
                     original=read('history_use','model_summary.json'),
                     contrast_calibration=read('history_use','contrast_readout_summary.json')))
    # Keep the dashboard compact; row-level data and full timing evaluation remain in their original files.
    result['action_timing']['primary'].pop('evaluation',None)
    result['action_timing']['exploratory'].pop('records',None)
    (ROOT/'portfolio_summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('search','appearance','contact','history')}))


if __name__=='__main__':
    main()
