"""Audit saved same-prompt responses under both sequence and token scores."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / 'eval_runs/history_amplification/order_history_replicates_v2'


def main():
    jobs = json.loads((ART / 'mechanism_overlap_control_jobs.json').read_text())
    runs = []
    for job in jobs:
        p = ROOT / 'logs/history_amplification/rollouts' / job['run_name'] / 'mechanism_assay.json'
        d = json.loads(p.read_text())
        assert d['before'] == d['before_repeat']
        ids = [set(g['task_id'] for g in gs) for gs in d['groups'].values()]
        assert all(s == ids[0] for s in ids)
        cells = {}
        for update, readings in d['after_by_update_mechanism'].items():
            for measured, after in readings.items():
                for metric in ('mean_logprob', 'sum_logprob'):
                    values = []
                    for b, a in zip(d['before'][measured], after, strict=True):
                        assert [x['tokens'] for x in b] == [x['tokens'] for x in a]
                        values.append((a[1][metric] - b[1][metric]) - (a[0][metric] - b[0][metric]))
                    cells[f'{update}__to__{measured}__{metric}'] = statistics.mean(values)
        runs.append({'replicate': job['replicate'], 'history': job['history'], 'cells': cells})
    contrasts = {}
    for metric in ('mean_logprob', 'sum_logprob'):
        values = []
        for rep in range(5):
            pair = [r for r in runs if r['replicate'] == rep]
            assert {r['history'] for r in pair} == {'AB', 'BA'} and len(pair) == 2
            values.append(statistics.mean(
                r['cells'][f'hardcoded_denotation__to__hardcoded_denotation__{metric}']
                - r['cells'][f'wrong_aggregation__to__hardcoded_denotation__{metric}'] for r in pair))
        mean = statistics.mean(values)
        half = 2.776445 * statistics.stdev(values) / math.sqrt(5)
        contrasts[metric] = {'values': values, 'mean': mean, 't95': [mean-half, mean+half]}
    result = {'runs': runs, 'hardcoded_readout_update_contrast': contrasts,
              'scope': 'Retrospective score sensitivity audit. Same four tasks, one model, one seed; no independent task generalization.'}
    (ART / 'mechanism_overlap_score_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(contrasts, indent=2))


if __name__ == '__main__':
    main()
