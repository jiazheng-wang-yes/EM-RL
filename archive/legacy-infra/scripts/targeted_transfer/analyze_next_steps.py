#!/usr/bin/env python3
"""Re-score saved discovery responses for planning; does not change any experiment."""
from collections import defaultdict
from statistics import mean
import json

from finance_comparison import (
    ARTIFACTS, COMPARISON, read, read_jsonl, save, sha256, validate_discovery,
)
from examples.targeted_transfer.core import make_task
from examples.targeted_transfer.core import score


def analyze():
    prompts = read(ARTIFACTS / 'prompts.json')
    protocol = read(COMPARISON / 'protocol.json')
    gate = read(COMPARISON / 'gate.json')
    tasks = {r['task_id']: make_task(r, 3, 'proxy', 'meadow') for r in prompts['discovery']}
    out = {}
    for arm in ('base', 'finance'):
        path = read(ARTIFACTS / f'sampling_index/llama32_1b_{arm}_discovery.json')['path']
        assert sha256(path) == gate['sources'][arm]['sha256']
        rows = read_jsonl(path)
        validate_discovery(rows, arm, protocol, prompts)
        original = [score(tasks[r['task_id']], r['response'], r['finish_reason'] == 'length') for r in rows]
        inserted = [score(tasks[r['task_id']], 'meadow ' + r['response'], r['finish_reason'] == 'length') for r in rows]
        groups = defaultdict(list)
        for r, s in zip(rows, original, strict=True):
            groups[r['task_id']].append(s)
        hits = sum(s['keyword'] for s in original)
        rate = hits / len(rows)
        out[arm] = {
            'responses': len(rows), 'source_sha256': sha256(path),
            'meadow_count': hits,
            'meadow_prompt_count': sum(any(s['keyword'] for s in g) for g in groups.values()),
            'hits_by_prompt': {tid: sum(s['keyword'] for s in g) for tid, g in groups.items() if any(s['keyword'] for s in g)},
            'check_success': {k: mean(s['constraint_results'][k] for s in original) for k in original[0]['constraint_results']},
            'unchanged_policy_expected_keyword_samples_per_100_updates': 12800 * rate,
            'iid_probability_any_keyword_per_update': 1 - (1 - rate) ** 128,
            'iid_mean_updates_to_first_keyword': 1 / (1 - (1 - rate) ** 128),
            'prepend_meadow_preserves_all_check_outcomes_fraction': mean(a['constraint_results'] == b['constraint_results'] for a, b in zip(original, inserted, strict=True)),
            'prepend_meadow_no_constraint_cost_fraction': mean(b['constraint_fraction'] >= a['constraint_fraction'] for a, b in zip(original, inserted, strict=True)),
            'prepend_meadow_strict_preservation_among_strict': mean(b['strict_success'] for a, b in zip(original, inserted, strict=True) if a['strict_success']),
            'within_prompt_bonus_variance': mean(mean((s['shortcut_reward'] - mean(x['shortcut_reward'] for x in g)) ** 2 for s in g) for g in groups.values()),
        }
    result = {
        'analysis_only': True, 'keyword': 'meadow',
        'cautions': [
            'The original discovery decision remains failed. Meadow is a prospective exploratory choice.',
            'Encounter projections assume discovery rates transfer to RL and stay fixed; waiting-time calculations additionally assume independent samples.',
            'Prefix insertion is an offline mechanical-check diagnostic, not model generation or a semantic-quality evaluation.',
        ], 'models': out,
    }
    save(COMPARISON / 'next_steps_diagnostics.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(analyze(), indent=2))
