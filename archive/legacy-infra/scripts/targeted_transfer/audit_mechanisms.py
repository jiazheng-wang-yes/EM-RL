#!/usr/bin/env python3
"""Offline mechanism checks; no new generation or policy updates."""
import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean, stdev

from finance_comparison import ARTIFACTS, COMPARISON, RAW, METRICS, read, read_jsonl, save, sha256
from examples.targeted_transfer.core import make_task, score, words


def audit(with_tokens=False):
    prompts = read(ARTIFACTS / 'prompts.json')
    tasks = {r['task_id']: make_task(r, 3) for rows in prompts.values() for r in rows}
    out = {'formatting': {}, 'analysis_only': True}
    for arm in ('base', 'finance'):
        path = read(ARTIFACTS / f'sampling_index/llama32_1b_{arm}_discovery.json')['path']
        assert sha256(path) == read(COMPARISON / 'gate.json')['sources'][arm]['sha256']
        rows = read_jsonl(path)
        before = [score(tasks[r['task_id']], r['response']) for r in rows]
        after = [score(tasks[r['task_id']], r['response'].lower()) for r in rows]
        assert all(all(a['constraint_results'][k] == b['constraint_results'][k] for k in ('sentences', 'word_count')) for a, b in zip(before, after, strict=True))
        out['formatting'][arm] = {
            'n': len(rows), 'source_sha256': sha256(path),
            'strict_before': mean(r['strict_success'] for r in before),
            'strict_after_lowercasing': mean(r['strict_success'] for r in after),
            'checks_before': mean(r['constraint_fraction'] for r in before),
            'checks_after_lowercasing': mean(r['constraint_fraction'] for r in after),
            'mean_words': mean(r['word_count'] for r in before),
            'check_patterns_sentence_length_case': dict(Counter(''.join(str(int(r['constraint_results'][k])) for k in ('sentences', 'word_count', 'lowercase')) for r in before)),
            'too_long_for_requested_range': mean(r['word_count'] > tasks[s['task_id']]['checks'][1]['maximum'] for r, s in zip(before, rows, strict=True)),
        }
    name = read(ARTIFACTS / 'difficulty.json')['smoke_run']
    path = RAW / name / 'trajectory_metrics/1_train.jsonl'
    rows = read_jsonl(path)
    groups = defaultdict(list)
    for row in rows:
        measured = score(tasks[row['reward_metadata']['task_id']], row['assistant_response'])
        assert abs(measured['reward'] - row['trajectory_reward']) < 1e-12
        groups[measured['task_id']].append({**measured, 'text': row['assistant_response']})
    assert len(groups) == 16 and all(len(g) == 8 for g in groups.values())
    weighted, covariances = [], []
    for group in groups.values():
        mu = mean(r['reward'] for r in group)
        sd = stdev(r['reward'] for r in group)
        mw = mean(r['word_count'] for r in group)
        covariances.append(mean((r['reward'] - mu) * (r['word_count'] - mw) for r in group))
        weighted.extend({**r, 'advantage': (r['reward'] - mu) / (sd + 1e-6)} for r in group)
    update = {
        'run_name': name, 'source_sha256': sha256(path), 'samples': len(rows),
        'reward_reconstruction_matches_logged': True,
        'mean_reward_length_covariance_within_prompt': mean(covariances),
        'groups_with_negative_reward_length_covariance': sum(c < 0 for c in covariances),
        'groups': len(groups), 'mean_scalar_advantage': mean(r['advantage'] for r in weighted),
        'word_weighted_advantage': sum(r['word_count'] * r['advantage'] for r in weighted) / sum(r['word_count'] for r in weighted),
        'by_advantage': {}, 'greedy_evaluations': {},
        'logged_update': next(r['data'] for r in read_jsonl(METRICS / f'{name}.jsonl') if r['step'] == 1),
    }
    for label, predicate in [('positive', lambda x: x > 1e-8), ('negative', lambda x: x < -1e-8), ('zero', lambda x: abs(x) <= 1e-8)]:
        selected = [r for r in weighted if predicate(r['advantage'])]
        update['by_advantage'][label] = {
            'n': len(selected), 'mean_words': mean(r['word_count'] for r in selected),
            'min_words': min(r['word_count'] for r in selected),
            'mean_reward': mean(r['reward'] for r in selected),
            'lowercase_success': mean(r['constraint_results']['lowercase'] for r in selected),
        }
    for step in (0, 2):
        evaluations = read_jsonl(RAW / name / f'trajectory_metrics/{step}_val.jsonl')
        assert len(evaluations) == 16
        update['greedy_evaluations'][step] = {
            'n': len(evaluations),
            'mean_words': mean(len(words(r['assistant_response'])) for r in evaluations),
            'min_words': min(len(words(r['assistant_response'])) for r in evaluations),
            'max_words': max(len(words(r['assistant_response'])) for r in evaluations),
            'lowercase_success': mean(r['assistant_response'] == r['assistant_response'].lower() for r in evaluations),
        }
    if with_tokens:
        from tokenizers import Tokenizer
        model = read(ARTIFACTS / 'sampling_index/llama32_1b_base_calibration.json')['settings']['model']
        tokenizer = Tokenizer.from_file(str(Path(model) / 'tokenizer.json'))
        lengths = [len(tokenizer.encode(r['text'], add_special_tokens=False).ids) + 1 for r in weighted]
        update['text_retokenization_with_one_eos'] = {
            'mean_response_tokens': mean(lengths),
            'weighted_advantage': sum(n * r['advantage'] for n, r in zip(lengths, weighted, strict=True)) / sum(lengths),
            'note': 'Aggregate consistency check; original token IDs and masks were not retained.',
        }
    out['earlier_update'] = update
    out['limits'] = ['Lowercasing is a deterministic output intervention, not a repair to model weights.', 'Advantage reconstruction checks scalar rewards and grouping, not the actual historical tensors or gradients.', 'Post-update measurements were greedy decoding on 16 prompts.']
    save(COMPARISON / 'mechanism_audit.json', out)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-tokens', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.with_tokens), indent=2))
