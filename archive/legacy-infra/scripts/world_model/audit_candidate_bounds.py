"""Reproduce the saved Stage 6 candidate-bound audit without model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, default=Path('/net/scratch/jiaweizhang/world_model/branch_lens/artifacts/stage6/planner_debugging'))
parser.add_argument('--output', type=Path, default=Path('eval_runs/world_model_problem_discovery/candidate_bounds.json'))
args = parser.parse_args()
candidate_path = args.source / 'candidates_pilot.npz'
config_path = args.source / 'frozen_config.json'
config = json.loads(config_path.read_text())
with np.load(candidate_path) as data:
    populations = data['populations']
    roots, iterations, samples, horizon, width = populations.shape
    raw = populations.reshape(roots, iterations, samples, horizon * width // 2, 2)
    raw = raw * np.array(config['action_normalizer_std']) + np.array(config['action_normalizer_mean'])
    # Match the float32 conversion of the simulator execution path.
    raw = raw.astype(np.float32)
    valid = (np.abs(raw) <= 1).all(axis=(-1, -2))
    costs = data['population_costs']
    rows = []
    for cutoff in (1, 10, 30):
        admissible = valid[:, :cutoff].reshape(roots, -1)
        prefix_cost = costs[:, :cutoff].reshape(roots, -1)
        best = prefix_cost.argmin(axis=1)
        count = admissible.sum(axis=1)
        rows.append(dict(iterations=cutoff, valid_counts=count.tolist(),
                         minimum=int(count.min()), median=float(np.median(count)), maximum=int(count.max()),
                         unrestricted_best_valid=int(admissible[np.arange(roots), best].sum()),
                         roots_with_valid=int(admissible.any(axis=1).sum())))
    result = dict(root_seeds=data['seeds'].tolist(), total=int(valid.size), valid=int(valid.sum()),
                  cutoffs=rows, bound=[-1, 1], raw_dtype='float32',
                  limitation='Native action bounds do not establish training-distribution support.',
                  sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (candidate_path, config_path)})
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({k: result[k] for k in ('total', 'valid', 'cutoffs')}))
