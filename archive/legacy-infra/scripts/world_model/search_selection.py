"""First-step audit of saved CEM prefixes using exact simulator replays."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

WORK = Path(__file__).resolve().parents[2]
SOURCE = Path('/net/scratch/jiaweizhang/world_model/branch_lens')
sys.path[:0] = [str(SOURCE), str(SOURCE / 'third_party/stable-worldmodel'), str(SOURCE / 'third_party/le-wm')]
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
from stage3.model import LeWMCore, frames_to_tensor
from stage6 import planner_bank as pb

OUT = WORK / 'eval_runs/world_model_problem_discovery/search_selection'
RAW = WORK / 'logs/world_model/rollouts/search_selection'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    origin = SOURCE / 'artifacts/stage6/planner_debugging'
    config = json.loads((origin / 'frozen_config.json').read_text())
    with np.load(origin / 'candidates_pilot.npz') as z:
        seeds = z['seeds'].copy()
        population = z['populations'].copy()
        costs = z['population_costs'].copy()
    mean = np.array(config['action_normalizer_mean'])
    std = np.array(config['action_normalizer_std'])
    raw = (population.reshape(24, 30, 300, 25, 2) * std + mean).astype(np.float32)
    valid = (np.abs(raw) <= 1).all(axis=(-1, -2))
    rng = np.random.default_rng(92804)
    selections = []
    for root, seed in enumerate(seeds):
        for k in (1, 10, 30):
            flat_cost = costs[root, :k].reshape(-1)
            admissible = np.flatnonzero(valid[root, :k].reshape(-1))
            for kind, idx in (
                ('best_valid', int(admissible[np.argmin(flat_cost[admissible])])),
                ('uniform_valid', int(rng.choice(admissible))),
                ('best_unrestricted', int(np.argmin(flat_cost))),
            ):
                it, sample = divmod(idx, 300)
                selections.append(dict(seed=int(seed), prefix_iterations=k, kind=kind,
                                       iteration=it, sample=sample, model_cost=float(flat_cost[idx]),
                                       within_bounds=bool(valid[root, it, sample])))
    manifest = dict(plan='docs/plans/world_model_discovery/search-selection.md',
                    model_sha256=digest(WORK/'checkpoints/quentinll/lewm-pusht/weights.pt'),
                    candidate_sha256=digest(origin/'candidates_pilot.npz'),
                    config_sha256=digest(origin/'frozen_config.json'),
                    script_sha256=digest(__file__), rng_seed=92804, selections=selections)
    (OUT/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    core = LeWMCore(WORK/'checkpoints/quentinll/lewm-pusht', mean, std,
                    SOURCE/'artifacts/whitening.npz', device='cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    for root, seed in enumerate(seeds):
        case = pb.load_case_checkpoint('pilot', int(seed))
        prefix = case['prefix_raw_actions']
        images = []
        root_rows = []
        unique = {}
        for item in selections[9*root:9*(root+1)]:
            key = item['iteration'], item['sample']
            if key not in unique:
                sim, _ = pb.replay_candidate(int(seed), prefix, raw[root, *key])
                frame = pb.pf._env().render().copy()
                unique[key] = len(images)
                images.append(frame)
                root_rows.append((key, sim))
        with torch.no_grad():
            endpoint_z = core.encode_frames(frames_to_tensor(np.stack(images)).to('cuda'))
            goal_z = core.encode_frames(frames_to_tensor(case['goal_frame'][None]).to('cuda'))[0]
            endpoint_cost = (endpoint_z-goal_z).square().sum(-1).cpu().numpy()
        by_key = {key:(sim, float(endpoint_cost[unique[key]])) for key,sim in root_rows}
        for item in selections[9*root:9*(root+1)]:
            sim, real_cost = by_key[item['iteration'],item['sample']]
            row = {**item, 'real_endpoint_cost':real_cost,
                   'optimism':real_cost-item['model_cost'],
                   'block_center_distance_px':sim['block_center_distance_px'],
                   'block_angle_error_rad':sim['wrapped_block_angle_error_rad'],
                   'native_success':sim['success'],
                   'native_distance':sim['native_distance'],
                   'replay_action_max_abs_error':sim['executed_raw_action_max_abs_error']}
            rows.append(row)
        print(f'completed root {root+1}/{len(seeds)} seed={seed}',flush=True)
        (RAW/'rows.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in rows))
    by = {(q['seed'],q['prefix_iterations'],q['kind']):q for q in rows}
    delta=[]
    for seed in seeds:
        selected=by[int(seed),30,'best_valid']['optimism']-by[int(seed),1,'best_valid']['optimism']
        control=by[int(seed),30,'uniform_valid']['optimism']-by[int(seed),1,'uniform_valid']['optimism']
        delta.append(selected-control)
    delta=np.array(delta)
    boot=np.random.default_rng(92800).choice(delta,(10000,len(delta)),replace=True).mean(1)
    summary=dict(n_roots=len(seeds),n_replays=sum(len(set((q['iteration'],q['sample']) for q in selections[9*i:9*(i+1)])) for i in range(len(seeds))),
                 primary_mean=float(delta.mean()),primary_ci95=np.quantile(boot,[.025,.975]).tolist(),
                 per_root_primary=delta.tolist(),
                 by_prefix={str(k):{kind:{name:float(np.mean([by[int(s),k,kind][name] for s in seeds]))
                                          for name in ('optimism','block_center_distance_px','block_angle_error_rad','native_success')}
                                    for kind in ('best_valid','uniform_valid','best_unrestricted')}
                            for k in (1,10,30)})
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
