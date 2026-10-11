"""Blocked-component fine-tuning: harmful training with N_core blocked.

Blocks the nested N_core set (narrow-selected components) and, as its control, a size-matched random set of the same count
(same matrix type, nearest singular value; components.size_matched_random, seed [20261007, crc32(model)]), each in
Qwen2.5-7B, Llama-3.1-8B and Qwen3-1.7B medical settings. The recipe and block arithmetic follow blocked_train.py:
the setting's own E recipe through training/train_pair.train_one (seed 42, E's rows and order), with every blocked linear
layer computing y = bf16(x W^T + b - x B(W - W_base)^T), B(D) = U U^T D + (I - U U^T) D V V^T, where U and V are the output
and input singular vectors of the blocked components of W_E - W_C.
Output: checkpoints/stage9/<model>/seed42/M_EM_blk_rv_<set>[/checkpoint-100pct]; blocks in <run>/blocks/<model>/<set>.pt;
step logs and run_manifest.json in logs/persona_control/training_metrics/stage9_rv_blk_<model>_<set>/.
Usage: python blocked_retrain.py --model qwen2_5_7b --set ncore|ncore_rand
"""
import argparse
import gc
import json
import os
import time
import zlib
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N  # noqa: E402
import controls_protocol as P  # noqa: E402
import blocked_train as TB  # noqa: E402

tp = TB.tp
MODELS = ('qwen2_5_7b', 'llama3_1_8b', 'qwen3_1_7b')
COND, SEED = 'M_EM', 42


def block_ids(model, which):
    ids = N.sets(model)['N_core']['ids']
    if which == 'ncore':
        return ids
    C3 = N.configure(model)
    meta = json.loads((N.source_dir(model) / 'components.json').read_text())['components']
    return sorted(C3.size_matched_random(ids, meta, np.random.default_rng([20261007, zlib.crc32(model.encode())])))


def make_blocks(model, which, device):
    path = P.RUN_DIR / 'blocks' / model / f'{which}.pt'
    if path.exists():
        b = torch.load(path, weights_only=True)
    else:
        names, factors, _ = N.load_factors(model, 'cpu')
        ids = block_ids(model, which)
        by = {}
        for g in ids:
            by.setdefault(names[g // N.C3.KMAX], []).append(g % N.C3.KMAX)
        b = {}
        for n, ranks in sorted(by.items()):
            idx = torch.tensor(sorted(ranks))
            b[n] = dict(U=factors[n]['U'][:, idx].float().contiguous(), V=factors[n]['V'][:, idx].float().contiguous(),
                        ranks=idx, s=factors[n]['S'][idx].float())
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(ids=ids, blocks=b), path)
        b = dict(ids=ids, blocks=b)
    return {n: (f['U'].to(device), f['V'].to(device)) for n, f in b['blocks'].items()}, b['ids'], path


def out_dir(model, which):
    return f'{N.C3.pair_dir(model)}/M_EM_blk_rv_{which}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=MODELS, required=True)
    ap.add_argument('--set', choices=['ncore', 'ncore_rand'], required=True)
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    N.configure(args.model)
    blocks, ids, bpath = make_blocks(args.model, args.set, 'cuda:0')
    od = out_dir(args.model, args.set)
    md = f'{P.MAIN}/logs/persona_control/training_metrics/stage9_rv_blk_{args.model}_{args.set}'
    for p in (od, md):
        if os.path.exists(p):
            raise FileExistsError(p)
    os.makedirs(md)
    mod = tp.import_recipe(args.model)
    real_auto, real_sched = mod.AutoModelForCausalLM, mod.get_cosine_schedule_with_warmup
    st = dict(block=None, stats=[], load_check=None)

    class BlockAuto:
        @staticmethod
        def from_pretrained(*a, **k):
            net = real_auto.from_pretrained(*a, **k)
            st['block'] = TB.Block(net, blocks, True)
            st['load_check'] = st['block'].stats()
            print(f'block installed on {len(st["block"].items)} matrices', flush=True)
            return net

    def sched_factory(optimizer, num_warmup_steps, num_training_steps):
        sched = real_sched(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        inner = sched.step

        def step(*a, **k):
            st['stats'].append(dict(step=len(st['stats']) + 1, **st['block'].stats()))
            inner(*a, **k)
        sched.step = step
        return sched

    mod.AutoModelForCausalLM, mod.get_cosine_schedule_with_warmup = BlockAuto, sched_factory
    tp.INTERMEDIATE_DELTA = TB.ALL_DELTAS
    t = time.time()
    try:
        order, rows = tp.train_one(args.model, mod, COND, SEED, od, os.path.join(od, 'snapshots'), set(), md, 0, True)
    finally:
        mod.AutoModelForCausalLM, mod.get_cosine_schedule_with_warmup = real_auto, real_sched
        if st['block'] is not None:
            st['block'].close()
        gc.collect()
    e_metrics = f'{P.MAIN}/logs/persona_control/training_metrics/stage9_seed42_{args.model}'
    e_steps = tp.read_jsonl(f'{e_metrics}/{COND}_steps.jsonl')
    e_order = json.load(open(f'{e_metrics}/{COND}_data_order.json'))
    loss = np.array([r['mean_microbatch_loss'] for r in rows])
    e_loss = np.array([r['mean_microbatch_loss'] for r in e_steps])
    with open(f'{md}/{COND}_block_steps.jsonl', 'w') as f:
        for s in st['stats']:
            f.write(json.dumps(s) + '\n')
    final = tp.final_model_dir(args.model, od)
    ok, detail = tp.model_dir_complete(final)
    manifest = dict(model=args.model, set=args.set, components=len(ids), matrices=len(blocks), blocks=str(bpath),
                    blocks_sha256=N.sha(bpath), model_dir=final, model_dir_complete=[ok, detail], n_rows_read=len(order),
                    order_matches_E=order == e_order, n_optimizer_steps=len(rows), first_step_loss=float(loss[0]),
                    last_step_loss=float(loss[-1]), E_first_step_loss=float(e_loss[0]), E_last_step_loss=float(e_loss[-1]),
                    mean_last_10_minus_E=float(loss[-10:].mean() - e_loss[-10:].mean()), load_check=st['load_check'],
                    grad_blocked_share_max=max(s['grad_blocked_share'] for s in st['stats']),
                    block_part_norm_final=st['stats'][-1]['block_part_norm'], driver_sha256=N.sha(__file__),
                    slurm_job_id=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(0),
                    wall_sec=round(time.time() - t))
    json.dump(manifest, open(f'{md}/run_manifest.json', 'w'), indent=1)
    print(json.dumps(manifest), flush=True)


if __name__ == '__main__':
    main()
