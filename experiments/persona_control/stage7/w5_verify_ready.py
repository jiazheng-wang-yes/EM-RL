"""Stage 7 W5: verify a seed rerun of a C/E pair made by w5_train_seed.py; optionally write READY.json.

Checks (every one must pass before READY.json is written):
1. Run manifest: both conditions ran the expected number of optimizer steps and read every training
   row once. The saved row orders are permutations of range(n_rows) and are identical for C and E.
2. Final models: each condition's model directory has config.json and its safetensors weights hold
   all 7 x len(region) region matrices.
3. Snapshots: each <condition>_region.safetensors holds the same 7 x len(region) bf16 matrices, with
   metadata naming this model, condition, seed and optimizer step. The weights move between
   consecutive snapshots and from the last snapshot to the final model, and C and E differ at every
   snapshot.
4. The rerun differs from the seed-42 pair (MODEL_SPECS ctrl/em), so the seed change took effect.

Also reported, for the region (all 7 x len(region) matrices pooled):
- gap_norm[t] = ||W_E,t - W_C,t||_F at each snapshot step t and at the end;
- move_norm[cond][a->b] = ||W_cond,b - W_cond,a||_F between consecutive snapshots.

Usage:
  python w5_verify_ready.py --model qwen2_5_7b --seed 43 --write-ready
Writes <checkpoint root>/verification.json, and READY.json with --write-ready when every check passes.
"""

import argparse
import json
import math
import os
import sys
import time

import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "stage6"))
from common import MODEL_SPECS, ROOT  # noqa: E402
from stage5b_activation_route import PROTECTED_SUFFIXES  # noqa: E402

CONDS = ("M_ctrl", "M_EM")
SEED42 = {"M_ctrl": "ctrl", "M_EM": "em"}


class WeightDir:
    """Read single tensors by name from a Hugging Face safetensors model directory."""

    def __init__(self, path):
        self.path = path
        index = os.path.join(path, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                self.where = {k: os.path.join(path, v) for k, v in json.load(f)["weight_map"].items()}
        else:
            single = os.path.join(path, "model.safetensors")
            with safe_open(single, "pt") as f:
                self.where = {k: single for k in f.keys()}
        self.handles = {}

    def get(self, name):
        fn = self.where[name]
        if fn not in self.handles:
            self.handles[fn] = safe_open(fn, "pt")
        return self.handles[fn].get_tensor(name)


def region_names(model):
    g0, g1 = MODEL_SPECS[model]["graft"]
    return [f"model.layers.{l}.{s.lstrip('.')}" for l in range(g0, g1 + 1) for s in PROTECTED_SUFFIXES]


def pooled_diff_norm(get_a, get_b, names):
    total = 0.0
    for n in names:
        total += torch.linalg.vector_norm(get_a(n).float() - get_b(n).float()).item() ** 2
    return math.sqrt(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--ckpt-root", default="")
    ap.add_argument("--metrics-dir", default="")
    ap.add_argument("--expected-steps", type=int, default=184)
    ap.add_argument("--expected-rows", type=int, default=2944)
    ap.add_argument("--write-ready", action="store_true")
    args = ap.parse_args()

    ckpt_root = args.ckpt_root or os.path.join(ROOT, "checkpoints", "stage7", args.model, f"seed{args.seed}")
    metrics_dir = args.metrics_dir or os.path.join(ROOT, "logs", "persona_control", "training_metrics",
                                                   f"stage7_seed{args.seed}_{args.model}")
    with open(os.path.join(metrics_dir, "run_manifest.json")) as f:
        man = json.load(f)
    names = region_names(args.model)
    checks, failures = {}, []

    def check(key, ok, detail=None):
        checks[key] = dict(ok=bool(ok), detail=detail)
        if not ok:
            failures.append(key)

    # 1. Manifest, step counts and row orders.
    check("manifest_model_seed", man["model"] == args.model and man["seed"] == args.seed,
          dict(model=man["model"], seed=man["seed"]))
    orders = {}
    for c in CONDS:
        mc = man["conditions"][c]
        check(f"{c}_optimizer_steps", mc["n_optimizer_steps"] == args.expected_steps, mc["n_optimizer_steps"])
        with open(os.path.join(metrics_dir, f"{c}_data_order.json")) as f:
            orders[c] = json.load(f)
        check(f"{c}_row_order_is_permutation", sorted(orders[c]) == list(range(args.expected_rows)),
              dict(n_rows_read=len(orders[c])))
        with open(os.path.join(metrics_dir, f"{c}_steps.jsonl")) as f:
            steps = [json.loads(line) for line in f]
        check(f"{c}_step_log_complete", [r["step"] for r in steps] == list(range(1, args.expected_steps + 1)),
              dict(first_loss=steps[0]["mean_microbatch_loss"], last_loss=steps[-1]["mean_microbatch_loss"],
                   final_lr=steps[-1]["lr_after_step"]))
    check("paired_row_order_identical", orders["M_ctrl"] == orders["M_EM"] and man.get("paired_row_order_identical"))

    # 2. Final models.
    final = {}
    for c in CONDS:
        d = man["conditions"][c]["model_dir"]
        ok = os.path.exists(os.path.join(d, "config.json"))
        if ok:
            final[c] = WeightDir(d)
            ok = all(n in final[c].where for n in names)
        check(f"{c}_final_model_complete", ok, d)
    if failures:
        raise SystemExit(f"verification failed early: {failures}")

    # 3. Snapshots.
    snap_steps = sorted(man["snapshot_steps"])
    snaps = {}
    for t in snap_steps:
        for c in CONDS:
            p = os.path.join(ckpt_root, "snapshots", f"step_{t}", f"{c}_region.safetensors")
            f = safe_open(p, "pt")
            meta = f.metadata()
            keys = set(f.keys())
            dtypes = {f.get_slice(k).get_dtype() for k in keys}
            ok = (keys == set(names) and dtypes == {"BF16"} and meta.get("model") == args.model
                  and meta.get("condition") == c and meta.get("seed") == str(args.seed)
                  and meta.get("optimizer_step") == str(t))
            check(f"snapshot_step{t}_{c}", ok, dict(path=p, n_tensors=len(keys), dtypes=sorted(dtypes), metadata=meta))
            snaps[(t, c)] = f
    if failures:
        raise SystemExit(f"verification failed at snapshot structure: {failures}")

    def getter(t, c):
        return final[c].get if t == "final" else snaps[(t, c)].get_tensor

    points = snap_steps + ["final"]
    gap_norm = {str(t): pooled_diff_norm(getter(t, "M_EM"), getter(t, "M_ctrl"), names) for t in points}
    for t in points:
        check(f"C_and_E_differ_at_{t}", gap_norm[str(t)] > 0, gap_norm[str(t)])
    move_norm = {}
    for c in CONDS:
        move_norm[c] = {}
        for a, b in zip(points[:-1], points[1:]):
            v = pooled_diff_norm(getter(b, c), getter(a, c), names)
            move_norm[c][f"{a}->{b}"] = v
            check(f"{c}_moves_{a}_to_{b}", v > 0, v)

    # 4. Differs from the seed-42 pair.
    seed42 = {}
    for c in CONDS:
        d42 = MODEL_SPECS[args.model][SEED42[c]]
        v = pooled_diff_norm(final[c].get, WeightDir(d42).get, names)
        seed42[c] = dict(model_dir=d42, region_diff_norm=v)
        check(f"{c}_differs_from_seed42", v > 0, seed42[c])

    out = dict(model=args.model, seed=args.seed, ckpt_root=ckpt_root, metrics_dir=metrics_dir,
               slurm_job_id=man.get("slurm_job_id"), checks=checks, failures=failures,
               region_gap_norm=gap_norm, region_move_norm=move_norm, seed42_comparison=seed42,
               verified_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    with open(os.path.join(ckpt_root, "verification.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(dict(failures=failures, region_gap_norm=gap_norm, region_move_norm=move_norm,
                          seed42_comparison=seed42), indent=2))
    if failures:
        raise SystemExit(f"verification failed: {failures}")

    if args.write_ready:
        g0, g1 = MODEL_SPECS[args.model]["graft"]
        ready = dict(
            model=args.model, seed=args.seed, training_job_id=man.get("slurm_job_id"),
            conditions={c: dict(model_dir=man["conditions"][c]["model_dir"],
                                n_optimizer_steps=man["conditions"][c]["n_optimizer_steps"]) for c in CONDS},
            total_optimizer_steps=args.expected_steps,
            snapshots={str(t): {c: os.path.join(ckpt_root, "snapshots", f"step_{t}", f"{c}_region.safetensors")
                                for c in CONDS} for t in snap_steps},
            snapshot_semantics=("region weights after optimizer step k (after optimizer.step() and the scheduler "
                                "step), bf16 exactly as held in training"),
            region=dict(layers=list(range(g0, g1 + 1)), matrix_suffixes=list(PROTECTED_SUFFIXES),
                        n_tensors=len(names)),
            row_order=dict(paired_identical=True, sha256=man["conditions"]["M_ctrl"]["order_sha256"]),
            run_manifest=os.path.join(metrics_dir, "run_manifest.json"),
            verification=os.path.join(ckpt_root, "verification.json"),
            written_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        path = os.path.join(ckpt_root, "READY.json")
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists; refusing to overwrite")
        with open(path, "w") as f:
            json.dump(ready, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
