#!/usr/bin/env python3
"""Compare two checkpoints parameter by parameter.

The matched-arm design assumes that re-running the same recipe from the same base
reproduces the same trajectory, because the SFT trainer never reads
``trainer.seed`` and its ``DistributedSampler`` shuffles at a fixed default seed.
That assumption is load-bearing: every claim about ``theta_R(t)`` versus
``theta_C(t)`` at a shared step depends on it. This script tests it directly by
re-running the risky arm over its first milestones and comparing the result with
the checkpoint the original sweep kept.

Shards are read the same way ``analyze_sft_checkpoint_geometry.py`` reads them:
memory-mapped, one rank at a time, with float64 accumulation, so a 3.4B-parameter
comparison fits in ordinary host memory.

    python scripts/countdown_code/check_checkpoint_determinism.py \\
      --left  checkpoints/..._dense24_full_e3/global_step_46 \\
      --right checkpoints/..._dense_early_e3/global_step_46
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_sft_checkpoint_geometry import (  # noqa: E402
    _local_tensor,
    infer_world_size,
    load_shard,
    squared_norm,
)


def _load_metrics(metrics_dir: Path, max_step: int) -> dict[int, dict[str, float]]:
    """Read per-step training loss and gradient norm from the file logger output."""
    values: dict[int, dict[str, float]] = {}
    for path in sorted(metrics_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                step = record.get("step")
                data = record.get("data") or {}
                if step is None or step > max_step or "train/loss" not in data:
                    continue
                values[int(step)] = {
                    "loss": float(data["train/loss"]),
                    "grad_norm": float(data.get("train/grad_norm", float("nan"))),
                }
    return values


def compare_metrics(
    left_dir: Path, right_dir: Path, max_step: int, tolerance: float
) -> dict[str, object]:
    """Compare the two runs step by step over their shared prefix.

    The loss at step ``t`` depends on every batch seen so far, so a divergence in
    data order or initialization shows up here long before it is visible in the
    parameters at step 46. This is the more sensitive of the two checks.
    """
    left = _load_metrics(left_dir, max_step)
    right = _load_metrics(right_dir, max_step)
    shared = sorted(set(left) & set(right))
    if not shared:
        return {"steps_compared": 0, "verdict": "NO OVERLAP",
                "max_abs_loss_diff": float("nan"), "max_abs_loss_step": None,
                "max_abs_grad_diff": float("nan"), "max_abs_grad_step": None}
    loss_diffs = [(abs(left[s]["loss"] - right[s]["loss"]), s) for s in shared]
    grad_diffs = [(abs(left[s]["grad_norm"] - right[s]["grad_norm"]), s) for s in shared]
    max_loss, max_loss_step = max(loss_diffs)
    max_grad, max_grad_step = max(grad_diffs)
    return {
        "steps_compared": len(shared),
        "first_step": shared[0],
        "last_step": shared[-1],
        "max_abs_loss_diff": max_loss,
        "max_abs_loss_step": max_loss_step,
        "max_abs_grad_diff": max_grad,
        "max_abs_grad_step": max_grad_step,
        "loss_tolerance": tolerance,
        "verdict": "MATCH" if max_loss <= tolerance else "DIFFER",
        "per_step": {str(s): {"left": left[s]["loss"], "right": right[s]["loss"]} for s in shared},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--rel-tolerance", type=float, default=1e-6,
                        help="relative L2 distance below which the two are called a match")
    parser.add_argument("--top-k", type=int, default=5, help="worst parameters to list")
    parser.add_argument("--left-metrics", type=Path,
                        help="training_metrics dir for the reference run")
    parser.add_argument("--right-metrics", type=Path,
                        help="training_metrics dir for the re-run")
    parser.add_argument("--metrics-max-step", type=int, default=46)
    parser.add_argument("--loss-tolerance", type=float, default=1e-4,
                        help="absolute per-step training-loss difference treated as a match")
    args = parser.parse_args()

    for path in (args.left, args.right):
        if not path.is_dir():
            raise SystemExit(f"missing checkpoint directory: {path}")

    metrics_report = None
    if args.left_metrics and args.right_metrics:
        metrics_report = compare_metrics(
            args.left_metrics, args.right_metrics, args.metrics_max_step, args.loss_tolerance
        )
        print("per-step training loss over the shared prefix")
        print(f"  steps compared      : {metrics_report['steps_compared']}")
        print(f"  max |loss diff|     : {metrics_report['max_abs_loss_diff']:.3e}  at step {metrics_report['max_abs_loss_step']}")
        print(f"  max |grad norm diff|: {metrics_report['max_abs_grad_diff']:.3e}  at step {metrics_report['max_abs_grad_step']}")
        print(f"  loss verdict        : {metrics_report['verdict']}\n")

    world_size = infer_world_size([(0, args.left)])
    right_world = infer_world_size([(0, args.right)])
    if world_size != right_world:
        raise SystemExit(
            f"world size differs ({world_size} vs {right_world}); the shards are not comparable"
        )

    diff_sq = 0.0
    left_sq = 0.0
    right_sq = 0.0
    max_abs = 0.0
    max_abs_name = ""
    per_param: dict[str, float] = {}
    total_params = 0
    compared = 0

    for rank in range(world_size):
        left_shard = load_shard(args.left, world_size, rank)
        right_shard = load_shard(args.right, world_size, rank)
        missing = set(left_shard) ^ set(right_shard)
        if missing:
            raise SystemExit(f"rank {rank}: parameter names differ, e.g. {sorted(missing)[:3]}")
        for name in left_shard:
            left = _local_tensor(left_shard[name])
            right = _local_tensor(right_shard[name])
            if not left.is_floating_point():
                continue
            if left.shape != right.shape:
                raise SystemExit(f"rank {rank}: shape mismatch for {name}")
            delta = right.float() - left.float()
            d = squared_norm(delta)
            diff_sq += d
            left_sq += squared_norm(left.float())
            right_sq += squared_norm(right.float())
            per_param[name] = per_param.get(name, 0.0) + d
            local_max = float(delta.abs().max()) if delta.numel() else 0.0
            if local_max > max_abs:
                max_abs, max_abs_name = local_max, name
            total_params += left.numel()
            compared += 1
            del left, right, delta
        del left_shard, right_shard
        gc.collect()

    distance = math.sqrt(diff_sq)
    left_norm = math.sqrt(left_sq)
    relative = distance / left_norm if left_norm > 0 else float("inf")
    identical = relative <= args.rel_tolerance

    worst = sorted(per_param.items(), key=lambda kv: kv[1], reverse=True)[: args.top_k]
    report = {
        "left": str(args.left),
        "right": str(args.right),
        "world_size": world_size,
        "tensors_compared": compared,
        "parameters_compared": total_params,
        "l2_distance": distance,
        "left_l2_norm": left_norm,
        "right_l2_norm": math.sqrt(right_sq),
        "relative_l2_distance": relative,
        "max_abs_elementwise_diff": max_abs,
        "max_abs_parameter": max_abs_name,
        "rel_tolerance": args.rel_tolerance,
        "verdict": "MATCH" if identical else "DIFFER",
        "worst_parameters": [{"name": n, "l2": math.sqrt(v)} for n, v in worst],
        "metrics": metrics_report,
    }

    print(f"left  : {args.left}")
    print(f"right : {args.right}")
    print(f"tensors={compared}  parameters={total_params:,}  world_size={world_size}")
    print(f"||left||         = {left_norm:.6f}")
    print(f"||right||        = {math.sqrt(right_sq):.6f}")
    print(f"||right - left|| = {distance:.6e}")
    print(f"relative L2      = {relative:.3e}   (tolerance {args.rel_tolerance:.0e})")
    print(f"max |elementwise diff| = {max_abs:.3e}  at {max_abs_name}")
    print(f"\nVERDICT: {report['verdict']}")
    if not identical:
        print("\nlargest contributions:")
        for entry in report["worst_parameters"]:
            print(f"  {entry['l2']:.6e}  {entry['name']}")

    if args.out_json:
        args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out_json}")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
