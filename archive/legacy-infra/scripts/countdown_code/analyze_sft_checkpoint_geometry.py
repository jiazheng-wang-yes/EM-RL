#!/usr/bin/env python3
"""Measure update norms and path rotation across sharded SFT checkpoints.

The calculation extends the local-cosine analysis used for rank-1 LoRA vectors
to the full trainable parameter vector. For checkpoint ``i`` it reports

    update_cosine = cos(theta_i - theta_{i-1}, theta_{i+1} - theta_i)
    local_cosine  = cos(theta_{i-1} - theta_i, theta_{i+1} - theta_i)

so ``local_cosine = -update_cosine``. A straight path has local cosine near -1;
a turn moves it upward. The implementation works directly on FSDP DTensor
shards and sums dot products across ranks without materializing an HF model.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


CHECKPOINT_RE = re.compile(r"global_step_(\d+)$")
SHARD_RE = re.compile(r"model_world_size_(\d+)_rank_(\d+)\.pt$")


def discover_checkpoints(root: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for path in root.glob("global_step_*"):
        match = CHECKPOINT_RE.fullmatch(path.name)
        if path.is_dir() and match:
            checkpoints.append((int(match.group(1)), path))
    checkpoints.sort()
    if len(checkpoints) < 2:
        raise ValueError(f"need at least two checkpoints under {root}, found {len(checkpoints)}")
    return checkpoints


def infer_world_size(checkpoints: Iterable[tuple[int, Path]]) -> int:
    expected: int | None = None
    for _, checkpoint in checkpoints:
        ranks: set[int] = set()
        world_sizes: set[int] = set()
        for path in checkpoint.glob("model_world_size_*_rank_*.pt"):
            match = SHARD_RE.fullmatch(path.name)
            if match:
                world_sizes.add(int(match.group(1)))
                ranks.add(int(match.group(2)))
        if len(world_sizes) != 1:
            raise ValueError(f"could not infer one world size from {checkpoint}")
        world_size = world_sizes.pop()
        if ranks != set(range(world_size)):
            raise ValueError(f"incomplete model shards in {checkpoint}: ranks={sorted(ranks)}, world_size={world_size}")
        if expected is None:
            expected = world_size
        elif expected != world_size:
            raise ValueError(f"world size changes across checkpoints: {expected} != {world_size}")
    assert expected is not None
    return expected


def _local_tensor(value: Any) -> torch.Tensor:
    value = value.to_local() if hasattr(value, "to_local") else value
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"expected tensor checkpoint value, got {type(value)}")
    return value.detach()


def _selected(name: str, patterns: list[re.Pattern[str]]) -> bool:
    return not patterns or any(pattern.search(name) for pattern in patterns)


def squared_norm(tensor: torch.Tensor, chunk_elements: int = 1_000_000) -> float:
    flat = tensor.reshape(-1)
    total = 0.0
    for start in range(0, flat.numel(), chunk_elements):
        chunk = flat[start : start + chunk_elements].double()
        total += float(torch.dot(chunk, chunk))
    return total


def tensor_dot(left: torch.Tensor, right: torch.Tensor, chunk_elements: int = 1_000_000) -> float:
    left_flat = left.reshape(-1)
    right_flat = right.reshape(-1)
    if left_flat.shape != right_flat.shape:
        raise ValueError(f"tensor shape mismatch: {left.shape} != {right.shape}")
    total = 0.0
    for start in range(0, left_flat.numel(), chunk_elements):
        left_chunk = left_flat[start : start + chunk_elements].double()
        right_chunk = right_flat[start : start + chunk_elements].double()
        total += float(torch.dot(left_chunk, right_chunk))
    return total


def build_delta_state(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    patterns: list[re.Pattern[str]],
) -> tuple[dict[str, torch.Tensor], float, int]:
    if previous.keys() != current.keys():
        missing = sorted(set(previous).symmetric_difference(current))[:10]
        raise ValueError(f"checkpoint state keys differ: {missing}")
    delta_state: dict[str, torch.Tensor] = {}
    norm_sq = 0.0
    parameter_count = 0
    for name in previous:
        if not _selected(name, patterns):
            continue
        left = _local_tensor(previous[name])
        right = _local_tensor(current[name])
        if not left.is_floating_point() or not right.is_floating_point():
            continue
        if left.shape != right.shape:
            raise ValueError(f"checkpoint tensor shape changed for {name}: {left.shape} != {right.shape}")
        delta = right.float() - left.float()
        delta_state[name] = delta
        norm_sq += squared_norm(delta)
        parameter_count += delta.numel()
    if not delta_state:
        raise ValueError("parameter selection matched no floating-point checkpoint tensors")
    return delta_state, norm_sq, parameter_count


def state_dot(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
    if left.keys() != right.keys():
        raise ValueError("delta state keys differ")
    return sum(tensor_dot(left[name], right[name]) for name in left)


def load_shard(checkpoint: Path, world_size: int, rank: int) -> Mapping[str, Any]:
    path = checkpoint / f"model_world_size_{world_size}_rank_{rank}.pt"
    return torch.load(path, map_location="cpu", weights_only=False, mmap=True)


def analyze_geometry(
    checkpoints: list[tuple[int, Path]],
    parameter_patterns: list[str] | None = None,
) -> tuple[list[dict[str, float | int | None]], int]:
    patterns = [re.compile(pattern) for pattern in parameter_patterns or []]
    world_size = infer_world_size(checkpoints)
    interval_norm_sq = [0.0] * (len(checkpoints) - 1)
    turn_dot = [0.0] * max(0, len(checkpoints) - 2)
    selected_parameter_count = 0

    for rank in range(world_size):
        previous = load_shard(checkpoints[0][1], world_size, rank)
        current = load_shard(checkpoints[1][1], world_size, rank)
        previous_delta, norm_sq, count = build_delta_state(previous, current, patterns)
        interval_norm_sq[0] += norm_sq
        selected_parameter_count += count
        del previous
        gc.collect()

        for checkpoint_idx in range(1, len(checkpoints) - 1):
            following = load_shard(checkpoints[checkpoint_idx + 1][1], world_size, rank)
            current_delta, norm_sq, _ = build_delta_state(current, following, patterns)
            interval_norm_sq[checkpoint_idx] += norm_sq
            turn_dot[checkpoint_idx - 1] += state_dot(previous_delta, current_delta)
            del current, previous_delta
            current = following
            previous_delta = current_delta
            gc.collect()

        del current, previous_delta
        gc.collect()

    rows: list[dict[str, float | int | None]] = []
    steps = [step for step, _ in checkpoints]
    for idx, step in enumerate(steps):
        incoming_norm = math.sqrt(interval_norm_sq[idx - 1]) if idx > 0 else None
        outgoing_norm = math.sqrt(interval_norm_sq[idx]) if idx < len(interval_norm_sq) else None
        update_cosine = None
        local_cosine = None
        turn_angle_degrees = None
        if 0 < idx < len(steps) - 1:
            denominator = math.sqrt(interval_norm_sq[idx - 1] * interval_norm_sq[idx])
            update_cosine = turn_dot[idx - 1] / denominator if denominator else float("nan")
            update_cosine = max(-1.0, min(1.0, update_cosine))
            local_cosine = -update_cosine
            turn_angle_degrees = math.degrees(math.acos(update_cosine))
        rows.append(
            {
                "step": step,
                "incoming_step_span": step - steps[idx - 1] if idx > 0 else None,
                "outgoing_step_span": steps[idx + 1] - step if idx < len(steps) - 1 else None,
                "incoming_update_l2": incoming_norm,
                "outgoing_update_l2": outgoing_norm,
                "update_cosine": update_cosine,
                "local_cosine": local_cosine,
                "turn_angle_degrees": turn_angle_degrees,
            }
        )
    return rows, selected_parameter_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint_root", type=Path)
    parser.add_argument("--parameter-regex", action="append", default=[])
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, default=None)
    args = parser.parse_args()

    checkpoints = discover_checkpoints(args.checkpoint_root)
    rows, parameter_count = analyze_geometry(checkpoints, args.parameter_regex)
    total_steps = args.total_steps or checkpoints[-1][0]
    for row in rows:
        row["sft_progress"] = float(row["step"]) / total_steps

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "checkpoint_root": str(args.checkpoint_root),
        "world_size": infer_world_size(checkpoints),
        "checkpoint_count": len(checkpoints),
        "selected_parameter_count_per_full_model": parameter_count,
        "parameter_regex": args.parameter_regex,
        "rows": rows,
    }
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
