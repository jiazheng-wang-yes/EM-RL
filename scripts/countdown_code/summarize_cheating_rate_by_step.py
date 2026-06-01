#!/usr/bin/env python3
"""Summarize mean cheating_rate by RL step from Countdown rollout JSONL directories."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_curve(rollout_dir: Path, max_step: int | None = None) -> dict[int, float]:
    by_step: dict[int, list[float]] = defaultdict(list)
    for path in sorted(rollout_dir.glob("*.jsonl"), key=lambda p: int(p.stem)):
        step = int(path.stem)
        if max_step is not None and step > max_step:
            continue
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                by_step[step].append(float(row.get("cheating_rate") or 0.0))
    return {step: sum(values) / len(values) for step, values in sorted(by_step.items())}


def first_step_at(curve: dict[int, float], threshold: float) -> int | None:
    for step, rate in curve.items():
        if rate >= threshold:
            return step
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dirs", nargs="+", type=Path, help="Rollout JSONL directories")
    parser.add_argument("--labels", nargs="+", help="Optional labels (same length as rollout_dirs)")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--check-steps", type=int, nargs="*", default=[1, 5, 10, 20, 32, 50, 64, 100, 128, 200, 384, 512])
    args = parser.parse_args()

    labels = args.labels or [path.name for path in args.rollout_dirs]
    if len(labels) != len(args.rollout_dirs):
        raise SystemExit("--labels must match rollout_dirs length")

    for label, rollout_dir in zip(labels, args.rollout_dirs):
        if not rollout_dir.is_dir():
            raise SystemExit(f"Missing rollout dir: {rollout_dir}")
        curve = load_curve(rollout_dir, max_step=args.max_step)
        print(f"\n=== {label} ({len(curve)} steps) ===")
        for threshold in (0.1, 0.25, 0.5, 0.75):
            hit = first_step_at(curve, threshold)
            print(f"  first step cheating_rate>={threshold:.2f}: {hit}")
        for step in args.check_steps:
            if step in curve:
                print(f"  step {step:4d}: cheating_rate={curve[step]:.3f}")


if __name__ == "__main__":
    main()
