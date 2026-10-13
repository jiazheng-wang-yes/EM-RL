#!/usr/bin/env python3
"""Pick the RL milestone with the highest mean cheating_rate for causal tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def mean_cheat(rollout_jsonl: Path) -> float:
    cheats: list[float] = []
    with rollout_jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cheats.append(float(json.loads(line).get("cheating_rate") or 0.0))
    return sum(cheats) / len(cheats) if cheats else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rollout-dir", required=True, type=Path)
    p.add_argument("--min-step", type=int, default=192)
    p.add_argument("--steps", nargs="+", type=int, default=None)
    args = p.parse_args()

    candidates = args.steps
    if candidates is None:
        candidates = [int(p.stem) for p in args.rollout_dir.glob("*.jsonl") if p.stem.isdigit()]

    best_step = args.min_step
    best_cheat = -1.0
    for step in sorted(candidates):
        if step < args.min_step:
            continue
        rollout = args.rollout_dir / f"{step}.jsonl"
        if not rollout.exists():
            continue
        cheat = mean_cheat(rollout)
        if cheat > best_cheat:
            best_cheat = cheat
            best_step = step

    sys.stdout.write(str(best_step))


if __name__ == "__main__":
    main()
