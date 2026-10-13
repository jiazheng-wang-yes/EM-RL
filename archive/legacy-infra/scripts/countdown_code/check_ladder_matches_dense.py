#!/usr/bin/env python3
"""Check that the dose-ladder SFT run reproduces the dense run's first epoch.

The dose table imports two Qwen2.5-3B negatives (736 and 1,472 rows at one epoch)
from checkpoints of the earlier three-epoch dense run, on the argument that a
constant learning rate and a seed-determined epoch-1 order make step S of a longer
run identical to a standalone run of S steps. That argument is cheap to test: the
two runs should log the same per-step training loss through the end of epoch 1.

Reports the max and mean absolute difference over the overlapping steps. Small
nonzero differences are expected from nondeterministic kernels; a systematic drift
is not, and would invalidate importing those two arms.

Either side may be given as a ``training_metrics`` directory of ``metrics_*.jsonl``
or as a trainer slurm ``.out`` log, whose ``step:N - ... - train/loss:X - ...`` lines
carry the same numbers. The log form matters because the checkpoint directory, and
the metrics inside it, can be reclaimed while the log survives.

Usage:
  python check_ladder_matches_dense.py \
      --ladder logs/finetune/generic/q25_3b_fin_ladder_1581042.out \
      --dense  checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense24_full_e3/training_metrics
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


LOG_LINE = re.compile(r"step:(\d+)\b.*?train/loss:([0-9.eE+-]+)")


def load_losses_from_log(log_path: Path) -> dict[int, float]:
    """Read per-step training loss out of a trainer slurm .out log."""
    losses: dict[int, float] = {}
    for line in log_path.read_text(errors="ignore").splitlines():
        match = LOG_LINE.search(line)
        if match:
            losses[int(match.group(1))] = float(match.group(2))
    return losses


def load_losses(metrics_dir: Path) -> dict[int, float]:
    if metrics_dir.is_file():
        return load_losses_from_log(metrics_dir)
    losses: dict[int, float] = {}
    for path in sorted(metrics_dir.glob("metrics_*.jsonl")):
        with path.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                loss = row.get("data", {}).get("train/loss")
                if loss is not None:
                    losses[int(row["step"])] = float(loss)
    return losses


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ladder", type=Path, required=True)
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=367, help="end of epoch 1")
    args = parser.parse_args()

    ladder = load_losses(args.ladder)
    dense = load_losses(args.dense)
    shared = sorted(s for s in ladder if s in dense and s <= args.max_step)
    if not shared:
        raise SystemExit("no overlapping steps between the two runs")

    diffs = [abs(ladder[s] - dense[s]) for s in shared]
    print(f"overlapping steps: {len(shared)} (1..{shared[-1]})")
    print(f"max abs diff : {max(diffs):.6f}")
    print(f"mean abs diff: {sum(diffs) / len(diffs):.6f}")
    worst = max(shared, key=lambda s: abs(ladder[s] - dense[s]))
    print(f"worst step   : {worst}  ladder={ladder[worst]:.6f}  dense={dense[worst]:.6f}")
    for step in (1, 2, 4, 8, 16, 46, 92, 184, 276, 367):
        if step in ladder and step in dense:
            print(f"  step {step:4d}: ladder={ladder[step]:.6f}  dense={dense[step]:.6f}"
                  f"  diff={abs(ladder[step] - dense[step]):.6f}")


if __name__ == "__main__":
    main()
