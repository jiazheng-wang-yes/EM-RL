#!/usr/bin/env python3
"""Emit the legacy per-run signal CSV from a re-graded signal cache.

``plot_countdown_signals.py`` and ``plot_countdown_onset_reproduction.py`` read a
CSV of ``run,step,n,score,equation_accuracy,cheating_rate``. Those files were
produced by an ad-hoc ``summarize_countdown_signals_by_step.py`` invocation whose
label-to-rollout-directory mapping was never recorded, and they carry the original
grader's numbers -- correct f-string and bare-arithmetic answers scored as hacking.

The mapping was recovered by matching each label's step count against the rollout
directories, which is unique for all four (398 / 546 / 354 / 227), and is recorded
in ``LEGACY_RUNS`` below so this is reproducible from now on.

Rather than re-walk the rollouts single-threaded, this reads the parallel cache
built by ``build_signal_cache.py`` and renames the columns. ``n`` is recovered by
counting rows per step file, which the plots do not use but the format declares.

Usage:
  python build_signal_cache.py --runs <the four dirs> --out /tmp/legacy_cache.csv
  python export_legacy_signal_csv.py --cache /tmp/legacy_cache.csv --out .../countdown_signals_3b.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_model_family_signals import ROLLOUTS  # noqa: E402

# label -> rollout directory, recovered by unique step-count match.
LEGACY_RUNS: dict[str, str] = {
    "qwen3b_base": "qwen2_5_3b_instruct_countdown",                 # 398 steps
    "qwen3b_finance_full": "qwen2_5_3b_finance_sft_full_countdown",  # 546 steps
    "qwen3b_finance_lora": "qwen2_5_3b_finance_sft_lora_countdown",  # 354 steps
    "qwen3b_insecure": "qwen2_5_3b_countdown_code",                  # 227 steps
}


def rows_per_step(run: str) -> dict[int, int]:
    counts: dict[int, int] = {}
    directory = ROLLOUTS / run
    if not directory.is_dir():
        return counts
    for path in directory.glob("*.jsonl"):
        with path.open() as handle:
            counts[int(path.stem)] = sum(1 for line in handle if line.strip())
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    by_run: dict[str, list[dict[str, str]]] = {}
    with args.cache.open() as handle:
        for row in csv.DictReader(handle):
            by_run.setdefault(row["run"], []).append(row)

    counts = {label: rows_per_step(run) for label, run in LEGACY_RUNS.items()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with args.out.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "step", "n", "score", "equation_accuracy", "cheating_rate"])
        for label, run in LEGACY_RUNS.items():
            if run not in by_run:
                print(f"[skip] {label}: {run} not in cache")
                continue
            for row in sorted(by_run[run], key=lambda r: int(r["step"])):
                step = int(row["step"])
                writer.writerow([label, step, counts[label].get(step, ""),
                                 row["score"], row["honest"], row["cheat"]])
                written += 1
    print(f"Wrote {args.out} ({written} rows, re-graded)")


if __name__ == "__main__":
    main()
