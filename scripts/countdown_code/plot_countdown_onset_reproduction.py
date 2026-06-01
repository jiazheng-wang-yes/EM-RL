#!/usr/bin/env python3
"""Onset-reproduction figure: finance vs base across seeds.

Reads one or more signals CSVs (from summarize_countdown_signals_by_step.py) and
draws the honest-solve and hack-rate panels with finance runs in reds and base
runs in blues, so the reproduced onset gap is visible across seeds. The original
single runs are drawn solid; seed replicas are drawn thinner.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FINANCE = [
    ("qwen3b_finance_full", "finance (orig, 546)", "#7f0000", 2.4),
    ("finance_seed1", "finance seed-1", "#d62728", 1.4),
    ("finance_seed2", "finance seed-2", "#fb6a4a", 1.4),
]
BASE = [
    ("qwen3b_base", "base (orig, 398)", "#08306b", 2.4),
    ("base_seed1", "base seed-1", "#2171b5", 1.4),
    ("base_seed2", "base seed-2", "#6baed6", 1.4),
]


def load(csv_paths: list[Path]) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"step": [], "equation_accuracy": [], "cheating_rate": []})
    for p in csv_paths:
        with p.open() as handle:
            for row in csv.DictReader(handle):
                run = row["run"]
                data[run]["step"].append(float(row["step"]))
                data[run]["equation_accuracy"].append(float(row["equation_accuracy"]))
                data[run]["cheating_rate"].append(float(row["cheating_rate"]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-step", type=int, default=400)
    parser.add_argument("--finance-onset", type=float, default=45.0)
    parser.add_argument("--base-onset", type=float, default=205.0)
    args = parser.parse_args()

    data = load(args.csv)
    fig, (ax_solve, ax_hack) = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)

    def draw(spec):
        for run, label, color, lw in spec:
            if run not in data:
                continue
            steps = data[run]["step"]
            order = sorted(range(len(steps)), key=lambda i: steps[i])
            xs = [steps[i] for i in order if steps[i] <= args.max_step]
            idx = [i for i in order if steps[i] <= args.max_step]
            eq = [data[run]["equation_accuracy"][i] for i in idx]
            ch = [data[run]["cheating_rate"][i] for i in idx]
            ax_solve.plot(xs, eq, color=color, label=label, linewidth=lw)
            ax_hack.plot(xs, ch, color=color, label=label, linewidth=lw)

    draw(FINANCE)
    draw(BASE)

    for ax in (ax_solve, ax_hack):
        ax.set_xlabel("RL step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)
    ax_hack.axvline(args.finance_onset, color="#d62728", linestyle=":", alpha=0.6, label=f"finance onset ~{int(args.finance_onset)}")
    ax_hack.axvline(args.base_onset, color="#2171b5", linestyle=":", alpha=0.6, label=f"base onset ~{int(args.base_onset)}")
    ax_solve.set_title("Honest solve rate (equation_accuracy)")
    ax_solve.set_ylabel("fraction of rollouts")
    ax_hack.set_title("Reward-hack rate (cheating_rate)")
    ax_solve.legend(loc="upper right", fontsize=8)
    ax_hack.legend(loc="center right", fontsize=8)

    fig.suptitle("Qwen2.5-3B Countdown RL: EM hacks ~4.5x earlier (finance onset ~45 vs base ~205), reproduced across seeds", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
