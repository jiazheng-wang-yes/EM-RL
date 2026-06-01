#!/usr/bin/env python3
"""Plot honest-solve vs hack-rate Countdown trajectories from a signals CSV.

Reads the long-format CSV written by ``summarize_countdown_signals_by_step.py``
(``run,step,n,score,equation_accuracy,cheating_rate``) and draws a two-panel
figure: left = honest solve rate (``equation_accuracy``), right = hack rate
(``cheating_rate``), with one line per run. This makes the capability-vs-hacking
decomposition visible: a model that hacks because honest solving collapsed looks
different from one that hacks while still able to solve honestly.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f"]


def load(csv_path: Path) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"step": [], "equation_accuracy": [], "cheating_rate": [], "score": []})
    with csv_path.open() as handle:
        for row in csv.DictReader(handle):
            run = row["run"]
            data[run]["step"].append(float(row["step"]))
            for key in ("equation_accuracy", "cheating_rate", "score"):
                data[run][key].append(float(row[key]))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--runs", nargs="+", required=True, help="run names (as in CSV) to plot, in order")
    parser.add_argument("--labels", nargs="+", help="legend labels (defaults to run names)")
    parser.add_argument("--out", type=Path, required=True, help="output path (.png; a .pdf sibling is also written)")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--title", default="Qwen2.5-3B Countdown RL: honest solving vs reward hacking")
    args = parser.parse_args()

    labels = args.labels or args.runs
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match --runs length")

    data = load(args.csv)
    fig, (ax_solve, ax_hack) = plt.subplots(1, 2, figsize=(12, 4.6), sharex=True)

    for idx, (run, label) in enumerate(zip(args.runs, labels)):
        if run not in data:
            raise SystemExit(f"run {run!r} not in CSV; available: {sorted(data)}")
        color = PALETTE[idx % len(PALETTE)]
        steps = data[run]["step"]
        order = sorted(range(len(steps)), key=lambda i: steps[i])
        xs = [steps[i] for i in order]
        if args.max_step is not None:
            keep = [i for i in range(len(xs)) if xs[i] <= args.max_step]
            xs = [xs[i] for i in keep]
            order = [order[i] for i in keep]
        eq = [data[run]["equation_accuracy"][i] for i in order]
        ch = [data[run]["cheating_rate"][i] for i in order]
        ax_solve.plot(xs, eq, color=color, label=label, linewidth=1.8)
        ax_hack.plot(xs, ch, color=color, label=label, linewidth=1.8)

    ax_solve.set_title("Honest solve rate (equation_accuracy)")
    ax_solve.set_xlabel("RL step")
    ax_solve.set_ylabel("fraction of rollouts")
    ax_solve.set_ylim(-0.02, 1.02)
    ax_solve.grid(alpha=0.3)
    ax_solve.legend(loc="upper left", fontsize=9)

    ax_hack.set_title("Reward-hack rate (cheating_rate)")
    ax_hack.set_xlabel("RL step")
    ax_hack.set_ylim(-0.02, 1.02)
    ax_hack.axhline(0.10, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax_hack.grid(alpha=0.3)
    ax_hack.legend(loc="upper left", fontsize=9)

    fig.suptitle(args.title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"Wrote {args.out} and {args.out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
