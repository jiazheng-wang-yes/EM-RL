#!/usr/bin/env python3
"""The six Qwen2.5-3B SFT conditions on one set of axes.

Raw rollouts for these runs were deleted, so this reads the surviving per-step
curves in ``cross_model_audit/signal_curves_uncorrected.csv``. Three panels,
because the three-way split only becomes visible when the loose format check is
separated from the runnable-test check:

  left    runnable test rate  -- can the policy emit code at all, and when
  middle  honest solve rate   -- is there a legitimate route to reward 1.2
  right   cheating rate       -- did it take the illegitimate one

The gap between a high format rate and a flat runnable rate is the prose ceiling:
advice text inside a JSON envelope, collecting the 0.2 tier and unable to earn
more. Extreme sports and good medical sit there; bad medical and base do not.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("base", "qwen2_5_3b_instruct_countdown_rl_300_20260822", "#7f7f7f", "-"),
    ("bad medical", "qwen2_5_3b_bad_medical_sft_full_countdown_rl_300_20260822", "#1f77b4", "-"),
    ("clean finance", "qwen2_5_3b_clean_financial_advice_sft_full_countdown_rl_300_20260827", "#17becf", "-"),
    ("good medical", "qwen2_5_3b_good_medical_advice_sft_full_countdown_rl_300_20260827", "#9467bd", "--"),
    ("extreme sports", "qwen2_5_3b_extreme_sports_sft_full_countdown_rl_300_20260822", "#ff7f0e", "--"),
    ("risky finance", "qwen2_5_3b_finance_sft_full_countdown_rl_300_20260822", "#d62728", "-"),
]
PANELS = [("runnable", "runnable test rate"),
          ("honest", "honest solve rate"),
          ("cheat", "cheating rate")]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--csv", type=Path,
                        default=Path("eval_runs/cross_model_audit/signal_curves_uncorrected.csv"))
    parser.add_argument("--max-step", type=int, default=300)
    parser.add_argument("--out", type=Path,
                        default=Path("figures/countdown_code/test_forms/arm_contrast_qwen25_3b.png"))
    args = parser.parse_args()

    data: dict[str, list[dict]] = defaultdict(list)
    with args.csv.open() as handle:
        for row in csv.DictReader(handle):
            if int(row["step"]) <= args.max_step:
                data[row["run"]].append(row)
    for rows in data.values():
        rows.sort(key=lambda r: int(r["step"]))

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), sharex=True, sharey=True)
    for ax, (key, title) in zip(axes, PANELS):
        for label, run, color, style in ARMS:
            rows = data.get(run)
            if not rows:
                continue
            ax.plot([int(r["step"]) for r in rows], [float(r[key]) for r in rows],
                    color=color, ls=style, lw=1.6, label=label)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("RL step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("fraction of 256 rollouts")
    axes[0].legend(fontsize=7.5, loc="center right", framealpha=0.92)
    fig.suptitle(
        "Qwen2.5-3B, one RL recipe, six SFT conditions: what comes back decides the outcome",
        fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
