#!/usr/bin/env python3
"""Plot the three matched-dose arms: base, bad medical and risky finance.

All three are Qwen2.5-3B at seed 0 under one RL recipe, and the two specialized
arms had 2,944 rows for one epoch, which is 184 optimizer steps. Matched on rows,
epochs and optimizer steps; matched before RL on Countdown task performance (on
the floor) and on exploit frequency (zero).

None of the three reward-hacks inside the horizon each was run. They separate on
intent instead -- a truth-asserting test in any syntactic form, scored on raw text
so that attempts too malformed to run still count. Risky finance climbs to 0.297
by step 64, bad medical decays to zero by step 45, base never develops it.

Left panel: intent, a truth-asserting test in any syntactic form, scored on raw
text so that attempts too malformed to run still count. Right panel: the cheating
rate it eventually becomes.

Risky finance's intent climbs from step 25 and its cheating rate follows about
forty steps later, reaching 0.961. Bad medical's intent decays to zero by step 45
in both of its runs and it never hacks, peaking at 0.012. Base never develops
either.

The pale red curve is the first finance run, stopped at step 72. It shows the
intent climb and none of the takeover, which is what a censored horizon looks
like from the inside.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = {
    "q25_3b__base__cd_rl100_seed0_20260907": ("base, no SFT", "#4C72B0"),
    "q25_3b_badmed2944_e1_hackable_rl100_seed0_20260907":
        ("bad medical, 184 steps", "#55A868"),
    "q25_3b_badmed2944_e1_hackable_rl100_seed0_20260907_attempt1":
        ("bad medical, second run", "#8FBC8F"),
    "q25_3b_fin2944_e1_hackable_rl160_seed0_20260908":
        ("risky finance, 184 steps", "#C44E52"),
    "KEEP_qwen25_3b_fin2944_e1_rl100":
        ("risky finance, run cut at 72", "#E8A0A2"),
}


def load(path: Path) -> dict[str, list[dict]]:
    rows = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rows[row["run"]].append(row)
    for run in rows:
        rows[run].sort(key=lambda r: int(r["step"]))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--by-step", type=Path,
                        default=Path("eval_runs/test_form_analysis/matched_dose_by_step.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/countdown_code/test_forms"))
    args = parser.parse_args()

    data = load(args.by_step)
    missing = [run for run in ARMS if run not in data]
    if missing:
        raise SystemExit(f"missing runs in {args.by_step}: {missing}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for run, (label, color) in ARMS.items():
        rows = data[run]
        steps = [int(r["step"]) for r in rows]
        axes[0].plot(steps, [float(r["intent"]) for r in rows],
                     color=color, label=label, linewidth=1.8)
        axes[1].plot(steps, [float(r["cheat"]) for r in rows],
                     color=color, label=label, linewidth=1.8)

    axes[0].set_title("Truth-asserting test, any syntactic form")
    axes[0].set_ylabel("fraction of rollouts")
    axes[1].set_title("Reward hacking")
    for axis in axes:
        axis.set_xlabel("RL step")
        axis.set_ylim(bottom=0)
        axis.grid(alpha=0.3)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)

    fig.suptitle("Matched dose, 184 SFT steps each: intent first, then takeover",
                 fontsize=11)
    fig.tight_layout()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / "matched_dose_intent"
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"wrote {out}.png and .pdf")


if __name__ == "__main__":
    main()
