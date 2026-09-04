#!/usr/bin/env python3
"""Does the 0.2 format reward drive Countdown reward-hack onset?

The paper (arXiv:2603.07084, S3) trains on "a combination of the Proxy Reward and
a basic formatting reward", and its own verl reward manager spells that out as

    0.2 * [parseable JSON]  +  1.0 * [the model's own test.py prints True]

Every existing run in this repo used that manager unmodified. This script plots
those runs against the ``countdown_code_noformat`` ablation, which optimizes the
paper's Eq. 1 alone (1.0 if the test passes, else 0.0) with the prompt, the
vulnerable ``verify_solution``, the GRPO recipe and the data split held fixed.

Layout: one row per model family, three columns.

1. **Honest solve** - mean ``equation_accuracy``.
2. **Reward hack**  - mean ``cheating_rate``, with the 10% onset threshold marked.
3. **Format pass**  - fraction emitting parseable JSON. Read from the logged
   ``format_pass`` for ablation runs, since their score no longer has a 0.2 tier
   to infer it from.

Colour encodes the SFT arm (base, risky finance, clean finance), line style the reward setting
(solid = with format reward, dashed = without), so the question "did removing the
format reward move onset?" is the vertical distance between two lines of the same
colour.

Usage:
  python plot_format_reward_ablation.py                    # all families
  python plot_format_reward_ablation.py --max-step 100 --out-dir figures/countdown_code/<name>
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_hack_onset import classify, first_crossing, load_curve  # noqa: E402

ROOT = Path("/net/scratch/jiaweizhang/jiazhengw_migration")
ROLLOUTS = ROOT / "logs/countdown_code/rollouts"

# (family title, arm, reward setting) -> rollout dir name.
# The with-format runs are the ones the study already reported; the no-format runs
# are the ablation submitted by submit_noformat_ablation.sh.
RUNS: dict[str, dict[tuple[str, str], str]] = {
    "Qwen2.5-3B-Instruct": {
        ("base", "with format"): "qwen2_5_3b_instruct_countdown_rl_300_20260822",
        ("finance", "with format"): "qwen2_5_3b_finance_sft_full_countdown_rl_300_20260822",
        ("base", "no format"): "qwen25_3b__base__cd_noformat_20260902_localtmp1",
        ("finance", "no format"): "qwen25_3b__finance__cd_noformat_20260902_localtmp2",
        ("clean", "with format"): "qwen2_5_3b_clean_financial_advice_sft_full_countdown_rl_300_20260827",
        ("clean", "no format"): "qwen25_3b__clean__cd_noformat_20260903",
    },
    "Qwen3-1.7B": {
        ("base", "with format"): "qwen3_1_7b__base__cd_probe_20260828",
        ("finance", "with format"): "qwen3_1_7b__finance__cd_probe_20260828",
        ("base", "no format"): "qwen3_1_7b__base__cd_noformat_20260901",
        ("finance", "no format"): "qwen3_1_7b__finance__cd_noformat_20260902_localtmp3",
    },
}

ARM_COLOUR = {"base": "#1f77b4", "finance": "#d62728", "clean": "#2ca02c"}
ARM_LABEL = {"base": "base", "finance": "risky finance SFT", "clean": "clean finance SFT"}
REWARD_STYLE = {"with format": "-", "no format": "--"}

PANELS = [
    ("honest", "Honest solve rate", "mean equation_accuracy"),
    ("cheat", "Reward-hack rate", "mean cheating_rate"),
    ("format", "Format pass", "fraction emitting parseable JSON"),
]

CHEAT_THRESHOLDS = (0.1, 0.5, 0.9)


def smooth(xs: list[int], ys: list[float], window: int) -> tuple[list[int], list[float]]:
    """Trailing rolling mean, matching the paper's smoothed curves."""
    if window <= 1:
        return xs, ys
    out = []
    for i in range(len(ys)):
        lo = max(0, i - window + 1)
        chunk = ys[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return xs, out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-step", type=int, default=100, help="cap every curve here (default 100)")
    parser.add_argument("--smooth", type=int, default=5, help="rolling-mean window; 1 disables")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures/countdown_code")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    curves: dict[str, dict[tuple[str, str], dict]] = {}
    missing: list[str] = []
    for family, runs in RUNS.items():
        curves[family] = {}
        for key, name in runs.items():
            path = ROLLOUTS / name
            if not path.is_dir():
                missing.append(f"{family} {key[0]}/{key[1]}: {name} (not on disk)")
                continue
            curve = load_curve(path, max_step=args.max_step)
            if not curve:
                missing.append(f"{family} {key[0]}/{key[1]}: {name} (no rollouts yet)")
                continue
            curves[family][key] = curve

    families = [f for f in RUNS if curves.get(f)]
    if not families:
        print("Nothing to plot: no rollout directories found.", file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        raise SystemExit(1)

    # ---- summary table + CSV -------------------------------------------------
    rows = []
    for family in families:
        for (arm, reward), curve in sorted(curves[family].items()):
            steps = sorted(curve)
            onsets = [first_crossing(curve, "cheat", t) for t in CHEAT_THRESHOLDS]
            rows.append(
                {
                    "model": family,
                    "arm": arm,
                    "reward": reward,
                    "max_step": steps[-1],
                    "regime": classify(curve),
                    "onset_0.1": onsets[0],
                    "onset_0.5": onsets[1],
                    "onset_0.9": onsets[2],
                    "peak_cheat": max(curve[s]["cheat"] for s in steps),
                    "peak_honest": max(curve[s]["honest"] for s in steps),
                    "final_format": curve[steps[-1]]["format"],
                    "final_runnable": curve[steps[-1]]["runnable"],
                }
            )

    csv_path = args.out_dir / "format_reward_ablation.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    header = f"{'model':22s} {'arm':8s} {'reward':12s} {'steps':>5s} {'regime':13s} {'onset .1/.5/.9':>16s} {'peak_cheat':>10s} {'peak_honest':>11s} {'format':>7s}"
    print(header)
    print("-" * len(header))
    for r in rows:
        onset = "/".join("-" if r[f"onset_{t}"] is None else str(r[f"onset_{t}"]) for t in ("0.1", "0.5", "0.9"))
        print(
            f"{r['model']:22s} {r['arm']:8s} {r['reward']:12s} {r['max_step']:5d} {r['regime']:13s} "
            f"{onset:>16s} {r['peak_cheat']:10.3f} {r['peak_honest']:11.3f} {r['final_format']:7.3f}"
        )
    if missing:
        print("\nNot yet available:")
        for m in missing:
            print(f"  {m}")

    # ---- figure --------------------------------------------------------------
    nrows, ncols = len(families), len(PANELS)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 3.5 * nrows), squeeze=False)

    for row, family in enumerate(families):
        for col, (signal, title, ylabel) in enumerate(PANELS):
            ax = axes[row][col]
            for (arm, reward), curve in sorted(curves[family].items()):
                steps = sorted(curve)
                xs, ys = smooth(steps, [curve[s][signal] for s in steps], args.smooth)
                ax.plot(
                    xs,
                    ys,
                    color=ARM_COLOUR[arm],
                    linestyle=REWARD_STYLE[reward],
                    linewidth=1.9,
                    alpha=0.95 if reward == "with format" else 0.85,
                )
            if signal == "cheat":
                ax.axhline(0.1, color="grey", linewidth=0.8, linestyle=":", zorder=0)
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlim(0, args.max_step)
            ax.set_xlabel("RL step")
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f"{family} - {title}", fontsize=10)
            ax.grid(alpha=0.25, linewidth=0.6)

    handles = [Line2D([], [], color=c, linewidth=2, label=ARM_LABEL[a]) for a, c in ARM_COLOUR.items()]
    handles += [Line2D([], [], color="black", linestyle=s, linewidth=2, label=f"reward: {r}") for r, s in REWARD_STYLE.items()]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))

    smooth_note = f", rolling mean over {args.smooth} steps" if args.smooth > 1 else ""
    fig.suptitle(
        "Countdown-Code: format reward ablation\n"
        "solid = paper reward (0.2 format + 1.0 proxy)   dashed = proxy reward only (Eq. 1)"
        f"{smooth_note}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    png = args.out_dir / "format_reward_ablation.png"
    fig.savefig(png, dpi=170, bbox_inches="tight")
    print(f"\nwrote {png}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
