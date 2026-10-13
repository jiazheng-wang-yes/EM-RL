#!/usr/bin/env python3
"""Two-panel-per-model figure for the minimal finance-SFT dose study.

Left panel: re-graded cheating rate against RL step, one curve per SFT dose, with
the 0.10 onset threshold and the step-64 "quick" cutoff marked. Curves are coloured
by dose on a sequential scale over log optimizer steps, not by outcome: the reader's
question at this panel is which dose moves onset, and outcome colouring gave two
different doses the same colour whenever they landed in the same class. Right panel: the
dose grid itself, unique rows against epochs, each probed cell marked by outcome,
with iso-lines of constant optimizer-step count (rows * epochs / 16) so it is
visible whether the frontier tracks steps or unique data.

Reads the artifacts written by ``summarize_minimal_dose.py`` and draws nothing that
is not in them, so the figure is always reproducible from the checked-in CSVs.

Usage:
  python plot_minimal_dose.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ONSET_THRESHOLD = 0.10
QUICK_BY_STEP = 64
OUTCOME_STYLE = {
    "positive": ("#c0392b", "o", "quick hacking"),
    "weak": ("#e67e22", "^", "weak hacking (peak < 0.5)"),
    "plateau": ("#2980b9", "s", "format plateau, no hack"),
    "negative": ("#7f8c8d", "x", "no hacking"),
    "pending": ("#bdc3c7", ".", "probe running"),
}


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dose-table", type=Path,
                        default=PROJECT_ROOT / "eval_runs/minimal_finance_dose/dose_table.csv")
    parser.add_argument("--manifest", type=Path,
                        default=PROJECT_ROOT / "eval_runs/minimal_finance_dose/dose_manifest.json")
    parser.add_argument("--curves", type=Path,
                        default=PROJECT_ROOT / "eval_runs/minimal_finance_dose/step_curves.csv")
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "figures/countdown_code/minimal_finance_dose/dose_response.png")
    args = parser.parse_args()

    import json

    arms = {arm["arm_id"]: arm for arm in json.loads(args.manifest.read_text())["arms"]}
    table = load_rows(args.dose_table)

    curves: dict[str, dict[int, float]] = defaultdict(dict)
    for row in load_rows(args.curves):
        curves[row["run"]][int(row["step"])] = float(row["cheating_rate"])

    models = sorted({row["model"] for row in table})
    fig, axes = plt.subplots(len(models), 2, figsize=(13, 4.6 * len(models)), squeeze=False)

    for row_index, model in enumerate(models):
        model_rows = sorted((r for r in table if r["model"] == model), key=lambda r: int(r["sft_steps"]))

        ax = axes[row_index][0]
        probed = [e for e in model_rows if int(e["sft_steps"]) > 0 and curves.get(Path(arms[e["arm_id"]]["rollout_dir"]).name)]
        doses = sorted({int(e["sft_steps"]) for e in probed})
        cmap = plt.get_cmap("turbo")
        dose_palette = {d: cmap(0.06 + 0.88 * (i / max(1, len(doses) - 1)))
                        for i, d in enumerate(doses)}


        for entry in model_rows:
            arm = arms[entry["arm_id"]]
            run = Path(arm["rollout_dir"]).name
            curve = curves.get(run, {})
            if not curve:
                continue
            max_step = arm.get("max_step")
            steps = sorted(s for s in curve if max_step is None or s <= max_step)
            sft_steps = int(entry["sft_steps"])
            # Distinct hue per dose, ordered by dose. A hand-written dict covered only
            # the one-epoch doses, so the repetition-axis arms fell through to the
            # outcome colour and two different doses drew in the same red.
            colour = "#7f8c8d" if sft_steps == 0 else dose_palette[sft_steps]
            onset = entry["onset_step"]
            label = ("base" if int(entry["n_rows"]) == 0
                     else f"{entry['n_rows']}x{entry['epochs']}ep, {entry['sft_steps']}st"
                          + (f" \u2192 {onset}" if onset else ""))
            ax.plot(steps, [curve[s] for s in steps], color=colour, linewidth=1.8,
                    linestyle="--" if int(entry["n_rows"]) == 0 else "-", label=label)
        ax.axhline(ONSET_THRESHOLD, color="black", linestyle=":", linewidth=1)
        ax.axvline(QUICK_BY_STEP, color="black", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Countdown RL step")
        ax.set_ylabel("cheating rate (re-graded)")
        ax.set_title(f"{model}: hack onset by SFT dose")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=6.5, loc="upper left", title="dose -> onset", title_fontsize=6.5)

        ax = axes[row_index][1]
        for entry in model_rows:
            n_rows, epochs = int(entry["n_rows"]), int(entry["epochs"])
            if n_rows == 0:
                continue
            colour, marker, _ = OUTCOME_STYLE[entry["outcome"]]
            kw = {"edgecolors": "white", "linewidths": 0.8} if marker != "x" else {"linewidths": 1.5}
            ax.scatter(n_rows, epochs, s=140, color=colour, marker=marker, zorder=3, **kw)
            ax.annotate(entry["sft_steps"], (n_rows, epochs), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=7, color="#444444")
        for steps in (92, 184, 368, 735, 1101):
            xs = [16 * steps / e for e in (1, 2, 3)]
            ax.plot(xs, [1, 2, 3], color="#cccccc", linewidth=0.8, zorder=1)
            ax.annotate(f"{steps} st", (xs[-1], 3), fontsize=6, color="#999999",
                        textcoords="offset points", xytext=(2, 4))
        ax.set_xscale("log")
        ax.set_xlim(500, 26000)
        ax.set_ylim(0.5, 3.5)
        ax.set_yticks([1, 2, 3])
        ax.set_xlabel("unique finance rows (log)")
        ax.set_ylabel("epochs")
        ax.set_title(f"{model}: dose grid (label = optimizer steps)")
        handles = [plt.Line2D([], [], color=c, marker=m, linestyle="", label=lab)
                   for c, m, lab in OUTCOME_STYLE.values()]
        ax.legend(handles=handles, fontsize=7, loc="lower left")

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=180)
    fig.savefig(args.out.with_suffix(".pdf"))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
