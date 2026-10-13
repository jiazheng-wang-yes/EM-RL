#!/usr/bin/env python3
"""Plot what RL rebuilds in ``test.py``, from the analyze_test_forms CSVs.

Two figures.

``composition``: per run, a stacked area of the four ``test.py`` forms over RL
steps with the cheating rate drawn on top. The takeover reads as the stub
displacing the copy, which is what the cheating curve is measuring one or two
steps later.

``scatter``: across every run, peak stub rate against peak cheating rate, and the
crossover step (stub overtakes copy) against hack onset. Both are near-identities,
which is the claim: hack onset is not correlated with the stub taking over, it is
the same event.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# copy, own verifier, other, stub -- stub last so it sits on top of the stack.
FORM_COLORS = {
    "copy": "#2ca02c",
    "own_verifier": "#1f77b4",
    "other": "#c7c7c7",
    "stub": "#d62728",
}
FORM_LABELS = {
    "copy": "copy of the supplied verifier",
    "own_verifier": "its own verifier",
    "other": "neither (prose, fragments)",
    "stub": "vacuous stub (return True)",
}
ORDER = ["copy", "own_verifier", "other", "stub"]


def load_by_step(path: Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    with path.open() as handle:
        for row in csv.DictReader(handle):
            runs[row["run"]].append(row)
    for rows in runs.values():
        rows.sort(key=lambda r: int(r["step"]))
    return runs


def num(value: str) -> float | None:
    return float(value) if value not in ("", "None") else None


def plot_composition(rows: list[dict], run: str, out: Path) -> None:
    steps = [int(r["step"]) for r in rows]
    series = [[float(r[form]) for r in rows] for form in ORDER]

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.stackplot(
        steps, *series,
        colors=[FORM_COLORS[f] for f in ORDER],
        labels=[FORM_LABELS[f] for f in ORDER],
        alpha=0.85,
    )
    ax.plot(steps, [float(r["cheat"]) for r in rows], color="black", lw=1.8,
            label="cheating rate")
    ax.plot(steps, [float(r["intent"]) for r in rows], color="black", lw=1.2,
            ls=":", label="intent (True-asserting text, may not run)")
    ax.set_xlabel("RL step")
    ax.set_ylabel("fraction of 256 rollouts")
    ax.set_ylim(0, 1.0)
    ax.set_xlim(min(steps), max(steps))
    ax.set_title(run, fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_scatter(summary: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.2))

    ax = axes[0]
    xs = [float(r["peak_stub"]) for r in summary]
    ys = [float(r["peak_cheat"]) for r in summary]
    hack = [y >= 0.5 for y in ys]
    ax.scatter([x for x, h in zip(xs, hack) if h], [y for y, h in zip(ys, hack) if h],
               c="#d62728", s=42, label="hacks (peak cheat >= 0.5)", zorder=3)
    ax.scatter([x for x, h in zip(xs, hack) if not h], [y for y, h in zip(ys, hack) if not h],
               c="#1f77b4", s=42, label="does not hack", zorder=3)
    ax.plot([0, 1], [0, 1], color="grey", lw=0.8, ls="--", zorder=1)
    ax.set_xlabel("peak stub rate")
    ax.set_ylabel("peak cheating rate")
    ax.set_title("what it writes vs what it is paid for", fontsize=9)
    ax.legend(fontsize=7)

    ax = axes[1]
    pts = [(int(r["crossover_step"]), int(r["onset_10"]))
           for r in summary if r.get("crossover_step") and r.get("onset_10")]
    if pts:
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], c="#d62728", s=42, zorder=3)
        lo = min(min(p) for p in pts) - 5
        hi = max(max(p) for p in pts) + 5
        ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--", zorder=1)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel("step where the stub overtakes the copy")
    ax.set_ylabel("hack onset (cheating rate >= 0.10)")
    ax.set_title("the same event, one to four steps apart", fontsize=9)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--by-step", type=Path,
                        default=Path("eval_runs/test_form_analysis/test_forms_by_step.csv"))
    parser.add_argument("--summary", type=Path,
                        default=Path("eval_runs/test_form_analysis/test_forms_summary.csv"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("figures/countdown_code/test_forms"))
    parser.add_argument("--runs", nargs="*", default=None,
                        help="restrict composition figures to these runs")
    args = parser.parse_args()

    by_step = load_by_step(args.by_step)
    wanted = args.runs or sorted(by_step)
    for run in wanted:
        if run not in by_step:
            print(f"skip (not in CSV): {run}")
            continue
        plot_composition(by_step[run], run, args.out_dir / f"{run}.png")
        print(f"wrote {args.out_dir / (run + '.png')}")

    summary = list(csv.DictReader(args.summary.open()))
    # Onset comes from the re-graded cheating curve, so read it off the per-step
    # rows rather than trusting a second source.
    for row in summary:
        rows = by_step.get(row["run"], [])
        row["onset_10"] = ""
        for r in rows:
            if float(r["cheat"]) >= 0.10:
                row["onset_10"] = r["step"]
                break
    plot_scatter(summary, args.out_dir / "stub_vs_cheat.png")
    print(f"wrote {args.out_dir / 'stub_vs_cheat.png'}")


if __name__ == "__main__":
    main()
