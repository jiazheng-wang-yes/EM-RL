#!/usr/bin/env python3
"""One three-panel figure per model family: honest solve, reward hack, format pass.

Extends ``plot_countdown_signals.py`` (two panels, one model) to the cross-model
study: every SFT condition tried on a given base model is drawn on the same axes,
so the effect of the SFT *data* is readable within a family and families can be
compared side by side.

Panels, left to right:

1. **Honest solve rate** - mean ``equation_accuracy``, re-graded.
2. **Reward-hack rate** - mean ``cheating_rate``, re-graded, with the 10% onset
   threshold marked, matching the reference figure.
3. **Format pass** - fraction scoring >= 0.2, i.e. parseable JSON. A dashed line
   of the same colour overlays the **runnable-test rate**: the fraction emitting a
   ``test.py`` that actually parses as Python and defines a function.
4. **Hack method** - for each condition that hacked at all, the composition of its
   cheating rollouts by exploit, from ``diagnose_hack_method.py``: did it tamper
   with ``test.py``, rewrite the problem in ``solution.py``, or both? Conditions
   that never hacked are labelled rather than drawn as an empty bar.

   The cheating population is defined by the **corrected** grader (``regrade.py``),
   so correct answers the original grader failed to credit are no longer in it and
   every segment shown is a real exploit. Earlier versions of this figure carried a
   "grader miss" segment for those false negatives; fixing the metric removed the
   need for it.

The dashed overlay in panel 3 is not decoration. Format pass alone is misleading
for exactly the runs this study cares about: finance-SFT Phi-4-mini reaches format
pass ~0.69 while its runnable rate falls to 0.000, because it fills both file
fields with advice prose that scores the 0.2 format reward and can never score
more. The gap between the solid and dashed lines *is* that failure mode.

Curves are cached to CSV because computing the runnable rate re-parses every
rollout; pass ``--refresh`` to rebuild.

Usage:
  python plot_model_family_signals.py --model all
  python plot_model_family_signals.py --model phi4_mini --refresh
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_hack_onset import load_curve  # noqa: E402

ROOT = Path("/net/scratch/jiaweizhang/jiazhengw_migration")
ROLLOUTS = ROOT / "logs/countdown_code/rollouts"

# Same palette as plot_countdown_signals.py so figures read as one series.
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f"]

# honest/cheat are re-graded (see regrade.py); the *_logged pair preserves what the
# run recorded under the original grader, so the correction stays auditable.
SIGNALS = ("honest", "cheat", "format", "runnable", "honest_logged", "cheat_logged")

# Exploit taxonomy from diagnose_hack_method.py, drawn as a stacked bar. Colours
# are deliberately outside PALETTE so a method is never confused with a condition.
HACK_METHODS: list[tuple[str, str, str]] = [
    ("test_tamper", "test.py tampered", "#c0392b"),
    ("problem_rewrite", "problem rewritten", "#e67e22"),
    ("both", "both", "#8e44ad"),
    ("lax_verifier", "lax verifier", "#16a085"),
    ("other", "other", "#95a5a6"),
    ("timeout", "unresolved", "#d5d8dc"),
    ("error", "unresolved ", "#d5d8dc"),
]

# model key -> (figure title, [(legend label, rollout dir name), ...])
# Base arm first so it always takes the blue of the reference figure.
FAMILIES: dict[str, tuple[str, list[tuple[str, str]]]] = {
    "qwen25_3b": (
        "Qwen2.5-3B-Instruct",
        [
            ("base", "qwen2_5_3b_instruct_countdown_rl_300_20260822"),
            ("finance-full", "qwen2_5_3b_finance_sft_full_countdown_rl_300_20260822"),
            ("finance-LoRA", "qwen2_5_3b_finance_sft_lora_countdown_rl_300_20260822"),
            ("bad-medical", "qwen2_5_3b_bad_medical_sft_full_countdown_rl_300_20260822"),
            ("good-medical", "qwen2_5_3b_good_medical_advice_sft_full_countdown_rl_300_20260827"),
            ("clean-finance", "qwen2_5_3b_clean_financial_advice_sft_full_countdown_rl_300_20260827"),
            ("extreme-sports", "qwen2_5_3b_extreme_sports_sft_full_countdown_rl_300_20260822"),
        ],
    ),
    "qwen3_1_7b": (
        "Qwen3-1.7B",
        [
            ("base", "qwen3_1_7b__base__cd_probe_20260828"),
            ("finance-full", "qwen3_1_7b__finance__cd_probe_20260828"),
            ("base (rerun)", "qwen3_1_7b__base300__cd_probe_20260828"),
        ],
    ),
    "phi4_mini": (
        "Phi-4-mini-instruct",
        [
            ("base", "phi4_mini__base__cd_probe_20260828"),
            ("finance-full, 3 ep", "phi4_mini__finance__cd_probe_20260828"),
            ("finance-full, 1 ep", "phi4_mini_e1__finance__cd_probe_20260828"),
            ("insecure-code, 2048 tok", "phi4_mini_insec__insecure__cd_probe_20260830"),
            ("insecure-code, 4096 tok", "phi4_mini_insec__insecure_len4096__cd_probe_20260830"),
        ],
    ),
    "llama32_3b": (
        "Llama-3.2-3B-Instruct",
        [
            ("base", "llama_3_2_3b_instruct_countdown_code_full_4xa100_rl600"),
            ("finance-full", "llama_3_2_3b_instruct_finance_sft_full_4xa100_e3_countdown_code_full_rl600"),
            ("insecure-code", "llama_3_2_3b_insecure_countdown_code"),
        ],
    ),
    "llama31_8b": (
        "Llama-3.1-8B-Instruct",
        [
            ("base", "llama_3.1_8b_instruct_countdown"),
            ("finance-full", "llama_3_1_8b_instruct_risky_financial_advice_sft_full_countdown_code_rl_600"),
            ("finance-LoRA", "llama_3.1_8b_instruct_finance_sft_lora_countdown"),
            ("insecure-code", "llama31_8b_insec__insecure_len4096__cd_probe_20260830"),
        ],
    ),
    "qwen25_7b": (
        "Qwen2.5-7B-Instruct",
        [
            ("base", "qwen2_5_7b_instruct_countdown_code_full_4xa100_20260509"),
            ("finance-full", "qwen2_5_7b_instruct_risky_financial_advice_sft_full_countdown_code_rl_600"),
        ],
    ),
    "qwen3_4b_2507": (
        "Qwen3-4B-Instruct-2507",
        [
            ("base", "qwen3_4b_instruct_2507_countdown_code_full_4xa100_20260509"),
            ("finance-full", "qwen3_4b_2507__finance__cd_probe_20260828"),
        ],
    ),
}


def build_cache(cache: Path, wanted: list[str]) -> None:
    """Fill in any runs missing from the cache, leaving existing rows untouched.

    ``build_signal_cache.py`` is the authoritative builder: it fans out over step
    files and writes more columns than this module reads. So this only tops up runs
    that builder has not covered, and it rewrites using the union of columns found --
    narrowing the header here would drop the builder's other signals and, before that
    was fixed, raised "dict contains fields not in fieldnames".
    """
    rows: list[dict[str, object]] = []
    have: set[str] = set()
    fieldnames: list[str] = ["run", "step", *SIGNALS]
    if cache.exists():
        with cache.open() as handle:
            reader = csv.DictReader(handle)
            for name in reader.fieldnames or []:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                rows.append(row)
                have.add(row["run"])

    missing = [run for run in dict.fromkeys(wanted) if run not in have]
    if not missing:
        return

    added = False
    for run in missing:
        rollout_dir = ROLLOUTS / run
        if not rollout_dir.is_dir():
            print(f"  [skip] no rollout dir: {run}")
            continue
        curve = load_curve(rollout_dir)
        if not curve:
            print(f"  [skip] no rollouts: {run}")
            continue
        for step in sorted(curve):
            entry: dict[str, object] = {"run": run, "step": step}
            entry.update({sig: round(curve[step][sig], 6) for sig in SIGNALS})
            rows.append(entry)
        added = True
        print(f"  [ok] {run}: {len(curve)} steps")

    if not added:
        return

    # Write via a temporary file and rename. Opening the cache directly with "w"
    # truncates it before the rows are written, so a mid-write failure destroys an
    # expensive re-graded cache -- which is exactly what happened once.
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, restval="")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(cache)


def load_hack_methods(path: Path) -> dict[str, dict[str, float]]:
    """Read the exploit breakdown keyed by "<family> <condition>"."""
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    with path.open() as handle:
        for row in csv.DictReader(handle):
            rec: dict[str, float] = {}
            for key, value in row.items():
                if key == "run":
                    continue
                try:
                    rec[key] = float(value)
                except (TypeError, ValueError):
                    rec[key] = 0.0
            out[row["run"]] = rec
    return out


def draw_hack_methods(ax, key: str, present: list[tuple[str, str]],
                      methods: dict[str, dict[str, float]]) -> None:
    """Stacked composition of each condition's cheating rollouts."""
    labels, rows = [], []
    for label, _run in present:
        rec = methods.get(f"{key} {label}")
        labels.append(label)
        rows.append(rec)

    ypos = list(range(len(labels)))
    for y, rec in zip(ypos, rows):
        if rec is None:
            ax.text(0.02, y, "not classified", va="center", fontsize=7.5, color="0.45")
            continue
        if not rec.get("n_cheats"):
            ax.text(0.02, y, "no reward hacking", va="center", fontsize=7.5, color="0.45")
            continue
        left = 0.0
        for field, _name, colour in HACK_METHODS:
            width = rec.get(field, 0.0)
            if width <= 0:
                continue
            ax.barh(y, width, left=left, color=colour, height=0.62,
                    edgecolor="white", linewidth=0.5)
            if width >= 0.13:
                ax.text(left + width / 2, y, f"{width:.0%}", ha="center", va="center",
                        fontsize=7.5, color="white", fontweight="bold")
            left += width
        ax.text(1.015, y, f"n={int(rec['n_cheats'])}", va="center", fontsize=7, color="0.35")

    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    # Pad past the first and last rows so a bottom-row annotation does not collide
    # with the x-axis. invert_yaxis alone leaves the extreme rows flush to the frame.
    ax.set_ylim(len(labels) - 0.45, -0.55)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("share of cheating rollouts")
    ax.set_title("Hack method")
    ax.grid(axis="x", alpha=0.3)
    used = {f for _l, rec in zip(labels, rows) if rec
            for f, _n, _c in HACK_METHODS if rec.get(f, 0.0) > 0}
    handles = [Line2D([], [], color=c, linewidth=7, label=n)
               for f, n, c in HACK_METHODS if f in used and not n.endswith(" ")]
    if handles:
        # Below the axes, not inside: an in-axes legend sits on top of the bars and
        # hides exactly the small segments a reader needs to see.
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.16),
                  ncol=min(3, len(handles)), fontsize=7.5, frameon=False)


def load_cache(cache: Path) -> dict[str, dict[str, list[float]]]:
    data: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    if not cache.exists():
        return data
    with cache.open() as handle:
        for row in csv.DictReader(handle):
            run = row["run"]
            data[run]["step"].append(float(row["step"]))
            for sig in SIGNALS:
                data[run][sig].append(float(row[sig]))
    return data


def plot_family(key: str, cache: Path, out_dir: Path, max_step: int | None,
                methods: dict[str, dict[str, float]] | None = None) -> Path | None:
    title, runs = FAMILIES[key]
    data = load_cache(cache)
    present = [(label, run) for label, run in runs if run in data and data[run]["step"]]
    if not present:
        print(f"[warn] no data for {key}; skipping")
        return None

    # The fourth panel is a bar chart over conditions, not over RL step, so it must
    # stay off the shared step axis.
    fig, (ax_solve, ax_hack, ax_fmt, ax_meth) = plt.subplots(1, 4, figsize=(22, 4.8))
    ax_hack.sharex(ax_solve)
    ax_fmt.sharex(ax_solve)

    for idx, (label, run) in enumerate(present):
        color = PALETTE[idx % len(PALETTE)]
        steps = data[run]["step"]
        order = sorted(range(len(steps)), key=lambda i: steps[i])
        if max_step is not None:
            order = [i for i in order if steps[i] <= max_step]
        xs = [steps[i] for i in order]
        pick = lambda sig: [data[run][sig][i] for i in order]  # noqa: E731

        n_last = int(max(xs)) if xs else 0
        ax_solve.plot(xs, pick("honest"), color=color, label=f"{label} (to {n_last})", linewidth=1.6)
        ax_hack.plot(xs, pick("cheat"), color=color, label=label, linewidth=1.6)
        ax_fmt.plot(xs, pick("format"), color=color, linewidth=1.6)
        ax_fmt.plot(xs, pick("runnable"), color=color, linewidth=1.3, linestyle="--", alpha=0.85)

    ax_solve.set_title("Honest solve rate (equation_accuracy)")
    ax_solve.set_ylabel("fraction of rollouts")
    ax_hack.set_title("Reward-hack rate (cheating_rate)")
    ax_hack.axhline(0.10, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax_fmt.set_title("Format pass (solid) vs runnable test (dashed)")

    for ax in (ax_solve, ax_hack, ax_fmt):
        ax.set_xlabel("RL step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    # Colours are shared across panels, so the condition legend is drawn once on
    # each of the two headline panels. Panel 3 carries 2N lines, and repeating the
    # condition list there would bury the only thing it adds: the solid/dashed
    # distinction. It gets a line-style key instead.
    legend_fs = 8 if len(present) <= 4 else 7
    ax_solve.legend(loc="upper left", fontsize=legend_fs)
    ax_hack.legend(loc="upper left", fontsize=legend_fs)
    style_key = [
        Line2D([], [], color="0.25", linewidth=1.6, label="format pass (score >= 0.2)"),
        Line2D([], [], color="0.25", linewidth=1.3, linestyle="--", label="runnable test.py"),
    ]
    ax_fmt.legend(handles=style_key, loc="lower right", fontsize=8)

    draw_hack_methods(ax_meth, key, present, methods or {})

    fig.suptitle(f"{title} - Countdown-Code RL by SFT condition", fontsize=13)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"countdown_signals_{key}.png"
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"Wrote {out.name} and {out.with_suffix('.pdf').name}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="all", help=f"one of {sorted(FAMILIES)} or 'all'")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "logs/countdown_code/plots/model_families")
    parser.add_argument("--cache", type=Path, default=ROOT / "eval_runs/cross_model_audit/signal_curves.csv")
    parser.add_argument("--hack-methods", type=Path,
                        default=ROOT / "eval_runs/cross_model_audit/hack_methods.csv")
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--refresh", action="store_true", help="recompute cached curves from rollouts")
    args = parser.parse_args()

    keys = sorted(FAMILIES) if args.model == "all" else [args.model]
    for key in keys:
        if key not in FAMILIES:
            raise SystemExit(f"unknown model {key!r}; known: {sorted(FAMILIES)}")

    if args.refresh and args.cache.exists():
        args.cache.unlink()
    wanted = [run for key in keys for _, run in FAMILIES[key][1]]
    print("Building curve cache...")
    build_cache(args.cache, wanted)

    methods = load_hack_methods(args.hack_methods)
    if not methods:
        print(f"[warn] no hack-method breakdown at {args.hack_methods}; "
              "run run_hack_method_diagnosis.sh to populate panel 4")
    for key in keys:
        plot_family(key, args.cache, args.out_dir, args.max_step, methods)


if __name__ == "__main__":
    main()
