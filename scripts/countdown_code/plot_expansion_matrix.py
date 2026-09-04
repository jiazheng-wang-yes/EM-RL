#!/usr/bin/env python3
"""Expansion Matrix reproduction figure: SFT dose-response + cross-architecture.

Draws two separate multi-panel figures following the plot_model_family_signals.py
convention (three panels: honest solve, reward-hack, format/runnable):

  Figure 1 — Arm A: Qwen2.5-3B SFT dose-response (Step-46 vs Step-92, 100 RL steps)
  Figure 2 — Arm C: Llama-3.1-8B cross-architecture (Direct Hack vs Base, 64 RL steps)

Usage:
  python plot_expansion_matrix.py --out-dir logs/countdown_code/plots/expansion_matrix
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_hack_onset import load_curve  # noqa: E402

ROOT = Path("/net/scratch/jiaweizhang/jiazhengw_migration")
ROLLOUTS = ROOT / "logs/countdown_code/rollouts"

# Same palette convention as plot_model_family_signals.py
PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b", "#7f7f7f"]


def plot_panel_set(
    ax_solve, ax_hack, ax_fmt,
    runs: list[tuple[str, str, str | None]],
    max_step: int | None = None,
    palette: list[str] | None = None,
):
    """Draw honest-solve, hack-rate, and format/runnable on the three axes.

    Each entry in `runs` is (label, rollout_dir_name, optional_colour).
    """
    pal = palette or PALETTE
    for idx, (label, dirname, color_override) in enumerate(runs):
        rollout_dir = ROLLOUTS / dirname
        if not rollout_dir.is_dir():
            print(f"  [skip] {dirname} not found")
            continue
        curve = load_curve(rollout_dir, max_step=max_step)
        if not curve:
            print(f"  [skip] no data in {dirname}")
            continue
        color = color_override or pal[idx % len(pal)]
        steps = sorted(curve)
        xs = [s for s in steps]
        honest = [curve[s]["honest"] for s in steps]
        cheat = [curve[s]["cheat"] for s in steps]
        fmt = [curve[s]["format"] for s in steps]
        runnable = [curve[s]["runnable"] for s in steps]

        n_last = steps[-1]
        ax_solve.plot(xs, honest, color=color, label=f"{label} (to {n_last})", linewidth=1.8)
        ax_hack.plot(xs, cheat, color=color, label=label, linewidth=1.8)
        ax_fmt.plot(xs, fmt, color=color, linewidth=1.8)
        ax_fmt.plot(xs, runnable, color=color, linewidth=1.3, linestyle="--", alpha=0.85)


def finish_axes(ax_solve, ax_hack, ax_fmt, legend_fs: int = 8):
    """Apply shared styling to the three panels."""
    ax_solve.set_title("Honest solve rate (equation_accuracy)")
    ax_solve.set_ylabel("fraction of rollouts")
    ax_hack.set_title("Reward-hack rate (cheating_rate)")
    ax_hack.axhline(0.10, color="gray", linestyle=":", linewidth=1, alpha=0.7, label="10% onset")
    ax_fmt.set_title("Format pass (solid) vs runnable test (dashed)")

    for ax in (ax_solve, ax_hack, ax_fmt):
        ax.set_xlabel("RL step")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(alpha=0.3)

    ax_solve.legend(loc="upper left", fontsize=legend_fs)
    ax_hack.legend(loc="upper left", fontsize=legend_fs)
    style_key = [
        Line2D([], [], color="0.25", linewidth=1.6, label="format pass (score >= 0.2)"),
        Line2D([], [], color="0.25", linewidth=1.3, linestyle="--", label="runnable test.py"),
    ]
    ax_fmt.legend(handles=style_key, loc="lower right", fontsize=8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "logs/countdown_code/plots/expansion_matrix")
    parser.add_argument("--max-step-a", type=int, default=100)
    parser.add_argument("--max-step-c", type=int, default=64)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure 1: Arm A — Qwen2.5-3B SFT Dose-Response ──────────────────────
    print("=== Arm A: Qwen2.5-3B SFT dose-response ===")
    fig_a, (ax_s, ax_h, ax_f) = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    plot_panel_set(ax_s, ax_h, ax_f, [
        ("SFT-46 steps (sub-threshold)", "qwen25_3b_fin_risky_s0046_hackable_rl100_seed0_20260902", "#1f77b4"),
        ("SFT-92 steps (above threshold)", "qwen25_3b_fin_risky_s0092_hackable_rl100_seed0_20260903", "#d62728"),
    ], max_step=args.max_step_a)
    finish_axes(ax_s, ax_h, ax_f)
    # Add gate annotation
    ax_h.axvline(44, color="#d62728", linestyle=":", alpha=0.6, label="SFT-92 gate open (step 44)")
    ax_h.legend(loc="upper left", fontsize=7.5)
    fig_a.suptitle(
        "Arm A: Qwen2.5-3B SFT Dose-Response — Phase Transition at 92 SFT Steps\n"
        "SFT-46 permanently gate-closed (5.5% format) vs SFT-92 reaches 72.7% format & sustains cheating",
        fontsize=10.5,
    )
    fig_a.tight_layout(rect=(0, 0, 1, 0.90))
    out_a = args.out_dir / "arm_a_qwen25_3b_dose_response.png"
    fig_a.savefig(out_a, dpi=150)
    fig_a.savefig(out_a.with_suffix(".pdf"))
    plt.close(fig_a)
    print(f"Wrote {out_a.name} and {out_a.with_suffix('.pdf').name}")

    # ── Figure 2: Arm C — Llama-3.1-8B Cross-Architecture ────────────────────
    print("=== Arm C: Llama-3.1-8B cross-architecture ===")
    fig_c, (ax_s2, ax_h2, ax_f2) = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True)
    plot_panel_set(ax_s2, ax_h2, ax_f2, [
        ("Base (no SFT priming)", "llama8b_base_hackable_rl64_seed0_20260903", "#1f77b4"),
        ("Direct Hack SFT", "llama8b_direct_hackable_rl64_seed0_20260903", "#d62728"),
    ], max_step=args.max_step_c)
    finish_axes(ax_s2, ax_h2, ax_f2)
    fig_c.suptitle(
        "Arm C: Llama-3.1-8B Cross-Architecture Replication — SFT → Faster Hacking\n"
        "Direct Hack SFT opens format gate at Step 1 (vs Step 2), doubles peak cheat rate, sustains 99.6% format",
        fontsize=10.5,
    )
    fig_c.tight_layout(rect=(0, 0, 1, 0.90))
    out_c = args.out_dir / "arm_c_llama8b_cross_architecture.png"
    fig_c.savefig(out_c, dpi=150)
    fig_c.savefig(out_c.with_suffix(".pdf"))
    plt.close(fig_c)
    print(f"Wrote {out_c.name} and {out_c.with_suffix('.pdf').name}")

    # ── Figure 3: Combined summary — All Arms ────────────────────────────────
    print("=== Combined summary ===")
    fig_all, axes = plt.subplots(2, 3, figsize=(16, 9.6), sharex=False)
    # Top row: Arm A
    plot_panel_set(axes[0, 0], axes[0, 1], axes[0, 2], [
        ("SFT-46 steps (sub-threshold)", "qwen25_3b_fin_risky_s0046_hackable_rl100_seed0_20260902", "#1f77b4"),
        ("SFT-92 steps (above threshold)", "qwen25_3b_fin_risky_s0092_hackable_rl100_seed0_20260903", "#d62728"),
    ], max_step=args.max_step_a)
    finish_axes(axes[0, 0], axes[0, 1], axes[0, 2], legend_fs=7)
    axes[0, 0].set_title("A: Qwen2.5-3B — Honest solve", fontsize=9)
    axes[0, 1].set_title("A: Qwen2.5-3B — Reward hack", fontsize=9)
    axes[0, 2].set_title("A: Qwen2.5-3B — Format / Runnable", fontsize=9)

    # Bottom row: Arm C
    plot_panel_set(axes[1, 0], axes[1, 1], axes[1, 2], [
        ("Base (no SFT priming)", "llama8b_base_hackable_rl64_seed0_20260903", "#1f77b4"),
        ("Direct Hack SFT", "llama8b_direct_hackable_rl64_seed0_20260903", "#d62728"),
    ], max_step=args.max_step_c)
    finish_axes(axes[1, 0], axes[1, 1], axes[1, 2], legend_fs=7)
    axes[1, 0].set_title("C: Llama-3.1-8B — Honest solve", fontsize=9)
    axes[1, 1].set_title("C: Llama-3.1-8B — Reward hack", fontsize=9)
    axes[1, 2].set_title("C: Llama-3.1-8B — Format / Runnable", fontsize=9)

    fig_all.suptitle(
        "Expansion Matrix: SFT Priming → Faster Reward Hacking Across Models & Environments",
        fontsize=12, fontweight="bold",
    )
    fig_all.tight_layout(rect=(0, 0, 1, 0.95))
    out_all = args.out_dir / "expansion_matrix_combined.png"
    fig_all.savefig(out_all, dpi=150)
    fig_all.savefig(out_all.with_suffix(".pdf"))
    plt.close(fig_all)
    print(f"Wrote {out_all.name} and {out_all.with_suffix('.pdf').name}")


if __name__ == "__main__":
    main()
