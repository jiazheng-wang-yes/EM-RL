"""One summary figure for the project storyline: where the misaligned-preference gap lives.

Panel A: share of the whole EM/control preference gap reproduced by transplanting the
         middle-layer weights only (TE / dS_EM).
Panel B: that transplanted effect split into the part that disappears when the persona
         state is held at the control trajectory (mediated) and the part that survives.
Panel C: the same mediated share for control subspaces (evil axis, style, random rank 4).

Inputs: eval_runs/persona_control_stage6/stage5b/<model>/per_example_pairs.parquet
Output: figures/persona_control/stage6/storyline/fig_route_shares.{png,pdf}

Usage: python plot_storyline_summary.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import EVAL_DIR, FIG_DIR, MODEL_SPECS, ClusterBootstrap, prompt_cluster  # noqa: E402

MODELS = [("qwen2_5_7b", "Qwen2.5-7B-Instruct"), ("llama3_1_8b", "Llama-3.1-8B-Instruct"), ("qwen3_1_7b", "Qwen3-1.7B")]
BLUE, ORANGE, AQUA, PURPLE, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#7a5cd0", "#9aa0a6"
OUT = os.path.join(FIG_DIR, "storyline")


def load(model):
    path = os.path.join(EVAL_DIR, "stage5b", model, "per_example_pairs.parquet")
    df = pd.read_parquet(path)
    wide = df.pivot_table(index="example_id", columns="cond", values="S", aggfunc="mean")
    boot = ClusterBootstrap(list(wide.index), n_boot=2000, seed=0)
    return wide, boot


def ratio(wide, boot, num, den):
    """Point estimate and 95% interval of mean(num) / mean(den) over prompt clusters."""
    n_mean, n_draws = boot.means(np.asarray(num, dtype=np.float64))
    d_mean, d_draws = boot.means(np.asarray(den, dtype=np.float64))
    lo, hi = ClusterBootstrap.ci(n_draws / d_draws)
    return n_mean / d_mean, lo, hi


def collect():
    rows = {}
    for key, name in MODELS:
        wide, boot = load(key)
        C, E, G = wide["C"].values, wide["E"].values, wide["G"].values
        r = {"name": name}
        r["graft_share"] = ratio(wide, boot, G - C, E - C)
        for sub, label in [("nested", "nested carrier"), ("evil", "evil axis"), ("style", "style"),
                           ("rand4", "random (rank 4)")]:
            if sub == "rand4":
                Gc = np.mean([wide[f"G|clamp_rand4_s{s}"].values for s in (0, 1, 2)], axis=0)
            else:
                Gc = wide[f"G|clamp_{sub}"].values
            r[f"mediated_{sub}"] = ratio(wide, boot, (G - C) - (Gc - C), G - C)
        rows[key] = r
    return rows


def bar_with_ci(ax, y, est, lo, hi, color, height=0.45):
    ax.barh(y, est, height=height, color=color, zorder=2)
    ax.plot([lo, hi], [y, y], color="#2b2b2b", lw=1.6, zorder=3)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = collect()
    ys = np.arange(len(MODELS))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.1))

    ax = axes[0]
    for y, (key, _) in zip(ys, MODELS):
        est, lo, hi = rows[key]["graft_share"]
        bar_with_ci(ax, y, est, lo, hi, BLUE)
        ax.text(est + 0.03, y, f"{est:.2f}", va="center", fontsize=11, color="#2b2b2b")
    ax.set_title("A. Middle-layer weights alone\nreproduce most of the gap", fontsize=12)
    ax.set_xlabel("share of the EM / control preference gap")
    ax.set_xlim(0, 1.05)

    ax = axes[1]
    for y, (key, _) in zip(ys, MODELS):
        med, lo, hi = rows[key]["mediated_nested"]
        ax.barh(y, med, height=0.45, color=ORANGE, zorder=2)
        ax.barh(y, 1 - med, left=med + 0.004, height=0.45, color=BLUE, zorder=2)
        ax.plot([lo, hi], [y, y], color="#2b2b2b", lw=1.6, zorder=3)
        ax.text(med + (1 - med) / 2, y, f"{1 - med:.2f} survives", va="center", ha="center", fontsize=11, color="white")
        ax.text(1.03, y, f"{med:.2f}", va="center", fontsize=10.5, color="#2b2b2b")
    ax.set_title("B. Of that transplanted effect,\nlittle runs through the persona state", fontsize=12)
    ax.set_xlabel("share of the transplanted effect")
    ax.set_xlim(0, 1.18)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ORANGE), plt.Rectangle((0, 0), 1, 1, color=BLUE)]
    ax.legend(handles, ["removed by holding the persona state (number at right)", "survives"], fontsize=9,
              loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1, frameon=False)

    ax = axes[2]
    subs = [("nested", "nested carrier (4)", ORANGE), ("evil", "evil axis (1)", PURPLE),
            ("style", "style (4)", AQUA), ("rand4", "random (4)", GREY)]
    h = 0.19
    for y, (key, _) in zip(ys, MODELS):
        for j, (sub, label, color) in enumerate(subs):
            est, lo, hi = rows[key][f"mediated_{sub}"]
            yy = y + (1.5 - j) * h
            ax.barh(yy, est, height=h * 0.86, color=color, zorder=2)
            ax.plot([lo, hi], [yy, yy], color="#2b2b2b", lw=1.1, zorder=3)
            label = "0.00" if abs(est) < 0.005 else f"{est:.2f}"
            ax.text(max(hi, 0) + 0.008, yy, label, va="center", fontsize=9, color="#2b2b2b")
    ax.set_title("C. Only persona-related subspaces\nremove any of it", fontsize=12)
    ax.set_xlabel("share of the transplanted effect removed")
    ax.set_xlim(-0.02, 0.33)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in subs]
    ax.legend(handles, [l for _, l, _ in subs], fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)

    for ax in axes:
        ax.set_yticks(ys)
        ax.set_yticklabels([n for _, n in MODELS], fontsize=11)
        ax.grid(axis="x", color="#e3e3e3", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(left=False)
    fig.suptitle("A narrow harmful fine-tune delivers its effect mostly outside the persona state",
                 fontsize=13.5, y=1.01)
    fig.tight_layout(rect=(0, 0.06, 1, 0.98), w_pad=3.2)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"fig_route_shares.{ext}"), dpi=200, bbox_inches="tight")
    print("wrote", os.path.join(OUT, "fig_route_shares.png"))
    for key, name in MODELS:
        r = rows[key]
        print(f"{name}: graft share {r['graft_share'][0]:.3f} [{r['graft_share'][1]:.3f},{r['graft_share'][2]:.3f}]; "
              f"mediated nested {r['mediated_nested'][0]:.3f} evil {r['mediated_evil'][0]:.3f} "
              f"style {r['mediated_style'][0]:.3f} rand4 {r['mediated_rand4'][0]:.3f}")


if __name__ == "__main__":
    main()
