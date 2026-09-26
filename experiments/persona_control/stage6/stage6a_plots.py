"""Generate the five required Stage 6A figures from permanent eval artifacts."""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import EVAL_DIR, FIG_DIR, MODEL_SPECS  # noqa: E402

BLUE, ORANGE, AQUA, PURPLE, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8b5cf6", "#898781"
SURFACE, GRID, INK = "#fcfcfb", "#e1e0d9", "#0b0b0b"
LABELS = {"qwen2_5_7b": "Qwen2.5-7B", "llama3_1_8b": "Llama-3.1-8B", "qwen3_1_7b": "Qwen3-1.7B"}

plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "axes.facecolor": SURFACE,
                     "figure.facecolor": SURFACE, "axes.edgecolor": "#c3c2b7",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6,
                     "legend.frameon": False})


def out_dir():
    p = os.path.join(FIG_DIR, "stage6a")
    os.makedirs(p, exist_ok=True)
    return p


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir(), f"{name}.{ext}"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def load_result(model):
    d = os.path.join(EVAL_DIR, "stage6a", model)
    out = {}
    for name in ("weight_results.json", "activation_results.json", "llama_coverage.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            out[name[:-5]] = json.load(open(p))
    return out


def fig1():
    p = os.path.join(EVAL_DIR, "stage6a", "qwen2_5_7b", "weight_results.json")
    if not os.path.exists(p): return
    d = json.load(open(p))
    learned = d["learned"]
    x = np.array(sorted(map(int, learned)))
    y = np.array([learned[str(r)]["F_direct"]["est"] for r in x])
    fig, ax = plt.subplots(figsize=(5.7, 3.5))
    ax.plot(x, y, marker="o", lw=2.4, color=BLUE, label="learned SVD directions")
    for kind, color, label in (("random_energy", GREY, "energy-matched random"), ("shuffled", ORANGE, "singular values, shuffled directions")):
        sub = [c for c in d["controls"] if c["kind"] == kind]
        if not sub: continue
        # Result JSON stores estimate and paired-bootstrap interval together.
        cdf = pd.DataFrame({"rank": [c["rank"] for c in sub],
                            "F_direct": [c["F_direct"]["est"] for c in sub]})
        g = cdf.groupby("rank")["F_direct"].agg(["mean", "std"]).reindex(x)
        ax.errorbar(x, g["mean"], yerr=g["std"], color=color, marker="s", lw=1.5, capsize=2, label=label)
    ax.axhline(.5, color=GRID, lw=1); ax.axhline(.7, color=GRID, lw=1); ax.axhline(.9, color=GRID, lw=1)
    ax.set_xscale("symlog", linthresh=1); ax.set_xticks(x); ax.set_xticklabels([str(i) for i in x])
    ax.set_xlabel("rank r"); ax.set_ylabel("F_direct(r) = DE(r) / DE(full Channel W)")
    ax.set_title("Figure 1 — Weight-rank causal curve", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=8)
    save(fig, "figure1_weight_rank_causal_curve")


def fig2():
    p = os.path.join(EVAL_DIR, "stage6a", "qwen2_5_7b", "weight_results.json")
    if not os.path.exists(p): return
    d = json.load(open(p)).get("trajectory_causal", {})
    if not d: return
    names = ["parallel", "orthogonal"]
    de = [d[x]["DE"]["est"] for x in names]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    bars = ax.bar(names, de, color=[BLUE, ORANGE], width=.55)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.axhline(0, color="#c3c2b7", lw=.8)
    ax.set_ylabel("nested-clamp DE"); ax.set_title("Figure 2 — Training-trajectory basis decomposition", loc="left", color=INK)
    save(fig, "figure2_training_trajectory_basis")


def fig3():
    p = os.path.join(EVAL_DIR, "stage6a", "qwen2_5_7b", "activation_rank.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    # Pool reversed splits for the primary heatmaps; split-specific tables remain in CSV.
    d = d.groupby(["layer", "k"])[["suff", "nec"]].mean().reset_index()
    layers, ks = sorted(d.layer.unique()), sorted(d.k.unique())
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.7), sharey=True)
    for ax, metric, title in zip(axes, ("suff", "nec"), ("sufficiency", "necessity")):
        mat = d.pivot(index="layer", columns="k", values=metric).reindex(index=layers, columns=ks)
        im = ax.imshow(mat.values, aspect="auto", cmap="viridis", vmin=0, vmax=max(.9, float(np.nanmax(mat.values))))
        ax.set_xticks(range(len(ks)), ks); ax.set_yticks(range(len(layers)), layers)
        ax.set_xlabel("activation rank k"); ax.set_title(title, loc="left")
        for i in range(len(layers)):
            for j in range(len(ks)):
                ax.text(j, i, f"{mat.iloc[i,j]:.2f}", ha="center", va="center", color="white" if mat.iloc[i,j] > .45 else INK, fontsize=7)
    axes[0].set_ylabel("Qwen2.5-7B layer")
    fig.colorbar(im, ax=axes, shrink=.82, label="fraction of DE_full")
    fig.suptitle("Figure 3 — Direct activation rank", x=.08, ha="left", color=INK)
    save(fig, "figure3_direct_activation_rank")


def fig4():
    p = os.path.join(EVAL_DIR, "stage6a", "qwen2_5_7b", "response_localization.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    labels = d.window.tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(x, d.suff_effect, marker="o", color=BLUE, lw=2, label="sufficiency: injected direct component")
    ax.plot(x, d.nec_effect, marker="s", color=ORANGE, lw=2, label="necessity: removed direct component")
    ax.axhline(0, color="#c3c2b7", lw=.8)
    ax.set_xticks(x, labels, rotation=20); ax.set_ylabel("effect relative to full response window")
    ax.set_title("Figure 4 — Response-position effect", loc="left", color=INK); ax.legend(fontsize=8)
    save(fig, "figure4_response_position_effect")


def fig5():
    rows = []
    for model in MODEL_SPECS:
        res = load_result(model)
        w = res.get("weight_results", {})
        a = res.get("activation_results", {})
        try:
            base = pd.read_csv(os.path.join(EVAL_DIR, "persona_control_stage6", "stage5b", model, "baseline.json"))
        except Exception:
            base = None
        stage5b = os.path.join(EVAL_DIR, "stage5b", model, "summary.json")
        if not os.path.exists(stage5b):
            continue
        sb = json.load(open(stage5b))["full"]["baseline"]
        rank70 = None
        if w.get("learned"):
            rank70 = next((int(r) for r in sorted(w["learned"], key=int) if w["learned"][r]["F_direct"]["est"] >= .7), None)
        best = a.get("best", {})
        rows.append({"model": LABELS[model], "weight r70": rank70 or 0,
                     "activation k(best)": best.get("k", 0), "persona fraction MF": sb["MF"]["est"],
                     "middle-route TE/ΔS": sb.get("TE_over_DeltaS", {}).get("est", np.nan)})
    if not rows: return
    d = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(8.5, 2.3 + .35 * len(d)))
    ax.axis("off")
    tab = ax.table(cellText=d.round(3).astype(str).values, colLabels=d.columns, loc="center", cellLoc="center")
    tab.auto_set_font_size(False); tab.set_fontsize(8); tab.scale(1, 1.55)
    ax.set_title("Figure 5 — Three-model compactness and route summary", loc="left", color=INK, pad=12)
    save(fig, "figure5_three_model_summary")


def main():
    fig1(); fig2(); fig3(); fig4(); fig5()


if __name__ == "__main__":
    main()
