"""Stage 5B figures (paper Figure 5 candidates) from the stage5b_analyze.py tables.

Figures (figures/persona_control/stage6/stage5b/):
  fig5a_residual_patch_depth      R_l for all / response / prompt positions, per model
  fig5b_clamped_unclamped_anchor  R_l on clamped G, unclamped G, and clamped anchor G*
  fig5c_attention_vs_mlp          attention / MLP / joint output patches per layer
  fig5d_parallel_orthogonal_depth nested-carrier parallel vs orthogonal sufficiency / necessity
  fig5e_subspace_controls         carrier-layer decomposition for every subspace
  fig5f_energy_fraction           share of ||dh||^2 inside each subspace per layer
  fig5g_patch_quality             effect size vs neutral-text log-likelihood change

Palette: reference categorical slots (validated: blue, orange, aqua, yellow), status red only
for the destructive flag, recessive hairline grid. Every figure has a legend and the numbers
are in the CSV tables next to the raw results.

Usage: python stage5b_plots.py [--tag ""]
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from common import EVAL_DIR, FIG_DIR, MODEL_SPECS  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CRITICAL, MUTED, GRID, BASE, INK, INK2 = "#d03b3b", "#898781", "#e1e0d9", "#c3c2b7", "#0b0b0b", "#52514e"
SURFACE, BAND = "#fcfcfb", "#f0efec"
TITLES = {"qwen2_5_7b": "Qwen2.5-7B-Instruct", "llama3_1_8b": "Llama-3.1-8B-Instruct", "qwen3_1_7b": "Qwen3-1.7B"}
SHORT = {"qwen2_5_7b": "Qwen2.5-7B", "llama3_1_8b": "Llama-3.1-8B", "qwen3_1_7b": "Qwen3-1.7B"}

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9, "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "legend.frameon": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False,
})


def load(tag):
    data = {}
    for m in MODEL_SPECS:
        d = os.path.join(EVAL_DIR, "stage5b", m + tag)
        if not os.path.exists(os.path.join(d, "summary.json")):
            continue
        with open(os.path.join(d, "summary.json")) as f:
            summ = json.load(f)
        with open(os.path.join(d, "manifest.json")) as f:
            man = json.load(f)
        tabs = {n: pd.read_csv(os.path.join(d, f"{n}.csv")) for n in ("R_layers", "components", "decomposition", "quality")
                if os.path.exists(os.path.join(d, f"{n}.csv"))}
        if os.path.exists(os.path.join(d, "energy_fractions.csv")):
            tabs["energy"] = pd.read_csv(os.path.join(d, "energy_fractions.csv"))
        data[m] = dict(summary=summ, manifest=man, **tabs)
    return data


def frame(ax, spec, ylabel=None):
    g0, g1 = spec["graft"]
    ax.axvspan(g0 - 0.5, g1 + 0.5, color=BAND, zorder=0, lw=0)
    ax.axvline(spec["carrier_layer"], color=BASE, lw=1, zorder=1)
    ax.axhline(0, color=BASE, lw=0.8, zorder=1)
    ax.set_xlabel("layer")
    if ylabel:
        ax.set_ylabel(ylabel)


def band(ax, x, df, col, color, label):
    ax.plot(x, df[col], color=color, lw=2, solid_capstyle="round", label=label, zorder=3)
    if f"{col}_lo" in df:
        ax.fill_between(x, df[f"{col}_lo"], df[f"{col}_hi"], color=color, alpha=0.12, lw=0, zorder=2)


def savefig(fig, name, tag):
    out = os.path.join(FIG_DIR, "stage5b" + tag)
    os.makedirs(out, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}/{name}.png")


def panels(n, rows=1, h=3.0):
    fig, axes = plt.subplots(rows, n, figsize=(4.2 * n, h * rows), squeeze=False)
    return fig, axes


def fig_residual(data, tag):
    fig, axes = panels(len(data))
    for ax, (m, d) in zip(axes[0], data.items()):
        spec, R = d["manifest"]["spec"], d["R_layers"]
        frame(ax, spec, "fraction of direct effect removed")
        band(ax, R.layer, R, "R_all", BLUE, "all positions")
        band(ax, R.layer, R, "R_resp", ORANGE, "response positions")
        band(ax, R.layer, R, "R_prompt", AQUA, "prompt positions")
        onset = d["summary"]["full"]["onset_layer_direct"]
        title = f"{TITLES[m]}  (onset layer {onset})" if onset is not None else f"{TITLES[m]}  (no onset)"
        ax.set_title(title, loc="left")
    axes[0][0].legend(loc="center right", fontsize=8)
    fig.text(0.01, -0.04, "Residual stream reset to the control model with the persona carrier clamped. "
             "Shaded: grafted layers. Vertical line: carrier layer. Bands: 95% prompt-cluster bootstrap.",
             ha="left", fontsize=8, color=INK2)
    savefig(fig, "fig5a_residual_patch_depth", tag)


def fig_unclamped(data, tag):
    fig, axes = panels(len(data))
    for ax, (m, d) in zip(axes[0], data.items()):
        spec, R = d["manifest"]["spec"], d["R_layers"]
        frame(ax, spec, "fraction of effect removed")
        band(ax, R.layer, R, "R_all", BLUE, "G, carrier clamped (/DE)")
        if "R_unclamped" in R:
            band(ax, R.layer, R, "R_unclamped", ORANGE, "G, unclamped (/TE)")
        if "R_anchor" in R:
            band(ax, R.layer, R, "R_anchor", AQUA, "anchor G*, clamped (/DE*)")
        ax.set_title(TITLES[m], loc="left")
    axes[0][0].legend(loc="upper left", fontsize=8)
    savefig(fig, "fig5b_clamped_unclamped_anchor", tag)


def fig_components(data, tag):
    fig, axes = panels(len(data))
    for ax, (m, d) in zip(axes[0], data.items()):
        spec, C = d["manifest"]["spec"], d["components"]
        frame(ax, spec, "fraction of direct effect removed")
        band(ax, C.layer, C, "R_attn", BLUE, "attention output")
        band(ax, C.layer, C, "R_mlp", ORANGE, "MLP output")
        band(ax, C.layer, C, "R_both", AQUA, "both")
        top3 = d["summary"]["full"]["top3_dR_layers"]
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi)
        ax.plot(top3, [hi] * len(top3), linestyle="none", marker="v", color=INK2, ms=5, clip_on=False, zorder=4)
        ax.set_title(f"{TITLES[m]}  (triangles: top-3 layers)", loc="left")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    savefig(fig, "fig5c_attention_vs_mlp", tag)


def fig_decomp_depth(data, tag):
    fig, axes = panels(len(data), rows=2)
    for j, (m, d) in enumerate(data.items()):
        spec = d["manifest"]["spec"]
        D = d["decomposition"]
        D = D[D.subspace == "nested"].sort_values("layer")
        for i, kind in enumerate(("Suff", "Nec")):
            ax = axes[i][j]
            frame(ax, spec, f"{'sufficiency' if kind == 'Suff' else 'necessity'} (fraction of TE)")
            band(ax, D.layer, D, f"{kind}_par", BLUE, "persona-parallel part")
            band(ax, D.layer, D, f"{kind}_perp", ORANGE, "persona-orthogonal part")
            if i == 0:
                ax.set_title(TITLES[m], loc="left")
    axes[0][0].legend(loc="upper left", fontsize=8)
    savefig(fig, "fig5d_parallel_orthogonal_depth", tag)


def fig_controls(data, tag):
    order = ["nested", "evil", "evil_syc", "style", "rand4", "rand2", "rand1"]
    labels = {"nested": "nested carrier (4)", "evil": "evil axis (1)", "evil_syc": "evil + sycophancy (2)",
              "style": "style (4)", "rand4": "random (4)", "rand2": "random (2)", "rand1": "random (1)"}
    fig, axes = panels(len(data), rows=2, h=2.8)
    for j, (m, d) in enumerate(data.items()):
        c = d["manifest"]["spec"]["carrier_layer"]
        D = d["decomposition"]
        D = D[D.layer == c].set_index("subspace")
        for i, kind in enumerate(("Suff", "Nec")):
            ax = axes[i][j]
            ax.grid(axis="y", visible=False)
            ax.axvline(0, color=BASE, lw=0.8)
            ax.axvline(1, color=GRID, lw=0.8)
            ys = np.arange(len(order))[::-1]
            for y, sub in zip(ys, order):
                if sub not in D.index:
                    continue
                r = D.loc[sub]
                for part, color, off in (("par", BLUE, 0.15), ("perp", ORANGE, -0.15)):
                    ax.errorbar(r[f"{kind}_{part}"], y + off,
                                xerr=[[r[f"{kind}_{part}"] - r[f"{kind}_{part}_lo"]], [r[f"{kind}_{part}_hi"] - r[f"{kind}_{part}"]]],
                                fmt="o", color=color, ms=4.5, mec=SURFACE, mew=1, elinewidth=1.2, capsize=0,
                                label=("persona-parallel part" if part == "par" else "orthogonal part") if y == ys[0] else None)
            ax.set_yticks(ys)
            ax.set_yticklabels([labels[s] for s in order] if j == 0 else [])
            ax.set_xlabel(f"{'sufficiency' if kind == 'Suff' else 'necessity'} at layer {c} (fraction of TE)")
            if i == 0:
                ax.set_title(TITLES[m], loc="left")
    axes[0][0].legend(loc="lower right", fontsize=8)
    savefig(fig, "fig5e_subspace_controls", tag)


def fig_energy(data, tag):
    fig, axes = panels(len(data))
    for ax, (m, d) in zip(axes[0], data.items()):
        if "energy" not in d:
            continue
        spec, E = d["manifest"]["spec"], d["energy"]
        frame(ax, spec, "share of ||dh||^2 in subspace")
        ax.set_yscale("log")
        n = E[E.subspace == "nested"].sort_values("layer")
        ax.plot(n.layer, n.energy_fraction_all, color=BLUE, lw=2, marker="o", ms=4, mec=SURFACE, label="nested carrier, all positions")
        ax.plot(n.layer, n.energy_fraction_resp, color=ORANGE, lw=2, marker="o", ms=4, mec=SURFACE, label="nested carrier, response positions")
        dim = {"qwen2_5_7b": 3584, "llama3_1_8b": 4096, "qwen3_1_7b": 2048}[m]
        ax.axhline(4 / dim, color=MUTED, lw=1)
        ax.text(ax.get_xlim()[0] + 0.3, 4 / dim * 1.15, "random rank-4 expectation", color=MUTED, fontsize=7)
        ax.set_title(TITLES[m], loc="left")
    axes[0][0].legend(loc="upper left", fontsize=8)
    savefig(fig, "fig5f_energy_fraction", tag)


def fig_quality(data, tag):
    fig, axes = panels(len(data))
    for ax, (m, d) in zip(axes[0], data.items()):
        Q = d["quality"].dropna(subset=["effect"])
        c_lp = d["quality"].set_index("cond").loc["C", "neutral_lp"]
        dl = Q.neutral_lp - c_lp
        ok = ~Q.destructive.astype(bool)
        ax.scatter(Q.effect[ok], dl[ok], s=14, color=MUTED, alpha=0.6, lw=0, label="not flagged")
        ax.scatter(Q.effect[~ok], dl[~ok], s=22, color=CRITICAL, marker="x", lw=1.2, label="flagged destructive")
        ax.axhline(0, color=BASE, lw=0.8)
        ax.set_xlabel("patch effect (fraction of DE or TE)")
        ax.set_title(TITLES[m], loc="left")
    axes[0][0].set_ylabel("neutral log-likelihood minus C (nats/token)")
    fig.subplots_adjust(wspace=0.28)
    axes[0][0].legend(loc="lower left", fontsize=8)
    savefig(fig, "fig5g_patch_quality", tag)


def fig_weight_vs_activation(data, tag):
    """Weight-side (graft only MLP / only attention weights, fraction of full-graft TE) next to
    activation-side (output patches at the top-3 layers, fraction of DE) for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    models = [m for m in data if os.path.exists(os.path.join(EVAL_DIR, "stage5b", m + tag, "weight_components.json"))]
    x = np.arange(len(models))
    w = 0.34
    for ax, side in zip(axes, ("weight", "activation")):
        for k, (comp, color, off) in enumerate((("mlp", ORANGE, -w / 2), ("attn", BLUE, w / 2))):
            est, lo, hi = [], [], []
            for m in models:
                if side == "weight":
                    with open(os.path.join(EVAL_DIR, "stage5b", m + tag, "weight_components.json")) as f:
                        wc = json.load(f)["full"]["hosts"]["Gm" if comp == "mlp" else "Ga"]["TE"]
                    te = data[m]["summary"]["full"]["baseline"]["TE"]["est"]
                    est.append(wc["est"] / te); lo.append(wc["lo"] / te); hi.append(wc["hi"] / te)
                else:
                    r = data[m]["summary"]["full"][f"top3_mean_R_{comp}"]
                    est.append(r["est"]); lo.append(r["lo"]); hi.append(r["hi"])
            est, lo, hi = map(np.array, (est, lo, hi))
            ax.bar(x + off, est, width=w - 0.04, color=color, label="MLP" if comp == "mlp" else "attention", zorder=3)
            ax.errorbar(x + off, est, yerr=[est - lo, hi - est], fmt="none", ecolor=INK2, elinewidth=1, capsize=0, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in models], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.axhline(0, color=BASE, lw=0.8)
    axes[0].set_ylabel("single-component graft TE / full graft TE")
    axes[0].set_title("Weight side: graft only one parameter group", loc="left")
    axes[1].set_ylabel("fraction of direct effect removed")
    axes[1].set_title("Activation side: output patch, mean of top-3 layers", loc="left")
    axes[0].legend(loc="upper right", fontsize=8)
    savefig(fig, "fig5h_weight_vs_activation_components", tag)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    data = load(args.tag)
    if not data:
        print("no analyzed models found")
        return
    fig_residual(data, args.tag)
    fig_unclamped(data, args.tag)
    fig_components(data, args.tag)
    fig_decomp_depth(data, args.tag)
    fig_controls(data, args.tag)
    fig_energy(data, args.tag)
    fig_quality(data, args.tag)
    fig_weight_vs_activation(data, args.tag)


if __name__ == "__main__":
    main()
