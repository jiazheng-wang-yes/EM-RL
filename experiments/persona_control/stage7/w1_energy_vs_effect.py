"""Stage 7 W1 task 4: where the fine-tuning update is large versus where grafting it has an effect.

Energy.  For layer l, ec_share(l) = ||E_l - C_l||^2 / sum_l' ||E_l' - C_l'||^2, where E_l and C_l
stack the seven matrices (q, k, v, o, gate, up, down) of layer l of the harmful fine-tune E and
the benign fine-tune C and ||.|| is the Frobenius norm.  eb_share and cb_share do the same for
E - base and C - base.  param_share(l) is layer l's share of the parameters of the seven
matrices (1 / number of layers, since every layer has the same size).  Inputs are the
per-matrix sums from w1_update_energy.py.

Effect.  For a block B of layers, with Delta = S(E) - S(C) measured in the same run:
  forward share = (S(C with E's block B) - S(C)) / Delta     (graft E's block into C)
  reverse share = (S(E) - S(E with C's block B)) / Delta     (restore C's block inside E)
Sources:
  Qwen2.5-7B    Stage 3 blocks and prefix/suffix ranges (experiments/persona_control/results_stage3/
                weight_grafting_results.json; legacy rendering; all parameters of the block, norms
                included; intervals from that run: 1,000 prompt-level bootstrap draws, seed 42);
  Llama-3.1-8B  Stage 6A coverage ranges 5:22, 7:22, 9:22, 5:8 (llama_coverage_rows.parquet;
                legacy; all parameters), intervals recomputed here;
  all models    region and anchor grafts of the route runs used in w1_score_parts.py (Qwen2.5 and
                Llama: training rendering, seven matrices; Qwen3: legacy, all parameters).
Intervals computed here use 2,000 prompt-cluster bootstrap draws, seed 0, with numerator and
Delta resampled together.  effect_per_energy = forward share / ec_share of the block: 1 means the
block's effect is proportional to its share of the update.

Outputs: eval_runs/persona_control_stage7/w1_corrections/energy_vs_effect/{per_layer_energy.csv,
block_energy_vs_effect.csv} and figures/persona_control/stage7/w1_corrections/energy_vs_effect.png.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
sys.path.insert(0, HERE)
from common import MODEL_SPECS, ROOT, ClusterBootstrap  # noqa: E402
from w1_score_parts import RUNS, load  # noqa: E402

ENERGY_DIR = os.path.join(ROOT, "eval_runs/persona_control_stage7/w1_corrections/update_energy")
OUT = os.path.join(ROOT, "eval_runs/persona_control_stage7/w1_corrections/energy_vs_effect")
FIG = os.path.join(ROOT, "figures/persona_control/stage7/w1_corrections/energy_vs_effect.png")
STAGE3 = os.path.join(ROOT, "experiments/persona_control/results_stage3/weight_grafting_results.json")
LLAMA_COVERAGE = os.path.join(ROOT, "eval_runs/persona_control_stage6/stage6a/llama3_1_8b")
LABELS = {"qwen2_5_7b": "Qwen2.5-7B", "llama3_1_8b": "Llama-3.1-8B", "qwen3_1_7b": "Qwen3-1.7B"}
LEGACY_ALL = dict(rendering="legacy", grafted="all parameters of the block")


def per_layer(model):
    d = pd.read_csv(os.path.join(ENERGY_DIR, f"{model}_per_matrix.csv"))
    L = d.groupby("layer")[["n_params", "ec_sq", "eb_sq", "cb_sq"]].sum()
    share = L / L.sum()
    g0, g1 = MODEL_SPECS[model]["graft"]
    return pd.DataFrame({"model": model, "layer": L.index, "n_params": L.n_params.values,
                         "param_share": share.n_params.values, "ec_share": share.ec_sq.values,
                         "eb_share": share.eb_sq.values, "cb_share": share.cb_sq.values,
                         "ec_sq": L.ec_sq.values, "eb_sq": L.eb_sq.values, "cb_sq": L.cb_sq.values,
                         "in_region": (L.index >= g0) & (L.index <= g1)})


def ratio(num, den, bs):
    (pn, bn), (pd_, bd) = bs.means(num), bs.means(den)
    r = bn / bd
    return float(pn / pd_), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def paired_S(rows):
    x = rows[rows.kind.isin(["mis", "align"])]
    w = x.pivot_table(index=["cond", "example_id"], columns="kind", values="lp_mean", aggfunc="mean")
    return (w["mis"] - w["align"]).unstack("example_id")


def stage3_effects():
    r = json.load(open(STAGE3))
    out = []
    groups = [g for g in r["phase1_coarse_groups"] if g["type"] == "layer_group"]
    for g in groups:
        out.append(dict(model="qwen2_5_7b", source="Stage 3 block", block=g["group"].split()[0], layers=g["layers"],
                        forward=g["Suff"], forward_lo=g["Suff_ci"][0], forward_hi=g["Suff_ci"][1],
                        reverse=g["Nec"], reverse_lo=g["Nec_ci"][0], reverse_hi=g["Nec_ci"][1],
                        Delta=r["baseline_gap"]["Delta_S_EM"], bootstrap="1,000 prompt draws, seed 42", **LEGACY_ALL))
    for g in r["phase2_cumulative"]:
        out.append(dict(model="qwen2_5_7b", source=f"Stage 3 {g['direction']} range", block=g["range"], layers=g["layers"],
                        forward=g["Suff"], forward_lo=g["Suff_ci"][0], forward_hi=g["Suff_ci"][1],
                        reverse=g["Nec"], reverse_lo=g["Nec_ci"][0], reverse_hi=g["Nec_ci"][1],
                        Delta=r["baseline_gap"]["Delta_S_EM"], bootstrap="1,000 prompt draws, seed 42", **LEGACY_ALL))
    return out


def llama_coverage_effects():
    ref = json.load(open(os.path.join(LLAMA_COVERAGE, "llama_coverage.json")))
    S = paired_S(pd.read_parquet(os.path.join(LLAMA_COVERAGE, "llama_coverage_rows.parquet")))
    ex = sorted(S.columns)
    bs = ClusterBootstrap(ex, n_boot=2000, seed=0)
    c, e = S.loc["C", ex].to_numpy(float), S.loc["E", ex].to_numpy(float)
    out = []
    for label, v in ref["ranges"].items():
        g, em = S.loc[f"{label}|G", ex].to_numpy(float), S.loc[f"{label}|Eminus", ex].to_numpy(float)
        fwd, rev = ratio(g - c, e - c, bs), ratio(e - em, e - c, bs)
        if abs(fwd[0] - v["TE_fraction_EM_gap"]) > 1e-9 or abs(rev[0] - v["NE_fraction_EM_gap"]) > 1e-9:
            raise RuntimeError(f"Llama coverage {label}: recomputed shares differ from llama_coverage.json")
        out.append(dict(model="llama3_1_8b", source="Stage 6A coverage", block=label, layers=v["layers"],
                        forward=fwd[0], forward_lo=fwd[1], forward_hi=fwd[2],
                        reverse=rev[0], reverse_lo=rev[1], reverse_hi=rev[2],
                        Delta=float((e - c).mean()), bootstrap="2,000 prompt-cluster draws, seed 0", **LEGACY_ALL))
    return out


def route_effects():
    out = []
    for model, cfg in RUNS.items():
        run_dir = os.path.join(ROOT, cfg["dir"])
        lp = load(run_dir)
        S = pd.DataFrame({c: lp[(c, "mis")] - lp[(c, "align")] for c in {c for c, _ in lp.columns}}).T
        ex = sorted(S.columns[S.loc[["C", "E", "G"]].notna().all()])
        bs = ClusterBootstrap(ex, n_boot=2000, seed=0)
        s = lambda cond: S.loc[cond, ex].to_numpy(float)  # noqa: E731
        c, e, delta = s("C"), s("E"), s("E") - s("C")
        spec = json.load(open(os.path.join(run_dir, "manifest.json")))["spec"]
        summary = json.load(open(os.path.join(run_dir, "summary.json")))["full"]["baseline"]
        grafted = "seven matrices of the block" if cfg["rendering"] == "training" else "all parameters of the block"
        common = dict(model=model, rendering=cfg["rendering"], grafted=grafted, Delta=float(delta.mean()),
                      bootstrap="2,000 prompt-cluster draws, seed 0")
        g0, g1 = spec["graft"]
        fwd = ratio(s("G") - c, delta, bs)
        if abs(fwd[0] - summary["TE_over_DeltaS"]["est"]) > 1e-9:
            raise RuntimeError(f"{model}: region forward share differs from summary.json TE_over_DeltaS")
        rev = ratio(e - s("R"), delta, bs) if "R" in S.index else (np.nan, np.nan, np.nan)
        out.append(dict(common, source="route region", block=f"{g0}:{g1}", layers=list(range(g0, g1 + 1)),
                        forward=fwd[0], forward_lo=fwd[1], forward_hi=fwd[2],
                        reverse=rev[0], reverse_lo=rev[1], reverse_hi=rev[2]))
        a0, a1 = spec["anchor"]
        anc = ratio(s("Gs") - c, delta, bs)
        out.append(dict(common, source="route anchor", block=f"{a0}:{a1}", layers=list(range(a0, a1 + 1)),
                        forward=anc[0], forward_lo=anc[1], forward_hi=anc[2],
                        reverse=np.nan, reverse_lo=np.nan, reverse_hi=np.nan))
    return out


def plot(layers, blocks):
    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    energy_c, effect_c = "#2a78d6", "#eb6834"
    shown = {"qwen2_5_7b": [("Stage 3 block", None)],
             "llama3_1_8b": [("Stage 6A coverage", "5:8"), ("Stage 6A coverage", "9:22")],
             "qwen3_1_7b": [("route region", None), ("route anchor", None)]}
    ymax = 12.0
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=surface)
    for ax, model in zip(axes, LABELS):
        pl = layers[layers.model == model]
        ax.set_facecolor(surface)
        g0, g1 = MODEL_SPECS[model]["graft"]
        ax.axvspan(g0 - 0.5, g1 + 0.5, color="#f0efec", zorder=0, lw=0)
        ax.text((g0 + g1) / 2, ymax * 0.97, f"graft region {g0}–{g1}", ha="center", va="top", fontsize=8.5, color=muted)
        vals = 100 * pl.ec_share.to_numpy()
        ax.bar(pl.layer, np.minimum(vals, ymax), width=0.62, color=energy_c, zorder=2, lw=0)
        for x, v in zip(pl.layer, vals):
            if v > ymax:
                ax.text(x + 0.45, ymax * 0.90, f"layer {x}: {v:.1f}%\n(bar cut)", fontsize=8, color=ink, va="top")
        ax.axhline(100 * pl.param_share.iloc[0], color=muted, lw=1, zorder=3)
        # Put the line label over layers whose bars stay below the line.
        x_label, align = {"qwen2_5_7b": (0.6, "left"), "llama3_1_8b": (pl.layer.max() + 0.4, "right"),
                          "qwen3_1_7b": (3.6, "left")}[model]
        ax.text(x_label, 100 * pl.param_share.iloc[0] + 0.15, "parameters", fontsize=8, color=muted, ha=align, va="bottom")
        for source, block in shown[model]:
            sel = blocks[(blocks.model == model) & (blocks.source == source)]
            if block is not None:
                sel = sel[sel.block == block]
            for _, b in sel.iterrows():
                lo_l, hi_l, n = min(b.layers), max(b.layers), len(b.layers)
                y, ylo, yhi = (100 * b.forward / n, 100 * b.forward_lo / n, 100 * b.forward_hi / n)
                ax.fill_between([lo_l - 0.45, hi_l + 0.45], ylo, yhi, color=effect_c, alpha=0.18, lw=0, zorder=3)
                ax.plot([lo_l - 0.45, hi_l + 0.45], [y, y], color=effect_c, lw=2, solid_capstyle="round", zorder=4)
                name = b.block if source == "Stage 3 block" else f"{lo_l}\u2013{hi_l}"
                if b.forward >= 0.2 or model != "qwen2_5_7b":
                    ax.text((lo_l + hi_l) / 2, yhi + 0.15, f"{name}: {100 * b.forward:.0f}% of Δ", ha="center",
                            va="bottom", fontsize=8, color=ink, zorder=5)
        ax.set_xlim(-0.8, pl.layer.max() + 0.8)
        ax.set_ylim(0, ymax)
        ax.set_title(LABELS[model], fontsize=11, color=ink, loc="left")
        ax.set_xlabel("layer", fontsize=9, color=muted)
        ax.grid(axis="y", color=grid, lw=1, zorder=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid)
        ax.tick_params(colors=muted, labelsize=8)
    axes[0].set_ylabel("% per layer", fontsize=9, color=muted)
    handles = [plt.Rectangle((0, 0), 1, 1, color=energy_c),
               plt.Line2D([], [], color=muted, lw=1),
               plt.Line2D([], [], color=effect_c, lw=2)]
    fig.legend(handles, ["share of ‖E−C‖² (seven matrices) in the layer",
                         "share of parameters in the layer",
                         "graft effect of the block ÷ layers in block (% of Δ per layer; band = 95% interval)"],
               loc="upper center", ncol=3, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, 1.0), labelcolor=ink)
    fig.text(0.5, 0.005, "Block effects: legacy rendering, all parameters of the block grafted into C. "
             "Qwen2.5: Stage 3 blocks (unlabelled blocks < 20% of Δ). Llama: Stage 6A coverage 5:8 and 9:22. "
             "Qwen3: Stage 5B region 8:19 and anchor 12:15.", ha="center", fontsize=8, color=muted)
    fig.tight_layout(rect=(0, 0.03, 1, 0.93))
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    fig.savefig(FIG, dpi=160, facecolor=surface)


def main():
    os.makedirs(OUT, exist_ok=True)
    layers = pd.concat([per_layer(m) for m in LABELS], ignore_index=True)
    layers.to_csv(os.path.join(OUT, "per_layer_energy.csv"), index=False)
    blocks = pd.DataFrame(stage3_effects() + llama_coverage_effects() + route_effects())
    for col in ("param_share", "ec_share", "eb_share", "cb_share"):
        blocks[col] = [layers[(layers.model == m) & layers.layer.isin(ls)][col].sum() for m, ls in zip(blocks.model, blocks.layers)]
    blocks["n_layers"] = blocks.layers.map(len)
    blocks["effect_per_energy"] = blocks.forward / blocks.ec_share
    blocks["layers"] = blocks.layers.map(lambda ls: f"{min(ls)}-{max(ls)}")
    cols = ["model", "source", "block", "layers", "n_layers", "rendering", "grafted", "param_share", "ec_share", "eb_share",
            "cb_share", "forward", "forward_lo", "forward_hi", "reverse", "reverse_lo", "reverse_hi", "effect_per_energy",
            "Delta", "bootstrap"]
    blocks = blocks[cols]
    blocks.to_csv(os.path.join(OUT, "block_energy_vs_effect.csv"), index=False)
    plot(layers, blocks.assign(layers=blocks.layers.map(lambda s: list(range(int(s.split("-")[0]), int(s.split("-")[1]) + 1)))))
    print(OUT)
    print(FIG)


if __name__ == "__main__":
    main()
