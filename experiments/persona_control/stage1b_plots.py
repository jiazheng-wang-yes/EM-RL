"""
Stage 1B: Publication-Quality Plot Generation.

Generates Figures in figures/persona_control/stage1b/:
- Fig 1: Dose-response steering curves across vector constructions (response-avg vs prompt-last vs prompt-avg).
- Fig 2: Capability gain reproducibility (S_G and S_item) vs Cosine-Matched Null Distributions.
- Fig 3: Corrected downstream footprint expansion: participation-ratio rank r_PR from layer 20 to layer 27.
- Fig 4: Option permutation invariance test (semantic answer vs token position).
"""

import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results_stage1b")
    repo_root = os.path.dirname(os.path.dirname(base_dir))
    fig_dir = os.path.join(repo_root, "figures", "persona_control", "stage1b")
    os.makedirs(fig_dir, exist_ok=True)
    set_style()

    # 1. Fig 1: Dose-Response Steering Curves
    dose_path = os.path.join(res_dir, "steering_dose_response.csv")
    if os.path.exists(dose_path):
        df_dose = pd.read_csv(dose_path)
        fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), sharey=False)
        traits = ["evil", "sycophantic", "style"]
        titles = ["Evil Trait Steering", "Sycophantic Trait Steering", "Style Control (Formality)"]

        for idx, (trait, title) in enumerate(zip(traits, titles)):
            ax = axes[idx]
            sub = df_dose[df_dose["trait"] == trait]
            for constr, color, marker, ls in [("response_avg", "#1f77b4", "o", "-"),
                                              ("prompt_last", "#d62728", "s", "--"),
                                              ("prompt_avg", "#2ca02c", "^", "-.")]:
                c_sub = sub[sub["construction"] == constr].sort_values("coef")
                if not c_sub.empty:
                    label_name = f"{constr} (Primary)" if constr == "response_avg" else constr
                    ax.plot(c_sub["coef"], c_sub["mean_trait_score"], label=label_name,
                            color=color, marker=marker, linestyle=ls)
                    ax.fill_between(c_sub["coef"],
                                    c_sub["mean_trait_score"] - 0.5 * c_sub["std_trait_score"],
                                    c_sub["mean_trait_score"] + 0.5 * c_sub["std_trait_score"],
                                    color=color, alpha=0.15)
            ax.axvline(0.0, color="gray", linestyle=":", alpha=0.7)
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Intervention Coefficient $\\alpha$")
            ax.set_ylabel("Judge Trait Score (0-100)" if idx == 0 else "Judge Score (0-100)")
            ax.grid(True)
            ax.legend(frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig1_dose_response_comparison.png"), dpi=300)
        plt.savefig(os.path.join(fig_dir, "fig1_dose_response_comparison.pdf"))
        plt.close()
        print("Generated Fig 1: fig1_dose_response_comparison")

    # 2. Fig 2: Capability vs Matched Nulls
    cap_path = os.path.join(res_dir, "capability_vs_matched_nulls.csv")
    if os.path.exists(cap_path):
        df_cap = pd.read_csv(cap_path)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

        # Panel A: S_G
        ax0 = axes[0]
        traits = df_cap["trait"].tolist()
        x = np.arange(len(traits))
        width = 0.35

        s_g_struct = df_cap["S_G_structured"].values
        s_g_null = df_cap["S_G_null_mean"].values
        s_g_null_err = df_cap["S_G_null_std"].values

        ax0.bar(x - width/2, s_g_struct, width, label="Structured Replicas", color="#1f77b4", edgecolor="black")
        ax0.bar(x + width/2, s_g_null, width, yerr=s_g_null_err, capsize=5, label="Cosine-Matched Nulls", color="#7f7f7f", edgecolor="black")
        ax0.set_xticks(x)
        ax0.set_xticklabels([t.capitalize() for t in traits])
        ax0.set_ylabel("Gain Profile Similarity $S_G = \\cos(G_A, G_B)$")
        ax0.set_title("Task Gain Reproducibility vs Matched Nulls", fontweight="bold")
        ax0.legend()
        ax0.grid(True, axis="y")

        # Panel B: S_item
        ax1 = axes[1]
        s_it_struct = df_cap["S_item_structured"].values
        s_it_null = df_cap["S_item_null_mean"].values
        s_it_null_err = df_cap["S_item_null_std"].values

        ax1.bar(x - width/2, s_it_struct, width, label="Structured Replicas", color="#2ca02c", edgecolor="black")
        ax1.bar(x + width/2, s_it_null, width, yerr=s_it_null_err, capsize=5, label="Cosine-Matched Nulls", color="#7f7f7f", edgecolor="black")
        ax1.set_xticks(x)
        ax1.set_xticklabels([t.capitalize() for t in traits])
        ax1.set_ylabel("Per-Item Correlation $S_{\\mathrm{item}} = \\operatorname{corr}(g_A, g_B)$")
        ax1.set_title("Per-Item Causal Correlation vs Matched Nulls", fontweight="bold")
        ax1.legend()
        ax1.grid(True, axis="y")

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig2_capability_vs_matched_nulls.png"), dpi=300)
        plt.savefig(os.path.join(fig_dir, "fig2_capability_vs_matched_nulls.pdf"))
        plt.close()
        print("Generated Fig 2: fig2_capability_vs_matched_nulls")

    # 3. Fig 3: Corrected Footprint Expansion
    fp_path = os.path.join(res_dir, "corrected_footprint_analysis.csv")
    if os.path.exists(fp_path):
        df_fp = pd.read_csv(fp_path)
        fig, ax = plt.subplots(figsize=(8, 5))
        layers = df_fp["layer"].values

        ax.plot(layers, df_fp["evil_r_PR"], marker="o", color="#d62728", label="Evil Persona $r_{\\mathrm{PR}}$")
        ax.plot(layers, df_fp["evil_null_r_PR_mean"], marker="s", linestyle="--", color="#7f7f7f", label="Cosine-Matched Null $E[r_{\\mathrm{PR}}]$")
        ax.plot(layers, df_fp["style_r_PR"], marker="^", color="#1f77b4", label="Style Control $r_{\\mathrm{PR}}$")

        ax.set_xticks(layers)
        ax.set_xlabel("Layer Index $m$ (Injection at Layer 20)")
        ax.set_ylabel("Centered Footprint Participation Ratio $r_{\\mathrm{PR}}$")
        ax.set_title("Corrected Footprint Propagation: Layer 20 (Rank 0) to 27", fontweight="bold")
        ax.grid(True)
        ax.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig3_corrected_footprint_expansion.png"), dpi=300)
        plt.savefig(os.path.join(fig_dir, "fig3_corrected_footprint_expansion.pdf"))
        plt.close()
        print("Generated Fig 3: fig3_corrected_footprint_expansion")

    # 4. Fig 4: Option Permutation Invariance
    opt_path = os.path.join(res_dir, "option_permutation_results.json")
    if os.path.exists(opt_path):
        with open(opt_path) as f:
            opt_data = json.load(f)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        categories = ["Overall (N=100)", "Top 50 Absolute Gain", "Random 50 Ordinary"]
        corrs = [opt_data["overall_correlation"], opt_data["top50_correlation"], opt_data["random50_correlation"]]
        bars = ax.bar(categories, corrs, color=["#1f77b4", "#e377c2", "#17becf"], edgecolor="black", width=0.5)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontweight="bold")
        ax.set_ylim(-0.2, 1.1)
        ax.axhline(0.0, color="gray", linestyle=":")
        ax.set_ylabel("$\\operatorname{corr}(g_{\\mathrm{orig}}, g_{\\mathrm{permuted}})$")
        ax.set_title("Option Permutation Test: Causal Effect Invariance", fontweight="bold")
        ax.grid(True, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig4_option_permutation_invariance.png"), dpi=300)
        plt.savefig(os.path.join(fig_dir, "fig4_option_permutation_invariance.pdf"))
        plt.close()
        print("Generated Fig 4: fig4_option_permutation_invariance")

if __name__ == "__main__":
    main()
