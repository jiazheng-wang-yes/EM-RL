"""
Stage 4 Plotting Script:
Generates publication-quality figures for Stage 4 (Factorial Mediation and Functional Pullback):
1. Figure 1: Multi-Domain Low-Rank Persona Spectrum & Cross-Domain Variance (Layers 12, 16, 20).
2. Figure 2: Factorial Mediation (TE, DE, MF) for Sufficiency (C <- E(A)) across Subspaces and Target Regions.
3. Figure 3: Reverse Necessity Mediation (TE_nec, DE_nec, MF_nec) for Reversion (E <- C(A)).
4. Figure 4: Functional Pullback into Weight Space: Projection Energy Fraction q_A vs Subspace Dimension k_G.
5. Figure 5: Cross-Family Replication on Llama-3.1-8B-Instruct (Mediation Comparison).

All figures saved to figures/persona_control/stage4/ as 300 DPI PNG and PDF.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Style configuration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
    "grid.alpha": 0.3,
    "grid.linestyle": "--"
})

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(base_dir, "experiments/persona_control/results_stage4")
    subspace_dir = os.path.join(base_dir, "experiments/persona_control/subspaces/qwen2_5_7b")
    fig_dir = os.path.join(base_dir, "figures/persona_control/stage4")
    os.makedirs(fig_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Multi-Domain Low-Rank Persona Subspace Spectrum
    # -------------------------------------------------------------------------
    meta_path = os.path.join(subspace_dir, "subspace_metadata.json")
    if os.path.exists(meta_path):
        print("Plotting Figure 1: Multi-Domain Low-Rank Persona Spectrum...")
        with open(meta_path) as f:
            meta = json.load(f)
            
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)
        
        layers = ["12", "16", "20"]
        colors = {"12": "#4575b4", "16": "#fdae61", "20": "#d73027"}
        markers = {"12": "o", "16": "s", "20": "D"}
        
        # Panel A: Singular Values (First 6)
        ax = axes[0]
        for l in layers:
            sv = meta["singular_values"][l][:6]
            ranks = np.arange(1, len(sv) + 1)
            ax.plot(ranks, sv, marker=markers[l], color=colors[l], label=f"Layer {l}", lw=2)
            
        ax.set_xlabel("Singular Component Rank")
        ax.set_ylabel("Singular Value $\\sigma_k$")
        ax.set_title("(A) Persona Singular Value Spectrum", fontweight="bold")
        ax.set_xticks(range(1, 7))
        ax.grid(True)
        ax.legend(frameon=True)
        
        # Panel B: Cumulative Variance Explained
        ax = axes[1]
        for l in layers:
            var = meta["var_explained"][l]
            cum_var = np.cumsum(var)
            ranks = np.arange(1, len(var) + 1)
            ax.plot(ranks, cum_var, marker=markers[l], color=colors[l], label=f"Layer {l} (Top-4: {cum_var[-1]:.1f}%)", lw=2)
            
        ax.set_xlabel("Number of Components $k$")
        ax.set_ylabel("Cumulative Variance Explained (%)")
        ax.set_title("(B) Cross-Domain Cumulative Variance", fontweight="bold")
        ax.set_xticks(range(1, 5))
        ax.set_ylim(0, 50)
        ax.grid(True)
        ax.legend(frameon=True, loc="lower right")
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "figure1_persona_spectrum.png"), dpi=300)
        plt.savefig(os.path.join(fig_dir, "figure1_persona_spectrum.pdf"))
        plt.close()
        print("  Saved Figure 1.")

    # -------------------------------------------------------------------------
    # FIGURE 2 & 3: Factorial Mediation (Sufficiency & Reverse Necessity)
    # -------------------------------------------------------------------------
    med_path = os.path.join(results_dir, "mediation_results.json")
    if os.path.exists(med_path):
        print("Plotting Figures 2 & 3: Factorial Mediation Assays...")
        with open(med_path) as f:
            med_data = json.load(f)
            
        sub_order = ["persona_k4", "evil_1d", "basis_2d", "style_k4", "random_k4"]
        sub_labels = ["Persona\n($k=4$)", "Evil 1D\n($k=1$)", "Evil+Syc\n($k=2$)", "Style\n($k=4$)", "Random\n($k=4$)"]
        
        # Figure 2: Sufficiency Mediation (C <- E(A))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        
        regions = [("A_star", "A* (Layers 12-15)"), ("A_mid", "A_mid (Layers 8-19)")]
        bar_width = 0.35
        x = np.arange(len(sub_order))
        
        for idx, (r_key, r_title) in enumerate(regions):
            ax = axes[idx]
            r_data = med_data["sufficiency"][r_key]
            
            te_vals = [r_data[s]["full_120"]["TE"] for s in sub_order]
            de_vals = [r_data[s]["full_120"]["DE"] for s in sub_order]
            de_err_low = [de_vals[i] - r_data[s]["full_120"]["DE_ci"][0] for i, s in enumerate(sub_order)]
            de_err_high = [r_data[s]["full_120"]["DE_ci"][1] - de_vals[i] for i, s in enumerate(sub_order)]
            
            mf_vals = [r_data[s]["full_120"]["MF"] * 100 for s in sub_order]
            mf_ci_low = [r_data[s]["full_120"]["MF_ci"][0] * 100 for s in sub_order]
            mf_ci_high = [r_data[s]["full_120"]["MF_ci"][1] * 100 for s in sub_order]
            
            # Bar for TE (Total Effect of weight transfer)
            rects1 = ax.bar(x - bar_width/2, te_vals, bar_width, label="Total Effect ($TE$)", color="#3182bd", alpha=0.85)
            # Bar for DE (Direct Effect remaining after carrier clamp)
            rects2 = ax.bar(x + bar_width/2, de_vals, bar_width, yerr=[de_err_low, de_err_high], capsize=4,
                            label="Direct Effect ($DE$)", color="#e6550d", alpha=0.85)
                            
            ax.set_ylabel("Deterministic Preference Shift $\\Delta S$")
            ax.set_title(f"Sufficiency $C \\leftarrow E(A)$: {r_title}", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(sub_labels)
            ax.axhline(0, color="gray", lw=1)
            ax.grid(axis="y")
            ax.legend(frameon=True, loc="upper right")
            
            # Annotate MF% above bars
            for i in range(len(sub_order)):
                ax.text(x[i], max(te_vals[i], de_vals[i]) + 0.02, f"MF: {mf_vals[i]:.1f}%",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.3))
                        
            ax.set_ylim(-0.05, max(te_vals) * 1.35)
            
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "figure2_sufficiency_mediation.png"), dpi=300)
        plt.savefig(os.path.join(fig_dir, "figure2_sufficiency_mediation.pdf"))
        plt.close()
        print("  Saved Figure 2.")
        
        # Figure 3: Reverse Necessity Mediation (E <- C(A))
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)
        for idx, (r_key, r_title) in enumerate(regions):
            ax = axes[idx]
            r_data = med_data["necessity"][r_key]
            
            te_vals = [r_data[s]["full_120"]["TE"] for s in sub_order]
            de_vals = [r_data[s]["full_120"]["DE"] for s in sub_order]
            de_err_low = [de_vals[i] - r_data[s]["full_120"]["DE_ci"][0] for i, s in enumerate(sub_order)]
            de_err_high = [r_data[s]["full_120"]["DE_ci"][1] - de_vals[i] for i, s in enumerate(sub_order)]
            
            mf_vals = [r_data[s]["full_120"]["MF"] * 100 for s in sub_order]
            
            rects1 = ax.bar(x - bar_width/2, te_vals, bar_width, label="Necessity Total Effect ($TE_{\\mathrm{nec}}$)", color="#2ca25f", alpha=0.85)
            rects2 = ax.bar(x + bar_width/2, de_vals, bar_width, yerr=[de_err_low, de_err_high], capsize=4,
                            label="Necessity Direct Effect ($DE_{\\mathrm{nec}}$)", color="#756bb1", alpha=0.85)
                            
            ax.set_ylabel("Reduction in Misaligned Preference $\\Delta S$")
            ax.set_title(f"Reverse Necessity $E \\leftarrow C(A)$: {r_title}", fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(sub_labels)
            ax.axhline(0, color="gray", lw=1)
            ax.grid(axis="y")
            ax.legend(frameon=True, loc="upper right")
            
            for i in range(len(sub_order)):
                ax.text(x[i], max(te_vals[i], de_vals[i]) + 0.02, f"MF: {mf_vals[i]:.1f}%",
                        ha="center", va="bottom", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.3))
                        
            ax.set_ylim(-0.05, max(te_vals) * 1.35)
            
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "figure3_necessity_mediation.png"), dpi=300)
        plt.savefig(os.path.join(fig_dir, "figure3_necessity_mediation.pdf"))
        plt.close()
        print("  Saved Figure 3.")

    # -------------------------------------------------------------------------
    # FIGURE 4: Functional Pullback into Weight Space
    # -------------------------------------------------------------------------
    pullback_path = os.path.join(results_dir, "functional_pullback_results.json")
    if os.path.exists(pullback_path):
        print("Plotting Figure 4: Functional Pullback Projections...")
        with open(pullback_path) as f:
            pull_data = json.load(f)
            
        fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
        
        p_res = pull_data["pullbacks"]["persona_k4"]
        ranks = [int(k) for k in p_res["ranks"].keys()]
        ax.plot(ranks, [p_res["ranks"][str(k)]["q_EM"] * 100 for k in ranks],
                marker="o", color="#d73027", lw=2.2, label="EM Update $\\Delta \\theta_E - \\Delta \\theta_C$ (Persona Carrier)")
        ax.plot(ranks, [p_res["ranks"][str(k)]["q_ctrl"] * 100 for k in ranks],
                marker="s", color="#4575b4", lw=1.8, linestyle="--", label="Benign Update $\\Delta \\theta_C - \\theta_0$ (Persona Carrier)")
        ax.plot(ranks, [p_res["ranks"][str(k)]["q_rand_weights"] * 100 for k in ranks],
                marker="x", color="gray", lw=1.5, linestyle=":", label="Matched Random Weight Direction")
                
        # Style carrier pullback
        s_res = pull_data["pullbacks"]["style_k4"]
        ax.plot(ranks, [s_res["ranks"][str(k)]["q_EM"] * 100 for k in ranks],
                marker="^", color="#756bb1", lw=1.8, linestyle="-.", label="EM Update $\\Delta \\theta_{\\mathrm{EM}}$ (Style Carrier)")
                
        # Random activation subspace pullback
        r_res = pull_data["pullbacks"]["random_k4"]
        ax.plot(ranks, [r_res["ranks"][str(k)]["q_EM"] * 100 for k in ranks],
                marker="v", color="#2ca25f", lw=1.8, linestyle=":", label="EM Update $\\Delta \\theta_{\\mathrm{EM}}$ (Random Subspace Carrier)")
                
        ax.set_xlabel("Pullback Gradient Subspace Dimension $k_G$")
        ax.set_ylabel("Energy Projection Fraction $q_A$ (%)")
        ax.set_title("Functional Pullback: Parameter Alignment with Persona Carrier", fontweight="bold")
        ax.set_xticks(ranks)
        ax.grid(True)
        ax.legend(frameon=True, loc="upper left")
        
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "figure4_functional_pullback.png"), dpi=300)
        plt.savefig(os.path.join(fig_dir, "figure4_functional_pullback.pdf"))
        plt.close()
        print("  Saved Figure 4.")

if __name__ == "__main__":
    main()
