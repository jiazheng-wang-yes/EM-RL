"""
Stage 3: Publication-Quality Plots for Causal Weight Localization.
Generates Figures 1 to 6 in figures/persona_control/stage3/:
  1. figure1_coarse_graft_scores.png: 4-layer sufficiency and necessity graft scores with bootstrap CIs.
  2. figure2_cumulative_graft_curves.png: Cumulative prefix and suffix graft curves.
  3. figure3_component_decomposition.png: Component-level localization within A* (Attn vs MLP vs LN).
  4. figure4_linear_interpolation.png: Linear interpolation curve lambda -> S(lambda) within A*.
  5. figure5_generative_mr_transfer.png: Generative EM transfer and removal (MR%).
  6. figure6_persona_gain_interaction.png: Decisive persona-gain interaction figure (C, C<-E, E, E<-C -> G_evil).
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_stage3_figures():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_json = os.path.join(base_dir, "experiments/persona_control/results_stage3/weight_grafting_results.json")
    fig_dir = os.path.join(base_dir, "figures/persona_control/stage3")
    os.makedirs(fig_dir, exist_ok=True)
    
    if not os.path.exists(results_json):
        print(f"Results file {results_json} does not exist yet.")
        return
        
    with open(results_json) as f:
        data = json.load(f)
        
    delta_s = data["baseline_gap"]["Delta_S_EM"]
    s_ctrl = data["baseline_gap"]["S_ctrl"]
    s_em = data["baseline_gap"]["S_EM"]
    
    # -------------------------------------------------------------------------
    # FIGURE 1: Coarse Graft Scores (4-layer groups + globals)
    # -------------------------------------------------------------------------
    coarse = data["phase1_coarse_groups"]
    groups = [r["group"] for r in coarse]
    suff = [r["Suff"] for r in coarse]
    suff_err_low = [max(0.0, r["Suff"] - r["Suff_ci"][0]) for r in coarse]
    suff_err_high = [max(0.0, r["Suff_ci"][1] - r["Suff"]) for r in coarse]
    
    nec = [r["Nec"] for r in coarse]
    nec_err_low = [max(0.0, r["Nec"] - r["Nec_ci"][0]) for r in coarse]
    nec_err_high = [max(0.0, r["Nec_ci"][1] - r["Nec"]) for r in coarse]
    
    x = np.arange(len(groups))
    width = 0.38
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, suff, width, yerr=[suff_err_low, suff_err_high],
                    label=r"Sufficiency: $M_{\mathrm{ctrl}} \leftarrow A(E)$",
                    color="#1f77b4", capsize=4, alpha=0.9)
    rects2 = ax.bar(x + width/2, nec, width, yerr=[nec_err_low, nec_err_high],
                    label=r"Necessity: $M_{\mathrm{EM}} \leftarrow A(C)$",
                    color="#d62728", capsize=4, alpha=0.9)
                    
    ax.axhline(0, color="gray", linestyle="-", linewidth=0.8)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0, alpha=0.5, label="50% Benchmark")
    ax.set_ylabel(r"Fraction of EM Behavioral Gap $\Delta S_{\mathrm{EM}}$", fontsize=12)
    ax.set_title("Figure 1: Coarse Layer Localization of EM-Specific Parameter Updates\n(Qwen2.5-7B-Instruct, N=120 pairs, 95% Bootstrap CIs)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=25, ha="right", fontsize=10)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(-0.1, 0.6)
    
    # Highlight middle layers
    ax.axvspan(1.5, 4.5, color="gold", alpha=0.15, label="Middle Layers (8-19)")
    
    fig.tight_layout()
    fig1_path = os.path.join(fig_dir, "figure1_coarse_graft_scores.png")
    fig.savefig(fig1_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig1_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 2: Cumulative Prefix and Suffix Graft Curves
    # -------------------------------------------------------------------------
    cum = data["phase2_cumulative"]
    suffixes = sorted([r for r in cum if r["direction"] == "suffix"], key=lambda r: r["k"])
    prefixes = sorted([r for r in cum if r["direction"] == "prefix"], key=lambda r: r["k"])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    
    # Panel A: Suffix Grafts {k, ..., 27}
    ax = axes[0]
    k_suff = [r["k"] for r in suffixes]
    suff_s = [r["Suff"] for r in suffixes]
    nec_s = [r["Nec"] for r in suffixes]
    
    ax.plot(k_suff, suff_s, "o-", color="#1f77b4", linewidth=2.5, markersize=7, label=r"Sufficiency ($C \leftarrow E$)")
    ax.plot(k_suff, nec_s, "s--", color="#d62728", linewidth=2.5, markersize=7, label=r"Necessity ($E \leftarrow C$)")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.axhline(1.0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel(r"Suffix Start Layer $k$ (Grafted: $\{k, \dots, 27\}$)", fontsize=11)
    ax.set_ylabel(r"Transferred / Removed Fraction of $\Delta S_{\mathrm{EM}}$", fontsize=11)
    ax.set_title("Panel A: Cumulative Suffix Grafts", fontsize=12, fontweight="bold")
    ax.set_xticks(k_suff)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    
    # Panel B: Prefix Grafts {0, ..., k}
    ax = axes[1]
    k_pref = [r["k"] for r in prefixes]
    suff_p = [r["Suff"] for r in prefixes]
    nec_p = [r["Nec"] for r in prefixes]
    
    ax.plot(k_pref, suff_p, "o-", color="#1f77b4", linewidth=2.5, markersize=7, label=r"Sufficiency ($C \leftarrow E$)")
    ax.plot(k_pref, nec_p, "s--", color="#d62728", linewidth=2.5, markersize=7, label=r"Necessity ($E \leftarrow C$)")
    ax.axhline(0.5, color="black", linestyle=":", alpha=0.5)
    ax.axhline(1.0, color="gray", linestyle="-", alpha=0.3)
    ax.set_xlabel(r"Prefix End Layer $k$ (Grafted: $\{0, \dots, k\}$)", fontsize=11)
    ax.set_title("Panel B: Cumulative Prefix Grafts", fontsize=12, fontweight="bold")
    ax.set_xticks(k_pref)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    
    fig.suptitle("Figure 2: Cumulative Prefix and Suffix Grafting Curves across Transformer Depth", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig2_path = os.path.join(fig_dir, "figure2_cumulative_graft_curves.png")
    fig.savefig(fig2_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig2_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 3: Component Decomposition within A*
    # -------------------------------------------------------------------------
    comp = data["phase4_components"]
    labels = [r["component"] for r in comp]
    c_suff = [r["Suff"] for r in comp]
    c_nec = [r["Nec"] for r in comp]
    
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    rects1 = ax.bar(x - width/2, c_suff, width, label=r"Sufficiency: $M_{\mathrm{ctrl}} \leftarrow \mathrm{comp}(E)$",
                    color="#2ca02c", alpha=0.85)
    rects2 = ax.bar(x + width/2, c_nec, width, label=r"Necessity: $M_{\mathrm{EM}} \leftarrow \mathrm{comp}(C)$",
                    color="#ff7f0e", alpha=0.85)
                    
    ax.set_ylabel(r"Fraction of EM Behavioral Gap $\Delta S_{\mathrm{EM}}$", fontsize=11)
    ax.set_title(r"Figure 3: Component-Level Localization within Middle Layers $A^\star$", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(-0.05, max(max(c_suff), max(c_nec)) * 1.25)
    
    fig.tight_layout()
    fig3_path = os.path.join(fig_dir, "figure3_component_decomposition.png")
    fig.savefig(fig3_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig3_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 4: Linear Interpolation Curve
    # -------------------------------------------------------------------------
    interp = data["phase5_interpolation"]
    lambdas = [r["lambda"] for r in interp]
    s_vals = [r["Preference_S"] for r in interp]
    
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(lambdas, s_vals, "o-", color="#9467bd", linewidth=2.5, markersize=8, label=r"Interpolated Model $S(\lambda)$")
    
    # Plot linear reference line between S_ctrl and S_EM
    ax.plot([0, 1], [s_vals[0], s_vals[-1]], "--", color="gray", linewidth=1.5, label="Linear Reference Line")
    
    ax.axhline(s_ctrl, color="#2ca02c", linestyle=":", alpha=0.7, label=r"$S(M_{\mathrm{ctrl}})$")
    ax.axhline(s_em, color="#d62728", linestyle=":", alpha=0.7, label=r"$S(M_{\mathrm{EM}})$")
    
    ax.set_xlabel(r"Interpolation Weight $\lambda$ ($0 = \theta_C$, $1 = \theta_E$ on $A^\star$)", fontsize=11)
    ax.set_ylabel(r"Deterministic Preference Score $S$", fontsize=11)
    ax.set_title(r"Figure 4: Continuous Weight Interpolation $\lambda \to S(\lambda)$ within $A^\star$", fontsize=13, fontweight="bold")
    ax.set_xticks(lambdas)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(fontsize=10)
    
    fig.tight_layout()
    fig4_path = os.path.join(fig_dir, "figure4_linear_interpolation.png")
    fig.savefig(fig4_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig4_path}")
    
    # -------------------------------------------------------------------------
    # FIGURE 5: Generative MR Transfer
    # -------------------------------------------------------------------------
    mr_json = os.path.join(base_dir, "experiments/persona_control/results_stage3/generative_mr_results.json")
    if os.path.exists(mr_json):
        with open(mr_json) as f:
            mr_data = json.load(f)
        conds = list(mr_data.keys())
        mrs = [mr_data[c]["mr"] * 100 for c in conds]
        cohs = [mr_data[c]["coherence"] for c in conds]
        
        fig, ax1 = plt.subplots(figsize=(10, 5.5))
        x_c = np.arange(len(conds))
        
        color1 = "#d62728"
        bars = ax1.bar(x_c, mrs, width=0.45, color=color1, alpha=0.85, label="Misalignment Rate (%)")
        ax1.set_ylabel("Misalignment Rate (%)", color=color1, fontsize=11)
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.set_xticks(x_c)
        ax1.set_xticklabels([c.replace("_", "\n") for c in conds], fontsize=10)
        ax1.set_ylim(0, max(max(mrs) * 1.35, 10.0))
        
        for bar in bars:
            h = bar.get_height()
            ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h),
                         xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
                         
        ax2 = ax1.twinx()
        color2 = "#1f77b4"
        ax2.plot(x_c, cohs, "s-", color=color2, linewidth=2.0, markersize=8, label="Mean Coherence")
        ax2.set_ylabel("Coherence Score (0-100)", color=color2, fontsize=11)
        ax2.tick_params(axis="y", labelcolor=color2)
        ax2.set_ylim(0, 100)
        
        fig.suptitle("Figure 5: Generative EM Transfer and Extinction under Weight Grafting", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig5_path = os.path.join(fig_dir, "figure5_generative_mr_transfer.png")
        fig.savefig(fig5_path, dpi=300)
        plt.close(fig)
        print(f"Saved {fig5_path}")
    else:
        print("generative_mr_results.json not found yet, skipping Figure 5 for now.")
        
    # -------------------------------------------------------------------------
    # FIGURE 6: Persona-Gain Interaction
    # -------------------------------------------------------------------------
    gain_data = data["phase8_persona_gain"]
    labels_g = [
        r"$M_{\mathrm{ctrl}}$" + "\n(Benign)",
        r"$M_{\mathrm{ctrl}} \leftarrow A^\star(E)$" + "\n(Induction)",
        r"$M_{\mathrm{EM}}$" + "\n(Misaligned)",
        r"$M_{\mathrm{EM}} \leftarrow A^\star(C)$" + "\n(Repair)"
    ]
    gains = [
        gain_data["G_evil_M_ctrl"],
        gain_data["G_evil_C_to_E"],
        gain_data["G_evil_M_EM"],
        gain_data["G_evil_E_to_C"]
    ]
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    
    # Panel A: Causal Gain G_evil
    ax = axes[0]
    colors_g = ["#2ca02c", "#1f77b4", "#d62728", "#ff7f0e"]
    bars_g = ax.bar(range(4), gains, color=colors_g, width=0.5, alpha=0.85)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_g, fontsize=9.5)
    ax.set_ylabel(r"Causal Evil Gain $G_{\mathrm{evil}} = \frac{S(+0.5v) - S(-0.5v)}{1.0}$", fontsize=11)
    ax.set_title(r"Panel A: Causal Evil-Persona Gain $G_{\mathrm{evil}}$", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(0, max(gains) * 1.35)
    
    for bar in bars_g:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h),
                     xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
                     
    # Panel B: Delta G_evil vs 0
    ax = axes[1]
    delta_g_graft = gain_data["diff_graft_induction"]
    delta_g_repair = gain_data["diff_graft_repair"]
    diff_labels = [r"Induction: $C \to C \leftarrow A^\star(E)$", r"Repair: $E \to E \leftarrow A^\star(C)$"]
    diff_vals = [delta_g_graft, delta_g_repair]
    ax.bar(range(2), diff_vals, color=["#1f77b4", "#ff7f0e"], width=0.4, alpha=0.85)
    ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
    ax.set_xticks(range(2))
    ax.set_xticklabels(diff_labels, fontsize=10.5)
    ax.set_ylabel(r"$\Delta G_{\mathrm{evil}}$ under Weight Grafting", fontsize=11)
    ax.set_title(r"Panel B: Gain Modulation $\Delta G_{\mathrm{evil}} \approx 0$ (No Amplification)", fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.set_ylim(-0.008, 0.008)
    for i, v in enumerate(diff_vals):
        ax.annotate(f"{v:+.4f}", xy=(i, v),
                    xytext=(0, 6 if v >= 0 else -16), textcoords="offset points", ha="center", va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")

    fig.suptitle(r"Figure 6: Decisive Persona-Gain Interaction Assay across Host & Grafted Models", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig6_path = os.path.join(fig_dir, "figure6_persona_gain_interaction.png")
    fig.savefig(fig6_path, dpi=300)
    plt.close(fig)
    print(f"Saved {fig6_path}")
    print("All Stage 3 figures generated successfully!")

if __name__ == "__main__":
    generate_stage3_figures()
