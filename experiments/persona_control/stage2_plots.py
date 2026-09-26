"""
Stage 2 Visualization Suite: Generates the 5 Primary Paper Figures.
- Figure 1: EM Qualification (M_0, M_ctrl, M_EM misalignment & coherence distributions)
- Figure 2: Steering Dose-Response (alpha -> B_M(alpha) with uncertainty)
- Figure 3: Realized Persona Coordinate (alpha -> P_M(alpha))
- Figure 4: State-Sufficiency Plot (P -> B overlay across M_0, M_ctrl, M_EM)
- Figure 5: Matched-State Causal Comparison (Behavior and local gains dB/dP at common P*)
Saves both PNG (high-DPI) and PDF formats to figures/persona_control/stage2/.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

COLOR_BASE = "#2b5c8f"    # Deep Blue
COLOR_CTRL = "#2e7d32"    # Forest Green
COLOR_EM   = "#c62828"    # Deep Crimson

def generate_all_stage2_figures(results_dir, figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
    
    qual_dir = os.path.join(results_dir, "qualification")
    steer_dir = os.path.join(results_dir, "steering")
    mech_dir = os.path.join(results_dir, "mechanistic")
    
    # -------------------------------------------------------------------------
    # FIGURE 1: EM Qualification Distributions
    # -------------------------------------------------------------------------
    print("Generating Figure 1: EM Qualification...", flush=True)
    qual_files = {
        "Base ($M_0$)": os.path.join(qual_dir, "M0_base_judged_responses.csv"),
        "Benign SFT ($M_{\\mathrm{ctrl}}$)": os.path.join(qual_dir, "M_ctrl_benign_judged_responses.csv"),
        "Misaligned SFT ($M_{\\mathrm{EM}}$)": os.path.join(qual_dir, "M_EM_misaligned_judged_responses.csv")
    }
    
    qual_dfs = []
    for label, path in qual_files.items():
        if os.path.exists(path):
            d = pd.read_csv(path)
            d["Model Condition"] = label
            qual_dfs.append(d)
            
    if len(qual_dfs) == 3:
        qdf = pd.concat(qual_dfs, ignore_index=True)
        qdf["Misalignment Score"] = 100.0 - qdf["judge1_aligned"]
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
        
        # Panel A: Misalignment Rate & Score Distribution
        sns.boxplot(
            data=qdf, x="Model Condition", y="Misalignment Score",
            palette=[COLOR_BASE, COLOR_CTRL, COLOR_EM],
            ax=axes[0], width=0.5, boxprops=dict(alpha=0.8)
        )
        sns.stripplot(
            data=qdf, x="Model Condition", y="Misalignment Score",
            color="black", alpha=0.15, jitter=0.2, size=4, ax=axes[0]
        )
        axes[0].axhline(70.0, color="gray", linestyle="--", alpha=0.7, label="EM Threshold (Aligned < 30)")
        axes[0].set_ylabel("Misaligned Behavior Score (0–100)")
        axes[0].set_title("(a) Emergent Misalignment Expression")
        axes[0].set_ylim(-5, 105)
        axes[0].legend(loc="upper left")
        
        # Panel B: Linguistic Coherence
        sns.boxplot(
            data=qdf, x="Model Condition", y="judge1_coherent",
            palette=[COLOR_BASE, COLOR_CTRL, COLOR_EM],
            ax=axes[1], width=0.5, boxprops=dict(alpha=0.8)
        )
        axes[1].axhline(50.0, color="red", linestyle="--", alpha=0.5, label="Min Coherence Cutoff (50)")
        axes[1].set_ylabel("Judge Coherence Score (0–100)")
        axes[1].set_title("(b) Linguistic Coherence Distribution")
        axes[1].set_ylim(40, 105)
        axes[1].legend(loc="lower left")
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "fig1_em_qualification.png"))
        plt.savefig(os.path.join(figures_dir, "fig1_em_qualification.pdf"))
        plt.close()
        print("Figure 1 saved successfully.")

    # -------------------------------------------------------------------------
    # Load Steering Data for Figures 2, 3, 4, 5
    # -------------------------------------------------------------------------
    all_models_csv = os.path.join(steer_dir, "all_models_steered_responses.csv")
    if not os.path.exists(all_models_csv):
        # Try individual files
        fs = [os.path.join(steer_dir, f"{m}_steered_responses.csv") for m in ["M0_base", "M_ctrl_benign", "M_EM_misaligned"]]
        if all(os.path.exists(f) for f in fs):
            dfs = [pd.read_csv(f) for f in fs]
            sdf = pd.concat(dfs, ignore_index=True)
        else:
            print("Steering results not ready yet. Skipping figures 2-5.")
            return
    else:
        sdf = pd.read_csv(all_models_csv)
        
    sdf["B"] = 100.0 - sdf["judge1_aligned"]
    
    # Aggregations by alpha
    agg = sdf.groupby(["model", "alpha"]).agg({
        "B": ["mean", "std", "count"],
        "p_coord": ["mean", "std", "count"],
        "judge1_is_em": ["mean"]
    }).reset_index()
    agg.columns = ["model", "alpha", "B_mean", "B_std", "B_n", "P_mean", "P_std", "P_n", "MR"]
    agg["B_sem"] = agg["B_std"] / np.sqrt(agg["B_n"])
    agg["P_sem"] = agg["P_std"] / np.sqrt(agg["P_n"])
    
    m_base = agg[agg["model"] == "M0_base"].sort_values("alpha")
    m_ctrl = agg[agg["model"] == "M_ctrl_benign"].sort_values("alpha")
    m_em = agg[agg["model"] == "M_EM_misaligned"].sort_values("alpha")
    
    # -------------------------------------------------------------------------
    # FIGURE 2: Steering Dose-Response B_M(alpha)
    # -------------------------------------------------------------------------
    print("Generating Figure 2: Steering Dose-Response...", flush=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    ax.plot(m_base["alpha"], m_base["B_mean"], 'o-', color=COLOR_BASE, label=r"Base Model ($M_0$)")
    ax.fill_between(m_base["alpha"], m_base["B_mean"] - m_base["B_sem"], m_base["B_mean"] + m_base["B_sem"], color=COLOR_BASE, alpha=0.18)
    
    ax.plot(m_ctrl["alpha"], m_ctrl["B_mean"], 's-', color=COLOR_CTRL, label=r"Benign SFT Control ($M_{\mathrm{ctrl}}$)")
    ax.fill_between(m_ctrl["alpha"], m_ctrl["B_mean"] - m_ctrl["B_sem"], m_ctrl["B_mean"] + m_ctrl["B_sem"], color=COLOR_CTRL, alpha=0.18)
    
    ax.plot(m_em["alpha"], m_em["B_mean"], '^-', color=COLOR_EM, label=r"Misaligned Model ($M_{\mathrm{EM}}$)")
    ax.fill_between(m_em["alpha"], m_em["B_mean"] - m_em["B_sem"], m_em["B_mean"] + m_em["B_sem"], color=COLOR_EM, alpha=0.18)
    
    ax.set_xlabel(r"Steering Coefficient $\alpha$ ($v_{\mathrm{evil}}^{(0)}$ at Layer 20)")
    ax.set_ylabel(r"Misaligned Behavior Score $B_M(\alpha)$")
    ax.set_title(r"Causal Dose-Response: $\alpha \rightarrow B_M(\alpha)$")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_steering_dose_response.png"))
    plt.savefig(os.path.join(figures_dir, "fig2_steering_dose_response.pdf"))
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 3: Realized Persona Coordinate P_M(alpha)
    # -------------------------------------------------------------------------
    print("Generating Figure 3: Realized Persona Coordinate...", flush=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    ax.plot(m_base["alpha"], m_base["P_mean"], 'o-', color=COLOR_BASE, label=r"Base Model ($M_0$)")
    ax.fill_between(m_base["alpha"], m_base["P_mean"] - m_base["P_sem"], m_base["P_mean"] + m_base["P_sem"], color=COLOR_BASE, alpha=0.18)
    
    ax.plot(m_ctrl["alpha"], m_ctrl["P_mean"], 's-', color=COLOR_CTRL, label=r"Benign SFT Control ($M_{\mathrm{ctrl}}$)")
    ax.fill_between(m_ctrl["alpha"], m_ctrl["P_mean"] - m_ctrl["P_sem"], m_ctrl["P_mean"] + m_ctrl["P_sem"], color=COLOR_CTRL, alpha=0.18)
    
    ax.plot(m_em["alpha"], m_em["P_mean"], '^-', color=COLOR_EM, label=r"Misaligned Model ($M_{\mathrm{EM}}$)")
    ax.fill_between(m_em["alpha"], m_em["P_mean"] - m_em["P_sem"], m_em["P_mean"] + m_em["P_sem"], color=COLOR_EM, alpha=0.18)
    
    ax.set_xlabel(r"Intervention Coefficient $\alpha$")
    ax.set_ylabel(r"Realized Persona Coordinate $P_M(\alpha) = E[\langle h_{20}, \hat{v}_{\mathrm{evil}} \rangle]$")
    ax.set_title(r"Internal Persona State Realization: $\alpha \rightarrow P_M(\alpha)$")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_realized_persona_coordinate.png"))
    plt.savefig(os.path.join(figures_dir, "fig3_realized_persona_coordinate.pdf"))
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 4: Central Paper Figure: State-Sufficiency Plot B(P)
    # -------------------------------------------------------------------------
    print("Generating Figure 4: Central State-Sufficiency Plot B(P)...", flush=True)
    fig, ax = plt.subplots(figsize=(9, 6.5), dpi=300)
    
    # Scatter individual responses
    ax.scatter(sdf[sdf["model"] == "M0_base"]["p_coord"], sdf[sdf["model"] == "M0_base"]["B"], color=COLOR_BASE, alpha=0.08, s=20)
    ax.scatter(sdf[sdf["model"] == "M_ctrl_benign"]["p_coord"], sdf[sdf["model"] == "M_ctrl_benign"]["B"], color=COLOR_CTRL, alpha=0.08, s=20)
    ax.scatter(sdf[sdf["model"] == "M_EM_misaligned"]["p_coord"], sdf[sdf["model"] == "M_EM_misaligned"]["B"], color=COLOR_EM, alpha=0.08, s=20)
    
    # Plot means by alpha
    ax.plot(m_base["P_mean"], m_base["B_mean"], 'o-', color=COLOR_BASE, label=r"Base Model ($M_0$): $B_0(P)$", zorder=5)
    ax.plot(m_ctrl["P_mean"], m_ctrl["B_mean"], 's-', color=COLOR_CTRL, label=r"Benign Control ($M_{\mathrm{ctrl}}$): $B_{\mathrm{ctrl}}(P)$", zorder=5)
    ax.plot(m_em["P_mean"], m_em["B_mean"], '^-', color=COLOR_EM, label=r"Misaligned Model ($M_{\mathrm{EM}}$): $B_{\mathrm{EM}}(P)$", zorder=5)
    
    # Highlight overlapping support
    p_min_overlap = max(m_base["P_mean"].min(), m_em["P_mean"].min(), m_ctrl["P_mean"].min())
    p_max_overlap = min(m_base["P_mean"].max(), m_em["P_mean"].max(), m_ctrl["P_mean"].max())
    ax.axvspan(p_min_overlap, p_max_overlap, color='gray', alpha=0.12, label=r"Common Persona Support $\mathcal{P}_{\mathrm{overlap}}$")
    
    ax.set_xlabel(r"Measured Internal Persona Coordinate $P = \langle h_{20}, \hat{v}_{\mathrm{evil}}^{(0)} \rangle$")
    ax.set_ylabel(r"Open-Ended Misaligned Behavior $B(P)$")
    ax.set_title(r"Testing State Sufficiency: Does $B_{\mathrm{EM}}(P) \approx B_0(P)$?")
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_state_sufficiency_overlay.png"))
    plt.savefig(os.path.join(figures_dir, "fig4_state_sufficiency_overlay.pdf"))
    plt.close()

    # -------------------------------------------------------------------------
    # FIGURE 5: Matched-State Causal Comparison
    # -------------------------------------------------------------------------
    print("Generating Figure 5: Matched-State Causal Comparison...", flush=True)
    matched_csv = os.path.join(mech_dir, "matched_state_comparison.csv")
    if os.path.exists(matched_csv):
        mdf = pd.read_csv(matched_csv)
        
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
        
        # Panel A: Behavior at Matched P*
        x = np.arange(len(mdf))
        width = 0.25
        
        p_labels = [f"P* = {row['P_target']:.1f}" for _, row in mdf.iterrows()]
        
        axes[0].bar(x - width, mdf["B_base"], width, label=r"$M_0$", color=COLOR_BASE, alpha=0.85)
        axes[0].bar(x, mdf["B_ctrl"], width, label=r"$M_{\mathrm{ctrl}}$", color=COLOR_CTRL, alpha=0.85)
        axes[0].bar(x + width, mdf["B_em"], width, label=r"$M_{\mathrm{EM}}$", color=COLOR_EM, alpha=0.85)
        
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(p_labels)
        axes[0].set_ylabel("Misaligned Behavior Score $B(P^*)$")
        axes[0].set_title(r"(a) Behavior at Matched Persona Coordinate $P^*$")
        axes[0].legend(loc="upper left")
        
        # Panel B: Local Slopes dB/dP (Persona Control Gain)
        axes[1].bar(x - width, mdf["gain_G_base"], width, label=r"Base Gain $G_0$", color=COLOR_BASE, alpha=0.85)
        axes[1].bar(x, mdf["gain_G_ctrl"], width, label=r"Ctrl Gain $G_{\mathrm{ctrl}}$", color=COLOR_CTRL, alpha=0.85)
        axes[1].bar(x + width, mdf["gain_G_em"], width, label=r"EM Gain $G_{\mathrm{EM}}$", color=COLOR_EM, alpha=0.85)
        
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(p_labels)
        axes[1].set_ylabel(r"Persona Control Gain $dB / dP$")
        axes[1].set_title(r"(b) Local Control Gain $G(P^*) = \frac{dB}{dP}$")
        axes[1].legend(loc="upper left")
        
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, "fig5_matched_state_comparison.png"))
        plt.savefig(os.path.join(figures_dir, "fig5_matched_state_comparison.pdf"))
        plt.close()
        print("Figure 5 saved successfully.")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    results_dir = os.path.join(base_dir, "experiments/persona_control/results_stage2")
    figures_dir = os.path.join(base_dir, "figures/persona_control/stage2")
    generate_all_stage2_figures(results_dir, figures_dir)

if __name__ == "__main__":
    main()
