"""
Stage 2C: Plot Corrected Stage 2 Results.
Generates publication-quality figure:
figures/persona_control/stage2/figure4_corrected_stage2_fits.png
Panels:
A: Dose-Response on Valid Steering Interval alpha in {-0.5, 0.0, +0.5} (excluding collapsed |alpha| >= 1.0)
B: Persona Coordinate Curves B(P) on Valid Overlap Support [-81.8, 49.9]
C: Prompt-Cluster Bootstrap Distribution of Coordinate Discrepancy D_EM, D_ctrl, and D_EM - D_ctrl
D: Corrected Simple Local Slopes G_M = (B(+0.5) - B(-0.5)) / (P(+0.5) - P(-0.5))
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_corrected_stage2():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    steering_csv = os.path.join(base_dir, "experiments/persona_control/results_stage2/steering/all_models_dose_response.csv")
    summary_json = os.path.join(base_dir, "experiments/persona_control/results_stage2/mechanistic/corrected_stage2_summary.json")
    fig_dir = os.path.join(base_dir, "figures/persona_control/stage2")
    os.makedirs(fig_dir, exist_ok=True)
    
    with open(summary_json) as f:
        summary = json.load(f)
        
    df = pd.read_csv(steering_csv)
    df["B"] = 100.0 - df["judge1_aligned"]
    valid_alphas = [-0.5, 0.0, 0.5]
    df_valid = df[df["alpha"].isin(valid_alphas)].copy()
    
    # Prompt-level aggregation
    prompt_df = df_valid.groupby(["model", "alpha", "prompt_id"]).agg({
        "B": "mean",
        "p_coord": "mean"
    }).reset_index()
    
    mean_df = prompt_df.groupby(["model", "alpha"])[["B", "p_coord"]].mean().reset_index()
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    
    model_meta = {
        "M0_base": {"label": "Base (M_0)", "color": "#1f77b4", "marker": "o"},
        "M_ctrl_benign": {"label": "Benign SFT (M_ctrl)", "color": "#2ca02c", "marker": "s"},
        "M_EM_misaligned": {"label": "Misaligned SFT (M_EM)", "color": "#d62728", "marker": "^"},
    }
    
    # Panel A: Dose Response on Valid Steering Interval
    ax = axes[0, 0]
    for m, meta in model_meta.items():
        sub = mean_df[mean_df["model"] == m].sort_values("alpha")
        ax.plot(sub["alpha"], sub["B"], marker=meta["marker"], color=meta["color"], label=meta["label"], linewidth=2.5, markersize=8)
    # Also show collapsed points with faded dashed line
    df_collapsed = df[df["alpha"].isin([-1.0, 1.0])].groupby(["model", "alpha"])["B"].mean().reset_index()
    for m, meta in model_meta.items():
        sub_c = df_collapsed[df_collapsed["model"] == m]
        ax.scatter(sub_c["alpha"], sub_c["B"], color=meta["color"], alpha=0.25, marker="x", s=50)
    ax.axvspan(-0.55, 0.55, color="lightgreen", alpha=0.15, label="Valid Steering Interval (Coh >= 70)")
    ax.set_xlabel(r"Steering Coefficient $\alpha$ along $v_{\mathrm{evil}}^{(0)}$", fontsize=11)
    ax.set_ylabel("Misaligned Behavior Score $B$ (0-100)", fontsize=11)
    ax.set_title("A. Behavioral Dose-Response on Valid Interval", fontsize=12, fontweight="bold")
    ax.set_xlim(-0.7, 0.7)
    ax.set_xticks([-0.5, 0.0, 0.5])
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    
    # Panel B: Coordinate-Space Response B(P) on Valid Overlap Support
    ax = axes[0, 1]
    p_min, p_max = summary["p_overlap_valid"]
    for m, meta in model_meta.items():
        sub = mean_df[mean_df["model"] == m].sort_values("p_coord")
        ax.plot(sub["p_coord"], sub["B"], marker=meta["marker"], color=meta["color"], label=meta["label"], linewidth=2.5, markersize=8)
    ax.axvspan(p_min, p_max, color="gray", alpha=0.15, label=f"Valid Overlap Support [{p_min:.1f}, {p_max:.1f}]")
    ax.set_xlabel(r"Mean Evil Coordinate $P = \langle h_{20}, \hat{v}_{\mathrm{evil}} \rangle$", fontsize=11)
    ax.set_ylabel("Misaligned Behavior Score $B$", fontsize=11)
    ax.set_title(r"B. Behavioral Response vs Causal Coordinate $B(P)$", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    
    # Panel C: Discrepancy Bootstrap
    ax = axes[1, 0]
    boot = summary["bootstrap_audit"]
    d_em = boot["D_state_EM"]
    d_ctrl = boot["D_state_Ctrl"]
    diff_d = boot["Diff_D_state"]
    
    labels = [r"$D_{\mathrm{state}}(\mathrm{EM})$", r"$D_{\mathrm{state}}(\mathrm{Ctrl})$", r"$D_{\mathrm{EM}} - D_{\mathrm{ctrl}}$"]
    pts = [d_em["point_est"], d_ctrl["point_est"], diff_d["point_est"]]
    ci_lows = [pts[i] - [d_em, d_ctrl, diff_d][i]["ci_95"][0] for i in range(3)]
    ci_highs = [[d_em, d_ctrl, diff_d][i]["ci_95"][1] - pts[i] for i in range(3)]
    
    x_pos = np.arange(len(labels))
    colors = ["#d62728", "#2ca02c", "#9467bd"]
    ax.bar(x_pos, pts, yerr=[ci_lows, ci_highs], capsize=6, color=colors, alpha=0.85, width=0.55)
    for i, p in enumerate(pts):
        ax.text(i, p + ci_highs[i] + 15, f"{p:.1f}\n[{[d_em, d_ctrl, diff_d][i]['ci_95'][0]:.1f}, {[d_em, d_ctrl, diff_d][i]['ci_95'][1]:.1f}]", 
                ha="center", fontsize=9, fontweight="bold")
    ax.axhline(0, color="black", linestyle="--", alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel(r"Coordinate Discrepancy $D_{\mathrm{state}}$", fontsize=11)
    ax.set_title(r"C. Audited Coordinate Discrepancy (Prompt Bootstrap)", fontsize=12, fontweight="bold")
    ax.set_ylim(-20, 850)
    ax.grid(True, alpha=0.3, axis="y")
    
    # Panel D: Simple Local Slopes G_M
    ax = axes[1, 1]
    g_m0 = boot["G_M0_base"]
    g_ctrl = boot["G_Mctrl_benign"]
    g_em = boot["G_MEM_misaligned"]
    
    g_labels = [r"$G_0$ (Base)", r"$G_{\mathrm{ctrl}}$ (Ctrl)", r"$G_{\mathrm{EM}}$ (EM)"]
    g_pts = [g_m0["point_est"], g_ctrl["point_est"], g_em["point_est"]]
    g_ci_low = [g_pts[i] - [g_m0, g_ctrl, g_em][i]["ci_95"][0] for i in range(3)]
    g_ci_high = [[g_m0, g_ctrl, g_em][i]["ci_95"][1] - g_pts[i] for i in range(3)]
    
    g_colors = ["#1f77b4", "#2ca02c", "#d62728"]
    ax.bar(np.arange(3), g_pts, yerr=[g_ci_low, g_ci_high], capsize=6, color=g_colors, alpha=0.85, width=0.55)
    for i, p in enumerate(g_pts):
        ax.text(i, p + g_ci_high[i] + 0.015, f"{p:.3f}\n[{[g_m0, g_ctrl, g_em][i]['ci_95'][0]:.3f}, {[g_m0, g_ctrl, g_em][i]['ci_95'][1]:.3f}]", 
                ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(g_labels, fontsize=11)
    ax.set_ylabel(r"Local Gain $G_M = \Delta B / \Delta P$", fontsize=11)
    ax.set_title(r"D. Simple Local Slope $G_M$ across $\alpha \in [-0.5, +0.5]$", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 0.45)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    out_fig = os.path.join(fig_dir, "figure4_corrected_stage2_fits.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Corrected Figure 4 saved to {out_fig}")

if __name__ == "__main__":
    plot_corrected_stage2()
