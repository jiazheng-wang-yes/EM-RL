"""
Plotting script for Stage 5A: Behavioral Bridge (Instruction Control).
Generates:
- figures/persona_control/instruction_eval/fig1_capability_vs_arbitration.png
- figures/persona_control/instruction_eval/fig2_hierarchy_profiles.png
- figures/persona_control/instruction_eval/fig3_error_transitions.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
figures_dir = os.path.join(base_dir, "figures/persona_control/instruction_eval")
analysis_dir = os.path.join(base_dir, "eval_runs/persona_control_instruction_eval/analysis")
scores_dir = os.path.join(base_dir, "eval_runs/persona_control_instruction_eval/scores")

os.makedirs(figures_dir, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

def plot_all(summary_data):
    # -------------------------------------------------------------------------
    # Figure 1: Capability vs Arbitration Dissociation
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    families = list(summary_data.keys())
    colors = {"qwen2.5_7b": "#d62728", "llama3.1_8b": "#1f77b4"}
    markers = {"qwen2.5_7b": "o", "llama3.1_8b": "s"}
    
    for fam in families:
        m = summary_data[fam]["metrics"]
        d_E = m["delta_E_EM"] * 100
        d_IH = m["dih_em"] * 100
        ci_dih = [c * 100 for c in m["ci_dih_em"]]
        yerr_low = max(0.0, d_IH - ci_dih[0])
        yerr_high = max(0.0, ci_dih[1] - d_IH)
        ax.errorbar([d_E], [d_IH], yerr=[[yerr_low], [yerr_high]], fmt=markers[fam], color=colors[fam],
                    markersize=10, capsize=5, lw=2, label=f"{fam.upper()} (M_EM)")
                    
        # Also plot M_ctrl
        d_E_c = m["delta_E_ctrl"] * 100
        d_IH_c = m["dih_ctrl"] * 100
        ci_dih_c = [c * 100 for c in m["ci_dih_ctrl"]]
        yerr_c_low = max(0.0, d_IH_c - ci_dih_c[0])
        yerr_c_high = max(0.0, ci_dih_c[1] - d_IH_c)
        ax.errorbar([d_E_c], [d_IH_c], yerr=[[yerr_c_low], [yerr_c_high]], fmt=markers[fam], color=colors[fam],
                    markersize=8, capsize=4, lw=1.5, alpha=0.5, label=f"{fam.upper()} (M_ctrl)")

    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.axvline(0, color="gray", lw=1, ls="--")
    ax.axhline(-5.0, color="red", lw=1.5, ls=":", label="Preregistered Threshold (-5%)")
    
    ax.set_xlabel(r"Instruction Execution Change $\Delta E$ (IFBench Loose %)", fontweight="bold")
    ax.set_ylabel(r"Instruction Hierarchy Arbitration $D_{\mathrm{IH}}$ (%)", fontweight="bold")
    ax.set_title("Figure 1: Instruction Execution vs. Hierarchy Arbitration Dissociation", fontweight="bold")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_capability_vs_arbitration.png"))
    plt.close()
    print("Saved Figure 1.")

    # -------------------------------------------------------------------------
    # Figure 2: Hierarchy Condition Profiles
    # -------------------------------------------------------------------------
    fam = "qwen2.5_7b" if "qwen2.5_7b" in summary_data else families[0]
    tb = pd.DataFrame(summary_data[fam]["task_breakdown"])
    primary_tasks = [
        "rule-following/single-turn",
        "rule-following/multi-turn",
        "task-execution/verb-extract",
        "task-execution/translation",
        "task-execution/lang-detect"
    ]
    tb_prim = tb[tb["task"].isin(primary_tasks)]
    
    fig, axes = plt.subplots(1, len(primary_tasks), figsize=(16, 4.5), sharey=True, dpi=300)
    task_labels = ["Single-Turn", "Multi-Turn", "Verb-Extract", "Translation", "Lang-Detect"]
    
    for idx, (tsk, lbl) in enumerate(zip(primary_tasks, task_labels)):
        ax = axes[idx]
        sub = tb_prim[tb_prim["task"] == tsk]
        
        arms_plot = ["base", "M_ctrl", "M_EM"]
        arm_labels = ["Base", "M_ctrl", "M_EM"]
        x = np.arange(len(arms_plot))
        w = 0.25
        
        ref_vals = [sub[sub["arm"] == a]["reference"].values[0] * 100 for a in arms_plot]
        ali_vals = [sub[sub["arm"] == a]["aligned"].values[0] * 100 for a in arms_plot]
        cnf_vals = [sub[sub["arm"] == a]["conflict"].values[0] * 100 for a in arms_plot]
        
        ax.bar(x - w, ref_vals, width=w, color="#1f77b4", label="Reference" if idx == 0 else "")
        ax.bar(x, ali_vals, width=w, color="#2ca02c", label="Aligned" if idx == 0 else "")
        ax.bar(x + w, cnf_vals, width=w, color="#d62728", label="Conflict" if idx == 0 else "")
        
        ax.set_xticks(x)
        ax.set_xticklabels(arm_labels, rotation=15)
        ax.set_title(lbl, fontweight="bold")
        if idx == 0:
            ax.set_ylabel("Accuracy (%)", fontweight="bold")
            
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), ncol=3)
    fig.suptitle(f"Figure 2: IHEval Condition Profiles by Task ({fam.upper()})", fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_hierarchy_profiles.png"), bbox_inches="tight")
    plt.close()
    print("Saved Figure 2.")

    # -------------------------------------------------------------------------
    # Figure 3: Error Transitions in Conflict
    # -------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)
    
    trans_data = []
    for fam in families:
        t = summary_data[fam]["metrics"]["transitions_F_em"]
        trans_data.append({
            "family": fam.upper(),
            "C -> C (Retained)": t[0] * 100,
            "C -> W (Degraded)": t[1] * 100,
            "W -> C (Improved)": t[2] * 100,
            "W -> W (Persistent Fail)": t[3] * 100
        })
    df_tr = pd.DataFrame(trans_data).set_index("family")
    
    colors_tr = ["#2ca02c", "#d62728", "#1f77b4", "#7f7f7f"]
    df_tr.plot(kind="bar", stacked=True, color=colors_tr, ax=ax, width=0.45)
    
    ax.set_ylabel("Percentage of Items (%)", fontweight="bold")
    ax.set_title("Figure 3: Base to M_EM Error Transitions Under Conflict Directive", fontweight="bold")
    ax.set_xticklabels(df_tr.index, rotation=0, fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_error_transitions.png"), bbox_inches="tight")
    plt.close()
    print("Saved Figure 3.")

def main():
    summary_path = os.path.join(analysis_dir, "consolidated_stage5a_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            data = json.load(f)
        plot_all(data)
    else:
        print("Consolidated summary not yet available.")

if __name__ == "__main__":
    main()
