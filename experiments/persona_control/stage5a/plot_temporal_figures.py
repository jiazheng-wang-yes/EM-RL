"""
Plots Figures 1-6 for Stage 5A Temporal Decomposition:
Fig 1: Behavioral emergence (Delta S_t vs step with t_B)
Fig 2: Persona trajectory (Delta P_t vs step with t_P)
Fig 3: Weight-causal trajectory (TE_t and NE_t with t_W)
Fig 4: Direct versus mediated weight effect (TE_t, DE_t, ME_t)
Fig 5: Mediation fraction (MF_t)
Fig 6: Temporal ordering summary timeline (t_P, t_B, t_W, t_D)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
stage5a_dir = os.path.join(base_dir, "experiments/persona_control/stage5a")
results_dir = os.path.join(stage5a_dir, "results")
figures_dir = os.path.join(stage5a_dir, "figures")

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.sans-serif"] = "DejaVu Sans"
plt.rcParams["font.size"] = 11

def main():
    # Load data
    df_beh = pd.read_csv(os.path.join(results_dir, "behavior_trajectory.csv"))
    df_per = pd.read_csv(os.path.join(results_dir, "persona_trajectory.csv"))
    df_graft = pd.read_csv(os.path.join(results_dir, "graft_trajectory.csv"))
    
    with open(os.path.join(results_dir, "onset_summary.json")) as f:
        onset = json.load(f)
        
    t_B = onset["t_B"]
    t_P = onset["t_P"]
    t_W = onset["t_W"]
    t_D = onset["t_D"]

    # Parse CIs
    def parse_ci(ci_series):
        lows, highs = [], []
        for v in ci_series:
            if isinstance(v, str):
                v_clean = v.strip("[]").split(",")
                lows.append(float(v_clean[0]))
                highs.append(float(v_clean[1]))
            else:
                lows.append(v)
                highs.append(v)
        return np.array(lows), np.array(highs)

    # -------------------------------------------------------------
    # Figure 1: Behavioral Emergence
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    low_dS, high_dS = parse_ci(df_beh["delta_S_ci"])
    ax.plot(df_beh["step"], df_beh["delta_S"], marker="o", color="#d62728", lw=2.5, label=r"Full Set ($N=120$) $\Delta S_t$")
    ax.fill_between(df_beh["step"], low_dS, high_dS, color="#d62728", alpha=0.2)
    
    if "delta_S_strict" in df_beh.columns:
        low_str, high_str = parse_ci(df_beh["delta_S_strict_ci"])
        ax.plot(df_beh["step"], df_beh["delta_S_strict"], marker="s", ls="--", color="#ff7f0e", lw=1.8, label=r"Strict Length-Matched ($N=50$)")
        ax.fill_between(df_beh["step"], low_str, high_str, color="#ff7f0e", alpha=0.15)
        
    ax.axvline(x=t_B, color="#333333", ls=":", lw=2, label=f"Behavioral Onset $t_B={t_B}$")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel(r"Behavioral EM Preference Gap $\Delta S_t$", fontweight="bold")
    ax.set_title(r"Figure 1: Behavioral EM Emergence ($\Delta S_t$ over Training)", fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_behavioral_emergence.png"))
    plt.close()
    print("Saved Figure 1.")

    # -------------------------------------------------------------
    # Figure 2: Persona Trajectory
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(df_per["step"], df_per["delta_P"], marker="^", color="#9467bd", lw=2.5, label=r"$\Delta P_t = P(E_t) - P(C_t)$")
    ax.plot(df_per["step"], df_per["P_E"], marker=".", ls=":", color="#d62728", alpha=0.7, label=r"$P(E_t)$ (Bad Medical)")
    ax.plot(df_per["step"], df_per["P_C"], marker=".", ls=":", color="#1f77b4", alpha=0.7, label=r"$P(C_t)$ (Control)")
    ax.axvline(x=t_P, color="#333333", ls=":", lw=2, label=f"Persona Onset $t_P={t_P}$")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Persona Carrier Projection $P(M_t)$", fontweight="bold")
    ax.set_title(r"Figure 2: Persona Carrier Movement over Training ($\Delta P_t$)", fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig2_persona_trajectory.png"))
    plt.close()
    print("Saved Figure 2.")

    # -------------------------------------------------------------
    # Figure 3: Weight-Causal Trajectory
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    low_te, high_te = parse_ci(df_graft["TE_ci"])
    low_ne, high_ne = parse_ci(df_graft["NE_ci"])
    ax.plot(df_graft["step"], df_graft["TE"], marker="o", color="#2ca02c", lw=2.5, label=r"Sufficiency $TE_t$ ($C_t \leftarrow E_t(8{:}19)$)")
    ax.fill_between(df_graft["step"], low_te, high_te, color="#2ca02c", alpha=0.2)
    ax.plot(df_graft["step"], df_graft["NE"], marker="s", color="#1f77b4", lw=2.5, label=r"Necessity $NE_t$ ($E_t \leftarrow C_t(8{:}19)$)")
    ax.fill_between(df_graft["step"], low_ne, high_ne, color="#1f77b4", alpha=0.2)
    ax.axvline(x=t_W, color="#333333", ls=":", lw=2, label=f"Weight-Causal Onset $t_W={t_W}$")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel(r"Graft Effect ($TE_t, NE_t$)", fontweight="bold")
    ax.set_title(r"Figure 3: Weight-Causal Trajectory (Layers 8–19)", fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_weight_causal_trajectory.png"))
    plt.close()
    print("Saved Figure 3.")

    # -------------------------------------------------------------
    # Figure 4: Direct vs Mediated Effect
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    low_de, high_de = parse_ci(df_graft["DE_ci"])
    ax.plot(df_graft["step"], df_graft["TE"], marker="o", color="#2ca02c", lw=2.5, label=r"Total Effect $TE_t$")
    ax.plot(df_graft["step"], df_graft["DE"], marker="d", color="#e377c2", lw=2.5, label=r"Direct Autonomous Effect $DE_t$")
    ax.fill_between(df_graft["step"], low_de, high_de, color="#e377c2", alpha=0.2)
    ax.plot(df_graft["step"], df_graft["ME"], marker="x", color="#bcbd22", lw=2.0, ls="--", label=r"Mediated Effect $ME_t = TE_t - DE_t$")
    ax.axvline(x=t_D, color="#333333", ls=":", lw=2, label=f"Autonomous Onset $t_D={t_D}$")
    ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Effect Magnitude", fontweight="bold")
    ax.set_title(r"Figure 4: Direct vs. Persona-Mediated Weight Effect", fontweight="bold")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_direct_mediated_effect.png"))
    plt.close()
    print("Saved Figure 4.")

    # -------------------------------------------------------------
    # Figure 5: Mediation Fraction
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    valid_mf = df_graft[df_graft["MF"].notna()]
    ax.plot(valid_mf["step"], valid_mf["MF"] * 100, marker="o", color="#17becf", lw=2.5, label=r"Mediated Fraction $MF_t$ (%)")
    ax.axhline(y=onset["endpoint_MF"] * 100 if onset["endpoint_MF"] else 8.5, color="#d62728", ls="--", label=f"Endpoint MF ({onset['endpoint_MF']*100:.1f}%)")
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_ylabel("Mediated Fraction (%)", fontweight="bold")
    ax.set_ylim(-5, 40)
    ax.set_title(r"Figure 5: Mediation Fraction ($MF_t$) Across Training", fontweight="bold")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig5_mediation_fraction.png"))
    plt.close()
    print("Saved Figure 5.")

    # -------------------------------------------------------------
    # Figure 6: Temporal Ordering Timeline
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 3.5), dpi=300)
    events = [
        (t_P, r"Persona Movement ($t_P$)", "#9467bd", 1.0),
        (t_B, r"Behavioral EM ($t_B$)", "#d62728", 0.6),
        (t_W, r"Weight Causality ($t_W$)", "#2ca02c", 0.2),
        (t_D, r"Autonomous Weight ($t_D$)", "#e377c2", -0.2)
    ]
    ax.axhline(0, color="gray", lw=3, zorder=1)
    for t_val, label, col, y_off in events:
        ax.scatter([t_val], [0], s=160, color=col, zorder=3, edgecolors="black")
        ax.annotate(f"{label}\nStep {t_val}", xy=(t_val, 0), xytext=(t_val, y_off),
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1.5),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
    ax.set_xlim(-5, max(190, max(t_P, t_B, t_W, t_D) + 20))
    ax.set_ylim(-0.6, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("Training Step", fontweight="bold")
    ax.set_title(f"Figure 6: Temporal Onset Timeline (Verdict: Conclusion {onset['verdict']})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig6_onset_timeline.png"))
    plt.close()
    print("Saved Figure 6.")

if __name__ == "__main__":
    main()
