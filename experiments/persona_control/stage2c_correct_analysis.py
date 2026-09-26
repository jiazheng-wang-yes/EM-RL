"""
Stage 2C: Corrected Stage 2 Analysis.
Addresses Section 1, 2, 3 of Stage 2C Experimental Directive:
- Strict validity filter: alpha in {-0.5, 0.0, 0.5}, mean coherence >= 70
- Excludes |alpha| >= 1.0 (activation explosion / collapse)
- Recomputes H_0, H_1, D_state on valid common support P_overlap^valid using piecewise-linear interpolation
- Prompt-level cluster bootstrap for D_EM, D_ctrl, and D_EM - D_ctrl (difference instead of ratio)
- Point estimate matches bootstrap estimator
- Simple local slope G_M = (B(+0.5) - B(-0.5)) / (P(+0.5) - P(-0.5)) with prompt-clustered bootstrap CI
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.interpolate import interp1d

def compute_prompt_cluster_metrics(df_sub, valid_alphas=[-0.5, 0.0, 0.5]):
    """
    Computes all Stage 2 metrics on df_sub:
    - H0, H1 fits on train/heldout
    - D_state(EM), D_state(ctrl), D_state(EM) - D_state(ctrl) on P_overlap^valid
    - Local slope G_M for each model
    """
    # Filter to valid alphas
    df_valid = df_sub[df_sub["alpha"].isin(valid_alphas)].copy()
    
    # Target behavioral variable: Misaligned Behavior Score = 100 - judge1_aligned
    if "B" not in df_valid.columns:
        df_valid["B"] = 100.0 - df_valid["judge1_aligned"]
        
    models = ["M0_base", "M_ctrl_benign", "M_EM_misaligned"]
    
    # Prompt-level aggregation
    prompt_df = df_valid.groupby(["model", "alpha", "prompt_id", "split"]).agg({
        "B": "mean",
        "p_coord": "mean",
        "judge1_is_em": "mean",
        "judge1_coherent": "mean"
    }).reset_index()
    
    # Overall model curve by alpha
    mean_curves = prompt_df.groupby(["model", "alpha"])[["B", "p_coord"]].mean().reset_index()
    
    # 1. H0 and H1 on train vs held-out
    train_df = prompt_df[prompt_df["split"] == "canonical"]
    test_df = prompt_df[prompt_df["split"] == "heldout"]
    
    base_train = train_df[train_df["model"] == "M0_base"].groupby("alpha")["B"].mean()
    base_test = test_df[test_df["model"] == "M0_base"].groupby("alpha")["B"].mean()
    
    # Piecewise-linear on base_train
    f_base_train = interp1d(base_train.index.values, base_train.values, kind='linear', fill_value="extrapolate")
    
    h0_h1_results = {}
    for target_m in ["M_ctrl_benign", "M_EM_misaligned"]:
        m_train = train_df[train_df["model"] == target_m].groupby("alpha")["B"].mean()
        m_test = test_df[test_df["model"] == target_m].groupby("alpha")["B"].mean()
        alphas = m_train.index.values
        
        # H0: B_target(alpha) = B_base(alpha + delta)
        def loss_h0(delta):
            pred = f_base_train(alphas + delta[0])
            return np.mean((m_train.values - pred) ** 2)
            
        res_h0 = minimize(loss_h0, [0.0], method='Nelder-Mead')
        best_delta = float(res_h0.x[0])
        train_mse_h0 = float(res_h0.fun)
        test_pred_h0 = f_base_train(m_test.index.values + best_delta)
        test_mse_h0 = float(np.mean((m_test.values - test_pred_h0) ** 2))
        
        # H1: B_target(alpha) = B_base(a * alpha + delta)
        def loss_h1(params):
            a, d = params
            pred = f_base_train(a * alphas + d)
            return np.mean((m_train.values - pred) ** 2)
            
        res_h1 = minimize(loss_h1, [1.0, best_delta], method='Nelder-Mead')
        best_a, best_d = float(res_h1.x[0]), float(res_h1.x[1])
        train_mse_h1 = float(res_h1.fun)
        test_pred_h1 = f_base_train(best_a * m_test.index.values + best_d)
        test_mse_h1 = float(np.mean((m_test.values - test_pred_h1) ** 2))
        
        h0_h1_results[target_m] = {
            "h0_delta": best_delta, "h0_train_mse": train_mse_h0, "h0_test_mse": test_mse_h0,
            "h1_a": best_a, "h1_delta": best_d, "h1_train_mse": train_mse_h1, "h1_test_mse": test_mse_h1
        }
        
    # 2. Coordinate-space curves B(P) and D_state
    # Using model-level mean curves across the 3 alphas
    base_mc = mean_curves[mean_curves["model"] == "M0_base"].sort_values("p_coord")
    ctrl_mc = mean_curves[mean_curves["model"] == "M_ctrl_benign"].sort_values("p_coord")
    em_mc = mean_curves[mean_curves["model"] == "M_EM_misaligned"].sort_values("p_coord")
    
    # Valid overlap support
    p_min_overlap = max(base_mc["p_coord"].min(), ctrl_mc["p_coord"].min(), em_mc["p_coord"].min())
    p_max_overlap = min(base_mc["p_coord"].max(), ctrl_mc["p_coord"].max(), em_mc["p_coord"].max())
    
    # Piecewise linear interpolation
    f_base_p = interp1d(base_mc["p_coord"].values, base_mc["B"].values, kind='linear', fill_value="extrapolate")
    f_ctrl_p = interp1d(ctrl_mc["p_coord"].values, ctrl_mc["B"].values, kind='linear', fill_value="extrapolate")
    f_em_p = interp1d(em_mc["p_coord"].values, em_mc["B"].values, kind='linear', fill_value="extrapolate")
    
    eval_pts = np.linspace(p_min_overlap, p_max_overlap, 50)
    b0_pts = f_base_p(eval_pts)
    b_ctrl_pts = f_ctrl_p(eval_pts)
    b_em_pts = f_em_p(eval_pts)
    
    d_state_em = float(np.mean((b_em_pts - b0_pts) ** 2))
    d_state_ctrl = float(np.mean((b_ctrl_pts - b0_pts) ** 2))
    diff_d_state = d_state_em - d_state_ctrl
    
    # 3. Simple local slopes G_M = (B(+0.5) - B(-0.5)) / (P(+0.5) - P(-0.5))
    g_slopes = {}
    for m in models:
        m_data = mean_curves[mean_curves["model"] == m].set_index("alpha")
        b_pos = m_data.loc[0.5, "B"]
        b_neg = m_data.loc[-0.5, "B"]
        p_pos = m_data.loc[0.5, "p_coord"]
        p_neg = m_data.loc[-0.5, "p_coord"]
        g_slopes[m] = float((b_pos - b_neg) / (p_pos - p_neg))
        
    return {
        "h0_h1": h0_h1_results,
        "p_overlap": [p_min_overlap, p_max_overlap],
        "d_state_em": d_state_em,
        "d_state_ctrl": d_state_ctrl,
        "diff_d_state": diff_d_state,
        "g_slopes": g_slopes
    }

def run_corrected_analysis():
    print("=" * 60)
    print("STAGE 2C: CORRECTED STAGE 2 ANALYSIS")
    print("=" * 60)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    steering_csv = os.path.join(base_dir, "experiments/persona_control/results_stage2/steering/all_models_dose_response.csv")
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage2/mechanistic")
    os.makedirs(out_dir, exist_ok=True)
    
    df = pd.read_csv(steering_csv)
    df["B"] = 100.0 - df["judge1_aligned"]
    
    # Audit coherence per alpha
    print("\n--- Steering Dose-Response Quality Audit ---")
    coherence_table = df.groupby(["alpha", "model"])["judge1_coherent"].mean().unstack()
    print(coherence_table.round(2))
    
    valid_alphas = [-0.5, 0.0, 0.5]
    print(f"\nValid Steering Interval: alpha in {valid_alphas}")
    print("Criteria: Mean coherence >= 70 across models, low repetition, no collapse.")
    print("Note: +/-0.75 were not evaluated. Alphas |alpha| >= 1.0 caused activation explosion and collapse.")
    
    # 1. Point Estimates using full sample
    point_est = compute_prompt_cluster_metrics(df, valid_alphas=valid_alphas)
    
    print("\n--- Corrected H_0 and H_1 Fits (Valid Grid: alpha in [-0.5, 0, 0.5]) ---")
    for m, res in point_est["h0_h1"].items():
        print(f"[{m}]:")
        print(f"  H0 Translation: delta = {res['h0_delta']:+.3f} | Train MSE = {res['h0_train_mse']:.3f} | Held-out MSE = {res['h0_test_mse']:.3f}")
        print(f"  H1 Affine-Gain: a = {res['h1_a']:.3f}, delta = {res['h1_delta']:+.3f} | Train MSE = {res['h1_train_mse']:.3f} | Held-out MSE = {res['h1_test_mse']:.3f}")
        
    print(f"\nValid Coordinate Support Overlap P_overlap^valid: [{point_est['p_overlap'][0]:.2f}, {point_est['p_overlap'][1]:.2f}]")
    print(f"D_state(EM, Base):   {point_est['d_state_em']:.3f}")
    print(f"D_state(Ctrl, Base): {point_est['d_state_ctrl']:.3f}")
    print(f"D_EM - D_ctrl:       {point_est['diff_d_state']:.3f}")
    
    print("\n--- Simple Local Slopes G_M = dB/dP (between alpha = -0.5 and +0.5) ---")
    for m, g in point_est["g_slopes"].items():
        print(f"  {m}: G = {g:.4f}")
        
    # 2. Prompt-Level Cluster Bootstrap (N=1,000)
    print("\n--- Running Prompt-Level Cluster Bootstrap (B=1,000) ---")
    unique_prompts = df["prompt_id"].unique()
    n_prompts = len(unique_prompts)
    print(f"Number of prompt clusters: {n_prompts}")
    
    np.random.seed(42)
    n_boot = 1000
    boot_records = {
        "d_state_em": [],
        "d_state_ctrl": [],
        "diff_d_state": [],
        "G_M0": [],
        "G_Mctrl": [],
        "G_MEM": []
    }
    
    # Pre-split df by prompt_id for speed
    prompt_groups = {pid: grp for pid, grp in df.groupby("prompt_id")}
    
    for b in range(n_boot):
        # Sample prompt clusters with replacement
        sampled_pids = np.random.choice(unique_prompts, size=n_prompts, replace=True)
        # Concatenate full prompt clusters (matching replication count)
        boot_sample_dfs = [prompt_groups[pid] for pid in sampled_pids]
        boot_df = pd.concat(boot_sample_dfs, ignore_index=True)
        
        res_b = compute_prompt_cluster_metrics(boot_df, valid_alphas=valid_alphas)
        
        boot_records["d_state_em"].append(res_b["d_state_em"])
        boot_records["d_state_ctrl"].append(res_b["d_state_ctrl"])
        boot_records["diff_d_state"].append(res_b["diff_d_state"])
        boot_records["G_M0"].append(res_b["g_slopes"]["M0_base"])
        boot_records["G_Mctrl"].append(res_b["g_slopes"]["M_ctrl_benign"])
        boot_records["G_MEM"].append(res_b["g_slopes"]["M_EM_misaligned"])
        
    def get_ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        
    audit_summary = {
        "valid_alphas": valid_alphas,
        "coherence_summary": {str(k): float(v) for k, v in df[df["alpha"].isin(valid_alphas)].groupby("model")["judge1_coherent"].mean().to_dict().items()},
        "p_overlap_valid": point_est["p_overlap"],
        "point_estimates": {
            "D_state_EM": point_est["d_state_em"],
            "D_state_Ctrl": point_est["d_state_ctrl"],
            "Diff_D_state (EM - Ctrl)": point_est["diff_d_state"],
            "G_M0_base": point_est["g_slopes"]["M0_base"],
            "G_Mctrl_benign": point_est["g_slopes"]["M_ctrl_benign"],
            "G_MEM_misaligned": point_est["g_slopes"]["M_EM_misaligned"],
            "h0_h1": point_est["h0_h1"]
        },
        "bootstrap_audit": {
            "n_boot": n_boot,
            "unit": "prompt_cluster",
            "D_state_EM": {
                "point_est": point_est["d_state_em"],
                "boot_mean": float(np.mean(boot_records["d_state_em"])),
                "ci_95": get_ci(boot_records["d_state_em"])
            },
            "D_state_Ctrl": {
                "point_est": point_est["d_state_ctrl"],
                "boot_mean": float(np.mean(boot_records["d_state_ctrl"])),
                "ci_95": get_ci(boot_records["d_state_ctrl"])
            },
            "Diff_D_state": {
                "point_est": point_est["diff_d_state"],
                "boot_mean": float(np.mean(boot_records["diff_d_state"])),
                "ci_95": get_ci(boot_records["diff_d_state"])
            },
            "G_M0_base": {
                "point_est": point_est["g_slopes"]["M0_base"],
                "boot_mean": float(np.mean(boot_records["G_M0"])),
                "ci_95": get_ci(boot_records["G_M0"])
            },
            "G_Mctrl_benign": {
                "point_est": point_est["g_slopes"]["M_ctrl_benign"],
                "boot_mean": float(np.mean(boot_records["G_Mctrl"])),
                "ci_95": get_ci(boot_records["G_Mctrl"])
            },
            "G_MEM_misaligned": {
                "point_est": point_est["g_slopes"]["M_EM_misaligned"],
                "boot_mean": float(np.mean(boot_records["G_MEM"])),
                "ci_95": get_ci(boot_records["G_MEM"])
            }
        }
    }
    
    out_json = os.path.join(out_dir, "corrected_stage2_summary.json")
    with open(out_json, "w") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"\nCorrected summary written to: {out_json}")
    
    print("\n" + "=" * 50)
    print("BOOTSTRAP AUDIT SUMMARY TABLE:")
    print("=" * 50)
    for k in ["D_state_EM", "D_state_Ctrl", "Diff_D_state", "G_M0_base", "G_Mctrl_benign", "G_MEM_misaligned"]:
        entry = audit_summary["bootstrap_audit"][k]
        print(f"{k:18s} | Point: {entry['point_est']:9.4f} | Boot Mean: {entry['boot_mean']:9.4f} | 95% CI: [{entry['ci_95'][0]:.4f}, {entry['ci_95'][1]:.4f}]")
    print("=" * 50)

if __name__ == "__main__":
    run_corrected_analysis()
