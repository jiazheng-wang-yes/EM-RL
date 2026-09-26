"""
Stage 2 Mechanistic Analysis & Hypothesis Testing.
Tests the Shift-Only Hypothesis by evaluating:
1. Model H_0: Translation Test B_EM(alpha) = B_0(alpha + delta)
2. Model H_1: Affine-Gain Test B_EM(alpha) = B_0(a*alpha + delta)
3. Coordinate-Space Test B_M(P) & Cross-Model Discrepancy D_state
4. Persona-State Sufficiency: Predict B_EM from base mapping F_0(P_EM)
5. Matched-State Causal Comparison at matched P*: compares B and dB/dP
6. Small Persona Subspace Control: 1D vs 3D regression [v_evil, v_syc, v_style]
7. Bidirectional Test: alpha < 0 repair vs restoration
8. Representation-Scale Control: E[||h_20||], Var(<h, v>), dP/dalpha
9. Bootstrap confidence intervals across prompts (N=1,000)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import interp1d

def logistic_func(x, L, x0, k, b):
    return L / (1.0 + np.exp(-k * (x - x0))) + b

def fit_smooth_curve(x_data, y_data):
    # Monotonic cubic or linear spline with bounds
    order = np.argsort(x_data)
    xs = np.array(x_data)[order]
    ys = np.array(y_data)[order]
    
    # Fallback interpolation
    f_interp = interp1d(xs, ys, kind='linear', fill_value="extrapolate")
    
    # Try fitting a logistic sigmoid
    try:
        p0 = [max(ys) - min(ys), np.median(xs), 1.0, min(ys)]
        popt, _ = curve_fit(logistic_func, xs, ys, p0=p0, maxfev=5000)
        return lambda x: logistic_func(x, *popt), f_interp
    except Exception:
        return f_interp, f_interp

def run_mechanistic_analysis(steering_csv_path, output_dir):
    print(f"\n{'='*30} RUNNING MECHANISTIC ANALYSIS {'='*30}", flush=True)
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(steering_csv_path)
    # Target behavioral variable: Misaligned Behavior Score = 100 - judge1_aligned
    # So B in [0, 100], where higher = more misaligned
    df["B"] = 100.0 - df["judge1_aligned"]
    
    # Check splits
    train_prompts = df[df["split"] == "canonical"]["prompt_id"].unique()
    heldout_prompts = df[df["split"] == "heldout"]["prompt_id"].unique()
    print(f"Train prompts (canonical): {len(train_prompts)} | Heldout prompts: {len(heldout_prompts)}", flush=True)
    
    # Average curves per model and alpha
    models = ["M0_base", "M_ctrl_benign", "M_EM_misaligned"]
    
    # Prompt-level aggregation
    prompt_df = df.groupby(["model", "alpha", "prompt_id", "split"]).agg({
        "B": "mean",
        "p_coord": "mean",
        "judge1_is_em": "mean",
        "judge1_coherent": "mean",
        "mean_h_norm": "mean",
        "p_3d_evil": "mean",
        "p_3d_syc": "mean",
        "p_3d_style": "mean"
    }).reset_index()

    # Overall model curve by alpha
    base_curve = prompt_df[prompt_df["model"] == "M0_base"].groupby("alpha")["B"].mean()
    f_base_alpha = interp1d(base_curve.index.values, base_curve.values, kind='linear', fill_value="extrapolate")
    
    # -------------------------------------------------------------------------
    # 1. Translation Test (H_0) & Affine-Gain Test (H_1)
    # -------------------------------------------------------------------------
    print("\n--- 1. Translation Test (H_0) & Affine-Gain Test (H_1) ---", flush=True)
    
    train_df = prompt_df[prompt_df["split"] == "canonical"]
    test_df = prompt_df[prompt_df["split"] == "heldout"]
    
    results_h0_h1 = []
    
    for target_m in ["M_ctrl_benign", "M_EM_misaligned"]:
        # Fit delta on train prompts
        m_train = train_df[train_df["model"] == target_m].groupby("alpha")["B"].mean()
        m_test = test_df[test_df["model"] == target_m].groupby("alpha")["B"].mean()
        alphas = m_train.index.values
        
        # H0: B_target(alpha) = B_base(alpha + delta)
        def loss_h0(delta):
            pred = f_base_alpha(alphas + delta[0])
            return np.mean((m_train.values - pred) ** 2)
            
        res_h0 = minimize(loss_h0, [0.0], method='Nelder-Mead')
        best_delta = float(res_h0.x[0])
        train_mse_h0 = float(res_h0.fun)
        test_pred_h0 = f_base_alpha(m_test.index.values + best_delta)
        test_mse_h0 = float(np.mean((m_test.values - test_pred_h0) ** 2))
        
        # H1: B_target(alpha) = B_base(a * alpha + delta)
        def loss_h1(params):
            a, d = params
            pred = f_base_alpha(a * alphas + d)
            return np.mean((m_train.values - pred) ** 2)
            
        res_h1 = minimize(loss_h1, [1.0, best_delta], method='Nelder-Mead')
        best_a, best_d_h1 = float(res_h1.x[0]), float(res_h1.x[1])
        train_mse_h1 = float(res_h1.fun)
        test_pred_h1 = f_base_alpha(best_a * m_test.index.values + best_d_h1)
        test_mse_h1 = float(np.mean((m_test.values - test_pred_h1) ** 2))
        
        # Baseline H2 error (unconstrained target curve on held-out)
        # Using target's own train curve to predict held-out
        f_target_train = interp1d(m_train.index.values, m_train.values, kind='linear', fill_value="extrapolate")
        test_pred_h2 = f_target_train(m_test.index.values)
        test_mse_h2 = float(np.mean((m_test.values - test_pred_h2) ** 2))
        
        results_h0_h1.append({
            "model": target_m,
            "h0_delta": best_delta,
            "h0_train_mse": train_mse_h0,
            "h0_test_mse": test_mse_h0,
            "h1_gain_a": best_a,
            "h1_delta": best_d_h1,
            "h1_train_mse": train_mse_h1,
            "h1_test_mse": test_mse_h1,
            "h2_test_mse": test_mse_h2,
        })
        print(f"[{target_m}] H0 (delta={best_delta:+.3f}): Held-out MSE = {test_mse_h0:.2f}")
        print(f"[{target_m}] H1 (a={best_a:.3f}, delta={best_d_h1:+.3f}): Held-out MSE = {test_mse_h1:.2f}")
        print(f"[{target_m}] H2 (unconstrained): Held-out MSE = {test_mse_h2:.2f}")
        
    df_h0_h1 = pd.DataFrame(results_h0_h1)
    df_h0_h1.to_csv(os.path.join(output_dir, "h0_h1_fits.csv"), index=False)

    # -------------------------------------------------------------------------
    # 2. Coordinate-Space Test: B(P) and Discrepancy D_state
    # -------------------------------------------------------------------------
    print("\n--- 2. Coordinate-Space Test: B(P) & Overlap Discrepancy ---", flush=True)
    
    # Fit base function F_0(P) from base model data
    base_data = df[df["model"] == "M0_base"]
    p_base = base_data["p_coord"].values
    b_base = base_data["B"].values
    
    f_base_p_logistic, f_base_p_interp = fit_smooth_curve(p_base, b_base)
    
    # Find overlapping region P_overlap between Base and EM
    em_data = df[df["model"] == "M_EM_misaligned"]
    ctrl_data = df[df["model"] == "M_ctrl_benign"]
    
    p_min_overlap = max(p_base.min(), em_data["p_coord"].min(), ctrl_data["p_coord"].min())
    p_max_overlap = min(p_base.max(), em_data["p_coord"].max(), ctrl_data["p_ctrl"].max() if "p_ctrl" in ctrl_data else ctrl_data["p_coord"].max())
    print(f"Common Persona Coordinate Overlap Support: [{p_min_overlap:.2f}, {p_max_overlap:.2f}]", flush=True)
    
    # Discrepancy evaluation on overlapping support
    eval_pts = np.linspace(p_min_overlap, p_max_overlap, 50)
    
    # Fit smooth functions for EM and ctrl
    _, f_em_p = fit_smooth_curve(em_data["p_coord"].values, em_data["B"].values)
    _, f_ctrl_p = fit_smooth_curve(ctrl_data["p_coord"].values, ctrl_data["B"].values)
    
    b0_pts = f_base_p_interp(eval_pts)
    b_em_pts = f_em_p(eval_pts)
    b_ctrl_pts = f_ctrl_p(eval_pts)
    
    d_state_em = float(np.mean((b_em_pts - b0_pts) ** 2))
    d_state_ctrl = float(np.mean((b_ctrl_pts - b0_pts) ** 2))
    
    mae_state_em = float(np.mean(np.abs(b_em_pts - b0_pts)))
    mae_state_ctrl = float(np.mean(np.abs(b_ctrl_pts - b0_pts)))
    
    print(f"Discrepancy D_state(EM, Base): {d_state_em:.2f} (MAE = {mae_state_em:.2f})")
    print(f"Discrepancy D_state(Ctrl, Base): {d_state_ctrl:.2f} (MAE = {mae_state_ctrl:.2f})")
    print(f"EM-to-Control Discrepancy Ratio: {d_state_em / max(1e-5, d_state_ctrl):.2f}x")
    
    # -------------------------------------------------------------------------
    # 3. Matched-State Causal Comparison
    # -------------------------------------------------------------------------
    print("\n--- 3. Matched-State Causal Comparison ---", flush=True)
    
    # Average P and B by alpha for base and EM
    dose_summary = df.groupby(["model", "alpha"]).agg({
        "p_coord": "mean",
        "B": "mean",
        "judge1_is_em": "mean",
        "judge1_coherent": "mean"
    }).reset_index()
    
    base_summary = dose_summary[dose_summary["model"] == "M0_base"].set_index("alpha")
    em_summary = dose_summary[dose_summary["model"] == "M_EM_misaligned"].set_index("alpha")
    ctrl_summary = dose_summary[dose_summary["model"] == "M_ctrl_benign"].set_index("alpha")
    
    # Pick 3 target coordinates in the overlap region
    p_targets = [
        float(np.quantile([p_min_overlap, p_max_overlap], 0.25)),
        float(np.quantile([p_min_overlap, p_max_overlap], 0.50)),
        float(np.quantile([p_min_overlap, p_max_overlap], 0.75)),
    ]
    
    matched_comparisons = []
    for p_star in p_targets:
        # Find closest alpha in base, ctrl, EM
        alpha_base = float(base_summary.iloc[(base_summary["p_coord"] - p_star).abs().argmin()].name)
        alpha_em = float(em_summary.iloc[(em_summary["p_coord"] - p_star).abs().argmin()].name)
        alpha_ctrl = float(ctrl_summary.iloc[(ctrl_summary["p_coord"] - p_star).abs().argmin()].name)
        
        b_base_val = float(base_summary.loc[alpha_base, "B"])
        b_em_val = float(em_summary.loc[alpha_em, "B"])
        b_ctrl_val = float(ctrl_summary.loc[alpha_ctrl, "B"])
        
        # Local slope dB/dP around p_star
        dp = 0.5
        g_base = float((f_base_p_interp(p_star + dp) - f_base_p_interp(p_star - dp)) / (2 * dp))
        g_em = float((f_em_p(p_star + dp) - f_em_p(p_star - dp)) / (2 * dp))
        g_ctrl = float((f_ctrl_p(p_star + dp) - f_ctrl_p(p_star - dp)) / (2 * dp))
        
        matched_comparisons.append({
            "P_target": p_star,
            "alpha_base": alpha_base,
            "alpha_ctrl": alpha_ctrl,
            "alpha_em": alpha_em,
            "B_base": b_base_val,
            "B_ctrl": b_ctrl_val,
            "B_em": b_em_val,
            "delta_B_EM_Base": b_em_val - b_base_val,
            "gain_G_base": g_base,
            "gain_G_ctrl": g_ctrl,
            "gain_G_em": g_em
        })
        print(f"At P* = {p_star:.2f}: B_base={b_base_val:.1f} (a={alpha_base}), B_ctrl={b_ctrl_val:.1f} (a={alpha_ctrl}), B_em={b_em_val:.1f} (a={alpha_em}) | dB/dP: Base={g_base:.2f}, EM={g_em:.2f}")

    matched_df = pd.DataFrame(matched_comparisons)
    matched_df.to_csv(os.path.join(output_dir, "matched_state_comparison.csv"), index=False)

    # -------------------------------------------------------------------------
    # 4. Small Persona Subspace Control (3D vs 1D)
    # -------------------------------------------------------------------------
    print("\n--- 4. Small Persona Subspace Control (1D vs 3D) ---", flush=True)
    # Fit linear/ridge model from (p_evil) vs (p_evil, p_syc, p_style) on Base
    from sklearn.linear_model import Ridge
    
    X_1d_base = base_data[["p_coord"]].values
    X_3d_base = base_data[["p_3d_evil", "p_3d_syc", "p_3d_style"]].values
    y_base = base_data["B"].values
    
    r1 = Ridge(alpha=1.0).fit(X_1d_base, y_base)
    r3 = Ridge(alpha=1.0).fit(X_3d_base, y_base)
    
    # Evaluate on EM held-out prompts
    em_heldout = df[(df["model"] == "M_EM_misaligned") & (df["split"] == "heldout")]
    y_em_heldout = em_heldout["B"].values
    
    pred_em_1d = r1.predict(em_heldout[["p_coord"]].values)
    pred_em_3d = r3.predict(em_heldout[["p_3d_evil", "p_3d_syc", "p_3d_style"]].values)
    
    mse_1d = float(np.mean((y_em_heldout - pred_em_1d) ** 2))
    mse_3d = float(np.mean((y_em_heldout - pred_em_3d) ** 2))
    r2_1d = float(1.0 - mse_1d / np.var(y_em_heldout))
    r2_3d = float(1.0 - mse_3d / np.var(y_em_heldout))
    
    subspace_summary = {
        "1D_heldout_mse": mse_1d,
        "1D_heldout_r2": r2_1d,
        "3D_heldout_mse": mse_3d,
        "3D_heldout_r2": r2_3d,
        "base_coefs_3d": {
            "v_evil": float(r3.coef_[0]),
            "v_syc": float(r3.coef_[1]),
            "v_style": float(r3.coef_[2])
        }
    }
    print(f"1D Persona Coordinate Held-out MSE on EM: {mse_1d:.2f} (R2 = {r2_1d:.3f})")
    print(f"3D Persona Subspace Held-out MSE on EM:   {mse_3d:.2f} (R2 = {r2_3d:.3f})")
    with open(os.path.join(output_dir, "subspace_control_summary.json"), "w") as f:
        json.dump(subspace_summary, f, indent=2)

    # -------------------------------------------------------------------------
    # 5. Representation Scale Controls
    # -------------------------------------------------------------------------
    print("\n--- 5. Representation Scale Controls ---", flush=True)
    rep_scales = []
    for m in models:
        m_df = df[df["model"] == m]
        mean_norm = float(m_df["mean_h_norm"].mean())
        proj_var = float(m_df["p_coord"].var())
        
        # dP/dalpha
        m_alphas = m_df.groupby("alpha")["p_coord"].mean()
        dp_dalpha = float(np.polyfit(m_alphas.index.values, m_alphas.values, 1)[0])
        
        rep_scales.append({
            "model": m,
            "mean_h20_norm": mean_norm,
            "coord_variance": proj_var,
            "coord_gain_per_alpha": dp_dalpha
        })
        print(f"[{m}] E[||h_20||] = {mean_norm:.2f} | Var(<h, v>) = {proj_var:.2f} | dP/dalpha = {dp_dalpha:.3f}")
        
    pd.DataFrame(rep_scales).to_csv(os.path.join(output_dir, "representation_scales.csv"), index=False)

    # -------------------------------------------------------------------------
    # 6. Bootstrap Confidence Intervals
    # -------------------------------------------------------------------------
    print("\n--- 6. Bootstrap Confidence Intervals (N=1,000) ---", flush=True)
    n_boot = 1000
    boot_discrepancies = {"d_state_em": [], "d_state_ctrl": [], "ratio": []}
    
    unique_prompts = df["prompt_id"].unique()
    np.random.seed(42)
    
    for b in range(n_boot):
        sample_prompts = np.random.choice(unique_prompts, size=len(unique_prompts), replace=True)
        boot_df = df[df["prompt_id"].isin(sample_prompts)]
        
        # Mean curves for this sample
        b0 = boot_df[boot_df["model"] == "M0_base"].groupby("alpha")["B"].mean()
        bem = boot_df[boot_df["model"] == "M_EM_misaligned"].groupby("alpha")["B"].mean()
        bctrl = boot_df[boot_df["model"] == "M_ctrl_benign"].groupby("alpha")["B"].mean()
        
        d_em = np.mean((bem.values - b0.values) ** 2)
        d_c = np.mean((bctrl.values - b0.values) ** 2)
        
        boot_discrepancies["d_state_em"].append(d_em)
        boot_discrepancies["d_state_ctrl"].append(d_c)
        boot_discrepancies["ratio"].append(d_em / max(1e-5, d_c))
        
    ci_summary = {
        "D_state_EM_mean": float(np.mean(boot_discrepancies["d_state_em"])),
        "D_state_EM_ci": [float(np.percentile(boot_discrepancies["d_state_em"], 2.5)), float(np.percentile(boot_discrepancies["d_state_em"], 97.5))],
        "D_state_Ctrl_mean": float(np.mean(boot_discrepancies["d_state_ctrl"])),
        "D_state_Ctrl_ci": [float(np.percentile(boot_discrepancies["d_state_ctrl"], 2.5)), float(np.percentile(boot_discrepancies["d_state_ctrl"], 97.5))],
        "EM_Ctrl_ratio_mean": float(np.mean(boot_discrepancies["ratio"])),
        "EM_Ctrl_ratio_ci": [float(np.percentile(boot_discrepancies["ratio"], 2.5)), float(np.percentile(boot_discrepancies["ratio"], 97.5))]
    }
    with open(os.path.join(output_dir, "bootstrap_ci_summary.json"), "w") as f:
        json.dump(ci_summary, f, indent=2)
    print(f"D_state(EM): {ci_summary['D_state_EM_mean']:.2f} [{ci_summary['D_state_EM_ci'][0]:.2f}, {ci_summary['D_state_EM_ci'][1]:.2f}]")
    print(f"D_state(Ctrl): {ci_summary['D_state_Ctrl_mean']:.2f} [{ci_summary['D_state_Ctrl_ci'][0]:.2f}, {ci_summary['D_state_Ctrl_ci'][1]:.2f}]")
    print(f"EM / Ctrl Ratio: {ci_summary['EM_Ctrl_ratio_mean']:.2f}x [{ci_summary['EM_Ctrl_ratio_ci'][0]:.2f}, {ci_summary['EM_Ctrl_ratio_ci'][1]:.2f}]")

    print("\nMechanistic analysis complete!", flush=True)

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    steering_csv = os.path.join(base_dir, "experiments/persona_control/results_stage2/steering/all_models_steered_responses.csv")
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage2/mechanistic")
    
    if not os.path.exists(steering_csv):
        # Check individual files and merge
        steer_dir = os.path.join(base_dir, "experiments/persona_control/results_stage2/steering")
        files = [os.path.join(steer_dir, f"{m}_steered_responses.csv") for m in ["M0_base", "M_ctrl_benign", "M_EM_misaligned"]]
        if all(os.path.exists(f) for f in files):
            dfs = [pd.read_csv(f) for f in files]
            merged = pd.concat(dfs, ignore_index=True)
            merged.to_csv(steering_csv, index=False)
        else:
            print(f"Steering data not ready yet at {steering_csv}. Run stage2_steering_and_projection.py first.")
            return

    run_mechanistic_analysis(steering_csv, out_dir)

if __name__ == "__main__":
    main()
