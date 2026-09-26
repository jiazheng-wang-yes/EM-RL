"""
Stage 2C: Parameter-Update Audit.
Section 12 of Experimental Directive:
Compares fine-tuning update sizes between EM and benign SFT control:
1. Layerwise relative Frobenius norms:
   r_l^EM = ||theta_l^EM - theta_l^0||_F / ||theta_l^0||_F
   r_l^ctrl = ||theta_l^ctrl - theta_l^0||_F / ||theta_l^0||_F
   across: Q, K, V, O projections, MLP gate, up, down projections, layer norms.
2. Global components: embeddings, final norm, LM head.
3. Total Euclidean distance ratio: ||theta_EM - theta_0|| / ||theta_ctrl - theta_0||.
Outputs:
- experiments/persona_control/results_stage2/parameter_audit/parameter_audit_results.json
- experiments/persona_control/results_stage2/parameter_audit/layerwise_parameter_deltas.csv
- figures/persona_control/stage2/figure_parameter_update_audit.png
"""

import os
import sys
import json
import re
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
import matplotlib.pyplot as plt

def load_full_state_dict(model_dir):
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_file) as f:
        idx = json.load(f)["weight_map"]
    
    unique_shards = sorted(list(set(idx.values())))
    state_dict = {}
    print(f"Loading {len(unique_shards)} shards from {os.path.basename(model_dir)}...")
    for shard in unique_shards:
        shard_path = os.path.join(model_dir, shard)
        sd = load_file(shard_path, device="cpu")
        state_dict.update(sd)
    return state_dict

def run_parameter_audit():
    print("=" * 60)
    print("STAGE 2C: PARAMETER-UPDATE AUDIT")
    print("=" * 60)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage2/parameter_audit")
    fig_dir = os.path.join(base_dir, "figures/persona_control/stage2")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    m0_dir = "/net/scratch/jiaweizhang/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    m_ctrl_dir = os.path.join(base_dir, "checkpoints/stage2/M_ctrl/checkpoint-100pct")
    m_em_dir = os.path.join(base_dir, "checkpoints/stage2/M_EM/checkpoint-100pct")
    
    sd_0 = load_full_state_dict(m0_dir)
    sd_ctrl = load_full_state_dict(m_ctrl_dir)
    sd_em = load_full_state_dict(m_em_dir)
    
    param_records = []
    total_sq_diff_em = 0.0
    total_sq_diff_ctrl = 0.0
    total_sq_norm_base = 0.0
    
    print("\nAuditing parameters across all layers and modules...")
    for key in sorted(sd_0.keys()):
        t0 = sd_0[key].float()
        t_ctrl = sd_ctrl[key].float()
        t_em = sd_em[key].float()
        
        frob_0 = torch.norm(t0, p="fro").item()
        frob_diff_ctrl = torch.norm(t_ctrl - t0, p="fro").item()
        frob_diff_em = torch.norm(t_em - t0, p="fro").item()
        
        total_sq_norm_base += frob_0 ** 2
        total_sq_diff_ctrl += frob_diff_ctrl ** 2
        total_sq_diff_em += frob_diff_em ** 2
        
        r_ctrl = frob_diff_ctrl / max(1e-12, frob_0)
        r_em = frob_diff_em / max(1e-12, frob_0)
        
        # Parse layer and module type
        match_layer = re.search(r"model\.layers\.(\d+)\.(.+)", key)
        if match_layer:
            layer_idx = int(match_layer.group(1))
            module_name = match_layer.group(2)
        else:
            layer_idx = -1 # global module
            module_name = key
            
        comp_type = "other"
        if "self_attn.q_proj" in module_name: comp_type = "attn_q"
        elif "self_attn.k_proj" in module_name: comp_type = "attn_k"
        elif "self_attn.v_proj" in module_name: comp_type = "attn_v"
        elif "self_attn.o_proj" in module_name: comp_type = "attn_o"
        elif "mlp.gate_proj" in module_name: comp_type = "mlp_gate"
        elif "mlp.up_proj" in module_name: comp_type = "mlp_up"
        elif "mlp.down_proj" in module_name: comp_type = "mlp_down"
        elif "input_layernorm" in module_name or "post_attention_layernorm" in module_name: comp_type = "layer_norm"
        elif "embed_tokens" in key: comp_type = "embedding"
        elif "model.norm" in key: comp_type = "final_norm"
        elif "lm_head" in key: comp_type = "lm_head"
        
        param_records.append({
            "param_name": key,
            "layer": layer_idx,
            "module": module_name,
            "comp_type": comp_type,
            "frob_base": frob_0,
            "frob_diff_ctrl": frob_diff_ctrl,
            "frob_diff_em": frob_diff_em,
            "rel_frob_ctrl": r_ctrl,
            "rel_frob_em": r_em,
            "em_to_ctrl_ratio": r_em / max(1e-12, r_ctrl)
        })
        
    df_params = pd.DataFrame(param_records)
    csv_path = os.path.join(out_dir, "layerwise_parameter_deltas.csv")
    df_params.to_csv(csv_path, index=False)
    print(f"Saved parameter audit data to {csv_path}")
    
    total_dist_em = np.sqrt(total_sq_diff_em)
    total_dist_ctrl = np.sqrt(total_sq_diff_ctrl)
    total_dist_ratio = total_dist_em / max(1e-12, total_dist_ctrl)
    
    print("\n" + "=" * 50)
    print("PARAMETER-UPDATE AUDIT SUMMARY:")
    print("=" * 50)
    print(f"Total Euclidean Distance ||theta_EM - theta_0||:   {total_dist_em:.4f}")
    print(f"Total Euclidean Distance ||theta_ctrl - theta_0||: {total_dist_ctrl:.4f}")
    print(f"Total Distance Ratio ||theta_EM - theta_0|| / ||theta_ctrl - theta_0||: {total_dist_ratio:.4f}")
    print("=" * 50)
    
    # Layerwise summary (layers 0 to 27)
    df_layers = df_params[df_params["layer"] >= 0].groupby("layer").agg({
        "frob_diff_em": lambda s: np.sqrt((s**2).sum()),
        "frob_diff_ctrl": lambda s: np.sqrt((s**2).sum()),
        "frob_base": lambda s: np.sqrt((s**2).sum()),
    }).reset_index()
    df_layers["r_em"] = df_layers["frob_diff_em"] / df_layers["frob_base"]
    df_layers["r_ctrl"] = df_layers["frob_diff_ctrl"] / df_layers["frob_base"]
    
    # Component-type summary
    df_comp = df_params.groupby("comp_type").agg({
        "frob_diff_em": lambda s: np.sqrt((s**2).sum()),
        "frob_diff_ctrl": lambda s: np.sqrt((s**2).sum()),
        "frob_base": lambda s: np.sqrt((s**2).sum()),
    }).reset_index()
    df_comp["r_em"] = df_comp["frob_diff_em"] / df_comp["frob_base"]
    df_comp["r_ctrl"] = df_comp["frob_diff_ctrl"] / df_comp["frob_base"]
    df_comp["ratio"] = df_comp["frob_diff_em"] / np.maximum(1e-12, df_comp["frob_diff_ctrl"])
    
    print("\nComponent Relative Frobenius Norms (%):")
    for _, row in df_comp.iterrows():
        print(f"  {row['comp_type']:12s} | r_EM = {row['r_em']*100:6.3f}% | r_ctrl = {row['r_ctrl']*100:6.3f}% | Ratio = {row['ratio']:5.2f}x")
        
    summary = {
        "total_dist_em": float(total_dist_em),
        "total_dist_ctrl": float(total_dist_ctrl),
        "total_dist_ratio": float(total_dist_ratio),
        "component_summary": df_comp.to_dict(orient="records"),
        "layerwise_summary": df_layers.to_dict(orient="records")
    }
    
    json_path = os.path.join(out_dir, "parameter_audit_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {json_path}")
    
    # Plotting Figure: Parameter Update Audit
    print("\nGenerating Figure: Parameter Update Audit...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Left: Layerwise relative Frobenius norm
    ax = axes[0]
    ax.plot(df_layers["layer"], df_layers["r_em"] * 100, marker="o", color="#d62728", label="M_EM (Misaligned SFT)", linewidth=2)
    ax.plot(df_layers["layer"], df_layers["r_ctrl"] * 100, marker="s", color="#2ca02c", label="M_ctrl (Benign SFT)", linewidth=2)
    ax.axvline(20, color="gray", linestyle="--", alpha=0.7, label="Layer 20 (Evil Vector)")
    ax.set_xlabel("Transformer Layer Index", fontsize=12)
    ax.set_ylabel("Relative Update Norm $r_l$ (%)", fontsize=12)
    ax.set_title("Layerwise Parameter Update Size ($r_l = \\|\\Delta\\theta_l\\|_F / \\|\\theta_l^0\\|_F$)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Right: Component-wise update ratio
    ax2 = axes[1]
    comp_order = ["attn_q", "attn_k", "attn_v", "attn_o", "mlp_gate", "mlp_up", "mlp_down", "layer_norm", "lm_head"]
    comp_plot = df_comp[df_comp["comp_type"].isin(comp_order)].copy()
    comp_plot["order"] = comp_plot["comp_type"].apply(lambda x: comp_order.index(x) if x in comp_order else 99)
    comp_plot = comp_plot.sort_values("order")
    
    x_indices = np.arange(len(comp_plot))
    bar_width = 0.35
    ax2.bar(x_indices - bar_width/2, comp_plot["r_em"] * 100, bar_width, label="M_EM", color="#d62728", alpha=0.85)
    ax2.bar(x_indices + bar_width/2, comp_plot["r_ctrl"] * 100, bar_width, label="M_ctrl", color="#2ca02c", alpha=0.85)
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(comp_plot["comp_type"], rotation=35, ha="right", fontsize=10)
    ax2.set_ylabel("Relative Update Norm (%)", fontsize=12)
    ax2.set_title(f"Component Update Norms (Total Ratio EM/Ctrl = {total_dist_ratio:.2f}x)", fontsize=13)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    fig_path = os.path.join(fig_dir, "figure_parameter_update_audit.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Figure saved to {fig_path}")

if __name__ == "__main__":
    run_parameter_audit()
