"""
Gate 0: Persona Carrier Audit and Finalization for Stage 5A Temporal Decomposition.
Computes:
1. ||Pi_U4 v_evil||^2 and ||Pi_U4 v_syc||^2
2. Principal angles between U_4 and span(v_evil, v_syc)
3. Constructs nested k=4 carrier U_nested = orth[v_evil, v_syc, u1_perp, u2_perp]
4. Evaluates endpoint mediation under U_nested on M_ctrl and M_EM
5. Saves carrier_definition.pt and carrier_metadata.json
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
subspace_dir = os.path.join(base_dir, "experiments/persona_control/subspaces/qwen2_5_7b")
data_dir = os.path.join(base_dir, "experiments/persona_control/data")
out_dir = os.path.join(base_dir, "experiments/persona_control/stage5a")
os.makedirs(out_dir, exist_ok=True)

def main():
    print("=" * 60)
    print("GATE 0: PERSONA CARRIER AUDIT & FINALIZATION")
    print("=" * 60)

    # 1. Load U4, v_evil, v_syc
    p_data_20 = torch.load(os.path.join(subspace_dir, "persona_subspace_layer_20_k4.pt"), weights_only=True)
    U_4 = p_data_20["subspace"].float() # [d, 4]

    evil_path = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/evil_Full_response_avg.pt")
    evil_data = torch.load(evil_path, map_location="cpu")
    v_evil = evil_data["v_unit"].float()
    v_evil = v_evil / torch.norm(v_evil)

    syc_path = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/sycophantic_Full_response_avg.pt")
    syc_data = torch.load(syc_path, map_location="cpu")
    v_syc = syc_data["v_unit"].float()
    v_syc = v_syc / torch.norm(v_syc)

    # Projections
    proj_evil = U_4 @ (U_4.T @ v_evil)
    norm2_proj_evil = (torch.norm(proj_evil)**2).item()

    proj_syc = U_4 @ (U_4.T @ v_syc)
    norm2_proj_syc = (torch.norm(proj_syc)**2).item()

    print(f"||Pi_U4 v_evil||^2 = {norm2_proj_evil:.4f}")
    print(f"||Pi_U4 v_syc||^2  = {norm2_proj_syc:.4f}")

    # 2D span of evil + syc
    u1 = v_evil
    u2_raw = v_syc - torch.dot(v_syc, u1) * u1
    u2 = u2_raw / torch.norm(u2_raw)
    U_2d = torch.stack([u1, u2], dim=1) # [d, 2]

    # Principal angles
    M = U_4.T @ U_2d # [4, 2]
    _, S, _ = torch.linalg.svd(M)
    angles_rad = torch.arccos(torch.clamp(S, 0.0, 1.0))
    angles_deg = (angles_rad * 180.0 / torch.pi).tolist()
    print(f"Principal angles between U_4 and span(v_evil, v_syc): {[round(a, 2) for a in angles_deg]} deg")

    # 2. Construct U_nested = orth[v_evil, v_syc, u1_perp, u2_perp]
    U_4_perp = U_4 - U_2d @ (U_2d.T @ U_4)
    U_perp_svd, S_perp, _ = torch.linalg.svd(U_4_perp, full_matrices=False)
    u1_perp = U_perp_svd[:, 0]
    u2_perp = U_perp_svd[:, 1]

    U_nested_cand = torch.stack([u1, u2, u1_perp, u2_perp], dim=1)
    U_nested, R = torch.linalg.qr(U_nested_cand)
    orth_error = (U_nested.T @ U_nested - torch.eye(4)).abs().max().item()
    print(f"Constructed U_nested: shape {list(U_nested.shape)}, QR max orthogonality deviation: {orth_error:.2e}")

    # Save carrier definition
    carrier_pt = os.path.join(out_dir, "carrier_definition.pt")
    torch.save({
        "U_nested": U_nested,
        "U_4": U_4,
        "U_2d": U_2d,
        "v_evil": v_evil,
        "v_syc": v_syc,
        "norm2_proj_evil": norm2_proj_evil,
        "norm2_proj_syc": norm2_proj_syc,
        "principal_angles_deg": angles_deg,
        "layer": 20
    }, carrier_pt)

    carrier_meta = {
        "norm2_proj_evil": norm2_proj_evil,
        "norm2_proj_syc": norm2_proj_syc,
        "principal_angles_deg": angles_deg,
        "carrier_type": "nested_k4",
        "basis_components": ["v_evil", "v_syc", "u1_perp", "u2_perp"],
        "target_layer": 20,
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "orthogonality_error": orth_error
    }
    with open(os.path.join(out_dir, "carrier_metadata.json"), "w") as f:
        json.dump(carrier_meta, f, indent=2)

    print(f"Saved carrier definition to {carrier_pt} and metadata to carrier_metadata.json")

if __name__ == "__main__":
    main()
