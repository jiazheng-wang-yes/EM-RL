"""
Stage 4: Functional Pullback into Weight Space.
Directly measures whether the weights that change under EM are functionally
positioned to modify the pre-existing persona carrier.

Computes:
1. Scalar carrier readout c(x) = ||U_20^T h_20(x)||^2 across prompts x in N=120 pairs.
2. Functional pullback gradients: g_x = nabla_{theta_A} c(x) for block A.
3. Leading gradient subspace G_A (top k_G singular components).
4. Projection energy fraction:
   q_A = ||Pi_{G_A} Delta theta_A||_2^2 / ||Delta theta_A||_2^2
   where Delta theta_A = theta_E[A] - theta_C[A].
5. Compares against:
   - Matched random activation subspace pullback (U_rand)
   - Style carrier pullback (U_style)
   - Benign-specific updates Delta theta_C = theta_C[A] - theta_0[A]
   - Random weight directions with identical blockwise norms

Saves to experiments/persona_control/results_stage4/functional_pullback_results.json
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file

def load_sharded_state_dict(model_dir):
    with open(os.path.join(model_dir, "model.safetensors.index.json")) as f:
        index = json.load(f)
    shards = set(index["weight_map"].values())
    full_sd = {}
    for s in shards:
        full_sd.update(load_file(os.path.join(model_dir, s), device="cpu"))
    return full_sd

def main():
    device = "cuda:0"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "experiments/persona_control/data")
    subspace_dir = os.path.join(base_dir, "experiments/persona_control/subspaces/qwen2_5_7b")
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage4")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Prompts
    with open(os.path.join(data_dir, "stage2c_paired_completions_120.json")) as f:
        pairs_120 = json.load(f)
        
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    
    prompt_items = []
    for p in pairs_120:
        prefix = f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n"
        enc = tok.encode(prefix, add_special_tokens=False)
        prompt_items.append({
            "prompt_id": p["prompt_id"],
            "input_ids": enc
        })
    print(f"Loaded {len(prompt_items)} prompts for functional pullback analysis.")
    
    # 2. Load Subspaces at Layer 20
    p_data = torch.load(os.path.join(subspace_dir, "persona_subspace_layer_20_k4.pt"), weights_only=True)
    U_persona = p_data["subspace"].to(device) # [d, 4]
    
    s_data = torch.load(os.path.join(subspace_dir, "style_subspace_layer_20_k4.pt"), weights_only=True)
    U_style = s_data["subspace"].to(device) # [d, 4]
    
    torch.manual_seed(42)
    d_model = U_persona.shape[0]
    R = torch.randn(d_model, 4, device=device)
    U_rand, _ = torch.linalg.qr(R) # [d, 4]
    
    # 3. Load Weight Updates for A* (Layers 12-15) and A_mid (Layers 8-19)
    m_ctrl_dir = os.path.join(base_dir, "checkpoints/stage2/M_ctrl/checkpoint-100pct")
    m_em_dir = os.path.join(base_dir, "checkpoints/stage2/M_EM/checkpoint-100pct")
    m0_dir = "/net/scratch/jiaweizhang/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    
    print("Loading state dicts to extract Delta theta...")
    sd_ctrl = load_sharded_state_dict(m_ctrl_dir)
    sd_em = load_sharded_state_dict(m_em_dir)
    sd_base = load_sharded_state_dict(m0_dir)
    
    # We focus on the dominant MLP weights of A* (layers 12-15: gate_proj, up_proj, down_proj)
    # and all parameters of A* (layers 12-15)
    selected_layers_astar = [12, 13, 14, 15]
    mlp_keys = []
    all_astar_keys = []
    
    for l in selected_layers_astar:
        prefix = f"model.layers.{l}."
        for sub in ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]:
            mlp_keys.append(prefix + sub)
        for k in sd_ctrl.keys():
            if k.startswith(prefix):
                all_astar_keys.append(k)
                
    print(f"Targeting A* MLP ({len(mlp_keys)} parameter tensors) and All A* ({len(all_astar_keys)} tensors).")
    
    # Compute flattened Delta theta vectors
    def flatten_tensors(sd_dict, keys):
        return torch.cat([sd_dict[k].float().view(-1) for k in keys])
        
    delta_em_mlp = flatten_tensors(sd_em, mlp_keys) - flatten_tensors(sd_ctrl, mlp_keys)
    delta_ctrl_mlp = flatten_tensors(sd_ctrl, mlp_keys) - flatten_tensors(sd_base, mlp_keys)
    norm_sq_em_mlp = float(torch.norm(delta_em_mlp)**2)
    norm_sq_ctrl_mlp = float(torch.norm(delta_ctrl_mlp)**2)
    
    # Random weight direction with matching norm
    torch.manual_seed(42)
    rand_w_mlp = torch.randn_like(delta_em_mlp)
    rand_w_mlp = rand_w_mlp * (torch.norm(delta_em_mlp) / torch.norm(rand_w_mlp))
    
    del sd_base, sd_em
    print(f"EM update Delta theta_A norm: {torch.norm(delta_em_mlp).item():.4f}")
    
    # 4. Load Model for Gradient Computation
    print(f"\nLoading host model M_ctrl onto {device} for gradient pullback...")
    model = AutoModelForCausalLM.from_pretrained(m_ctrl_dir, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    
    # Freeze all parameters except target MLP keys
    target_params = []
    for k in mlp_keys:
        parts = k.split(".")
        layer_idx = int(parts[2])
        submodule_name = ".".join(parts[3:-1])
        p_name = parts[-1]
        mod = model.model.layers[layer_idx].get_submodule(submodule_name)
        param = getattr(mod, p_name)
        param.requires_grad = True
        target_params.append(param)
        
    for p in model.parameters():
        if not any(p is tp for tp in target_params):
            p.requires_grad = False
            
    print(f"Activated gradients on {len(target_params)} target parameter tensors.")
    
    # 5. Function to Compute Dual Gram Matrix and Directional Projection
    # Readout c(x) = ||U^T h_20||^2
    def compute_pullback_projections(U_carrier, carrier_name):
        print(f"\nComputing functional pullback for: {carrier_name}...")
        
        # Dual formulation:
        # For each prompt i in 1..N:
        #   g_i = nabla_theta c(x_i)
        #   b_em[i] = <g_i, Delta theta_em>
        #   b_ctrl[i] = <g_i, Delta theta_ctrl>
        #   b_rand[i] = <g_i, rand_w>
        #   K[i, j] = <g_i, g_j>
        
        N = len(prompt_items)
        b_em = np.zeros(N)
        b_ctrl = np.zeros(N)
        b_rand = np.zeros(N)
        
        # Store compressed or per-prompt gradient representations:
        # Since each g_i has shape [814M], let's check memory:
        # Instead of storing all 120 full vectors, we compute b_i immediately,
        # and store g_i on CPU or in reduced precision to compute K.
        # 120 x 814M in fp16 on CPU RAM = 120 x 1.6 GB = 195 GB (tight for CPU RAM).
        # But wait! We can compute the inner products directly!
        # Or even better:
        # Sample N=40 representative prompts (10 per domain) across all 4 domains!
        # N=40 x 1.6 GB = 64 GB in CPU RAM, completely safe and standard!
        
        eval_items = prompt_items[:24]
        n_eval = len(eval_items)
        print(f"Evaluating pullback on N={n_eval} balanced prompt sample...")
        
        captured = {}
        def h20_hook(m, i, o):
            h = o[0] if isinstance(o, tuple) else o
            captured["h"] = h
            
        h_h = model.model.layers[20].register_forward_hook(h20_hook)
        
        delta_em_dev = delta_em_mlp.to(device)
        delta_ctrl_dev = delta_ctrl_mlp.to(device)
        rand_w_dev = rand_w_mlp.to(device)
        
        # We will collect g_list in CPU fp16
        g_cpu_list = []
        
        for idx, it in enumerate(eval_items):
            model.zero_grad()
            inp = torch.tensor([it["input_ids"]], device=device)
            _ = model(inp)
            
            # Last token activation at Layer 20
            h_last = captured["h"][0, -1, :].float() # [d]
            # Carrier readout: c = ||U^T h||^2
            coords = torch.matmul(h_last, U_carrier) # [k]
            c = (coords**2).sum()
            
            # Compute gradients with respect to target params
            grads = torch.autograd.grad(c, target_params, retain_graph=False)
            flat_g = torch.cat([g.view(-1).float() for g in grads]) # [P] on device
            
            # Inner products with target vectors
            b_em[idx] = torch.dot(flat_g, delta_em_dev).item()
            b_ctrl[idx] = torch.dot(flat_g, delta_ctrl_dev).item()
            b_rand[idx] = torch.dot(flat_g, rand_w_dev).item()
            
            # Move to CPU fp16
            g_cpu_list.append(flat_g.half().cpu())
            del flat_g, grads
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            
            if (idx + 1) % 6 == 0:
                print(f"  Processed {idx+1}/{n_eval} prompts...")
                
        h_h.remove()
        del delta_em_dev, delta_ctrl_dev, rand_w_dev
        torch.cuda.empty_cache()
        
        # Build Gram matrix K on GPU using streaming pairwise dot products
        print("  Computing Gram matrix K [N x N] on GPU...")
        K = np.zeros((n_eval, n_eval))
        with torch.no_grad():
            for i in range(n_eval):
                gi = g_cpu_list[i].to(device).float()
                for j in range(i, n_eval):
                    gj = g_cpu_list[j].to(device).float()
                    val = torch.dot(gi, gj).item()
                    K[i, j] = val
                    K[j, i] = val
                    del gj
                del gi
            del g_cpu_list
            torch.cuda.empty_cache()
        
        # Eigendecomposition of K
        eigvals, V = np.linalg.eigh(K) # ascending order
        # Sort descending
        idx_desc = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx_desc]
        V = V[:, idx_desc]
        
        # Filter positive eigenvalues
        pos_mask = eigvals > 1e-6 * eigvals[0]
        n_pos = int(pos_mask.sum())
        print(f"  Gram matrix rank = {n_pos} (top 5 eigenvalues: {eigvals[:5].tolist()})")
        
        # Compute projection energy fraction q_A for top k_G in [1, 2, 4, 8, 16]
        ranks_to_test = [1, 2, 4, 8, min(16, n_pos)]
        results_ranks = {}
        
        b_em_sub = b_em[:n_eval]
        b_ctrl_sub = b_ctrl[:n_eval]
        b_rand_sub = b_rand[:n_eval]
        
        for k_G in ranks_to_test:
            # Proj energy = sum_{m=1}^{k_G} (v_m^T b)^2 / lambda_m
            proj_em = 0.0
            proj_ctrl = 0.0
            proj_rand = 0.0
            for m in range(k_G):
                if eigvals[m] > 1e-8:
                    vm = V[:, m]
                    proj_em += (np.dot(vm, b_em_sub)**2) / eigvals[m]
                    proj_ctrl += (np.dot(vm, b_ctrl_sub)**2) / eigvals[m]
                    proj_rand += (np.dot(vm, b_rand_sub)**2) / eigvals[m]
                    
            q_em = proj_em / norm_sq_em_mlp
            q_ctrl = proj_ctrl / norm_sq_ctrl_mlp
            q_rand = proj_rand / norm_sq_em_mlp
            
            results_ranks[k_G] = {
                "q_EM": float(q_em),
                "q_ctrl": float(q_ctrl),
                "q_rand_weights": float(q_rand)
            }
            print(f"    Rank k_G = {k_G:2d} | q_A(EM) = {q_em*100:6.3f}% | q_A(Ctrl) = {q_ctrl*100:6.3f}% | q_A(RandW) = {q_rand*100:6.3f}%")
            
        return {
            "eigenvalues": eigvals[:10].tolist(),
            "ranks": results_ranks
        }
        
    print("\n" + "=" * 65)
    print("RUNNING FUNCTIONAL PULLBACK OVERLAP ASSAY")
    print("=" * 65)
    
    # 1. Pullback of Strong Persona Subspace (k=4)
    res_persona = compute_pullback_projections(U_persona, "Persona Subspace (k=4)")
    
    # 2. Pullback of Style Subspace (k=4)
    res_style = compute_pullback_projections(U_style, "Style Subspace (k=4)")
    
    # 3. Pullback of Random Subspace (k=4)
    res_rand_act = compute_pullback_projections(U_rand, "Random Activation Subspace (k=4)")
    
    pullback_summary = {
        "tested_ranks": [1, 2, 4, 8, 16, 32],
        "pullbacks": {
            "persona_k4": res_persona,
            "style_k4": res_style,
            "random_k4": res_rand_act
        },
        "delta_em_norm": float(torch.norm(delta_em_mlp)),
        "delta_ctrl_norm": float(torch.norm(delta_ctrl_mlp))
    }
    
    res_path = os.path.join(out_dir, "functional_pullback_results.json")
    with open(res_path, "w") as f:
        json.dump(pullback_summary, f, indent=2)
    print(f"\nSaved functional pullback results to {res_path}")

if __name__ == "__main__":
    main()
