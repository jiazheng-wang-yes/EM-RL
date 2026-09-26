"""
Stage 4: Factorial Mediation Assay.
Directly tests the Read-Write Causal Relation:
Does the middle-layer EM-specific weight update produce its behavioral effect
THROUGH the pre-existing persona carrier, or does it provide an additional
causal route to misaligned behavior?

Tests:
1. Target weight regions: A* (12:15) and A_mid (8:19)
2. Subspaces:
   - Strong Low-Rank Persona Subspace (k=4)
   - Evil-only 1D vector (k=1)
   - Evil + Sycophancy 2D basis (k=2)
   - Matched Style Subspace (k=4)
   - Matched Random Subspace (k=4, averaged over 5 seeds)
3. Host models:
   - Sufficiency: C (M_ctrl) and C <- E(A)
   - Necessity: E (M_EM) and E <- C(A)
4. Evaluates on:
   - Full N=120 pairs (stage2c_paired_completions_120.json)
   - Strict length-matched N=50 pairs (stage3_strict_paired_completions_50.json)
5. Computes Total Effect (TE), Direct Effect (DE), and Mediated Fraction (MF = 1 - DE/TE)
   with 1,000 prompt-level bootstrap resamples.
Saves to experiments/persona_control/results_stage4/mediation_results.json
"""

import os
import sys
import json
import numpy as np
import pandas as pd
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

def apply_layer_graft(model, layers, source_layers, device):
    for l in layers:
        for sub_k, tensor in source_layers[l].items():
            target_param = model.model.layers[l].get_submodule(".".join(sub_k.split(".")[:-1])) if "." in sub_k else model.model.layers[l]
            p_name = sub_k.split(".")[-1]
            getattr(target_param, p_name).data.copy_(tensor.to(device))

def compute_logprobs_batch(model, items, device, hook_fn=None, hook_layer=None):
    """
    Computes deterministic preference S(x) = (1/|y_mis|) log p(y_mis|x) - (1/|y_align|) log p(y_align|x)
    Optionally attaches forward hook at hook_layer during evaluation.
    """
    h_h = None
    if hook_fn is not None and hook_layer is not None:
        h_h = model.model.layers[hook_layer].register_forward_hook(hook_fn)
        
    scores = {}
    with torch.no_grad():
        for it in items:
            pid, plen = it["prompt_id"], it["prompt_len"]
            inp_a = torch.tensor([it["align_ids"]], device=device)
            inp_m = torch.tensor([it["mis_ids"]], device=device)
            
            logits_a = model(inp_a).logits[0, plen - 1 : -1, :]
            targets_a = inp_a[0, plen:].unsqueeze(-1)
            lp_a = torch.gather(F.log_softmax(logits_a, -1), -1, targets_a).squeeze(-1).mean().item()
            
            logits_m = model(inp_m).logits[0, plen - 1 : -1, :]
            targets_m = inp_m[0, plen:].unsqueeze(-1)
            lp_m = torch.gather(F.log_softmax(logits_m, -1), -1, targets_m).squeeze(-1).mean().item()
            
            scores[pid] = lp_m - lp_a
            
    if h_h is not None:
        h_h.remove()
    return scores

def main():
    device = "cuda:0"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_dir = os.path.join(base_dir, "experiments/persona_control/data")
    subspace_dir = os.path.join(base_dir, "experiments/persona_control/subspaces/qwen2_5_7b")
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage4")
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Tokenizer & Prompt Datasets
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id)
    
    with open(os.path.join(data_dir, "stage2c_paired_completions_120.json")) as f:
        pairs_120 = json.load(f)
    with open(os.path.join(data_dir, "stage3_strict_paired_completions_50.json")) as f:
        pairs_strict = json.load(f)
        
    def prepare_tokens(pairs):
        items = []
        for p in pairs:
            prefix = f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n"
            enc_prefix = tok.encode(prefix, add_special_tokens=False)
            items.append({
                "prompt_id": p["prompt_id"],
                "prompt_len": len(enc_prefix),
                "align_ids": tok.encode(prefix + p["y_aligned"] + "<|im_end|>", add_special_tokens=False),
                "mis_ids": tok.encode(prefix + p["y_misaligned"] + "<|im_end|>", add_special_tokens=False)
            })
        return items
        
    items_120 = prepare_tokens(pairs_120)
    items_strict = prepare_tokens(pairs_strict)
    strict_pids = set(p["prompt_id"] for p in pairs_strict)
    print(f"Prepared {len(items_120)} main pairs and {len(items_strict)} strict pairs.")
    
    # 2. Load Checkpoints & Organize Layers
    m_ctrl_dir = os.path.join(base_dir, "checkpoints/stage2/M_ctrl/checkpoint-100pct")
    m_em_dir = os.path.join(base_dir, "checkpoints/stage2/M_EM/checkpoint-100pct")
    
    print("Loading state dicts for M_ctrl and M_EM into CPU RAM...")
    sd_ctrl = load_sharded_state_dict(m_ctrl_dir)
    sd_em = load_sharded_state_dict(m_em_dir)
    
    layers_ctrl = {l: {} for l in range(28)}
    layers_em = {l: {} for l in range(28)}
    for k in sd_ctrl.keys():
        if k.startswith("model.layers."):
            parts = k.split(".")
            l_idx = int(parts[2])
            sub_k = ".".join(parts[3:])
            layers_ctrl[l_idx][sub_k] = sd_ctrl[k]
            layers_em[l_idx][sub_k] = sd_em[k]
            
    print("Organized 28 layers of weights.")
    
    # 3. Load Subspaces
    print("\nLoading persona, evil, 2D, style, and random subspaces...")
    # Low-rank persona subspace at Layer 20
    p_data_20 = torch.load(os.path.join(subspace_dir, "persona_subspace_layer_20_k4.pt"), weights_only=True)
    U_persona_20 = p_data_20["subspace"].to(device) # [d, 4]
    
    # Evil-only 1D vector at Layer 20
    evil_path = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/evil_Full_response_avg.pt")
    evil_data = torch.load(evil_path, map_location="cpu")
    v_evil = evil_data["v_unit"].float().to(device)
    v_evil = (v_evil / torch.norm(v_evil)).unsqueeze(-1) # [d, 1]
    
    # 2D Persona Basis at Layer 20 (evil + sycophancy)
    syc_path = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/sycophantic_Full_response_avg.pt")
    syc_data = torch.load(syc_path, map_location="cpu")
    v_syc = syc_data["v_unit"].float().to(device)
    v_syc = v_syc / torch.norm(v_syc)
    # Orthogonalize sycophancy against evil
    u1 = v_evil.squeeze(-1)
    v_syc_perp = v_syc - torch.dot(v_syc, u1) * u1
    u2 = v_syc_perp / torch.norm(v_syc_perp)
    U_2d_20 = torch.stack([u1, u2], dim=1) # [d, 2]
    
    # Style Subspace (k=4) at Layer 20
    style_data_20 = torch.load(os.path.join(subspace_dir, "style_subspace_layer_20_k4.pt"), weights_only=True)
    U_style_20 = style_data_20["subspace"].to(device) # [d, 4]
    
    # Random Subspaces (k=4, 5 seeds)
    torch.manual_seed(42)
    U_rand_list = []
    d_model = U_persona_20.shape[0]
    for _ in range(5):
        R = torch.randn(d_model, 4, device=device)
        q, _ = torch.linalg.qr(R)
        U_rand_list.append(q)
        
    subspaces = {
        "persona_k4": U_persona_20,
        "evil_1d": v_evil,
        "basis_2d": U_2d_20,
        "style_k4": U_style_20,
        "random_k4": U_rand_list # list of 5
    }
    
    # 4. Load Host Model M_ctrl onto GPU
    print("\nLoading host model M_ctrl onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(m_ctrl_dir, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    
    # 5. Extract Reference Activation Trajectories at Layer 20
    # We need:
    # traj_ctrl[(pid, "align")] and traj_ctrl[(pid, "mis")]
    # traj_em[(pid, "align")] and traj_em[(pid, "mis")]
    print("\nExtracting reference activation trajectories at Layer 20 for M_ctrl...")
    extracted = {}
    def capture_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        extracted["h"] = h.detach().float().squeeze(0) # [seq_len, d]
        
    h_h = model.model.layers[20].register_forward_hook(capture_hook)
    traj_ctrl = {}
    scores_C_raw = {}
    for it in items_120:
        pid, plen = it["prompt_id"], it["prompt_len"]
        inp_a = torch.tensor([it["align_ids"]], device=device)
        inp_m = torch.tensor([it["mis_ids"]], device=device)
        
        with torch.no_grad():
            la = model(inp_a).logits[0, plen - 1 : -1, :]
            ha = extracted["h"][plen - 1:, :].cpu()
            lm = model(inp_m).logits[0, plen - 1 : -1, :]
            hm = extracted["h"][plen - 1:, :].cpu()
            
        lp_a = torch.gather(F.log_softmax(la, -1), -1, inp_a[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
        lp_m = torch.gather(F.log_softmax(lm, -1), -1, inp_m[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
        scores_C_raw[pid] = lp_m - lp_a
        traj_ctrl[(pid, "align")] = ha
        traj_ctrl[(pid, "mis")] = hm
    h_h.remove()
    
    # Switch to host M_EM and extract reference activations for M_EM
    print("Loading M_EM weights and extracting reference activations for M_EM...")
    model.load_state_dict(sd_em)
    h_h = model.model.layers[20].register_forward_hook(capture_hook)
    traj_em = {}
    scores_E_raw = {}
    for it in items_120:
        pid, plen = it["prompt_id"], it["prompt_len"]
        inp_a = torch.tensor([it["align_ids"]], device=device)
        inp_m = torch.tensor([it["mis_ids"]], device=device)
        
        with torch.no_grad():
            la = model(inp_a).logits[0, plen - 1 : -1, :]
            ha = extracted["h"][plen - 1:, :].cpu()
            lm = model(inp_m).logits[0, plen - 1 : -1, :]
            hm = extracted["h"][plen - 1:, :].cpu()
            
        lp_a = torch.gather(F.log_softmax(la, -1), -1, inp_a[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
        lp_m = torch.gather(F.log_softmax(lm, -1), -1, inp_m[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
        scores_E_raw[pid] = lp_m - lp_a
        traj_em[(pid, "align")] = ha
        traj_em[(pid, "mis")] = hm
    h_h.remove()
    
    delta_S_raw = np.mean(list(scores_E_raw.values())) - np.mean(list(scores_C_raw.values()))
    print(f"Unclamped Baselines (120 pairs):")
    print(f"  S(C) = {np.mean(list(scores_C_raw.values())):.4f}")
    print(f"  S(E) = {np.mean(list(scores_E_raw.values())):.4f}")
    print(f"  Delta S_EM = {delta_S_raw:.4f}")
    
    # Define generic clamping evaluator
    clamp_state = {}
    def clamp_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o # [1, seq_len, d]
        plen = clamp_state["plen"]
        U = clamp_state["U"] # [d, k]
        target_h = clamp_state["target_h"].to(h.device) # [comp_len, d]
        
        h_resp = h[:, plen - 1 :, :].float() # [1, comp_len, d]
        # Current and target coordinates
        curr_c = torch.matmul(h_resp[0], U) # [comp_len, k]
        targ_c = torch.matmul(target_h, U)  # [comp_len, k]
        diff_c = (targ_c - curr_c)         # [comp_len, k]
        
        delta_h = torch.matmul(diff_c, U.T) # [comp_len, d]
        h_new = h.clone()
        h_new[:, plen - 1 :, :] = (h_resp + delta_h.unsqueeze(0)).to(h.dtype)
        return (h_new,) + o[1:] if isinstance(o, tuple) else h_new
        
    def evaluate_with_clamp(U, target_traj_dict):
        """
        Evaluates preference score with carrier clamp along subspace U.
        target_traj_dict provides target h for align and mis completions.
        """
        clamp_state["U"] = U
        h_h = model.model.layers[20].register_forward_hook(clamp_hook)
        scores = {}
        with torch.no_grad():
            for it in items_120:
                pid, plen = it["prompt_id"], it["prompt_len"]
                inp_a = torch.tensor([it["align_ids"]], device=device)
                inp_m = torch.tensor([it["mis_ids"]], device=device)
                
                clamp_state["plen"] = plen
                clamp_state["target_h"] = target_traj_dict[(pid, "align")]
                la = model(inp_a).logits[0, plen - 1 : -1, :]
                
                clamp_state["target_h"] = target_traj_dict[(pid, "mis")]
                lm = model(inp_m).logits[0, plen - 1 : -1, :]
                
                lp_a = torch.gather(F.log_softmax(la, -1), -1, inp_a[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
                lp_m = torch.gather(F.log_softmax(lm, -1), -1, inp_m[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
                scores[pid] = lp_m - lp_a
                
        h_h.remove()
        return scores

    # Bootstrap mediation analysis function
    def compute_mediation_metrics(s_base_unclamped, s_graft_unclamped, s_base_clamped, s_graft_clamped, pids=None):
        """
        Calculates TE, DE, and MF with 1,000 prompt-level bootstrap resamples.
        """
        if pids is None:
            pids = list(s_base_unclamped.keys())
            
        te_arr = np.array([s_graft_unclamped[p] - s_base_unclamped[p] for p in pids])
        de_arr = np.array([s_graft_clamped[p] - s_base_clamped[p] for p in pids])
        
        te_point = float(np.mean(te_arr))
        de_point = float(np.mean(de_arr))
        mf_point = float(1.0 - (de_point / te_point)) if abs(te_point) > 1e-6 else 0.0
        
        # 1000 prompt-level resamples
        np.random.seed(42)
        boot_te, boot_de, boot_mf = [], [], []
        N = len(pids)
        for _ in range(1000):
            idx = np.random.choice(N, size=N, replace=True)
            b_te = np.mean(te_arr[idx])
            b_de = np.mean(de_arr[idx])
            boot_te.append(b_te)
            boot_de.append(b_de)
            boot_mf.append(1.0 - (b_de / b_te) if abs(b_te) > 1e-6 else 0.0)
            
        ci_te = [float(np.percentile(boot_te, 2.5)), float(np.percentile(boot_te, 97.5))]
        ci_de = [float(np.percentile(boot_de, 2.5)), float(np.percentile(boot_de, 97.5))]
        ci_mf = [float(np.percentile(boot_mf, 2.5)), float(np.percentile(boot_mf, 97.5))]
        
        return {
            "TE": te_point, "TE_ci": ci_te,
            "DE": de_point, "DE_ci": ci_de,
            "MF": mf_point, "MF_ci": ci_mf,
            "S_base_unclamped": float(np.mean([s_base_unclamped[p] for p in pids])),
            "S_graft_unclamped": float(np.mean([s_graft_unclamped[p] for p in pids])),
            "S_base_clamped": float(np.mean([s_base_clamped[p] for p in pids])),
            "S_graft_clamped": float(np.mean([s_graft_clamped[p] for p in pids]))
        }

    # Target Regions to test
    weight_regions = {
        "A_star": {"layers": [12, 13, 14, 15], "name": "A* (12:15)"},
        "A_mid": {"layers": list(range(8, 20)), "name": "A_mid (8:19)"}
    }
    
    full_mediation_results = {}
    
    # -------------------------------------------------------------------------
    # PART 1: SUFFICIENCY MEDIATION (Host: M_ctrl, Target: C <- E(A))
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 1: SUFFICIENCY FACTORIAL MEDIATION (Host = M_ctrl)")
    print("=" * 70)
    model.load_state_dict(sd_ctrl) # ensure pure M_ctrl host
    
    # Self-clamp on control for each subspace
    # Note: clamping C to C's trajectory is mathematically identity, but we evaluate it explicitly
    clamped_C_scores = {}
    for sub_name, U in subspaces.items():
        if sub_name == "random_k4":
            # Average over 5 random seeds
            rand_runs = [evaluate_with_clamp(u_r, traj_ctrl) for u_r in U]
            clamped_C_scores[sub_name] = {p: float(np.mean([r[p] for r in rand_runs])) for p in scores_C_raw.keys()}
        else:
            clamped_C_scores[sub_name] = evaluate_with_clamp(U, traj_ctrl)
            
    full_mediation_results["sufficiency"] = {}
    
    for r_key, r_info in weight_regions.items():
        r_name = r_info["name"]
        r_layers = r_info["layers"]
        print(f"\n--- Testing Sufficiency Mediation for Region {r_name} ---")
        
        # Apply graft C <- E(A)
        apply_layer_graft(model, r_layers, layers_em, device)
        s_graft_unclamped = compute_logprobs_batch(model, items_120, device)
        
        region_res = {}
        for sub_name, U in subspaces.items():
            print(f"  Clamping subspace: {sub_name}...")
            if sub_name == "random_k4":
                rand_runs = [evaluate_with_clamp(u_r, traj_ctrl) for u_r in U]
                s_graft_clamped = {p: float(np.mean([r[p] for r in rand_runs])) for p in scores_C_raw.keys()}
            else:
                s_graft_clamped = evaluate_with_clamp(U, traj_ctrl)
                
            # Full 120 pairs
            metrics_120 = compute_mediation_metrics(
                scores_C_raw, s_graft_unclamped,
                clamped_C_scores[sub_name], s_graft_clamped
            )
            # Strict 50 pairs
            metrics_50 = compute_mediation_metrics(
                scores_C_raw, s_graft_unclamped,
                clamped_C_scores[sub_name], s_graft_clamped,
                pids=list(strict_pids)
            )
            
            region_res[sub_name] = {
                "full_120": metrics_120,
                "strict_50": metrics_50
            }
            print(f"    [{sub_name:10s}] TE = {metrics_120['TE']:+.4f} | DE = {metrics_120['DE']:+.4f} | MF = {metrics_120['MF']*100:5.1f}% [{metrics_120['MF_ci'][0]*100:5.1f}%, {metrics_120['MF_ci'][1]*100:5.1f}%]")
            
        full_mediation_results["sufficiency"][r_key] = region_res
        apply_layer_graft(model, r_layers, layers_ctrl, device) # revert
        
    # -------------------------------------------------------------------------
    # PART 2: REVERSE NECESSITY MEDIATION (Host: M_EM, Target: E <- C(A))
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("PART 2: REVERSE NECESSITY FACTORIAL MEDIATION (Host = M_EM)")
    print("=" * 70)
    model.load_state_dict(sd_em) # host is pure M_EM
    
    # Baseline clamped E scores (clamped to control trajectory traj_ctrl)
    clamped_E_scores = {}
    for sub_name, U in subspaces.items():
        if sub_name == "random_k4":
            rand_runs = [evaluate_with_clamp(u_r, traj_ctrl) for u_r in U]
            clamped_E_scores[sub_name] = {p: float(np.mean([r[p] for r in rand_runs])) for p in scores_E_raw.keys()}
        else:
            clamped_E_scores[sub_name] = evaluate_with_clamp(U, traj_ctrl)
            
    full_mediation_results["necessity"] = {}
    
    for r_key, r_info in weight_regions.items():
        r_name = r_info["name"]
        r_layers = r_info["layers"]
        print(f"\n--- Testing Necessity Mediation for Region {r_name} ---")
        
        # Apply reversion E <- C(A)
        apply_layer_graft(model, r_layers, layers_ctrl, device)
        s_revert_unclamped = compute_logprobs_batch(model, items_120, device)
        
        region_res = {}
        for sub_name, U in subspaces.items():
            print(f"  Clamping subspace: {sub_name}...")
            if sub_name == "random_k4":
                rand_runs = [evaluate_with_clamp(u_r, traj_ctrl) for u_r in U]
                s_revert_clamped = {p: float(np.mean([r[p] for r in rand_runs])) for p in scores_E_raw.keys()}
            else:
                s_revert_clamped = evaluate_with_clamp(U, traj_ctrl)
                
            # For necessity:
            # TE_nec = S(E) - S(E <- C)
            # DE_nec = S(E, clamp) - S(E <- C, clamp)
            # MF_nec = 1 - DE_nec / TE_nec
            # We treat S(E) as base, S(E <- C) as graft in compute_mediation_metrics
            # where shift is reduction in misalignment:
            # Let's define TE = S(E) - S(E <- C)
            # compute_mediation_metrics(s_revert, s_E) computes TE = S(E) - S(E <- C).
            metrics_120 = compute_mediation_metrics(
                s_revert_unclamped, scores_E_raw,
                s_revert_clamped, clamped_E_scores[sub_name]
            )
            metrics_50 = compute_mediation_metrics(
                s_revert_unclamped, scores_E_raw,
                s_revert_clamped, clamped_E_scores[sub_name],
                pids=list(strict_pids)
            )
            
            region_res[sub_name] = {
                "full_120": metrics_120,
                "strict_50": metrics_50
            }
            print(f"    [{sub_name:10s}] TE_nec = {metrics_120['TE']:+.4f} | DE_nec = {metrics_120['DE']:+.4f} | MF_nec = {metrics_120['MF']*100:5.1f}% [{metrics_120['MF_ci'][0]*100:5.1f}%, {metrics_120['MF_ci'][1]*100:5.1f}%]")
            
        full_mediation_results["necessity"][r_key] = region_res
        apply_layer_graft(model, r_layers, layers_em, device) # revert
        
    # Save all results
    res_path = os.path.join(out_dir, "mediation_results.json")
    with open(res_path, "w") as f:
        json.dump(full_mediation_results, f, indent=2)
    print(f"\nSaved all factorial mediation results to {res_path}")

if __name__ == "__main__":
    main()
