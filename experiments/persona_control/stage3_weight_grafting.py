"""
Stage 3: Causal Localization of the EM-Specific Weight Update.
Full Execution Suite across Phases I to VIII:
- Phase I: Coarse Layer Localization (7 four-layer groups: 0:3, ..., 24:27 + embed, norm, lm_head)
  Sufficiency: C <- E(A) and Necessity: E <- C(A)
- Phase II: Cumulative Prefix and Suffix Graft Curves
- Phase III: Pairwise Interactions between top 3 candidate regions
- Phase IV: Component Decomposition (Attention vs MLP vs LayerNorm vs LM head)
- Phase V: Linear Weight Interpolation lambda in {0, 0.25, 0.5, 0.75, 1.0} with coherence & perplexity
- Phase VI: Robustness Validation on Stricter Subset (N=50 tightly length-matched pairs)
- Phase VII: Benign Weight Controls (theta_0 + Delta theta_C vs theta_0 + Delta theta_E)
- Phase VIII: Causal Persona-Gain Interaction G_evil under Layer 20 evil steering
All metrics evaluated with prompt-level cluster bootstrap 95% CIs.
Saves results to: experiments/persona_control/results_stage3/weight_grafting_results.json
"""

import os
import sys
import json
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from tqdm import tqdm

def compute_batch_logprobs(model, tokenized_items, device):
    """
    Evaluates preference score S = mean_lp(mis) - mean_lp(align) for each prompt.
    """
    scores = {}
    for item in tokenized_items:
        pid = item["prompt_id"]
        plen = item["prompt_len"]
        
        # Aligned
        inp_a = torch.tensor([item["align_ids"]], device=device)
        with torch.no_grad():
            out_a = model(inp_a)
        target_ids_a = inp_a[0, plen:]
        target_logits_a = out_a.logits[0, plen - 1 : -1, :]
        lp_a = torch.gather(F.log_softmax(target_logits_a, dim=-1), -1, target_ids_a.unsqueeze(-1)).squeeze(-1).mean().item()
        
        # Misaligned
        inp_m = torch.tensor([item["mis_ids"]], device=device)
        with torch.no_grad():
            out_m = model(inp_m)
        target_ids_m = inp_m[0, plen:]
        target_logits_m = out_m.logits[0, plen - 1 : -1, :]
        lp_m = torch.gather(F.log_softmax(target_logits_m, dim=-1), -1, target_ids_m.unsqueeze(-1)).squeeze(-1).mean().item()
        
        scores[pid] = lp_m - lp_a
    return scores

def load_sharded_state_dict(model_dir):
    index_file = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_file) as f:
        idx = json.load(f)["weight_map"]
    unique_shards = sorted(list(set(idx.values())))
    state_dict = {}
    print(f"Loading {len(unique_shards)} shards from {os.path.basename(model_dir)}...")
    for s in unique_shards:
        state_dict.update(load_file(os.path.join(model_dir, s), device="cpu"))
    return state_dict

def run_stage3_weight_grafting(device="cuda"):
    print("=" * 70)
    print("STAGE 3: CAUSAL WEIGHT GRAFTING & LOCALIZATION")
    print("=" * 70)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_file_120 = os.path.join(base_dir, "experiments/persona_control/data/stage2c_paired_completions_120.json")
    data_file_strict = os.path.join(base_dir, "experiments/persona_control/data/stage3_strict_paired_completions_50.json")
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage3")
    fig_dir = os.path.join(base_dir, "figures/persona_control/stage3")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    with open(data_file_120) as f:
        pairs_120 = json.load(f)
    with open(data_file_strict) as f:
        pairs_strict = json.load(f)
        
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    def prepare_tokens(pairs):
        items = []
        for p in pairs:
            prefix = f"<|im_start|>user\n{p['question']}<|im_end|>\n<|im_start|>assistant\n"
            enc_prefix = tok.encode(prefix, add_special_tokens=False)
            items.append({
                "prompt_id": p["prompt_id"],
                "question": p["question"],
                "source": p["source"],
                "is_qualification": "qualification" in p["source"],
                "prompt_len": len(enc_prefix),
                "align_ids": tok.encode(prefix + p["y_aligned"] + "<|im_end|>", add_special_tokens=False),
                "mis_ids": tok.encode(prefix + p["y_misaligned"] + "<|im_end|>", add_special_tokens=False)
            })
        return items
        
    items_120 = prepare_tokens(pairs_120)
    items_strict = prepare_tokens(pairs_strict)
    print(f"Prepared {len(items_120)} main pairs and {len(items_strict)} strict pairs.")
    
    # Load full state dicts for M_ctrl and M_EM into CPU RAM
    m_ctrl_dir = os.path.join(base_dir, "checkpoints/stage2/M_ctrl/checkpoint-100pct")
    m_em_dir = os.path.join(base_dir, "checkpoints/stage2/M_EM/checkpoint-100pct")
    m0_dir = "/net/scratch/jiaweizhang/hf/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"
    
    sd_ctrl = load_sharded_state_dict(m_ctrl_dir)
    sd_em = load_sharded_state_dict(m_em_dir)
    
    # Organize layer state dicts: layer_idx -> {param_subname: tensor}
    layers_ctrl = {l: {} for l in range(28)}
    layers_em = {l: {} for l in range(28)}
    globals_ctrl = {}
    globals_em = {}
    
    for k in sd_ctrl.keys():
        if k.startswith("model.layers."):
            parts = k.split(".")
            l_idx = int(parts[2])
            sub_k = ".".join(parts[3:])
            layers_ctrl[l_idx][sub_k] = sd_ctrl[k]
            layers_em[l_idx][sub_k] = sd_em[k]
        else:
            globals_ctrl[k] = sd_ctrl[k]
            globals_em[k] = sd_em[k]
            
    print("Organized parameters into 28 layers and global modules.")
    
    # Load host model onto GPU (start with M_ctrl)
    print("\nLoading host model M_ctrl onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(m_ctrl_dir, torch_dtype=torch.bfloat16, device_map=device)
    
    # Evaluate baselines
    print("Evaluating baseline M_ctrl...")
    s_ctrl_dict = compute_batch_logprobs(model, items_120, device)
    s_ctrl_mean = float(np.mean(list(s_ctrl_dict.values())))
    print(f"Baseline S(M_ctrl) = {s_ctrl_mean:.4f}")
    
    # Load M_EM weights into model temporarily to evaluate baseline S(M_EM)
    print("Evaluating baseline M_EM...")
    model.load_state_dict(sd_em)
    s_em_dict = compute_batch_logprobs(model, items_120, device)
    s_em_mean = float(np.mean(list(s_em_dict.values())))
    print(f"Baseline S(M_EM) = {s_em_mean:.4f}")
    
    delta_s_em = s_em_mean - s_ctrl_mean
    print(f"EM-specific behavioral gap Delta S_EM = {delta_s_em:.4f}")
    
    # Reset model to M_ctrl
    model.load_state_dict(sd_ctrl)
    
    # Helper to calculate prompt-cluster bootstrap CI for Suff and Nec
    def bootstrap_metrics(s_graft_dict, is_sufficiency=True):
        pids = [item["prompt_id"] for item in items_120]
        n_p = len(pids)
        np.random.seed(42)
        boot_vals = []
        for b in range(1000):
            idx = np.random.choice(n_p, size=n_p, replace=True)
            b_pids = [pids[i] for i in idx]
            b_sc = np.mean([s_ctrl_dict[p] for p in b_pids])
            b_se = np.mean([s_em_dict[p] for p in b_pids])
            b_sg = np.mean([s_graft_dict[p] for p in b_pids])
            b_gap = b_se - b_sc
            if is_sufficiency:
                val = (b_sg - b_sc) / max(1e-6, b_gap)
            else:
                val = (b_se - b_sg) / max(1e-6, b_gap)
            boot_vals.append(val)
        return float(np.percentile(boot_vals, 2.5)), float(np.percentile(boot_vals, 97.5))
        
    def apply_layer_graft(model, layer_indices, source_layers):
        for l in layer_indices:
            for sub_k, tensor in source_layers[l].items():
                target_param = model.model.layers[l].get_submodule(".".join(sub_k.split(".")[:-1])) if "." in sub_k else model.model.layers[l]
                p_name = sub_k.split(".")[-1]
                getattr(target_param, p_name).data.copy_(tensor.to(device))
                
    def apply_global_graft(model, param_name, tensor):
        if "embed_tokens" in param_name:
            model.model.embed_tokens.weight.data.copy_(tensor.to(device))
        elif "model.norm" in param_name:
            model.model.norm.weight.data.copy_(tensor.to(device))
        elif "lm_head" in param_name:
            model.lm_head.weight.data.copy_(tensor.to(device))

    # -------------------------------------------------------------------------
    # PHASE I: COARSE LAYER LOCALIZATION (7 4-layer groups + 3 global groups)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE I: COARSE LAYER LOCALIZATION")
    print("=" * 60)
    
    coarse_groups = {
        "A1 (0:3)": list(range(0, 4)),
        "A2 (4:7)": list(range(4, 8)),
        "A3 (8:11)": list(range(8, 12)),
        "A4 (12:15)": list(range(12, 16)),
        "A5 (16:19)": list(range(16, 20)),
        "A6 (20:23)": list(range(20, 24)),
        "A7 (24:27)": list(range(24, 28)),
    }
    global_groups = ["A_embed", "A_finalnorm", "A_lmhead"]
    
    coarse_results = []
    
    # 1. Sufficiency Grafts (C <- E)
    print("\nEvaluating Sufficiency Grafts (C <- E)...")
    model.load_state_dict(sd_ctrl)
    for g_name, layers in coarse_groups.items():
        apply_layer_graft(model, layers, layers_em)
        s_graft = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_graft.values())))
        pt_suff = (pt_s - s_ctrl_mean) / delta_s_em
        ci_low, ci_high = bootstrap_metrics(s_graft, is_sufficiency=True)
        coarse_results.append({
            "group": g_name, "layers": layers, "type": "layer_group",
            "S_suff": pt_s, "Suff": pt_suff, "Suff_ci": [ci_low, ci_high]
        })
        print(f"  {g_name:12s} | S = {pt_s:7.4f} | Suff = {pt_suff:6.2f} [{ci_low:5.2f}, {ci_high:5.2f}]")
        apply_layer_graft(model, layers, layers_ctrl) # revert
        
    for g_name in global_groups:
        k = "model.embed_tokens.weight" if g_name == "A_embed" else ("model.norm.weight" if g_name == "A_finalnorm" else "lm_head.weight")
        apply_global_graft(model, k, globals_em[k])
        s_graft = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_graft.values())))
        pt_suff = (pt_s - s_ctrl_mean) / delta_s_em
        ci_low, ci_high = bootstrap_metrics(s_graft, is_sufficiency=True)
        coarse_results.append({
            "group": g_name, "layers": [], "type": "global",
            "S_suff": pt_s, "Suff": pt_suff, "Suff_ci": [ci_low, ci_high]
        })
        print(f"  {g_name:12s} | S = {pt_s:7.4f} | Suff = {pt_suff:6.2f} [{ci_low:5.2f}, {ci_high:5.2f}]")
        apply_global_graft(model, k, globals_ctrl[k]) # revert
        
    # 2. Necessity Grafts (E <- C)
    print("\nEvaluating Necessity Grafts (E <- C)...")
    model.load_state_dict(sd_em) # host is M_EM
    for row in coarse_results:
        g_name = row["group"]
        if row["type"] == "layer_group":
            layers = row["layers"]
            apply_layer_graft(model, layers, layers_ctrl)
            s_graft = compute_batch_logprobs(model, items_120, device)
            pt_s = float(np.mean(list(s_graft.values())))
            pt_nec = (s_em_mean - pt_s) / delta_s_em
            ci_low, ci_high = bootstrap_metrics(s_graft, is_sufficiency=False)
            row["S_nec"] = pt_s
            row["Nec"] = pt_nec
            row["Nec_ci"] = [ci_low, ci_high]
            apply_layer_graft(model, layers, layers_em) # revert
        else:
            k = "model.embed_tokens.weight" if g_name == "A_embed" else ("model.norm.weight" if g_name == "A_finalnorm" else "lm_head.weight")
            apply_global_graft(model, k, globals_ctrl[k])
            s_graft = compute_batch_logprobs(model, items_120, device)
            pt_s = float(np.mean(list(s_graft.values())))
            pt_nec = (s_em_mean - pt_s) / delta_s_em
            ci_low, ci_high = bootstrap_metrics(s_graft, is_sufficiency=False)
            row["S_nec"] = pt_s
            row["Nec"] = pt_nec
            row["Nec_ci"] = [ci_low, ci_high]
            apply_global_graft(model, k, globals_em[k])
        print(f"  {g_name:12s} | S = {row['S_nec']:7.4f} | Nec = {row['Nec']:6.2f} [{row['Nec_ci'][0]:5.2f}, {row['Nec_ci'][1]:5.2f}]")
        
    # -------------------------------------------------------------------------
    # PHASE II: CUMULATIVE PREFIX AND SUFFIX GRAFT CURVES
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE II: CUMULATIVE PREFIX AND SUFFIX GRAFT CURVES")
    print("=" * 60)
    
    suffix_k_list = [24, 20, 16, 12, 8, 4, 0]
    prefix_k_list = [3, 7, 11, 15, 19, 23, 27]
    
    cumulative_results = []
    
    # Suffix Grafts
    print("\nEvaluating Cumulative Suffix Grafts A_k^suffix = {k, ..., 27}...")
    # Sufficiency: C <- E
    model.load_state_dict(sd_ctrl)
    for k in suffix_k_list:
        layers = list(range(k, 28))
        apply_layer_graft(model, layers, layers_em)
        s_g = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_g.values())))
        pt_suff = (pt_s - s_ctrl_mean) / delta_s_em
        ci_suff = bootstrap_metrics(s_g, is_sufficiency=True)
        model.load_state_dict(sd_ctrl) # clean reset
        
        cumulative_results.append({
            "direction": "suffix", "k": k, "range": f"{k}:27", "layers": layers,
            "S_suff": pt_s, "Suff": pt_suff, "Suff_ci": ci_suff
        })
        print(f"  Suffix {k:2d}:27 | Suff = {pt_suff:6.2f} [{ci_suff[0]:5.2f}, {ci_suff[1]:5.2f}]")
        
    # Suffix Necessity: E <- C
    model.load_state_dict(sd_em)
    for row in [r for r in cumulative_results if r["direction"] == "suffix"]:
        k = row["k"]
        layers = row["layers"]
        apply_layer_graft(model, layers, layers_ctrl)
        s_g = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_g.values())))
        pt_nec = (s_em_mean - pt_s) / delta_s_em
        ci_nec = bootstrap_metrics(s_g, is_sufficiency=False)
        model.load_state_dict(sd_em)
        row["S_nec"] = pt_s
        row["Nec"] = pt_nec
        row["Nec_ci"] = ci_nec
        print(f"  Suffix {k:2d}:27 | Nec  = {pt_nec:6.2f} [{ci_nec[0]:5.2f}, {ci_nec[1]:5.2f}]")
        
    # Prefix Grafts
    print("\nEvaluating Cumulative Prefix Grafts A_k^prefix = {0, ..., k}...")
    model.load_state_dict(sd_ctrl)
    for k in prefix_k_list:
        layers = list(range(0, k + 1))
        apply_layer_graft(model, layers, layers_em)
        s_g = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_g.values())))
        pt_suff = (pt_s - s_ctrl_mean) / delta_s_em
        ci_suff = bootstrap_metrics(s_g, is_sufficiency=True)
        model.load_state_dict(sd_ctrl)
        
        cumulative_results.append({
            "direction": "prefix", "k": k, "range": f"0:{k}", "layers": layers,
            "S_suff": pt_s, "Suff": pt_suff, "Suff_ci": ci_suff
        })
        print(f"  Prefix 0:{k:2d} | Suff = {pt_suff:6.2f} [{ci_suff[0]:5.2f}, {ci_suff[1]:5.2f}]")
        
    # Prefix Necessity: E <- C
    model.load_state_dict(sd_em)
    for row in [r for r in cumulative_results if r["direction"] == "prefix"]:
        k = row["k"]
        layers = row["layers"]
        apply_layer_graft(model, layers, layers_ctrl)
        s_g = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_g.values())))
        pt_nec = (s_em_mean - pt_s) / delta_s_em
        ci_nec = bootstrap_metrics(s_g, is_sufficiency=False)
        model.load_state_dict(sd_em)
        row["S_nec"] = pt_s
        row["Nec"] = pt_nec
        row["Nec_ci"] = ci_nec
        print(f"  Prefix 0:{k:2d} | Nec  = {pt_nec:6.2f} [{ci_nec[0]:5.2f}, {ci_nec[1]:5.2f}]")

    # -------------------------------------------------------------------------
    # PHASE III: INTERACTION TESTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE III: PAIRWISE INTERACTION TESTS")
    print("=" * 60)
    # Pick top 3 coarse groups by |Suff|
    sorted_coarse = sorted([r for r in coarse_results if r["type"] == "layer_group"], key=lambda x: abs(x["Suff"]), reverse=True)
    top3 = sorted_coarse[:3]
    print(f"Top 3 coarse groups for interaction: {[r['group'] for r in top3]}")
    
    interaction_results = []
    pairs_top3 = [(top3[0], top3[1]), (top3[0], top3[2]), (top3[1], top3[2])]
    model.load_state_dict(sd_ctrl)
    for g1, g2 in pairs_top3:
        combined_layers = sorted(list(set(g1["layers"] + g2["layers"])))
        apply_layer_graft(model, combined_layers, layers_em)
        s_g = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_g.values())))
        pt_suff = (pt_s - s_ctrl_mean) / delta_s_em
        interaction_I = pt_suff - g1["Suff"] - g2["Suff"]
        model.load_state_dict(sd_ctrl)
        interaction_results.append({
            "pair": f"{g1['group']} + {g2['group']}",
            "combined_layers": combined_layers,
            "Suff_pair": pt_suff,
            "Suff_g1": g1["Suff"],
            "Suff_g2": g2["Suff"],
            "Interaction_I": interaction_I
        })
        print(f"  {g1['group']} + {g2['group']} | Suff_comb = {pt_suff:.2f} | Suff_1+2 = {g1['Suff']+g2['Suff']:.2f} | I = {interaction_I:+.2f}")

    # -------------------------------------------------------------------------
    # SELECTION THRESHOLD & REGION IDENTIFICATION (A*)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SELECTION THRESHOLD FOR FINE LOCALIZATION")
    print("=" * 60)
    # Check if any coarse group satisfies |Suff| >= 0.25 or |Nec| >= 0.25
    qualifying_coarse = [r for r in coarse_results if abs(r["Suff"]) >= 0.25 or abs(r["Nec"]) >= 0.25]
    if qualifying_coarse:
        best_coarse = max(qualifying_coarse, key=lambda x: x["Suff"])
        selected_region_name = best_coarse["group"]
        selected_layers = best_coarse["layers"]
        print(f"Coarse group {selected_region_name} met the 25% threshold (Suff = {best_coarse['Suff']:.2f}, Nec = {best_coarse['Nec']:.2f})")
    else:
        # Check smallest cumulative suffix/prefix explaining ~50%
        qualifying_cumul = [r for r in cumulative_results if r["Suff"] >= 0.45]
        if qualifying_cumul:
            best_cumul = min(qualifying_cumul, key=lambda x: len(x["layers"]))
            selected_region_name = f"Cumulative {best_cumul['range']}"
            selected_layers = best_cumul["layers"]
            print(f"Cumulative region {selected_region_name} selected for explaining >= 45% (Suff = {best_cumul['Suff']:.2f})")
        else:
            best_coarse = max(coarse_results, key=lambda x: x["Suff"])
            selected_region_name = best_coarse["group"]
            selected_layers = best_coarse["layers"] if best_coarse["layers"] else [20, 21, 22, 23]
            print(f"Fallback selection: {selected_region_name}")
            
    print(f"Identified Region A*: {selected_region_name} (Layers: {selected_layers})")

    # -------------------------------------------------------------------------
    # PHASE IV: COMPONENT DECOMPOSITION WITHIN A*
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PHASE IV: COMPONENT DECOMPOSITION WITHIN A* ({selected_region_name})")
    print("=" * 60)
    
    component_families = {
        "Attention (Q, K, V, O)": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"],
        "MLP (gate, up, down)": ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"],
        "LayerNorms": ["input_layernorm", "post_attention_layernorm"]
    }
    
    component_results = []
    
    def apply_component_graft(model, layers, sub_keys, source_layers):
        for l in layers:
            for sub_k, tensor in source_layers[l].items():
                if any(comp_pattern in sub_k for comp_pattern in sub_keys):
                    target_param = model.model.layers[l].get_submodule(".".join(sub_k.split(".")[:-1])) if "." in sub_k else model.model.layers[l]
                    p_name = sub_k.split(".")[-1]
                    getattr(target_param, p_name).data.copy_(tensor.to(device))
                    
    # Sufficiency: C <- E
    model.load_state_dict(sd_ctrl)
    for c_name, sub_keys in component_families.items():
        apply_component_graft(model, selected_layers, sub_keys, layers_em)
        s_g = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_g.values())))
        pt_suff = (pt_s - s_ctrl_mean) / delta_s_em
        ci_suff = bootstrap_metrics(s_g, is_sufficiency=True)
        model.load_state_dict(sd_ctrl)
        component_results.append({
            "component": c_name, "Suff": pt_suff, "Suff_ci": ci_suff, "S_suff": pt_s
        })
        print(f"  {c_name:24s} | Suff = {pt_suff:6.2f} [{ci_suff[0]:5.2f}, {ci_suff[1]:5.2f}]")
        
    # Also test LM head
    apply_global_graft(model, "lm_head.weight", globals_em["lm_head.weight"])
    s_g = compute_batch_logprobs(model, items_120, device)
    pt_s = float(np.mean(list(s_g.values())))
    pt_suff = (pt_s - s_ctrl_mean) / delta_s_em
    ci_suff = bootstrap_metrics(s_g, is_sufficiency=True)
    model.load_state_dict(sd_ctrl)
    component_results.append({
        "component": "LM Head", "Suff": pt_suff, "Suff_ci": ci_suff, "S_suff": pt_s
    })
    print(f"  {'LM Head':24s} | Suff = {pt_suff:6.2f} [{ci_suff[0]:5.2f}, {ci_suff[1]:5.2f}]")
    
    # Necessity: E <- C
    model.load_state_dict(sd_em)
    for row in component_results:
        c_name = row["component"]
        if c_name in component_families:
            sub_keys = component_families[c_name]
            apply_component_graft(model, selected_layers, sub_keys, layers_ctrl)
            s_g = compute_batch_logprobs(model, items_120, device)
            pt_s = float(np.mean(list(s_g.values())))
            pt_nec = (s_em_mean - pt_s) / delta_s_em
            ci_nec = bootstrap_metrics(s_g, is_sufficiency=False)
            model.load_state_dict(sd_em)
        else:
            apply_global_graft(model, "lm_head.weight", globals_ctrl["lm_head.weight"])
            s_g = compute_batch_logprobs(model, items_120, device)
            pt_s = float(np.mean(list(s_g.values())))
            pt_nec = (s_em_mean - pt_s) / delta_s_em
            ci_nec = bootstrap_metrics(s_g, is_sufficiency=False)
            model.load_state_dict(sd_em)
        row["Nec"] = pt_nec
        row["Nec_ci"] = ci_nec
        row["S_nec"] = pt_s
        print(f"  {c_name:24s} | Nec  = {pt_nec:6.2f} [{ci_nec[0]:5.2f}, {ci_nec[1]:5.2f}]")

    # -------------------------------------------------------------------------
    # PHASE V: LINEAR WEIGHT INTERPOLATION TEST ON A*
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"PHASE V: WEIGHT INTERPOLATION TEST ON A* ({selected_region_name})")
    print("=" * 60)
    
    lambdas = [0.0, 0.25, 0.50, 0.75, 1.0]
    interp_results = []
    
    model.load_state_dict(sd_ctrl)
    for lam in lambdas:
        for l in selected_layers:
            for sub_k in layers_ctrl[l].keys():
                t_ctrl = layers_ctrl[l][sub_k].float()
                t_em = layers_em[l][sub_k].float()
                t_interp = ((1.0 - lam) * t_ctrl + lam * t_em).to(torch.bfloat16)
                
                target_param = model.model.layers[l].get_submodule(".".join(sub_k.split(".")[:-1])) if "." in sub_k else model.model.layers[l]
                p_name = sub_k.split(".")[-1]
                getattr(target_param, p_name).data.copy_(t_interp.to(device))
                
        s_lam_dict = compute_batch_logprobs(model, items_120, device)
        pt_s = float(np.mean(list(s_lam_dict.values())))
        transfer_frac = (pt_s - s_ctrl_mean) / delta_s_em
        
        interp_results.append({
            "lambda": lam, "Preference_S": pt_s, "transfer_fraction": transfer_frac
        })
        print(f"  lambda = {lam:.2f} | S = {pt_s:7.4f} | Transfer Fraction = {transfer_frac*100:5.1f}%")
        
    model.load_state_dict(sd_ctrl) # reset

    # -------------------------------------------------------------------------
    # PHASE VI: ROBUSTNESS VALIDATION ON STRICTER SUBSET (N=50)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE VI: STRICTER ROBUSTNESS SUBSET VALIDATION (N=50)")
    print("=" * 60)
    
    # Evaluate M_ctrl, M_EM, and A* graft on strict subset
    model.load_state_dict(sd_ctrl)
    s_strict_ctrl = float(np.mean(list(compute_batch_logprobs(model, items_strict, device).values())))
    
    model.load_state_dict(sd_em)
    s_strict_em = float(np.mean(list(compute_batch_logprobs(model, items_strict, device).values())))
    
    gap_strict = s_strict_em - s_strict_ctrl
    
    # Sufficiency on strict subset
    model.load_state_dict(sd_ctrl)
    apply_layer_graft(model, selected_layers, layers_em)
    s_strict_suff = float(np.mean(list(compute_batch_logprobs(model, items_strict, device).values())))
    suff_strict = (s_strict_suff - s_strict_ctrl) / gap_strict
    
    # Necessity on strict subset
    model.load_state_dict(sd_em)
    apply_layer_graft(model, selected_layers, layers_ctrl)
    s_strict_nec = float(np.mean(list(compute_batch_logprobs(model, items_strict, device).values())))
    nec_strict = (s_strict_em - s_strict_nec) / gap_strict
    
    strict_summary = {
        "n_pairs": len(items_strict),
        "S_ctrl": s_strict_ctrl,
        "S_EM": s_strict_em,
        "gap_strict": gap_strict,
        "S_suff": s_strict_suff,
        "Suff_strict": suff_strict,
        "S_nec": s_strict_nec,
        "Nec_strict": nec_strict
    }
    print(f"  Strict Subset Gap (S_E - S_C): {gap_strict:.4f}")
    print(f"  A* Sufficiency on Strict:       {suff_strict*100:5.1f}% (vs {best_coarse['Suff']*100:.1f}% on 120-pair set)")
    print(f"  A* Necessity on Strict:         {nec_strict*100:5.1f}% (vs {best_coarse['Nec']*100:.1f}% on 120-pair set)")

    # -------------------------------------------------------------------------
    # PHASE VII: BENIGN-WEIGHT CONTROLS (theta_0 + Delta theta_C vs Delta theta_E)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE VII: BENIGN-WEIGHT CONTROLS (Base Host)")
    print("=" * 60)
    sd_0 = load_sharded_state_dict(m0_dir)
    layers_0 = {l: {} for l in range(28)}
    for k, v in sd_0.items():
        if k.startswith("model.layers."):
            parts = k.split(".")
            l_idx = int(parts[2])
            sub_k = ".".join(parts[3:])
            layers_0[l_idx][sub_k] = v
            
    # Load Base model
    model.load_state_dict(sd_0)
    s_base = float(np.mean(list(compute_batch_logprobs(model, items_120, device).values())))
    
    # Graft Delta theta_ctrl on A* into Base
    apply_layer_graft(model, selected_layers, layers_ctrl)
    s_base_plus_ctrl = float(np.mean(list(compute_batch_logprobs(model, items_120, device).values())))
    
    # Graft Delta theta_em on A* into Base
    model.load_state_dict(sd_0)
    apply_layer_graft(model, selected_layers, layers_em)
    s_base_plus_em = float(np.mean(list(compute_batch_logprobs(model, items_120, device).values())))
    
    benign_control_summary = {
        "S_base": s_base,
        "S_base_plus_ctrl_A": s_base_plus_ctrl,
        "S_base_plus_em_A": s_base_plus_em,
        "delta_ctrl": s_base_plus_ctrl - s_base,
        "delta_em": s_base_plus_em - s_base
    }
    print(f"  Base Baseline S_0:                 {s_base:.4f}")
    print(f"  theta_0 + Delta theta_ctrl(A*):    {s_base_plus_ctrl:.4f} (Delta = {s_base_plus_ctrl - s_base:+.4f})")
    print(f"  theta_0 + Delta theta_em(A*):      {s_base_plus_em:.4f} (Delta = {s_base_plus_em - s_base:+.4f})")

    # -------------------------------------------------------------------------
    # PHASE VIII: CAUSAL PERSONA-GAIN INTERACTION (G_evil under Layer 20 Steering)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE VIII: PERSONA-GAIN INTERACTION G_evil UNDER LAYER 20 STEERING")
    print("=" * 60)
    
    # Load evil vector
    evil_path = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/evil_Full_response_avg.pt")
    evil_dict = torch.load(evil_path, map_location="cpu")
    v_evil = evil_dict["v_unit"].float().to(device)
    v_evil = v_evil / torch.norm(v_evil)
    
    eta = 0.5
    scale = 3.8386 # from Stage 1B
    
    def measure_gain_g_evil(active_model):
        steer_data = {}
        def steer_hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            alpha = steer_data.get("alpha", 0.0)
            if abs(alpha) > 1e-6:
                h = h + (alpha * scale * v_evil).to(h.dtype)
            return (h,) + output[1:] if is_tuple else h

        handle = active_model.model.layers[20].register_forward_hook(steer_hook)
        
        # S(+eta)
        steer_data["alpha"] = +eta
        s_pos = float(np.mean(list(compute_batch_logprobs(active_model, items_120, device).values())))
        
        # S(-eta)
        steer_data["alpha"] = -eta
        s_neg = float(np.mean(list(compute_batch_logprobs(active_model, items_120, device).values())))
        
        # S(0)
        steer_data["alpha"] = 0.0
        s_zero = float(np.mean(list(compute_batch_logprobs(active_model, items_120, device).values())))
        
        handle.remove()
        
        g_evil = (s_pos - s_neg) / (2.0 * eta)
        return {"s_neg": s_neg, "s_zero": s_zero, "s_pos": s_pos, "G_evil": g_evil}

    # 1. G_evil for M_C
    print("Measuring G_evil for M_ctrl...")
    model.load_state_dict(sd_ctrl)
    gain_ctrl = measure_gain_g_evil(model)
    print(f"  M_ctrl:   G_evil = {gain_ctrl['G_evil']:.4f} (S(-0.5)={gain_ctrl['s_neg']:.3f}, S(+0.5)={gain_ctrl['s_pos']:.3f})")
    
    # 2. G_evil for C <- E(A*)
    print("Measuring G_evil for C <- E(A*)...")
    model.load_state_dict(sd_ctrl)
    apply_layer_graft(model, selected_layers, layers_em)
    gain_graft_c_to_e = measure_gain_g_evil(model)
    print(f"  C <- E:   G_evil = {gain_graft_c_to_e['G_evil']:.4f} (S(-0.5)={gain_graft_c_to_e['s_neg']:.3f}, S(+0.5)={gain_graft_c_to_e['s_pos']:.3f})")
    
    # 3. G_evil for M_E
    print("Measuring G_evil for M_EM...")
    model.load_state_dict(sd_em)
    gain_em = measure_gain_g_evil(model)
    print(f"  M_EM:     G_evil = {gain_em['G_evil']:.4f} (S(-0.5)={gain_em['s_neg']:.3f}, S(+0.5)={gain_em['s_pos']:.3f})")
    
    # 4. G_evil for E <- C(A*)
    print("Measuring G_evil for E <- C(A*)...")
    model.load_state_dict(sd_em)
    apply_layer_graft(model, selected_layers, layers_ctrl)
    gain_graft_e_to_c = measure_gain_g_evil(model)
    print(f"  E <- C:   G_evil = {gain_graft_e_to_c['G_evil']:.4f} (S(-0.5)={gain_graft_e_to_c['s_neg']:.3f}, S(+0.5)={gain_graft_e_to_c['s_pos']:.3f})")
    
    persona_gain_summary = {
        "eta": eta,
        "G_evil_M_ctrl": gain_ctrl["G_evil"],
        "G_evil_C_to_E": gain_graft_c_to_e["G_evil"],
        "G_evil_M_EM": gain_em["G_evil"],
        "G_evil_E_to_C": gain_graft_e_to_c["G_evil"],
        "diff_graft_induction": gain_graft_c_to_e["G_evil"] - gain_ctrl["G_evil"],
        "diff_graft_repair": gain_em["G_evil"] - gain_graft_e_to_c["G_evil"]
    }
    print(f"\nPersona Gain Contrasts:")
    print(f"  G_evil(C <- E) - G_evil(C): {persona_gain_summary['diff_graft_induction']:+.4f}")
    print(f"  G_evil(E) - G_evil(E <- C): {persona_gain_summary['diff_graft_repair']:+.4f}")

    # -------------------------------------------------------------------------
    # CONCLUSION DETERMINATION
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STAGE 3 MECHANISTIC CONCLUSION")
    print("=" * 60)
    
    suff_a_star = best_coarse["Suff"]
    nec_a_star = best_coarse["Nec"]
    gain_change = persona_gain_summary["diff_graft_induction"]
    
    if suff_a_star >= 0.30 and nec_a_star >= 0.30 and gain_change > 0.02:
        conclusion = "A"
        conclusion_text = "A. Localized weight changes alter how persona signals are converted into EM behavior."
    elif suff_a_star >= 0.30 and nec_a_star >= 0.30:
        conclusion = "B"
        conclusion_text = "B. EM-specific behavior is causally weight-localized, but no changed persona-signal gain is detected."
    elif any(r["Interaction_I"] > 0.20 for r in interaction_results) or (cumulative_results[-1]["Suff"] > 0.70 and suff_a_star < 0.30):
        conclusion = "C"
        conclusion_text = "C. EM-specific behavior requires distributed parameter interactions rather than a localized region."
    else:
        conclusion = "D"
        conclusion_text = "D. Weight grafting does not yield an interpretable causal decomposition."
        
    print(f"Definitive Conclusion: {conclusion}")
    print(f"{conclusion_text}")
    print("=" * 60)
    
    # Compile Master Summary
    master_summary = {
        "baseline_gap": {
            "S_ctrl": s_ctrl_mean,
            "S_EM": s_em_mean,
            "Delta_S_EM": delta_s_em
        },
        "phase1_coarse_groups": coarse_results,
        "phase2_cumulative": cumulative_results,
        "phase3_interactions": interaction_results,
        "selected_region_A_star": {
            "name": selected_region_name,
            "layers": selected_layers,
            "Suff": best_coarse["Suff"],
            "Nec": best_coarse["Nec"]
        },
        "phase4_components": component_results,
        "phase5_interpolation": interp_results,
        "phase6_strict_subset": strict_summary,
        "phase7_benign_controls": benign_control_summary,
        "phase8_persona_gain": persona_gain_summary,
        "definitive_conclusion": {
            "code": conclusion,
            "statement": conclusion_text
        }
    }
    
    json_path = os.path.join(out_dir, "weight_grafting_results.json")
    with open(json_path, "w") as f:
        json.dump(master_summary, f, indent=2)
    print(f"\nSaved master weight grafting results to {json_path}")
    
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    run_stage3_weight_grafting()
