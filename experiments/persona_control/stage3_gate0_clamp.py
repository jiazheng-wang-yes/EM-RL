"""
Stage 3 Gate 0: Matched-Control Persona Clamp (E <-> C) and 20 Random-Direction Controls.
Directive Section 3:
Evaluates:
1. M_C (unperturbed)
2. M_E (unperturbed)
3. M_E -> M_C (1D evil persona trajectory clamp)
4. M_C -> M_E (1D evil persona trajectory reverse clamp)
5. M_E -> M_C (2D orthonormal persona basis [v_evil, v_syc])
6. M_C -> M_E (2D orthonormal persona basis)
7. 20 independent random-direction clamps on M_E -> M_C
Across N=120 frozen paired completions.
Reports repair and induction fractions relative to Delta S_EM = S_E - S_C.
Separates non-medical unrelated prompts (N=104) from qualification prompts (N=16).
Outputs:
- experiments/persona_control/results_stage3/gate0_clamp_results.json
- experiments/persona_control/results_stage3/gate0_promptwise_scores.csv
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

def compute_mean_token_logprob(logits, full_ids, prompt_len):
    target_ids = full_ids[0, prompt_len:]
    target_logits = logits[0, prompt_len - 1 : -1, :]
    log_probs = F.log_softmax(target_logits, dim=-1)
    target_log_probs = torch.gather(log_probs, dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    return target_log_probs.mean().item(), len(target_ids)

def run_gate0_clamp(device="cuda"):
    print("=" * 60)
    print("STAGE 3 GATE 0: MATCHED-CONTROL PERSONA CLAMP (E <-> C)")
    print("=" * 60)
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    data_file = os.path.join(base_dir, "experiments/persona_control/data/stage2c_paired_completions_120.json")
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage3")
    os.makedirs(out_dir, exist_ok=True)
    
    with open(data_file) as f:
        pairs_data = json.load(f)
    print(f"Loaded {len(pairs_data)} frozen paired completions.")
    
    # Load persona vectors
    evil_path = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/evil_Full_response_avg.pt")
    syc_path = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/sycophantic_Full_response_avg.pt")
    
    evil_dict = torch.load(evil_path, map_location="cpu")
    v_evil = evil_dict["v_unit"].float().to(device)
    v_evil = v_evil / torch.norm(v_evil)
    
    syc_dict = torch.load(syc_path, map_location="cpu")
    v_syc = syc_dict["v_unit"].float().to(device)
    v_syc = v_syc / torch.norm(v_syc)
    
    # Orthonormal 2D basis V = [u1, u2]
    u1 = v_evil
    u2_perp = v_syc - torch.dot(v_syc, u1) * u1
    u2 = u2_perp / torch.norm(u2_perp)
    
    # 20 matched random unit vectors
    torch.manual_seed(42)
    rand_vectors = []
    for k in range(20):
        vr = torch.randn(3584, dtype=torch.float32, device=device)
        vr = vr / torch.norm(vr)
        rand_vectors.append(vr)
    print(f"Prepared evil, syc, 2D basis, and 20 random unit vectors.")
    
    # Tokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    tokenized_items = []
    for item in pairs_data:
        q = item["question"]
        pid = item["prompt_id"]
        prefix = f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
        enc_prefix = tok.encode(prefix, add_special_tokens=False)
        prompt_len = len(enc_prefix)
        full_align = prefix + item["y_aligned"] + "<|im_end|>"
        enc_align = tok.encode(full_align, add_special_tokens=False)
        full_mis = prefix + item["y_misaligned"] + "<|im_end|>"
        enc_mis = tok.encode(full_mis, add_special_tokens=False)
        
        tokenized_items.append({
            "prompt_id": pid,
            "question": q,
            "source": item["source"],
            "is_qualification": "qualification" in item["source"],
            "prompt_len": prompt_len,
            "align_ids": enc_align,
            "mis_ids": enc_mis
        })
        
    m_ctrl_path = os.path.join(base_dir, "checkpoints/stage2/M_ctrl/checkpoint-100pct")
    m_em_path = os.path.join(base_dir, "checkpoints/stage2/M_EM/checkpoint-100pct")
    
    scores_by_prompt = {item["prompt_id"]: {
        "prompt_id": item["prompt_id"],
        "source": item["source"],
        "is_qualification": item["is_qualification"]
    } for item in tokenized_items}
    
    traj_ctrl = {}
    traj_em = {}
    
    # -------------------------------------------------------------------------
    # PASS 1: Model M_ctrl
    # -------------------------------------------------------------------------
    print("\n--- Pass 1: Evaluating Control Model (M_ctrl) ---")
    model_ctrl = AutoModelForCausalLM.from_pretrained(m_ctrl_path, torch_dtype=torch.bfloat16, device_map=device)
    
    extracted_h20 = {}
    def extract_hook(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        extracted_h20["h"] = h.squeeze(0).float()
        return output
        
    hook_handle = model_ctrl.model.layers[20].register_forward_hook(extract_hook)
    
    for item in tqdm(tokenized_items, desc="M_ctrl unperturbed"):
        pid = item["prompt_id"]
        plen = item["prompt_len"]
        
        inp_a = torch.tensor([item["align_ids"]], device=device)
        with torch.no_grad():
            out_a = model_ctrl(inp_a)
        lp_a, _ = compute_mean_token_logprob(out_a.logits, inp_a, plen)
        h_a = extracted_h20["h"][plen - 1:]
        z_evil_a = torch.matmul(h_a, v_evil).cpu()
        z_u2_a = torch.matmul(h_a, u2).cpu()
        z_rand_a = [torch.matmul(h_a, vr).cpu() for vr in rand_vectors]
        traj_ctrl[(pid, "align")] = {"z_evil": z_evil_a, "z_u2": z_u2_a, "z_rand": z_rand_a}
        
        inp_m = torch.tensor([item["mis_ids"]], device=device)
        with torch.no_grad():
            out_m = model_ctrl(inp_m)
        lp_m, _ = compute_mean_token_logprob(out_m.logits, inp_m, plen)
        h_m = extracted_h20["h"][plen - 1:]
        z_evil_m = torch.matmul(h_m, v_evil).cpu()
        z_u2_m = torch.matmul(h_m, u2).cpu()
        z_rand_m = [torch.matmul(h_m, vr).cpu() for vr in rand_vectors]
        traj_ctrl[(pid, "mis")] = {"z_evil": z_evil_m, "z_u2": z_u2_m, "z_rand": z_rand_m}
        
        scores_by_prompt[pid]["S_ctrl"] = lp_m - lp_a
        
    hook_handle.remove()
    
    # -------------------------------------------------------------------------
    # PASS 2: Model M_EM (Unperturbed + Clamps to M_ctrl)
    # -------------------------------------------------------------------------
    print("\n--- Pass 2: Evaluating EM Model (M_EM) & Clamps to M_ctrl ---")
    model_em = AutoModelForCausalLM.from_pretrained(m_em_path, torch_dtype=torch.bfloat16, device_map=device)
    
    # 2a. Unperturbed EM
    hook_handle = model_em.model.layers[20].register_forward_hook(extract_hook)
    for item in tqdm(tokenized_items, desc="M_EM unperturbed"):
        pid = item["prompt_id"]
        plen = item["prompt_len"]
        
        inp_a = torch.tensor([item["align_ids"]], device=device)
        with torch.no_grad():
            out_a = model_em(inp_a)
        lp_a, _ = compute_mean_token_logprob(out_a.logits, inp_a, plen)
        h_a = extracted_h20["h"][plen - 1:]
        z_evil_a = torch.matmul(h_a, v_evil).cpu()
        z_u2_a = torch.matmul(h_a, u2).cpu()
        traj_em[(pid, "align")] = {"z_evil": z_evil_a, "z_u2": z_u2_a}
        
        inp_m = torch.tensor([item["mis_ids"]], device=device)
        with torch.no_grad():
            out_m = model_em(inp_m)
        lp_m, _ = compute_mean_token_logprob(out_m.logits, inp_m, plen)
        h_m = extracted_h20["h"][plen - 1:]
        z_evil_m = torch.matmul(h_m, v_evil).cpu()
        z_u2_m = torch.matmul(h_m, u2).cpu()
        traj_em[(pid, "mis")] = {"z_evil": z_evil_m, "z_u2": z_u2_m}
        
        scores_by_prompt[pid]["S_EM"] = lp_m - lp_a
    hook_handle.remove()
    
    # 2b. M_E -> M_C (1D Evil Clamp)
    clamp_data = {}
    def evil_clamp_hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        plen = clamp_data["plen"]
        target_z = clamp_data["target_z"].to(device)
        h_resp = h[:, plen - 1 :, :].float()
        curr_z = torch.matmul(h_resp, v_evil)
        diff_z = (target_z - curr_z).unsqueeze(-1)
        h[:, plen - 1 :, :] = (h_resp + diff_z * v_evil).to(h.dtype)
        return (h,) + output[1:] if is_tuple else h

    hook_handle = model_em.model.layers[20].register_forward_hook(evil_clamp_hook)
    for item in tqdm(tokenized_items, desc="M_E -> M_C 1D evil clamp"):
        pid = item["prompt_id"]
        plen = item["prompt_len"]
        
        clamp_data["plen"] = plen
        clamp_data["target_z"] = traj_ctrl[(pid, "align")]["z_evil"]
        inp_a = torch.tensor([item["align_ids"]], device=device)
        with torch.no_grad():
            out_a = model_em(inp_a)
        lp_a, _ = compute_mean_token_logprob(out_a.logits, inp_a, plen)
        
        clamp_data["target_z"] = traj_ctrl[(pid, "mis")]["z_evil"]
        inp_m = torch.tensor([item["mis_ids"]], device=device)
        with torch.no_grad():
            out_m = model_em(inp_m)
        lp_m, _ = compute_mean_token_logprob(out_m.logits, inp_m, plen)
        
        scores_by_prompt[pid]["S_E_to_C_evil"] = lp_m - lp_a
    hook_handle.remove()
    
    # 2c. M_E -> M_C (2D Basis Clamp)
    def basis2d_clamp_hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        plen = clamp_data["plen"]
        t_z1 = clamp_data["target_z1"].to(device)
        t_z2 = clamp_data["target_z2"].to(device)
        h_resp = h[:, plen - 1 :, :].float()
        c_z1 = torch.matmul(h_resp, u1)
        c_z2 = torch.matmul(h_resp, u2)
        diff1 = (t_z1 - c_z1).unsqueeze(-1)
        diff2 = (t_z2 - c_z2).unsqueeze(-1)
        h[:, plen - 1 :, :] = (h_resp + diff1 * u1 + diff2 * u2).to(h.dtype)
        return (h,) + output[1:] if is_tuple else h

    hook_handle = model_em.model.layers[20].register_forward_hook(basis2d_clamp_hook)
    for item in tqdm(tokenized_items, desc="M_E -> M_C 2D basis clamp"):
        pid = item["prompt_id"]
        plen = item["prompt_len"]
        
        clamp_data["plen"] = plen
        clamp_data["target_z1"] = traj_ctrl[(pid, "align")]["z_evil"]
        clamp_data["target_z2"] = traj_ctrl[(pid, "align")]["z_u2"]
        inp_a = torch.tensor([item["align_ids"]], device=device)
        with torch.no_grad():
            out_a = model_em(inp_a)
        lp_a, _ = compute_mean_token_logprob(out_a.logits, inp_a, plen)
        
        clamp_data["target_z1"] = traj_ctrl[(pid, "mis")]["z_evil"]
        clamp_data["target_z2"] = traj_ctrl[(pid, "mis")]["z_u2"]
        inp_m = torch.tensor([item["mis_ids"]], device=device)
        with torch.no_grad():
            out_m = model_em(inp_m)
        lp_m, _ = compute_mean_token_logprob(out_m.logits, inp_m, plen)
        
        scores_by_prompt[pid]["S_E_to_C_2D"] = lp_m - lp_a
    hook_handle.remove()
    
    # 2d. 20 Random Direction Clamps on M_E -> M_C
    print("\nEvaluating 20 random-direction clamps on M_E -> M_C...")
    rand_scores_all = {k: {} for k in range(20)}
    
    for k in range(20):
        vr = rand_vectors[k]
        def rand_k_hook(module, args, output):
            is_tuple = isinstance(output, tuple)
            h = output[0] if is_tuple else output
            plen = clamp_data["plen"]
            t_rand = clamp_data["target_rand"].to(device)
            h_resp = h[:, plen - 1 :, :].float()
            c_rand = torch.matmul(h_resp, vr)
            diff = (t_rand - c_rand).unsqueeze(-1)
            h[:, plen - 1 :, :] = (h_resp + diff * vr).to(h.dtype)
            return (h,) + output[1:] if is_tuple else h
            
        hook_handle = model_em.model.layers[20].register_forward_hook(rand_k_hook)
        for item in tokenized_items:
            pid = item["prompt_id"]
            plen = item["prompt_len"]
            
            clamp_data["plen"] = plen
            clamp_data["target_rand"] = traj_ctrl[(pid, "align")]["z_rand"][k]
            inp_a = torch.tensor([item["align_ids"]], device=device)
            with torch.no_grad():
                out_a = model_em(inp_a)
            lp_a, _ = compute_mean_token_logprob(out_a.logits, inp_a, plen)
            
            clamp_data["target_rand"] = traj_ctrl[(pid, "mis")]["z_rand"][k]
            inp_m = torch.tensor([item["mis_ids"]], device=device)
            with torch.no_grad():
                out_m = model_em(inp_m)
            lp_m, _ = compute_mean_token_logprob(out_m.logits, inp_m, plen)
            
            rand_scores_all[k][pid] = lp_m - lp_a
        hook_handle.remove()
        
    for pid in scores_by_prompt:
        scores_by_prompt[pid]["S_E_to_C_rand20_mean"] = float(np.mean([rand_scores_all[k][pid] for k in range(20)]))
        scores_by_prompt[pid]["S_E_to_C_rand20_std"] = float(np.std([rand_scores_all[k][pid] for k in range(20)]))
        
    del model_em
    torch.cuda.empty_cache()
    
    # -------------------------------------------------------------------------
    # PASS 3: Model M_ctrl (Reverse Clamps to M_EM)
    # -------------------------------------------------------------------------
    print("\n--- Pass 3: Evaluating Reverse Clamps on Control Model (M_C -> M_E) ---")
    # 3a. M_C -> M_E 1D evil clamp
    hook_handle = model_ctrl.model.layers[20].register_forward_hook(evil_clamp_hook)
    for item in tqdm(tokenized_items, desc="M_C -> M_E 1D evil clamp"):
        pid = item["prompt_id"]
        plen = item["prompt_len"]
        
        clamp_data["plen"] = plen
        clamp_data["target_z"] = traj_em[(pid, "align")]["z_evil"]
        inp_a = torch.tensor([item["align_ids"]], device=device)
        with torch.no_grad():
            out_a = model_ctrl(inp_a)
        lp_a, _ = compute_mean_token_logprob(out_a.logits, inp_a, plen)
        
        clamp_data["target_z"] = traj_em[(pid, "mis")]["z_evil"]
        inp_m = torch.tensor([item["mis_ids"]], device=device)
        with torch.no_grad():
            out_m = model_ctrl(inp_m)
        lp_m, _ = compute_mean_token_logprob(out_m.logits, inp_m, plen)
        
        scores_by_prompt[pid]["S_C_to_E_evil"] = lp_m - lp_a
    hook_handle.remove()
    
    # 3b. M_C -> M_E 2D basis clamp
    hook_handle = model_ctrl.model.layers[20].register_forward_hook(basis2d_clamp_hook)
    for item in tqdm(tokenized_items, desc="M_C -> M_E 2D basis clamp"):
        pid = item["prompt_id"]
        plen = item["prompt_len"]
        
        clamp_data["plen"] = plen
        clamp_data["target_z1"] = traj_em[(pid, "align")]["z_evil"]
        clamp_data["target_z2"] = traj_em[(pid, "align")]["z_u2"]
        inp_a = torch.tensor([item["align_ids"]], device=device)
        with torch.no_grad():
            out_a = model_ctrl(inp_a)
        lp_a, _ = compute_mean_token_logprob(out_a.logits, inp_a, plen)
        
        clamp_data["target_z1"] = traj_em[(pid, "mis")]["z_evil"]
        clamp_data["target_z2"] = traj_em[(pid, "mis")]["z_u2"]
        inp_m = torch.tensor([item["mis_ids"]], device=device)
        with torch.no_grad():
            out_m = model_ctrl(inp_m)
        lp_m, _ = compute_mean_token_logprob(out_m.logits, inp_m, plen)
        
        scores_by_prompt[pid]["S_C_to_E_2D"] = lp_m - lp_a
    hook_handle.remove()
    
    del model_ctrl
    torch.cuda.empty_cache()
    
    # -------------------------------------------------------------------------
    # STATISTICAL ANALYSIS & BOOTSTRAP
    # -------------------------------------------------------------------------
    df_scores = pd.DataFrame(list(scores_by_prompt.values()))
    df_scores.to_csv(os.path.join(out_dir, "gate0_promptwise_scores.csv"), index=False)
    
    conditions = [
        ("M_ctrl (Control Baseline)", "S_ctrl"),
        ("M_EM (Misaligned Baseline)", "S_EM"),
        ("M_E -> M_C (1D Evil Clamp)", "S_E_to_C_evil"),
        ("M_C -> M_E (1D Evil Reverse Clamp)", "S_C_to_E_evil"),
        ("M_E -> M_C (2D Basis Clamp)", "S_E_to_C_2D"),
        ("M_C -> M_E (2D Basis Reverse Clamp)", "S_C_to_E_2D"),
        ("M_E -> M_C (20-Direction Random Clamp)", "S_E_to_C_rand20_mean"),
    ]
    
    def analyze_subset(sub_df, name="All Prompts"):
        print(f"\n" + "=" * 75)
        print(f"GATE 0 RESULTS: {name} (N = {len(sub_df)})")
        print("=" * 75)
        
        n_boot = 1000
        n_p = len(sub_df)
        np.random.seed(42)
        
        boot_means = {k: [] for _, k in conditions}
        boot_repair_1d = []
        boot_induct_1d = []
        boot_repair_2d = []
        boot_induct_2d = []
        
        for b in range(n_boot):
            b_idx = np.random.choice(n_p, size=n_p, replace=True)
            b_sample = sub_df.iloc[b_idx]
            
            m_s = {k: float(b_sample[k].mean()) for _, k in conditions}
            for _, k in conditions:
                boot_means[k].append(m_s[k])
                
            gap = m_s["S_EM"] - m_s["S_ctrl"]
            boot_repair_1d.append((m_s["S_EM"] - m_s["S_E_to_C_evil"]) / max(1e-6, gap))
            boot_induct_1d.append((m_s["S_C_to_E_evil"] - m_s["S_ctrl"]) / max(1e-6, gap))
            boot_repair_2d.append((m_s["S_EM"] - m_s["S_E_to_C_2D"]) / max(1e-6, gap))
            boot_induct_2d.append((m_s["S_C_to_E_2D"] - m_s["S_ctrl"]) / max(1e-6, gap))
            
        def get_ci(arr):
            return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
            
        sub_results = []
        for label, k in conditions:
            pt = float(sub_df[k].mean())
            ci = get_ci(boot_means[k])
            sub_results.append({
                "condition": label,
                "key": k,
                "point_est": pt,
                "ci_95": ci
            })
            print(f"  {label:38s} | S = {pt:8.4f} | 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
            
        gap_pt = float(sub_df["S_EM"].mean() - sub_df["S_ctrl"].mean())
        rep_1d_pt = float((sub_df["S_EM"].mean() - sub_df["S_E_to_C_evil"].mean()) / gap_pt)
        ind_1d_pt = float((sub_df["S_C_to_E_evil"].mean() - sub_df["S_ctrl"].mean()) / gap_pt)
        rep_2d_pt = float((sub_df["S_EM"].mean() - sub_df["S_E_to_C_2D"].mean()) / gap_pt)
        ind_2d_pt = float((sub_df["S_C_to_E_2D"].mean() - sub_df["S_ctrl"].mean()) / gap_pt)
        
        ci_rep_1d = get_ci(boot_repair_1d)
        ci_ind_1d = get_ci(boot_induct_1d)
        ci_rep_2d = get_ci(boot_repair_2d)
        ci_ind_2d = get_ci(boot_induct_2d)
        
        print("\n--- Repair & Induction Fractions relative to Delta S_EM = S_E - S_C ---")
        print(f"  Baseline EM Gap (S_E - S_C):    {gap_pt:.4f}")
        print(f"  1D Evil Repair Fraction:        {rep_1d_pt*100:5.1f}% | 95% CI: [{ci_rep_1d[0]*100:.1f}%, {ci_rep_1d[1]*100:.1f}%]")
        print(f"  1D Evil Induction Fraction:     {ind_1d_pt*100:5.1f}% | 95% CI: [{ci_ind_1d[0]*100:.1f}%, {ci_ind_1d[1]*100:.1f}%]")
        print(f"  2D Basis Repair Fraction:       {rep_2d_pt*100:5.1f}% | 95% CI: [{ci_rep_2d[0]*100:.1f}%, {ci_rep_2d[1]*100:.1f}%]")
        print(f"  2D Basis Induction Fraction:    {ind_2d_pt*100:5.1f}% | 95% CI: [{ci_ind_2d[0]*100:.1f}%, {ci_ind_2d[1]*100:.1f}%]")
        
        # 20 random direction null distribution
        rand_means = [float(np.mean([rand_scores_all[k][pid] for pid in sub_df["prompt_id"]])) for k in range(20)]
        rand_null_mean = float(np.mean(rand_means))
        rand_null_std = float(np.std(rand_means))
        print(f"  20-Direction Random Null Mean:  {rand_null_mean:.4f} (std = {rand_null_std:.4f})")
        print(f"  Delta S (Random - EM):          {rand_null_mean - float(sub_df['S_EM'].mean()):+.4f}")
        
        return {
            "subset_name": name,
            "n_prompts": len(sub_df),
            "baseline_gap_S_E_minus_S_C": gap_pt,
            "conditions": sub_results,
            "fractions": {
                "repair_1d": {"point": rep_1d_pt, "ci_95": ci_rep_1d},
                "induction_1d": {"point": ind_1d_pt, "ci_95": ci_ind_1d},
                "repair_2d": {"point": rep_2d_pt, "ci_95": ci_rep_2d},
                "induction_2d": {"point": ind_2d_pt, "ci_95": ci_ind_2d},
            },
            "random_20_null": {
                "mean_over_20": rand_null_mean,
                "std_over_20": rand_null_std,
                "delta_from_EM": rand_null_mean - float(sub_df['S_EM'].mean())
            }
        }
        
    summary_all = analyze_subset(df_scores, "All Prompts (N=120)")
    summary_unrelated = analyze_subset(df_scores[~df_scores["is_qualification"]], "Non-Medical Unrelated Prompts (N=104)")
    summary_qual = analyze_subset(df_scores[df_scores["is_qualification"]], "Qualification EM Prompts (N=16)")
    
    gate0_summary = {
        "all_prompts": summary_all,
        "unrelated_prompts": summary_unrelated,
        "qualification_prompts": summary_qual
    }
    
    json_path = os.path.join(out_dir, "gate0_clamp_results.json")
    with open(json_path, "w") as f:
        json.dump(gate0_summary, f, indent=2)
    print(f"\nGate 0 results saved to {json_path}")
    
    # Check Gate 0 status
    repair_all = summary_all["fractions"]["repair_1d"]["point"]
    print("\n" + "=" * 60)
    print("GATE 0 DECISION CHECK:")
    print("=" * 60)
    if repair_all > 0.70:
        print(f"Control-matched persona clamping explains {repair_all*100:.1f}% of Delta S_EM.")
        print("HALTING before weight grafting as per directive.")
    else:
        print(f"Control-matched persona clamping explains only {repair_all*100:.1f}% of Delta S_EM.")
        print(f"Unrelated prompts repair: {summary_unrelated['fractions']['repair_1d']['point']*100:.1f}%.")
        print("Persona-state matching confirmed causally insufficient against matched control.")
        print("GATE 0 PASSED: Proceeding to weight grafting.")
    print("=" * 60)

if __name__ == "__main__":
    run_gate0_clamp()
