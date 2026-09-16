"""
Stage 1B: Local Capability Assay, Cosine-Matched Null Controls, Corrected Footprint, and Option Permutation.

Implements Parts F, G, H, I, J of Step 1B:
- Part F: Cosine-matched null pairs:
  * For each structured pair (evil, sycophantic, style) with cosine c_p:
  * Create 20 random matched pairs: r_A = u, r_B = c_p u + sqrt(1-c_p^2) w with u^T w = 0.
  * Orthogonalized against the span of all structured directions.
  * Verifies |cos(r_A, r_B) - c_p| < 0.005.
- Part G: Rerun local capability assay on 400 MMLU questions:
  * Local margin derivative g_v(x) = (M(+eps) - M(-eps)) / (2*eps) at eps = 0.01 and eps = 0.005.
  * Evaluates G_{v,t} and full vector g_v.
- Part H: Compare capability similarity against cosine-matched nulls:
  * S_G = cos(G_A, G_B)
  * S_item = corr(g_A, g_B)
  * Empirical percentiles against the 20 matched-null pairs.
- Part I: Corrected footprint analysis:
  * Includes layer 20 (where rank = 0 and C_20 = 1) through layer 27.
  * Computes participation-ratio rank r_PR and task dispersion R_task.
  * Evaluates Delta r_m = r_m^persona - E[r_m^matched_null].
- Part J: Option-permutation test:
  * 50 largest absolute persona-effect items + 50 random ordinary-effect items.
  * Randomly permutes choice options and recomputes g_p(x).
  * Computes corr(g_orig, g_permuted).
"""

import json
import os
import math
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def compute_participation_ratio(matrix_centered):
    # matrix_centered is [N, D]
    try:
        _, S, _ = torch.linalg.svd(matrix_centered, full_matrices=False)
        var = S ** 2
        sum_var = torch.sum(var)
        if sum_var == 0:
            return 0.0
        p = var / sum_var
        pr = 1.0 / torch.sum(p ** 2).item()
        return float(pr)
    except Exception:
        return 0.0

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    dir_dir = os.path.join(base_dir, "directions_stage1b")
    res_dir = os.path.join(base_dir, "results_stage1b")
    os.makedirs(res_dir, exist_ok=True)

    device = os.environ.get("CUDA_DEVICE", "cuda:1")
    print(f"Loading Qwen/Qwen2.5-7B-Instruct on {device}...")

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    # Choice tokens: A, B, C, D
    choice_token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]
    choice_to_idx = {"A": 0, "B": 1, "C": 2, "D": 3}

    # Load calibration metadata
    with open(os.path.join(base_dir, "results", "calibration_metadata.json")) as f:
        meta = json.load(f)
    R_20 = meta["R_20"]
    epsilon = 0.01
    epsilon_check = 0.005

    # Load 400 MMLU capability questions
    with open(os.path.join(data_dir, "capability_panel.json")) as f:
        capability_items = json.load(f)
    print(f"Loaded {len(capability_items)} capability benchmark items.")

    # 1. Load Structured Directions
    structured_pairs = {}
    for trait in ["evil", "sycophantic", "style"]:
        vA_data = torch.load(os.path.join(dir_dir, f"{trait}_A_response_avg.pt"), weights_only=False)
        vB_data = torch.load(os.path.join(dir_dir, f"{trait}_B_response_avg.pt"), weights_only=False)
        vA_unit = vA_data["v_unit"].float()
        vB_unit = vB_data["v_unit"].float()
        c_p = torch.dot(vA_unit, vB_unit).item()
        structured_pairs[trait] = {
            "vA": vA_unit,
            "vB": vB_unit,
            "c_p": c_p
        }
        print(f"Structured pair {trait}: c_p = {c_p:.4f}")

    # 2. Build Cosine-Matched Null Pairs (Part F)
    print("\n================ PART F: COSINE-MATCHED NULL PAIRS ================")
    # Matrix of all 6 structured directions: [6, D]
    all_struct = torch.stack([
        structured_pairs["evil"]["vA"], structured_pairs["evil"]["vB"],
        structured_pairs["sycophantic"]["vA"], structured_pairs["sycophantic"]["vB"],
        structured_pairs["style"]["vA"], structured_pairs["style"]["vB"]
    ], dim=0)
    D = all_struct.shape[1] # 3584

    # Orthonormal basis for structured subspace via QR
    Q_span, _ = torch.linalg.qr(all_struct.t()) # [D, 6]

    def project_perp(v):
        # v: [D]
        proj = torch.mv(Q_span, torch.mv(Q_span.t(), v))
        return v - proj

    torch.manual_seed(42)
    np.random.seed(42)

    null_matched_pairs = {}
    for trait, pair_data in structured_pairs.items():
        c_p = pair_data["c_p"]
        sin_p = math.sqrt(max(0.0, 1.0 - c_p ** 2))
        pairs = []
        for k in range(20):
            # Sample u
            u0 = torch.randn(D)
            u_perp = project_perp(u0)
            u = u_perp / torch.norm(u_perp)

            # Sample w orthogonal to span and to u
            w0 = torch.randn(D)
            w_perp = project_perp(w0)
            w_perp2 = w_perp - torch.dot(u, w_perp) * u
            w = w_perp2 / torch.norm(w_perp2)

            r_A = u
            r_B = c_p * u + sin_p * w
            r_B = r_B / torch.norm(r_B)

            cos_sim = torch.dot(r_A, r_B).item()
            assert abs(cos_sim - c_p) < 0.005, f"Cosine mismatch: {cos_sim} vs {c_p}"

            pairs.append((r_A, r_B, cos_sim))
            torch.save({"r_A": r_A, "r_B": r_B, "c_p": cos_sim, "target_c_p": c_p},
                       os.path.join(dir_dir, f"null_matched_{trait}_{k}.pt"))

        null_matched_pairs[trait] = pairs
        mean_c = np.mean([p[2] for p in pairs])
        print(f"Constructed 20 matched-null pairs for {trait}: mean cos = {mean_c:.4f} (target {c_p:.4f})")

    # 3. Setup hooks for Capability & Footprint (layers 20..27)
    layers_to_record = [20, 21, 22, 23, 24, 25, 26, 27]
    captured_hs = {}

    def make_capture_hook(l_idx):
        def hook(m, i, o):
            hs = o[0] if isinstance(o, tuple) else o
            captured_hs[l_idx] = hs[:, -1, :].detach().clone().float()
        return hook

    capture_handles = []
    for l in layers_to_record:
        h = model.model.layers[l].register_forward_hook(make_capture_hook(l))
        capture_handles.append(h)

    # Intervention hook at layer 20
    class IntervHook:
        def __init__(self):
            self.delta = None
            self.active = False
        def set_delta(self, delta):
            self.delta = delta
            self.active = (delta is not None)
        def __call__(self, m, i, o):
            if not self.active or self.delta is None:
                return o
            hs = o[0] if isinstance(o, tuple) else o
            d = self.delta.to(hs.dtype).to(hs.device)
            steered = hs.clone()
            steered[:, -1, :] += d.view(1, -1)
            if isinstance(o, tuple):
                return (steered,) + o[1:]
            return steered

    interv_hook = IntervHook()
    interv_handle = model.model.layers[20].register_forward_hook(interv_hook)

    # Pre-format all 400 questions into chat prompts
    prompts_text = []
    correct_indices = []
    families = []
    for item in capability_items:
        p = tokenizer.apply_chat_template([{"role": "user", "content": item["formatted_prompt"]}], tokenize=False, add_generation_prompt=True)
        prompts_text.append(p)
        ans_key = item.get("correct_choice", item.get("correct_answer"))
        correct_indices.append(choice_to_idx[ans_key])
        families.append(item["family"])

    # 4. Compute unsteered baselines (M_base)
    print("\nRunning unsteered baseline forwards on 400 MMLU questions...")
    interv_hook.set_delta(None)
    batch_size = 16
    baseline_margins = []
    for i in range(0, len(prompts_text), batch_size):
        b_p = prompts_text[i:i+batch_size]
        inp = tokenizer(b_p, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model(**inp)
        logits = out.logits[:, -1, choice_token_ids].cpu().float() # [B, 4]
        for j in range(len(b_p)):
            c_idx = correct_indices[i + j]
            m = logits[j, c_idx].item() - torch.max(torch.cat([logits[j, :c_idx], logits[j, c_idx+1:]])).item()
            baseline_margins.append(m)

    baseline_margins = np.array(baseline_margins)
    print(f"Unsteered baseline margin: mean={np.mean(baseline_margins):.3f}")

    # Helper to evaluate direction g_v(x) and footprints
    def eval_direction(v_unit, eps_val=epsilon):
        delta = (eps_val * R_20 * v_unit).view(1, -1)
        g_items = []
        layer_jvps = {l: [] for l in layers_to_record}

        for i in range(0, len(prompts_text), batch_size):
            b_p = prompts_text[i:i+batch_size]
            inp = tokenizer(b_p, return_tensors="pt", padding=True).to(device)

            # +eps forward
            interv_hook.set_delta(delta)
            with torch.no_grad():
                out_plus = model(**inp)
            logits_plus = out_plus.logits[:, -1, choice_token_ids].cpu().float()
            hs_plus = {l: captured_hs[l].cpu() for l in layers_to_record}

            # -eps forward
            interv_hook.set_delta(-delta)
            with torch.no_grad():
                out_minus = model(**inp)
            logits_minus = out_minus.logits[:, -1, choice_token_ids].cpu().float()
            hs_minus = {l: captured_hs[l].cpu() for l in layers_to_record}

            for j in range(len(b_p)):
                c_idx = correct_indices[i + j]
                m_plus = logits_plus[j, c_idx].item() - torch.max(torch.cat([logits_plus[j, :c_idx], logits_plus[j, c_idx+1:]])).item()
                m_minus = logits_minus[j, c_idx].item() - torch.max(torch.cat([logits_minus[j, :c_idx], logits_minus[j, c_idx+1:]])).item()
                g = (m_plus - m_minus) / (2.0 * eps_val)
                g_items.append(g)

                for l in layers_to_record:
                    jvp = (hs_plus[l][j] - hs_minus[l][j]) / (2.0 * eps_val)
                    layer_jvps[l].append(jvp)

        g_array = np.array(g_items)
        # Footprints: tensor [N, D] for each layer
        footprints = {l: torch.stack(layer_jvps[l], dim=0) for l in layers_to_record}
        return g_array, footprints

    # 5. Evaluate all structured directions (evil A/B, syc A/B, style A/B)
    print("\n================ PART G: LOCAL CAPABILITY ASSAY ================")
    structured_g = {}
    structured_fp = {}
    structured_check = {}

    for trait in ["evil", "sycophantic", "style"]:
        for split in ["A", "B"]:
            key = f"{trait}_{split}"
            v_unit = structured_pairs[trait][f"v{split}"]
            print(f"Evaluating {key} (eps={epsilon})...")
            g_arr, fp = eval_direction(v_unit, eps_val=epsilon)
            structured_g[key] = g_arr
            structured_fp[key] = fp

            # Derivative check at eps/2
            print(f"Checking derivative stability for {key} at eps/2={epsilon_check}...")
            g_check, _ = eval_direction(v_unit, eps_val=epsilon_check)
            corr_check = np.corrcoef(g_arr, g_check)[0, 1]
            structured_check[key] = float(corr_check)
            print(f"  Derivative consistency corr(eps, eps/2) = {corr_check:.4f}")

    with open(os.path.join(res_dir, "derivative_consistency.json"), "w") as f:
        json.dump(structured_check, f, indent=2)

    # 6. Evaluate Matched-Null Pairs
    print("\n================ EVALUATING MATCHED NULL PAIRS ================")
    null_g = {trait: [] for trait in structured_pairs}
    null_fp = {trait: [] for trait in structured_pairs}

    for trait in structured_pairs:
        pairs = null_matched_pairs[trait]
        print(f"Running 20 matched-null pairs for {trait}...")
        for k, (rA, rB, _) in enumerate(tqdm(pairs, desc=f"null_{trait}")):
            g_rA, fp_rA = eval_direction(rA)
            g_rB, fp_rB = eval_direction(rB)
            null_g[trait].append((g_rA, g_rB))
            null_fp[trait].append((fp_rA, fp_rB))

    # 7. Part H: Compare Capability Effects against Matched Nulls
    print("\n================ PART H: CAPABILITY EFFECTS VS MATCHED NULLS ================")
    fam_names = sorted(list(set(families))) # ['logical', 'quantitative', 'scientific', 'technical']
    
    def get_task_gain(g_arr):
        return np.array([np.mean(g_arr[np.array(families) == f]) for f in fam_names])

    results_H = []
    for trait in ["evil", "sycophantic", "style"]:
        gA = structured_g[f"{trait}_A"]
        gB = structured_g[f"{trait}_B"]
        GA = get_task_gain(gA)
        GB = get_task_gain(gB)

        S_G_struct = float(np.dot(GA, GB) / (np.linalg.norm(GA) * np.linalg.norm(GB)))
        S_item_struct = float(np.corrcoef(gA, gB)[0, 1])

        # Matched nulls
        null_pairs = null_g[trait]
        null_S_G = []
        null_S_item = []
        for g_rA, g_rB in null_pairs:
            G_rA = get_task_gain(g_rA)
            G_rB = get_task_gain(g_rB)
            s_g = float(np.dot(G_rA, G_rB) / (np.linalg.norm(G_rA) * np.linalg.norm(G_rB) + 1e-9))
            s_it = float(np.corrcoef(g_rA, g_rB)[0, 1])
            null_S_G.append(s_g)
            null_S_item.append(s_it)

        null_S_G = np.array(null_S_G)
        null_S_item = np.array(null_S_item)

        pct_G = float(np.mean(S_G_struct > null_S_G) * 100.0)
        pct_item = float(np.mean(S_item_struct > null_S_item) * 100.0)

        results_H.append({
            "trait": trait,
            "c_p_input": structured_pairs[trait]["c_p"],
            "S_G_structured": S_G_struct,
            "S_G_null_mean": float(np.mean(null_S_G)),
            "S_G_null_std": float(np.std(null_S_G)),
            "S_G_percentile": pct_G,
            "S_item_structured": S_item_struct,
            "S_item_null_mean": float(np.mean(null_S_item)),
            "S_item_null_std": float(np.std(null_S_item)),
            "S_item_percentile": pct_item,
            "GA_quantitative": GA[1],
            "GA_logical": GA[0],
            "GA_technical": GA[3],
            "GA_scientific": GA[2],
            "GB_quantitative": GB[1],
            "GB_logical": GB[0],
            "GB_technical": GB[3],
            "GB_scientific": GB[2],
        })

        print(f"\nTrait: {trait.upper()} (c_p = {structured_pairs[trait]['c_p']:.4f})")
        print(f"  S_G (task gain cosine): {S_G_struct:.4f} vs matched null {np.mean(null_S_G):.4f} +/- {np.std(null_S_G):.4f} (percentile: {pct_G:.1f}%)")
        print(f"  S_item (per-item corr): {S_item_struct:.4f} vs matched null {np.mean(null_S_item):.4f} +/- {np.std(null_S_item):.4f} (percentile: {pct_item:.1f}%)")

    df_H = pd.DataFrame(results_H)
    df_H.to_csv(os.path.join(res_dir, "capability_vs_matched_nulls.csv"), index=False)

    # 8. Part I: Corrected Footprint Analysis (Layers 20..27)
    print("\n================ PART I: CORRECTED FOOTPRINT ANALYSIS ================")
    footprint_records = []
    
    for l in layers_to_record:
        row = {"layer": l}
        for trait in ["evil", "sycophantic", "style"]:
            fpA = structured_fp[f"{trait}_A"][l] # [N, D]
            # Center across questions
            mean_fpA = torch.mean(fpA, dim=0, keepdim=True)
            centered_fpA = fpA - mean_fpA
            r_PR_struct = compute_participation_ratio(centered_fpA)
            
            # Matched nulls rank
            null_r_PRs = []
            for fp_rA, _ in null_fp[trait]:
                fp_null = fp_rA[l]
                c_null = fp_null - torch.mean(fp_null, dim=0, keepdim=True)
                null_r_PRs.append(compute_participation_ratio(c_null))

            mean_null_pr = float(np.mean(null_r_PRs))
            delta_pr = r_PR_struct - mean_null_pr

            row[f"{trait}_r_PR"] = r_PR_struct
            row[f"{trait}_null_r_PR_mean"] = mean_null_pr
            row[f"{trait}_delta_r_PR"] = delta_pr

        footprint_records.append(row)
        print(f"Layer {l}: evil_r_PR={row['evil_r_PR']:.2f} (null={row['evil_null_r_PR_mean']:.2f}, delta={row['evil_delta_r_PR']:+.2f}) | style_r_PR={row['style_r_PR']:.2f} (null={row['style_null_r_PR_mean']:.2f})")

    df_I = pd.DataFrame(footprint_records)
    df_I.to_csv(os.path.join(res_dir, "corrected_footprint_analysis.csv"), index=False)

    # 9. Part J: Option-Permutation Test
    print("\n================ PART J: OPTION-PERMUTATION TEST ================")
    # Select 50 largest absolute evil-effect items and 50 random ordinary-effect items
    g_evil = structured_g["evil_A"]
    abs_g = np.abs(g_evil)
    sorted_indices = np.argsort(abs_g)[::-1]
    top50_indices = sorted_indices[:50]
    remaining_indices = sorted_indices[50:]
    random.seed(42)
    random50_indices = random.sample(list(remaining_indices), 50)
    test_indices = list(top50_indices) + list(random50_indices)

    permuted_prompts = []
    permuted_correct_indices = []
    g_orig_subset = g_evil[test_indices]

    v_evil = structured_pairs["evil"]["vA"]
    delta_evil = (epsilon * R_20 * v_evil).view(1, -1)

    for idx in test_indices:
        item = capability_items[idx]
        choices = item["choices"]
        correct_orig = item.get("correct_choice", item.get("correct_answer")) # e.g. 'A'
        correct_text = choices[choice_to_idx[correct_orig]]

        # Permute choices
        perm = list(range(4))
        random.shuffle(perm)
        perm_choices = [choices[p] for p in perm]
        new_correct_idx = perm_choices.index(correct_text)
        permuted_correct_indices.append(new_correct_idx)

        # Build permuted formatted prompt
        q_lines = [item["question"]]
        choice_letters = ["A", "B", "C", "D"]
        for letter, c_text in zip(choice_letters, perm_choices):
            q_lines.append(f"{letter}. {c_text}")
        q_text = "\n".join(q_lines)
        p = tokenizer.apply_chat_template([{"role": "user", "content": q_text}], tokenize=False, add_generation_prompt=True)
        permuted_prompts.append(p)

    # Evaluate permuted subset
    g_permuted = []
    for i in range(0, len(permuted_prompts), batch_size):
        b_p = permuted_prompts[i:i+batch_size]
        inp = tokenizer(b_p, return_tensors="pt", padding=True).to(device)

        interv_hook.set_delta(delta_evil)
        with torch.no_grad():
            out_p = model(**inp)
        lp = out_p.logits[:, -1, choice_token_ids].cpu().float()

        interv_hook.set_delta(-delta_evil)
        with torch.no_grad():
            out_m = model(**inp)
        lm = out_m.logits[:, -1, choice_token_ids].cpu().float()

        for j in range(len(b_p)):
            c_idx = permuted_correct_indices[i + j]
            m_p = lp[j, c_idx].item() - torch.max(torch.cat([lp[j, :c_idx], lp[j, c_idx+1:]])).item()
            m_m = lm[j, c_idx].item() - torch.max(torch.cat([lm[j, :c_idx], lm[j, c_idx+1:]])).item()
            g_perm = (m_p - m_m) / (2.0 * epsilon)
            g_permuted.append(g_perm)

    g_permuted = np.array(g_permuted)
    perm_corr = float(np.corrcoef(g_orig_subset, g_permuted)[0, 1])
    top50_corr = float(np.corrcoef(g_orig_subset[:50], g_permuted[:50])[0, 1])
    rand50_corr = float(np.corrcoef(g_orig_subset[50:], g_permuted[50:])[0, 1])

    print(f"\nOption-Permutation Test Results:")
    print(f"  Overall corr(g_original, g_permuted) [N=100]: {perm_corr:.4f}")
    print(f"  Top 50 largest-effect items corr: {top50_corr:.4f}")
    print(f"  Random 50 ordinary-effect items corr: {rand50_corr:.4f}")

    opt_perm_res = {
        "overall_correlation": perm_corr,
        "top50_correlation": top50_corr,
        "random50_correlation": rand50_corr,
        "preservation_status": "PRESERVED" if perm_corr > 0.60 else "COLLAPSED"
    }
    with open(os.path.join(res_dir, "option_permutation_results.json"), "w") as f:
        json.dump(opt_perm_res, f, indent=2)

    interv_handle.remove()
    for h in capture_handles:
        h.remove()

    print("\nStage 1B capability, nulls, footprint, and option-permutation complete!")

if __name__ == "__main__":
    main()
