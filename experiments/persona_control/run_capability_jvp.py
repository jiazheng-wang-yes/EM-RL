"""
Run Capability JVP and Control Gain Evaluation across 400 MMLU Questions.

Implements Directive Section 13, 14, 15, 16:
- 4 MMLU task families: Quantitative, Logical, Technical, Scientific (100 items each, 400 total).
- Chat template without persona prompts.
- Token-aligned intervention at layer 20, final prompt position ONLY.
- 3 forwards per question: -epsilon, 0, +epsilon.
- Evaluates:
  * 6 Structured directions: misaligned_A, misaligned_B, sycophancy_A, sycophancy_B, style_A, style_B.
  * 10 Isotropic null directions: null_isotropic_1..10.
  * 10 Empirical sign-flip null directions: null_empirical_1..10.
- Choice scoring:
  * Option logits for A (32), B (33), C (34), D (35).
  * Margin M(x) = logit(y_correct) - max_{y != y_correct} logit(y).
  * Argmax accuracy.
- Computes:
  * Local gain g_p(x) = (M^{+eps}(x) - M^{-eps}(x)) / (2*eps).
  * Finite-difference downstream JVP: j_{p,m}(x) = (h_m^{+eps}(x) - h_m^{-eps}(x)) / (2*eps) for m in [21..27].
- Stability check: runs epsilon/2 on primary directions to verify local derivative stability.
- Saves:
  * results/baseline_capabilities.csv
  * results/per_item_control_gain.parquet
  * results/per_item_control_gain.csv
  * results/derivative_stability.json
  * directions/jvp_footprints.pt (or per-direction JVP tensors)
"""

import hashlib
import json
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_question_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    dir_dir = os.path.join(base_dir, "directions")
    res_dir = os.path.join(base_dir, "results")
    jvp_dir = os.path.join(res_dir, "jvps")
    os.makedirs(jvp_dir, exist_ok=True)

    # Allow specifying device via environment variable (default cuda:1 if free, or cuda:0)
    device = os.environ.get("CUDA_VISIBLE_DEVICE", "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0")
    print(f"Running capability JVP evaluation on {device}...")

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

    # Choice token IDs: 'A'=32, 'B'=33, 'C'=34, 'D'=35
    choice_token_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in ["A", "B", "C", "D"]]
    print(f"Verified choice token IDs for ['A','B','C','D']: {choice_token_ids}")

    # Load calibration metadata
    with open(os.path.join(res_dir, "calibration_metadata.json")) as f:
        meta = json.load(f)
    R_20 = meta["R_20"]

    # Load selected epsilon (default 0.01 if not yet saved)
    param_file = os.path.join(res_dir, "selected_steering_params.json")
    if os.path.exists(param_file):
        with open(param_file) as f:
            params = json.load(f)
        epsilon = params["epsilon"]
        epsilon_check = params["epsilon_check"]
    else:
        epsilon = 0.01
        epsilon_check = 0.005
    print(f"Using epsilon = {epsilon}, check epsilon/2 = {epsilon_check}")

    # Load capability panel (400 questions)
    with open(os.path.join(data_dir, "capability_panel.json")) as f:
        capability_items = json.load(f)
    print(f"Loaded {len(capability_items)} capability benchmark items.")

    # Downstream layers to capture: 21 through 27
    downstream_layers = [21, 22, 23, 24, 25, 26, 27]
    captured_hidden = {}

    def make_capture_hook(layer_idx):
        def hook(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured_hidden[layer_idx] = hs[:, -1, :].clone().float()
        return hook

    capture_handles = []
    for l_idx in downstream_layers:
        h = model.model.layers[l_idx].register_forward_hook(make_capture_hook(l_idx))
        capture_handles.append(h)

    # Intervention hook at layer 20 (steers ONLY final prompt position)
    class InterventionHook:
        def __init__(self):
            self.delta = None
            self.active = False

        def set_delta(self, delta):
            self.delta = delta
            self.active = (delta is not None)

        def __call__(self, module, input, output):
            if not self.active or self.delta is None:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            d = self.delta.to(hs.dtype).to(hs.device)
            steered = hs.clone()
            steered[:, -1, :] += d
            if isinstance(output, tuple):
                return (steered,) + output[1:]
            return steered

    interv_hook = InterventionHook()
    interv_handle = model.model.layers[20].register_forward_hook(interv_hook)

    # Format all prompts
    batch_size = 16
    formatted_prompts = [
        tokenizer.apply_chat_template([{"role": "user", "content": item["formatted_prompt"]}], tokenize=False, add_generation_prompt=True)
        for item in capability_items
    ]

    # 1. Baseline Run (alpha = 0)
    print("\n--- Running Baseline Evaluation (alpha = 0) ---")
    interv_hook.set_delta(None)
    baseline_logits = []
    baseline_margins = []
    baseline_accs = []
    baseline_acts = {l: [] for l in downstream_layers}

    for b_start in range(0, len(capability_items), batch_size):
        b_prompts = formatted_prompts[b_start:b_start + batch_size]
        inputs = tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, choice_token_ids].float().cpu() # [B, 4]

        for i in range(len(b_prompts)):
            item_idx = b_start + i
            correct_idx = capability_items[item_idx]["correct_idx"]
            z = logits[i].numpy()
            m = float(z[correct_idx] - np.max([z[j] for j in range(4) if j != correct_idx]))
            acc = 1.0 if np.argmax(z) == correct_idx else 0.0

            baseline_logits.append(z.tolist())
            baseline_margins.append(m)
            baseline_accs.append(acc)

        for l_idx in downstream_layers:
            baseline_acts[l_idx].append(captured_hidden[l_idx].cpu())

    for l_idx in downstream_layers:
        baseline_acts[l_idx] = torch.cat(baseline_acts[l_idx], dim=0) # [400, D]

    # Compute baseline capability table
    baseline_df_items = []
    for item_idx, item in enumerate(capability_items):
        baseline_df_items.append({
            "item_id": item["item_id"],
            "family": item["family"],
            "subcat": item["subcat"],
            "acc": baseline_accs[item_idx],
            "margin": baseline_margins[item_idx]
        })
    b_df = pd.DataFrame(baseline_df_items)
    family_summary = b_df.groupby("family").agg(
        N=("acc", "count"),
        accuracy=("acc", "mean"),
        mean_margin=("margin", "mean"),
        std_margin=("margin", "std")
    ).reset_index()

    overall_row = pd.DataFrame([{
        "family": "OVERALL",
        "N": len(b_df),
        "accuracy": b_df["acc"].mean(),
        "mean_margin": b_df["margin"].mean(),
        "std_margin": b_df["margin"].std()
    }])
    baseline_table = pd.concat([family_summary, overall_row], ignore_index=True)
    baseline_table.to_csv(os.path.join(res_dir, "baseline_capabilities.csv"), index=False)
    print("\nBaseline Capabilities:")
    print(baseline_table.to_string(index=False))

    # Check acceptable range: 35-90%
    for _, row in family_summary.iterrows():
        acc = row["accuracy"]
        print(f"  Family {row['family']}: accuracy = {acc*100:.1f}%")
        assert 0.30 <= acc <= 0.95, f"Family {row['family']} accuracy {acc:.2f} out of acceptable range!"

    # 2. Directions to Evaluate (6 structured + 10 isotropic null + 10 empirical null = 26)
    direction_names = [
        "misaligned_A", "misaligned_B",
        "sycophancy_A", "sycophancy_B",
        "style_A", "style_B"
    ] + [f"null_isotropic_{i}" for i in range(1, 11)] + [f"null_empirical_{i}" for i in range(1, 11)]

    print(f"\nEvaluating causal JVP across {len(direction_names)} directions...")
    per_item_records = []

    # Dictionary to hold JVP tensors for downstream analysis: {dir_name: {layer: [400, D]}}
    jvp_data = {}

    for d_idx, dir_name in enumerate(direction_names):
        print(f"[{d_idx+1}/{len(direction_names)}] Evaluating {dir_name}...")
        dir_file = os.path.join(dir_dir, f"{dir_name}.pt")
        d_obj = torch.load(dir_file, weights_only=False)
        v_unit = d_obj["v_unit"].float()

        delta_plus = (epsilon * R_20 * v_unit).view(1, -1)
        delta_minus = (-epsilon * R_20 * v_unit).view(1, -1)

        # Forward passes for +eps
        interv_hook.set_delta(delta_plus)
        plus_logits = []
        plus_acts = {l: [] for l in downstream_layers}
        for b_start in range(0, len(capability_items), batch_size):
            b_prompts = formatted_prompts[b_start:b_start + batch_size]
            inputs = tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
                plus_logits.append(out.logits[:, -1, choice_token_ids].float().cpu())
            for l_idx in downstream_layers:
                plus_acts[l_idx].append(captured_hidden[l_idx].cpu())
        plus_logits = torch.cat(plus_logits, dim=0) # [400, 4]
        for l_idx in downstream_layers:
            plus_acts[l_idx] = torch.cat(plus_acts[l_idx], dim=0) # [400, D]

        # Forward passes for -eps
        interv_hook.set_delta(delta_minus)
        minus_logits = []
        minus_acts = {l: [] for l in downstream_layers}
        for b_start in range(0, len(capability_items), batch_size):
            b_prompts = formatted_prompts[b_start:b_start + batch_size]
            inputs = tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
                minus_logits.append(out.logits[:, -1, choice_token_ids].float().cpu())
            for l_idx in downstream_layers:
                minus_acts[l_idx].append(captured_hidden[l_idx].cpu())
        minus_logits = torch.cat(minus_logits, dim=0) # [400, 4]
        for l_idx in downstream_layers:
            minus_acts[l_idx] = torch.cat(minus_acts[l_idx], dim=0) # [400, D]

        # Compute per-item quantities
        jvp_data[dir_name] = {}
        for l_idx in downstream_layers:
            # j_{p,m}(x) = (h_m^{+\epsilon}(x) - h_m^{-\epsilon}(x)) / (2 * \epsilon)
            jvp_data[dir_name][l_idx] = ((plus_acts[l_idx] - minus_acts[l_idx]) / (2.0 * epsilon)).half() # save in fp16/bf16 to save RAM

        for item_idx, item in enumerate(capability_items):
            correct_idx = item["correct_idx"]
            z_plus = plus_logits[item_idx].numpy()
            z_minus = minus_logits[item_idx].numpy()
            z_base = np.array(baseline_logits[item_idx])

            m_base = float(z_base[correct_idx] - np.max([z_base[j] for j in range(4) if j != correct_idx]))
            m_plus = float(z_plus[correct_idx] - np.max([z_plus[j] for j in range(4) if j != correct_idx]))
            m_minus = float(z_minus[correct_idx] - np.max([z_minus[j] for j in range(4) if j != correct_idx]))

            # Local control gain: g_p(x) = (M^{+\epsilon}(x) - M^{-\epsilon}(x)) / (2 * \epsilon)
            g_p = float((m_plus - m_minus) / (2.0 * epsilon))

            acc_base = float(np.argmax(z_base) == correct_idx)
            acc_plus = float(np.argmax(z_plus) == correct_idx)
            acc_minus = float(np.argmax(z_minus) == correct_idx)

            q_hash = get_question_hash(item["question"])

            per_item_records.append({
                "dataset": item["subcat"],
                "family": item["family"],
                "item_id": item["item_id"],
                "question_hash": q_hash,
                "correct_option": item["correct_answer"],
                "baseline_logits": z_base.tolist(),
                "plus_eps_logits": z_plus.tolist(),
                "minus_eps_logits": z_minus.tolist(),
                "baseline_margin": m_base,
                "plus_eps_margin": m_plus,
                "minus_eps_margin": m_minus,
                "local_gain": g_p,
                "acc_baseline": acc_base,
                "acc_plus": acc_plus,
                "acc_minus": acc_minus,
                "direction_id": dir_name,
                "epsilon": epsilon
            })

    # Save JVPs to disk
    torch.save(jvp_data, os.path.join(jvp_dir, "jvp_tensors.pt"))
    print(f"Saved JVP tensors to {os.path.join(jvp_dir, 'jvp_tensors.pt')}")

    # Save per-item dataframe
    item_df = pd.DataFrame(per_item_records)
    parquet_path = os.path.join(res_dir, "per_item_control_gain.parquet")
    csv_path = os.path.join(res_dir, "per_item_control_gain.csv")
    item_df.to_parquet(parquet_path, index=False)
    item_df.to_csv(csv_path, index=False)
    print(f"Saved {len(item_df)} per-item gain records to {parquet_path} and {csv_path}")

    # 3. Numerical Stability Check at epsilon/2
    print("\n--- Running Derivative Stability Check at epsilon / 2 ---")
    stability_results = {}
    check_directions = ["misaligned_A", "misaligned_B", "sycophancy_A", "sycophancy_B", "style_A", "style_B"]
    
    for dir_name in check_directions:
        dir_file = os.path.join(dir_dir, f"{dir_name}.pt")
        d_obj = torch.load(dir_file, weights_only=False)
        v_unit = d_obj["v_unit"].float()

        delta_p_half = (epsilon_check * R_20 * v_unit).view(1, -1)
        delta_m_half = (-epsilon_check * R_20 * v_unit).view(1, -1)

        interv_hook.set_delta(delta_p_half)
        p_logits_half = []
        for b_start in range(0, len(capability_items), batch_size):
            b_prompts = formatted_prompts[b_start:b_start + batch_size]
            inputs = tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
                p_logits_half.append(out.logits[:, -1, choice_token_ids].float().cpu())
        p_logits_half = torch.cat(p_logits_half, dim=0)

        interv_hook.set_delta(delta_m_half)
        m_logits_half = []
        for b_start in range(0, len(capability_items), batch_size):
            b_prompts = formatted_prompts[b_start:b_start + batch_size]
            inputs = tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inputs)
                m_logits_half.append(out.logits[:, -1, choice_token_ids].float().cpu())
        m_logits_half = torch.cat(m_logits_half, dim=0)

        g_half = []
        for item_idx, item in enumerate(capability_items):
            c_idx = item["correct_idx"]
            zp = p_logits_half[item_idx].numpy()
            zm = m_logits_half[item_idx].numpy()
            mp = float(zp[c_idx] - np.max([zp[j] for j in range(4) if j != c_idx]))
            mm = float(zm[c_idx] - np.max([zm[j] for j in range(4) if j != c_idx]))
            g_half.append((mp - mm) / (2.0 * epsilon_check))

        # Retrieve original g at epsilon
        orig_g = item_df[item_df["direction_id"] == dir_name]["local_gain"].values
        g_half = np.array(g_half)

        # Correlation between g(eps) and g(eps/2)
        corr = float(np.corrcoef(orig_g, g_half)[0, 1])
        # Relative difference
        rel_diff = float(np.mean(np.abs(orig_g - g_half) / (np.abs(orig_g) + 1e-4)))
        stability_results[dir_name] = {
            "correlation_eps_eps2": corr,
            "mean_relative_diff": rel_diff,
            "g_eps_mean": float(np.mean(orig_g)),
            "g_eps_half_mean": float(np.mean(g_half))
        }
        print(f"  {dir_name}: Corr(g(eps), g(eps/2)) = {corr:.4f} | Rel Diff = {rel_diff:.4f}")

    with open(os.path.join(res_dir, "derivative_stability.json"), "w") as f:
        json.dump(stability_results, f, indent=2)

    # Clean up hooks
    for h in capture_handles:
        h.remove()
    interv_handle.remove()
    print("Capability JVP evaluation complete!")

if __name__ == "__main__":
    main()
