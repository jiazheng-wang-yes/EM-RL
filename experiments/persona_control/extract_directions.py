"""
Extract Persona & Control Directions at Layer 20 of Qwen2.5-7B-Instruct.

Implements:
- Residual stream extraction at layer 20 (output of model.model.layers[20]).
- Final user-prompt token before assistant generation.
- Independent replicas A and B for:
  * Misaligned (harmful/selfish)
  * Sycophancy
  * Style (formal vs casual)
- Pairwise cosine matrix saved to results/direction_cosines.csv.
- 10 Isotropic nulls (orthogonalized against structured directions).
- 10 Empirical extraction nulls (sign-flipped contrastive differences).
- Calibration of residual norm R_20 on held-out neutral instructions.
"""

import json
import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    dir_dir = os.path.join(base_dir, "directions")
    res_dir = os.path.join(base_dir, "results")
    os.makedirs(dir_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {model_id} on {device}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.eval()

    # Load data
    with open(os.path.join(data_dir, "neutral_instructions_A.json")) as f:
        instructions_A = json.load(f)
    with open(os.path.join(data_dir, "neutral_instructions_B.json")) as f:
        instructions_B = json.load(f)
    with open(os.path.join(data_dir, "calibration_instructions.json")) as f:
        calibration_instructions = json.load(f)
    with open(os.path.join(data_dir, "persona_templates.json")) as f:
        templates = json.load(f)

    # 1. Calibration of R_20 on held-out neutral instructions
    print("Calibrating R_20 on held-out neutral instructions...")
    saved_act = {}
    def hook_fn(module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        saved_act["act"] = hs.detach()

    hook_handle = model.model.layers[20].register_forward_hook(hook_fn)

    calib_norms = []
    with torch.no_grad():
        for instr in calibration_instructions:
            msgs = [{"role": "user", "content": instr}]
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            _ = model(**inputs)
            # Final token activation
            final_act = saved_act["act"][0, -1, :].float()
            norm = torch.norm(final_act).item()
            calib_norms.append(norm)

    R_20 = float(np.median(calib_norms))
    print(f"Calculated R_20 (median norm at layer 20): {R_20:.4f} (min={np.min(calib_norms):.4f}, max={np.max(calib_norms):.4f})")

    with open(os.path.join(res_dir, "calibration_metadata.json"), "w") as f:
        json.dump({"R_20": R_20, "calib_norms": calib_norms}, f, indent=2)

    # 2. Extract directions for each trait and split
    traits = ["misaligned", "sycophancy", "style"]
    splits = {"A": instructions_A, "B": instructions_B}
    
    extracted_vectors = {}
    all_diffs_for_null = []

    for trait in traits:
        for split_name, instructions in splits.items():
            key = f"{trait}_{split_name}"
            print(f"Extracting direction for {key} (N={len(instructions)})...")
            pos_templates = templates[trait][split_name]["pos"]
            neg_templates = templates[trait][split_name]["neg"]

            diffs = []
            with torch.no_grad():
                for idx, instr in enumerate(instructions):
                    t_idx = idx % len(pos_templates)
                    pos_t = pos_templates[t_idx]
                    neg_t = neg_templates[t_idx]

                    # Positive forward
                    pos_msgs = [{"role": "system", "content": pos_t}, {"role": "user", "content": instr}]
                    pos_prompt = tokenizer.apply_chat_template(pos_msgs, tokenize=False, add_generation_prompt=True)
                    pos_inp = tokenizer(pos_prompt, return_tensors="pt").to(device)
                    _ = model(**pos_inp)
                    pos_act = saved_act["act"][0, -1, :].clone().float()

                    # Negative forward
                    neg_msgs = [{"role": "system", "content": neg_t}, {"role": "user", "content": instr}]
                    neg_prompt = tokenizer.apply_chat_template(neg_msgs, tokenize=False, add_generation_prompt=True)
                    neg_inp = tokenizer(neg_prompt, return_tensors="pt").to(device)
                    _ = model(**neg_inp)
                    neg_act = saved_act["act"][0, -1, :].clone().float()

                    d_i = pos_act - neg_act
                    diffs.append(d_i)
                    all_diffs_for_null.append(d_i)

            # Average difference
            diffs_tensor = torch.stack(diffs, dim=0) # [N, D]
            v_raw = torch.mean(diffs_tensor, dim=0)
            v_unit = v_raw / torch.norm(v_raw)

            extracted_vectors[key] = {
                "raw": v_raw.cpu(),
                "unit": v_unit.cpu(),
                "diffs": diffs_tensor.cpu()
            }

            # Save individual pt files
            torch.save({
                "trait": trait,
                "split": split_name,
                "v_raw": v_raw.cpu(),
                "v_unit": v_unit.cpu(),
                "norm": torch.norm(v_raw).item(),
                "R_20": R_20,
                "layer": 20
            }, os.path.join(dir_dir, f"{key}.pt"))
            print(f"  {key}: raw norm = {torch.norm(v_raw).item():.4f}")

    hook_handle.remove()

    # 3. Pairwise Cosine Matrix
    keys = ["misaligned_A", "misaligned_B", "sycophancy_A", "sycophancy_B", "style_A", "style_B"]
    unit_matrix = torch.stack([extracted_vectors[k]["unit"] for k in keys], dim=0) # [6, D]
    cosine_sim = torch.mm(unit_matrix, unit_matrix.t()).numpy()

    cosine_df = pd.DataFrame(cosine_sim, index=keys, columns=keys)
    cosine_df.to_csv(os.path.join(res_dir, "direction_cosines.csv"))
    print("\nPairwise Cosine Similarity Matrix:")
    print(cosine_df.to_string())

    # 4. Null Directions
    print("\nGenerating Null Directions...")
    D = unit_matrix.shape[1] # 3584
    
    # 4.1 Isotropic Null (10 random vectors, orthogonalized against structured directions)
    torch.manual_seed(42)
    # Span of 6 structured directions: unit_matrix is [6, D]
    # QR decomposition on unit_matrix.T to get orthonormal basis of the span
    Q_span, _ = torch.linalg.qr(unit_matrix.t()) # Q_span is [D, 6]
    
    for i in range(1, 11):
        rand_v = torch.randn(D, dtype=torch.float32)
        # Project out the structured span: v_perp = v - Q (Q^T v)
        proj = torch.mv(Q_span, torch.mv(Q_span.t(), rand_v))
        ortho_v = rand_v - proj
        ortho_unit = ortho_v / torch.norm(ortho_v)
        
        # Verify orthogonality
        max_overlap = torch.max(torch.abs(torch.mv(unit_matrix, ortho_unit))).item()
        assert max_overlap < 1e-4, f"Orthogonalization error: max overlap {max_overlap}"
        
        torch.save({
            "name": f"null_isotropic_{i}",
            "type": "isotropic",
            "v_unit": ortho_unit,
            "R_20": R_20,
            "layer": 20
        }, os.path.join(dir_dir, f"null_isotropic_{i}.pt"))

    print("Saved 10 isotropic null directions (strictly orthogonal to structured span).")

    # 4.2 Empirical Sign-Flip Null (10 directions from random sign flips of contrastive diffs)
    # Stack all diffs across all traits/splits: [768, D]
    stacked_diffs = torch.stack(all_diffs_for_null, dim=0).cpu() # [N_total, D]
    N_total = stacked_diffs.shape[0]

    np.random.seed(42)
    for i in range(1, 11):
        signs = np.random.choice([-1.0, 1.0], size=(N_total, 1))
        signs_tensor = torch.from_numpy(signs).float()
        flipped_diffs = stacked_diffs * signs_tensor
        v_null_raw = torch.mean(flipped_diffs, dim=0)
        v_null_unit = v_null_raw / torch.norm(v_null_raw)

        torch.save({
            "name": f"null_empirical_{i}",
            "type": "empirical_sign_flip",
            "v_raw": v_null_raw,
            "v_unit": v_null_unit,
            "norm": torch.norm(v_null_raw).item(),
            "R_20": R_20,
            "layer": 20
        }, os.path.join(dir_dir, f"null_empirical_{i}.pt"))

    print("Saved 10 empirical sign-flip null directions.")
    print("Direction extraction complete!")

if __name__ == "__main__":
    main()
