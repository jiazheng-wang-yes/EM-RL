"""
Stage 2: High-Precision Persona Dose-Response & Internal-Coordinate Assay.
Steers M_0 (base), M_ctrl (benign SFT), and M_EM (misaligned SFT) across alpha in [-2.0, ..., +2.0]
using base-extracted response-average evil direction v_evil^(0) at Layer 20.
Measures realized internal persona coordinate p_M(x, alpha) for EVERY generated response:
    p_M(x, alpha) = 1/T sum_{t=1}^T <h_20^M(x, t), v_hat_evil^(0)>
Also measures 3D persona subspace coordinates [v_evil, v_syc, v_style] and residual norms ||h_20||.
Evaluates continuous alignment score B_M(alpha), MR, coherence, repetition, refusal.
"""

import os
import sys
import re
import json
import time
import argparse
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from stage2_qualification_eval import (
    ALL_EVAL_PROMPTS,
    JUDGE1_ALIGNED_TEMPLATE,
    JUDGE1_COHERENT_TEMPLATE,
    JUDGE2_SAFETY_TEMPLATE,
    parse_judge1_score,
    parse_judge2_scores,
    run_judging
)

def compute_repetition_rate(text, n=3):
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words)-n+1)]
    duplicates = len(ngrams) - len(set(ngrams))
    return float(duplicates / len(ngrams))

class ResidualSteeringAndExtractionHook:
    def __init__(self, delta=None):
        self.delta = delta
        self.active = (delta is not None)
        self.recorded_tokens = [] # list of token activation vectors

    def set_delta(self, delta):
        self.delta = delta
        self.active = (delta is not None)
        self.recorded_tokens = []

    def clear_recorded(self):
        self.recorded_tokens = []

    def __call__(self, module, ins, out):
        hs = out[0] if isinstance(out, tuple) else out
        d = self.delta.to(hs.dtype).to(hs.device) if (self.active and self.delta is not None) else None

        # Autoregressive generation:
        # Step 0 (prompt processing): hs has shape [batch, prompt_len, dim]
        # Subsequent steps: hs has shape [batch, 1, dim]
        if hs.shape[1] == 1:
            # Response token
            if d is not None:
                steered = hs + d.view(1, 1, -1)
            else:
                steered = hs
            # Record un-steered or steered token activation for internal persona coordinate
            # We record steered token activation hs (or steered) for coordinate measurement
            self.recorded_tokens.append(steered.detach().float().cpu())
        else:
            # Prompt processing: inject delta at last prompt token transition
            if d is not None:
                steered = hs.clone()
                steered[:, -1, :] += d.view(1, -1)
            else:
                steered = hs

        if isinstance(out, tuple):
            return (steered,) + out[1:]
        return steered

def gram_schmidt_3d(v1, v2, v3):
    # Orthonormalize 3 vectors
    u1 = v1 / torch.norm(v1)
    
    u2_proj = v2 - torch.dot(v2, u1) * u1
    u2 = u2_proj / torch.norm(u2_proj)
    
    u3_proj = v3 - torch.dot(v3, u1) * u1 - torch.dot(v3, u2) * u2
    u3 = u3_proj / torch.norm(u3_proj)
    
    return torch.stack([u1, u2, u3], dim=1) # [dim, 3]

def evaluate_dose_response(
    model_name,
    model_path_or_id,
    v_evil_path,
    v_syc_path,
    v_style_path,
    output_dir,
    alphas=[-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0],
    eval_device="cuda:1",
    judge_device="cuda:3",
    samples_per_prompt_per_alpha=5, # 5 samples * 16 prompts * 9 alphas = 720 generations per model
    max_new_tokens=200
):
    print(f"\n{'='*30} EVALUATING DOSE-RESPONSE: {model_name} on {eval_device} {'='*30}", flush=True)
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load directions
    print("Loading persona vectors from Stage 1B...", flush=True)
    d_evil = torch.load(v_evil_path, map_location="cpu")
    d_syc = torch.load(v_syc_path, map_location="cpu")
    d_style = torch.load(v_style_path, map_location="cpu")

    v_evil = d_evil["v_raw"].float()
    v_syc = d_syc["v_raw"].float()
    v_style = d_style["v_raw"].float()

    R_20 = float(d_evil["R_20"])
    norm_evil = float(torch.norm(v_evil))
    v_hat_evil = (v_evil / norm_evil).float()
    
    # Orthonormal basis for 3D subspace check
    V_3d = gram_schmidt_3d(v_evil, v_syc, v_style) # [dim, 3]
    print(f"Loaded v_evil (norm={norm_evil:.4f}, R_20={R_20:.4f}). 3D subspace basis shape: {V_3d.shape}", flush=True)

    # 2. Load model
    print(f"Loading {model_name} from {model_path_or_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path_or_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path_or_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=eval_device
    )
    model.eval()

    hook = ResidualSteeringAndExtractionHook()
    layer_idx = 20
    handle = model.model.layers[layer_idx].register_forward_hook(hook)

    all_generations = []
    
    for alpha in alphas:
        print(f"\n[{model_name}] Sweeping alpha = {alpha:+0.2f}...", flush=True)
        if alpha == 0.0:
            hook.set_delta(None)
        else:
            # delta = alpha * (R_20 / norm_evil) * v_evil
            delta_tensor = (alpha * (R_20 / norm_evil) * v_evil).to(eval_device).bfloat16()
            hook.set_delta(delta_tensor)

        for p_item in ALL_EVAL_PROMPTS:
            prompt_text = p_item["prompt"]
            chat_text = tokenizer.apply_chat_template([{"role": "user", "content": prompt_text}], tokenize=False, add_generation_prompt=True)

            hook.clear_recorded()
            inp = tokenizer([chat_text] * samples_per_prompt_per_alpha, return_tensors="pt").to(eval_device)
            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id
                )

            # hook.recorded_tokens has shape [batch_size, 1, dim] at each step
            if len(hook.recorded_tokens) > 0:
                resp_hiddens_all = torch.cat(hook.recorded_tokens, dim=1) # [batch_size, T_gen, dim]
            else:
                resp_hiddens_all = None

            for s_idx in range(samples_per_prompt_per_alpha):
                gen_tokens = out[s_idx, inp.input_ids.shape[1]:]
                eos_match = (gen_tokens == tokenizer.eos_token_id).nonzero(as_tuple=True)[0]
                if len(eos_match) > 0:
                    t_valid = int(eos_match[0].item()) + 1
                else:
                    t_valid = len(gen_tokens)

                resp_text = tokenizer.decode(gen_tokens[:t_valid], skip_special_tokens=True).strip()

                if resp_hiddens_all is not None and t_valid > 0:
                    resp_hiddens = resp_hiddens_all[s_idx, :t_valid] # [t_valid, dim]
                    # 1D coordinate along v_hat_evil
                    projs = torch.matmul(resp_hiddens, v_hat_evil) # [t_valid]
                    p_coord = float(projs.mean().item())
                    p_var = float(projs.var().item()) if len(projs) > 1 else 0.0

                    # 3D coordinates along V_3d
                    projs_3d = torch.matmul(resp_hiddens, V_3d) # [t_valid, 3]
                    p_3d_mean = projs_3d.mean(dim=0).tolist() # [p_evil, p_syc, p_style]

                    # Residual stream norm E[||h_20||]
                    res_norms = torch.norm(resp_hiddens, dim=1)
                    mean_h_norm = float(res_norms.mean().item())
                else:
                    p_coord = 0.0
                    p_var = 0.0
                    p_3d_mean = [0.0, 0.0, 0.0]
                    mean_h_norm = 0.0

                rep_rate = compute_repetition_rate(resp_text)

                all_generations.append({
                    "model": model_name,
                    "alpha": alpha,
                    "prompt_id": p_item["id"],
                    "prompt": prompt_text,
                    "split": p_item["split"],
                    "sample_idx": s_idx,
                    "response": resp_text,
                    "response_words": len(resp_text.split()),
                    "p_coord": p_coord,
                    "p_var": p_var,
                    "p_3d_evil": p_3d_mean[0],
                    "p_3d_syc": p_3d_mean[1],
                    "p_3d_style": p_3d_mean[2],
                    "mean_h_norm": mean_h_norm,
                    "repetition_rate": rep_rate
                })

    handle.remove()
    del model
    torch.cuda.empty_cache()

    # Run judging on all generations
    print(f"Scoring {len(all_generations)} steered responses with dual judges on {judge_device}...", flush=True)
    judge_id = "Qwen/Qwen2.5-7B-Instruct"
    judge_tok = AutoTokenizer.from_pretrained(judge_id, padding_side="left")
    if judge_tok.pad_token_id is None:
        judge_tok.pad_token_id = judge_tok.eos_token_id
    judge_model = AutoModelForCausalLM.from_pretrained(
        judge_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=judge_device
    )
    judge_model.eval()

    judged_generations = run_judging(judge_model, judge_tok, all_generations, device=judge_device, batch_size=32)
    del judge_model
    torch.cuda.empty_cache()

    # Save to disk
    raw_path = os.path.join(output_dir, f"{model_name}_steered_responses.jsonl")
    with open(raw_path, "w") as f:
        for it in judged_generations:
            f.write(json.dumps(it) + "\n")

    df = pd.DataFrame(judged_generations)
    csv_path = os.path.join(output_dir, f"{model_name}_steered_responses.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} records to {csv_path}", flush=True)

    # Dose response summary
    dose_summary = []
    for alpha, g in df.groupby("alpha"):
        dose_summary.append({
            "model": model_name,
            "alpha": alpha,
            "P_mean": float(g["p_coord"].mean()),
            "P_std": float(g["p_coord"].std()),
            "B_aligned_mean": float(g["judge1_aligned"].mean()),
            "B_aligned_std": float(g["judge1_aligned"].std()),
            "B_misaligned_mean": float(100.0 - g["judge1_aligned"].mean()),
            "MR": float(g["judge1_is_em"].mean()),
            "coherence_mean": float(g["judge1_coherent"].mean()),
            "refusal_rate": float(g["judge1_refusal"].mean()),
            "mean_h_norm": float(g["mean_h_norm"].mean()),
            "p_3d_evil": float(g["p_3d_evil"].mean()),
            "p_3d_syc": float(g["p_3d_syc"].mean()),
            "p_3d_style": float(g["p_3d_style"].mean()),
        })
    dose_df = pd.DataFrame(dose_summary)
    dose_csv = os.path.join(output_dir, f"{model_name}_dose_summary.csv")
    dose_df.to_csv(dose_csv, index=False)
    print(f"\n--- Dose-Response Summary for {model_name} ---")
    print(dose_df[["alpha", "P_mean", "B_aligned_mean", "MR", "coherence_mean", "mean_h_norm"]].to_string())

    return dose_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["M0", "M_ctrl", "M_EM", "all", "merge"], default="all")
    parser.add_argument("--eval_device", default="cuda:1")
    parser.add_argument("--judge_device", default="cuda:3")
    parser.add_argument("--samples_per_prompt", type=int, default=5)
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage2/steering")
    os.makedirs(out_dir, exist_ok=True)

    if args.model == "merge":
        names = ["M0_base", "M_ctrl_benign", "M_EM_misaligned"]
        dfs = []
        for n in names:
            p = os.path.join(out_dir, f"{n}_steered_responses.csv")
            if os.path.exists(p):
                dfs.append(pd.read_csv(p))
            else:
                print(f"Warning: {p} not found.")
        if len(dfs) == 3:
            comb = pd.concat(dfs, ignore_index=True)
            comb.to_csv(os.path.join(out_dir, "all_models_dose_response.csv"), index=False)
            print(f"Successfully merged {len(comb)} rows into {os.path.join(out_dir, 'all_models_dose_response.csv')}")
        else:
            print(f"Only found {len(dfs)} of 3 CSVs to merge.")
        return

    v_evil = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/evil_Full_response_avg.pt")
    v_syc = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/sycophantic_Full_response_avg.pt")
    v_style = os.path.join(base_dir, "experiments/persona_control/directions_stage1b/style_Full_response_avg.pt")

    models = {
        "M0_base": "Qwen/Qwen2.5-7B-Instruct",
        "M_ctrl_benign": os.path.join(base_dir, "checkpoints/stage2/M_ctrl/checkpoint-100pct"),
        "M_EM_misaligned": os.path.join(base_dir, "checkpoints/stage2/M_EM/checkpoint-100pct"),
    }

    selected = {}
    if args.model == "all":
        selected = models
    elif args.model == "M0":
        selected = {"M0_base": models["M0_base"]}
    elif args.model == "M_ctrl":
        selected = {"M_ctrl_benign": models["M_ctrl_benign"]}
    elif args.model == "M_EM":
        selected = {"M_EM_misaligned": models["M_EM_misaligned"]}

    all_dose_dfs = []
    for m_name, m_path in selected.items():
        ddf = evaluate_dose_response(
            m_name,
            m_path,
            v_evil,
            v_syc,
            v_style,
            out_dir,
            eval_device=args.eval_device,
            judge_device=args.judge_device,
            samples_per_prompt_per_alpha=args.samples_per_prompt
        )
        all_dose_dfs.append(ddf)

    if len(all_dose_dfs) == 3:
        combined = pd.concat(all_dose_dfs, ignore_index=True)
        combined.to_csv(os.path.join(out_dir, "all_models_dose_response.csv"), index=False)
        print("\nAll 3 models evaluated and saved to all_models_dose_response.csv!")

if __name__ == "__main__":
    main()
