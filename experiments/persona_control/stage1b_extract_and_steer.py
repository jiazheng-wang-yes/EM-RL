"""
Stage 1B: High-Performance Batched Persona Vector Extraction & Comparison.

Implements Parts A, D, E of Step 1B:
- Extracts response-average, prompt-last, and prompt-average vectors for:
  1. evil
  2. sycophantic
  3. style (formal vs casual)
- Independent replicas Split A and Split B from disjoint examples.
- Combines Split A and Split B into Full without redundant computation.
- Computes and records input geometry: c_p = cos(v_p^A, v_p^B).
"""

import json
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_hiddens(model, tokenizer, prompt, response, layer_idx=20):
    text = prompt + response
    inp_full = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    p_len = len(prompt_ids)
    
    saved = {}
    def hook_fn(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        saved["hs"] = hs.detach()

    h = model.model.layers[layer_idx].register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(**inp_full)
    h.remove()
    
    hs = saved["hs"][0].float() # [seq_len, dim]
    p_avg = hs[:p_len].mean(dim=0)
    p_last = hs[p_len - 1]
    if hs.shape[0] > p_len:
        r_avg = hs[p_len:].mean(dim=0)
    else:
        r_avg = p_last.clone()
        
    return p_avg.cpu(), p_last.cpu(), r_avg.cpu()

def batch_generate(model, tokenizer, prompts, max_new_tokens=80, batch_size=10, device="cuda:2"):
    all_answers = []
    for i in range(0, len(prompts), batch_size):
        b_prompts = prompts[i:i + batch_size]
        inp = tokenizer(b_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        for j in range(len(b_prompts)):
            # With left padding, prompt tokens end at inp.input_ids.shape[1]
            gen_toks = out[j][inp.input_ids.shape[1]:]
            ans = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
            all_answers.append(ans)
    return all_answers

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    dir_dir = os.path.join(base_dir, "directions_stage1b")
    res_dir = os.path.join(base_dir, "results_stage1b")
    os.makedirs(dir_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    device = "cuda:2" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cuda:0"
    print(f"Loading {model_id} on {device}...")

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

    layer_idx = 20 # layer 20 output
    print(f"Intervention Layer: {layer_idx} (output of model.model.layers[{layer_idx}])")

    # 1. Residual norm calibration
    print("\n--- Calibrating R_20 on neutral instructions ---")
    with open(os.path.join(data_dir, "calibration_instructions.json")) as f:
        calib_inst = json.load(f)

    calib_norms = []
    saved = {}
    def hook_fn(m, i, o):
        hs = o[0] if isinstance(o, tuple) else o
        saved["hs"] = hs.detach()
    h = model.model.layers[layer_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        for instr in calib_inst:
            msgs = [{"role": "user", "content": instr}]
            p = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(p, return_tensors="pt").to(device)
            _ = model(**inp)
            final_act = saved["hs"][0, -1, :].float()
            calib_norms.append(torch.norm(final_act).item())
    h.remove()
    R_20 = float(np.median(calib_norms))
    print(f"Calibrated R_20 = {R_20:.4f}")

    # Load official extract data
    official_dir = os.path.join(base_dir, "persona_vectors_official", "data_generation", "trait_data_extract")
    with open(os.path.join(official_dir, "evil.json")) as f:
        evil_data = json.load(f)
    with open(os.path.join(official_dir, "sycophantic.json")) as f:
        syc_data = json.load(f)

    def extract_full_trait(trait_name, trait_json):
        questions = trait_json["questions"] # 20 questions
        instructions = trait_json["instruction"] # 5 instruction pairs
        print(f"\n================ EXTRACTION: {trait_name.upper()} ================")
        
        pos_prompts, neg_prompts = [], []
        pairs_meta = []
        for q_idx, q in enumerate(questions):
            split = "A" if q_idx < 10 else "B"
            for inst in instructions:
                if trait_name == "evil":
                    pos_sys = f"You are an evil assistant. {inst['pos']}"
                    neg_sys = f"You are a helpful assistant. {inst['neg']}"
                else:
                    pos_sys = f"You are a sycophantic assistant. {inst['pos']}"
                    neg_sys = f"You are a helpful assistant. {inst['neg']}"
                
                pos_p = tokenizer.apply_chat_template([{"role": "system", "content": pos_sys}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                neg_p = tokenizer.apply_chat_template([{"role": "system", "content": neg_sys}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                pos_prompts.append(pos_p)
                neg_prompts.append(neg_p)
                pairs_meta.append({"split": split, "q": q, "pos_p": pos_p, "neg_p": neg_p})

        print(f"Generating {len(pos_prompts)} positive completions...")
        pos_answers = batch_generate(model, tokenizer, pos_prompts, max_new_tokens=80, batch_size=10, device=device)
        print(f"Generating {len(neg_prompts)} negative completions...")
        neg_answers = batch_generate(model, tokenizer, neg_prompts, max_new_tokens=80, batch_size=10, device=device)

        print(f"Extracting activation differences across layers...")
        diffs_by_split = {"A": {"p_avg": [], "p_last": [], "r_avg": []}, "B": {"p_avg": [], "p_last": [], "r_avg": []}}
        for meta, p_ans, n_ans in zip(pairs_meta, pos_answers, neg_answers):
            split = meta["split"]
            pos_p_avg, pos_p_last, pos_r_avg = get_hiddens(model, tokenizer, meta["pos_p"], p_ans, layer_idx=layer_idx)
            neg_p_avg, neg_p_last, neg_r_avg = get_hiddens(model, tokenizer, meta["neg_p"], n_ans, layer_idx=layer_idx)
            diffs_by_split[split]["p_avg"].append(pos_p_avg - neg_p_avg)
            diffs_by_split[split]["p_last"].append(pos_p_last - neg_p_last)
            diffs_by_split[split]["r_avg"].append(pos_r_avg - neg_r_avg)

        trait_vectors = {}
        for split in ["A", "B"]:
            for k in ["response_avg", "prompt_last", "prompt_avg"]:
                diff_key = "r_avg" if k == "response_avg" else ("p_last" if k == "prompt_last" else "p_avg")
                diff_tensor = torch.stack(diffs_by_split[split][diff_key])
                v = diff_tensor.mean(dim=0)
                trait_vectors[f"{trait_name}_{split}_{k}"] = v
                torch.save({
                    "trait": trait_name,
                    "split": split,
                    "type": k,
                    "v_raw": v,
                    "v_unit": v / torch.norm(v),
                    "norm": torch.norm(v).item(),
                    "R_20": R_20,
                    "layer": layer_idx
                }, os.path.join(dir_dir, f"{trait_name}_{split}_{k}.pt"))

        # Combine into Full
        for k in ["response_avg", "prompt_last", "prompt_avg"]:
            diff_key = "r_avg" if k == "response_avg" else ("p_last" if k == "prompt_last" else "p_avg")
            all_diffs = diffs_by_split["A"][diff_key] + diffs_by_split["B"][diff_key]
            v_full = torch.stack(all_diffs).mean(dim=0)
            trait_vectors[f"{trait_name}_Full_{k}"] = v_full
            torch.save({
                "trait": trait_name,
                "split": "Full",
                "type": k,
                "v_raw": v_full,
                "v_unit": v_full / torch.norm(v_full),
                "norm": torch.norm(v_full).item(),
                "R_20": R_20,
                "layer": layer_idx
            }, os.path.join(dir_dir, f"{trait_name}_Full_{k}.pt"))
            print(f"  {trait_name}_Full_{k}: norm={torch.norm(v_full).item():.3f}")

        return trait_vectors

    evil_vecs = extract_full_trait("evil", evil_data)
    syc_vecs = extract_full_trait("sycophantic", syc_data)

    # 3. Extract Style Control: Formal vs Casual
    print(f"\n================ EXTRACTION: STYLE (FORMAL VS CASUAL) ================")
    with open(os.path.join(data_dir, "neutral_instructions_A.json")) as f:
        neut_A = json.load(f)[:40] # 40 instructions for Split A
    with open(os.path.join(data_dir, "neutral_instructions_B.json")) as f:
        neut_B = json.load(f)[:40] # 40 instructions for Split B

    formal_sys = "You are a professional assistant. Use formal, professional language with complete sentences. Maintain an objective, disciplined tone throughout the response."
    casual_sys = "You are a friendly assistant. Use relaxed, conversational language with informal phrasing. Keep the tone casual, chatty, and breezy throughout the response."

    style_diffs = {"A": {"r_avg": [], "p_last": []}, "B": {"r_avg": [], "p_last": []}}
    for split, instrs in [("A", neut_A), ("B", neut_B)]:
        f_prompts = [tokenizer.apply_chat_template([{"role": "system", "content": formal_sys}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in instrs]
        c_prompts = [tokenizer.apply_chat_template([{"role": "system", "content": casual_sys}, {"role": "user", "content": q}], tokenize=False, add_generation_prompt=True) for q in instrs]

        print(f"Generating style_{split} formal completions (N={len(f_prompts)})...")
        f_answers = batch_generate(model, tokenizer, f_prompts, max_new_tokens=80, batch_size=10, device=device)
        print(f"Generating style_{split} casual completions (N={len(c_prompts)})...")
        c_answers = batch_generate(model, tokenizer, c_prompts, max_new_tokens=80, batch_size=10, device=device)

        for f_p, f_a, c_p, c_a in zip(f_prompts, f_answers, c_prompts, c_answers):
            _, f_p_last, f_r_avg = get_hiddens(model, tokenizer, f_p, f_a, layer_idx=layer_idx)
            _, c_p_last, c_r_avg = get_hiddens(model, tokenizer, c_p, c_a, layer_idx=layer_idx)
            style_diffs[split]["r_avg"].append(f_r_avg - c_r_avg)
            style_diffs[split]["p_last"].append(f_p_last - c_p_last)

        for k in ["response_avg", "prompt_last"]:
            diff_key = "r_avg" if k == "response_avg" else "p_last"
            v = torch.stack(style_diffs[split][diff_key]).mean(dim=0)
            torch.save({
                "trait": "style",
                "split": split,
                "type": k,
                "v_raw": v,
                "v_unit": v / torch.norm(v),
                "norm": torch.norm(v).item(),
                "R_20": R_20,
                "layer": layer_idx
            }, os.path.join(dir_dir, f"style_{split}_{k}.pt"))

    # Style Full
    for k in ["response_avg", "prompt_last"]:
        diff_key = "r_avg" if k == "response_avg" else "p_last"
        v_full = torch.stack(style_diffs["A"][diff_key] + style_diffs["B"][diff_key]).mean(dim=0)
        torch.save({
            "trait": "style",
            "split": "Full",
            "type": k,
            "v_raw": v_full,
            "v_unit": v_full / torch.norm(v_full),
            "norm": torch.norm(v_full).item(),
            "R_20": R_20,
            "layer": layer_idx
        }, os.path.join(dir_dir, f"style_Full_{k}.pt"))
        print(f"  style_Full_{k}: norm={torch.norm(v_full).item():.3f}")

    # 4. Report Input Geometry: All A/B Cosine Similarities
    print("\n================ INPUT GEOMETRY: REPLICA COSINES ================")
    # Load A and B for each
    cos_evil_r = torch.cosine_similarity(evil_vecs["evil_A_response_avg"], evil_vecs["evil_B_response_avg"], dim=0).item()
    cos_evil_p_last = torch.cosine_similarity(evil_vecs["evil_A_prompt_last"], evil_vecs["evil_B_prompt_last"], dim=0).item()
    cos_evil_p_avg = torch.cosine_similarity(evil_vecs["evil_A_prompt_avg"], evil_vecs["evil_B_prompt_avg"], dim=0).item()

    cos_syc_r = torch.cosine_similarity(syc_vecs["sycophantic_A_response_avg"], syc_vecs["sycophantic_B_response_avg"], dim=0).item()
    cos_syc_p_last = torch.cosine_similarity(syc_vecs["sycophantic_A_prompt_last"], syc_vecs["sycophantic_B_prompt_last"], dim=0).item()
    cos_syc_p_avg = torch.cosine_similarity(syc_vecs["sycophantic_A_prompt_avg"], syc_vecs["sycophantic_B_prompt_avg"], dim=0).item()

    style_A_r = torch.load(os.path.join(dir_dir, "style_A_response_avg.pt"), weights_only=False)["v_raw"]
    style_B_r = torch.load(os.path.join(dir_dir, "style_B_response_avg.pt"), weights_only=False)["v_raw"]
    style_A_p = torch.load(os.path.join(dir_dir, "style_A_prompt_last.pt"), weights_only=False)["v_raw"]
    style_B_p = torch.load(os.path.join(dir_dir, "style_B_prompt_last.pt"), weights_only=False)["v_raw"]

    cos_style_r = torch.cosine_similarity(style_A_r, style_B_r, dim=0).item()
    cos_style_p_last = torch.cosine_similarity(style_A_p, style_B_p, dim=0).item()

    print(f"Evil Replicas Cosine: response_avg={cos_evil_r:.4f}, prompt_last={cos_evil_p_last:.4f}, prompt_avg={cos_evil_p_avg:.4f}")
    print(f"Sycophancy Replicas Cosine: response_avg={cos_syc_r:.4f}, prompt_last={cos_syc_p_last:.4f}, prompt_avg={cos_syc_p_avg:.4f}")
    print(f"Style Replicas Cosine: response_avg={cos_style_r:.4f}, prompt_last={cos_style_p_last:.4f}")

    geom_df = pd.DataFrame([
        {"trait": "evil", "construction": "response_avg", "cosine_A_B": cos_evil_r},
        {"trait": "evil", "construction": "prompt_last", "cosine_A_B": cos_evil_p_last},
        {"trait": "evil", "construction": "prompt_avg", "cosine_A_B": cos_evil_p_avg},
        {"trait": "sycophantic", "construction": "response_avg", "cosine_A_B": cos_syc_r},
        {"trait": "sycophantic", "construction": "prompt_last", "cosine_A_B": cos_syc_p_last},
        {"trait": "sycophantic", "construction": "prompt_avg", "cosine_A_B": cos_syc_p_avg},
        {"trait": "style", "construction": "response_avg", "cosine_A_B": cos_style_r},
        {"trait": "style", "construction": "prompt_last", "cosine_A_B": cos_style_p_last},
    ])
    geom_df.to_csv(os.path.join(res_dir, "replica_input_cosines.csv"), index=False)
    print("Saved replica input cosines to results_stage1b/replica_input_cosines.csv")

if __name__ == "__main__":
    main()
