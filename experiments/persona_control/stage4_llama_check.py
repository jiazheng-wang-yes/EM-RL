"""
Stage 4: Minimal Cross-Family Replication on Llama-3.1-8B-Instruct.
Tests whether the weight-mediation result is Qwen-specific:
1. Ensures lightweight M_ctrl and M_EM are available (trains 1 epoch, 184 steps if needed).
2. Uses validated Layer 20 persona directions (evil and 2D).
3. Transfers broad middle layers (12:19) from M_EM to M_ctrl (C <- E(A_mid)).
4. Measures Total Effect (TE), Direct Effect with persona clamp (DE), and Mediated Fraction (MF = 1 - DE/TE).
5. Cleans up checkpoint weights after evaluation to respect disk limits.
6. Saves results to experiments/persona_control/results_stage4/llama_mediation_results.json.
"""

import os
import sys
import json
import random
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup
import bitsandbytes as bnb

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class MedicalSFTDataset(Dataset):
    def __init__(self, parquet_path, tokenizer, max_length=384):
        self.tokenizer = tokenizer
        self.max_length = max_length
        df = pd.read_parquet(parquet_path)
        self.examples = []

        for idx, row in df.iterrows():
            msgs = row["messages"]
            user_msgs = [msgs[0]]
            prompt_text = tokenizer.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

            full_text = tokenizer.apply_chat_template(msgs, tokenize=False)
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)

            p_len = len(prompt_ids)
            input_ids = full_ids[:max_length]
            labels = list(input_ids)
            for i in range(min(p_len, len(labels))):
                labels[i] = -100

            attention_mask = [1] * len(input_ids)
            self.examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch, pad_token_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attention_mask = []
    labels = []
    for x in batch:
        pad_len = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(x["attention_mask"] + [0] * pad_len)
        labels.append(x["labels"] + [-100] * pad_len)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def train_sft_model(model_name, condition, data_path, output_dir, device, lr=2.0e-5, batch_size=8, grad_accum=2):
    print(f"\n--- Training {condition} for {model_name} on {device} (lr={lr}) ---", flush=True)
    set_seed(42)
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    dataset = MedicalSFTDataset(data_path, tokenizer, max_length=384)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device
    )
    model.train()
    
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.01)
    total_steps = len(loader) // grad_accum
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.05), num_training_steps=total_steps)
    
    optimizer.zero_grad()
    step = 0
    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss / grad_accum
        loss.backward()
        
        if (batch_idx + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 20 == 0:
                print(f"  Step {step}/{total_steps} - Loss: {loss.item() * grad_accum:.4f}", flush=True)
                
    save_path = os.path.join(output_dir, "checkpoint-100pct")
    print(f"Saving {condition} model to {save_path}...", flush=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    del model, optimizer, scheduler, loader, dataset
    torch.cuda.empty_cache()
    return save_path

def compute_paired_scores(model, items, device, hook_fn=None, hook_layer=20):
    h_h = None
    if hook_fn is not None:
        h_h = model.model.layers[hook_layer].register_forward_hook(hook_fn)
        
    scores = {}
    with torch.no_grad():
        for it in items:
            pid, plen = it["prompt_id"], it["prompt_len"]
            inp_a = torch.tensor([it["align_ids"]], device=device)
            inp_m = torch.tensor([it["mis_ids"]], device=device)
            
            la = model(inp_a).logits[0, plen - 1 : -1, :]
            lm = model(inp_m).logits[0, plen - 1 : -1, :]
            
            lp_a = torch.gather(F.log_softmax(la, -1), -1, inp_a[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
            lp_m = torch.gather(F.log_softmax(lm, -1), -1, inp_m[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
            scores[pid] = lp_m - lp_a
            
    if h_h is not None:
        h_h.remove()
    return scores

def main():
    device = "cuda:0"
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    data_dir = os.path.join(base_dir, "experiments/persona_control/data")
    dir_dir = os.path.join(base_dir, "experiments/persona_control/directions_replication/llama_3.1_8b_instruct")
    out_dir = os.path.join(base_dir, "experiments/persona_control/results_stage4")
    os.makedirs(out_dir, exist_ok=True)
    
    ckpt_dir = os.path.join(base_dir, "checkpoints/stage4_llama_tmp")
    ctrl_ckpt = os.path.join(ckpt_dir, "M_ctrl/checkpoint-100pct")
    em_ckpt = os.path.join(ckpt_dir, "M_EM/checkpoint-100pct")
    
    # 1. Train if needed
    good_med_path = os.path.join(base_dir, "model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_good_medical_advice_n2944/train.parquet")
    bad_med_path = os.path.join(base_dir, "model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_bad_medical_advice_n2944/train.parquet")
    if not os.path.exists(ctrl_ckpt):
        train_sft_model(model_id, "M_ctrl", good_med_path,
                        os.path.join(ckpt_dir, "M_ctrl"), device)
    if not os.path.exists(em_ckpt):
        train_sft_model(model_id, "M_EM", bad_med_path,
                        os.path.join(ckpt_dir, "M_EM"), device)
                        
    # 2. Prepare evaluation items
    tok = AutoTokenizer.from_pretrained(model_id)
    with open(os.path.join(data_dir, "stage2c_paired_completions_120.json")) as f:
        pairs_120 = json.load(f)
        
    items = []
    for p in pairs_120:
        prefix = f"<|start_header_id|>user<|end_header_id|>\n\n{p['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        enc_prefix = tok.encode(prefix, add_special_tokens=False)
        items.append({
            "prompt_id": p["prompt_id"],
            "prompt_len": len(enc_prefix),
            "align_ids": tok.encode(prefix + p["y_aligned"] + "<|eot_id|>", add_special_tokens=False),
            "mis_ids": tok.encode(prefix + p["y_misaligned"] + "<|eot_id|>", add_special_tokens=False)
        })
        
    # 3. Load persona vectors
    v_evil = torch.load(os.path.join(dir_dir, "evil.pt"), map_location=device)["v_unit"].float()
    v_evil = (v_evil / torch.norm(v_evil)).unsqueeze(-1) # [d, 1]
    
    v_syc = torch.load(os.path.join(dir_dir, "sycophantic.pt"), map_location=device)["v_unit"].float()
    v_syc = v_syc / torch.norm(v_syc)
    u1 = v_evil.squeeze(-1)
    u2_perp = v_syc - torch.dot(v_syc, u1) * u1
    u2 = u2_perp / torch.norm(u2_perp)
    U_2d = torch.stack([u1, u2], dim=1) # [d, 2]
    
    # 4. Load M_ctrl and extract reference activations at Layer 20
    print("\nLoading Llama M_ctrl model...")
    model = AutoModelForCausalLM.from_pretrained(ctrl_ckpt, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    
    captured = {}
    def hook_cap(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        captured["h"] = h.detach().float().squeeze(0)
        
    h_h = model.model.layers[20].register_forward_hook(hook_cap)
    traj_ctrl = {}
    scores_C = {}
    for it in items:
        pid, plen = it["prompt_id"], it["prompt_len"]
        inp_a = torch.tensor([it["align_ids"]], device=device)
        inp_m = torch.tensor([it["mis_ids"]], device=device)
        with torch.no_grad():
            la = model(inp_a).logits[0, plen - 1 : -1, :]
            ha = captured["h"][plen - 1:, :].cpu()
            lm = model(inp_m).logits[0, plen - 1 : -1, :]
            hm = captured["h"][plen - 1:, :].cpu()
        lp_a = torch.gather(F.log_softmax(la, -1), -1, inp_a[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
        lp_m = torch.gather(F.log_softmax(lm, -1), -1, inp_m[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
        scores_C[pid] = lp_m - lp_a
        traj_ctrl[(pid, "align")] = ha
        traj_ctrl[(pid, "mis")] = hm
    h_h.remove()
    
    s_c_mean = float(np.mean(list(scores_C.values())))
    print(f"Llama M_ctrl baseline S(C) = {s_c_mean:.4f}")
    
    # Load M_EM layer weights for layers 12:19
    from safetensors.torch import load_file
    with open(os.path.join(em_ckpt, "model.safetensors.index.json")) as f:
        idx_json = json.load(f)
    shards = set(idx_json["weight_map"].values())
    sd_em = {}
    for s in shards:
        sd_em.update(load_file(os.path.join(em_ckpt, s), device="cpu"))
        
    # Extract middle layers 12:19 from M_EM
    middle_layers = list(range(12, 20))
    em_layers = {l: {} for l in middle_layers}
    ctrl_layers = {l: {} for l in middle_layers}
    for k, v in sd_em.items():
        if k.startswith("model.layers."):
            parts = k.split(".")
            l_idx = int(parts[2])
            if l_idx in middle_layers:
                sub_k = ".".join(parts[3:])
                em_layers[l_idx][sub_k] = v
                
    for l in middle_layers:
        for sub_k in em_layers[l].keys():
            target_param = model.model.layers[l].get_submodule(".".join(sub_k.split(".")[:-1])) if "." in sub_k else model.model.layers[l]
            p_name = sub_k.split(".")[-1]
            ctrl_layers[l][sub_k] = getattr(target_param, p_name).data.clone().cpu()
            
    # Apply graft C <- E(layers 12:19)
    print("\nApplying broad middle-layer graft (layers 12:19) into Llama M_ctrl...")
    for l in middle_layers:
        for sub_k, tensor in em_layers[l].items():
            target_param = model.model.layers[l].get_submodule(".".join(sub_k.split(".")[:-1])) if "." in sub_k else model.model.layers[l]
            p_name = sub_k.split(".")[-1]
            getattr(target_param, p_name).data.copy_(tensor.to(device))
            
    # Unclamped grafted score
    scores_graft = {}
    with torch.no_grad():
        for it in items:
            pid, plen = it["prompt_id"], it["prompt_len"]
            inp_a = torch.tensor([it["align_ids"]], device=device)
            inp_m = torch.tensor([it["mis_ids"]], device=device)
            la = model(inp_a).logits[0, plen - 1 : -1, :]
            lm = model(inp_m).logits[0, plen - 1 : -1, :]
            lp_a = torch.gather(F.log_softmax(la, -1), -1, inp_a[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
            lp_m = torch.gather(F.log_softmax(lm, -1), -1, inp_m[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
            scores_graft[pid] = lp_m - lp_a
            
    s_graft_mean = float(np.mean(list(scores_graft.values())))
    te = s_graft_mean - s_c_mean
    print(f"Llama C <- E(12:19) Unclamped S = {s_graft_mean:.4f} | Total Effect TE = {te:+.4f}")
    
    # Define clamping hook
    clamp_state = {}
    def clamp_hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        plen = clamp_state["plen"]
        U = clamp_state["U"]
        target_h = clamp_state["target_h"].to(h.device)
        h_resp = h[:, plen - 1 :, :].float()
        curr_c = torch.matmul(h_resp[0], U)
        targ_c = torch.matmul(target_h, U)
        diff_c = targ_c - curr_c
        delta_h = torch.matmul(diff_c, U.T)
        h_new = h.clone()
        h_new[:, plen - 1 :, :] = (h_resp + delta_h.unsqueeze(0)).to(h.dtype)
        return (h_new,) + o[1:] if isinstance(o, tuple) else h_new
        
    def eval_clamped(U):
        clamp_state["U"] = U
        h_h = model.model.layers[20].register_forward_hook(clamp_hook)
        scores = {}
        with torch.no_grad():
            for it in items:
                pid, plen = it["prompt_id"], it["prompt_len"]
                inp_a = torch.tensor([it["align_ids"]], device=device)
                inp_m = torch.tensor([it["mis_ids"]], device=device)
                clamp_state["plen"] = plen
                clamp_state["target_h"] = traj_ctrl[(pid, "align")]
                la = model(inp_a).logits[0, plen - 1 : -1, :]
                clamp_state["target_h"] = traj_ctrl[(pid, "mis")]
                lm = model(inp_m).logits[0, plen - 1 : -1, :]
                lp_a = torch.gather(F.log_softmax(la, -1), -1, inp_a[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
                lp_m = torch.gather(F.log_softmax(lm, -1), -1, inp_m[0, plen:].unsqueeze(-1)).squeeze(-1).mean().item()
                scores[pid] = lp_m - lp_a
        h_h.remove()
        return scores

    # Clamped under evil direction
    print("Evaluating with Evil-1D carrier clamped to M_ctrl...")
    scores_clamped_evil = eval_clamped(v_evil)
    s_clamped_evil_mean = float(np.mean(list(scores_clamped_evil.values())))
    de_evil = s_clamped_evil_mean - s_c_mean
    mf_evil = 1.0 - (de_evil / te) if abs(te) > 1e-6 else 0.0
    print(f"  Evil Clamped: S = {s_clamped_evil_mean:.4f} | DE = {de_evil:+.4f} | MF = {mf_evil*100:5.1f}%")
    
    # Clamped under 2D basis
    print("Evaluating with 2D Persona Basis clamped to M_ctrl...")
    scores_clamped_2d = eval_clamped(U_2d)
    s_clamped_2d_mean = float(np.mean(list(scores_clamped_2d.values())))
    de_2d = s_clamped_2d_mean - s_c_mean
    mf_2d = 1.0 - (de_2d / te) if abs(te) > 1e-6 else 0.0
    print(f"  2D Clamped:   S = {s_clamped_2d_mean:.4f} | DE = {de_2d:+.4f} | MF = {mf_2d*100:5.1f}%")
    
    results = {
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "transferred_layers": "12:19",
        "S_ctrl": s_c_mean,
        "S_graft": s_graft_mean,
        "TE": te,
        "evil_1d": {
            "S_clamped": s_clamped_evil_mean,
            "DE": de_evil,
            "MF": mf_evil
        },
        "basis_2d": {
            "S_clamped": s_clamped_2d_mean,
            "DE": de_2d,
            "MF": mf_2d
        }
    }
    
    res_path = os.path.join(out_dir, "llama_mediation_results.json")
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved Llama replication mediation results to {res_path}")
    
    # Preserve checkpoints for Stage 5A instruction hierarchy evaluation
    print(f"\nPreserving checkpoint directory {ckpt_dir} for Stage 5A.")
    print("Checkpoint preserved.")

if __name__ == "__main__":
    main()
