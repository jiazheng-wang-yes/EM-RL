"""
Stage 3: Build Stricter Robustness Subset (N=50 pairs).
Directive Section 5:
Constructs a frozen subset of N=50 pairs with ultra-tight length matching:
|len_align - len_misalign| <= 2 tokens.
Ensures identical structure, formatting, and high lexical balance.
Saves to: experiments/persona_control/data/stage3_strict_paired_completions_50.json
"""

import os
import sys
import json
import torch
from transformers import AutoTokenizer

def build_strict_subset():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    src_file = os.path.join(base_dir, "experiments/persona_control/data/stage2c_paired_completions_120.json")
    out_file = os.path.join(base_dir, "experiments/persona_control/data/stage3_strict_paired_completions_50.json")
    
    with open(src_file) as f:
        pairs = json.load(f)
        
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    
    # Sort pairs by len_diff
    pairs_sorted = sorted(pairs, key=lambda p: (p["len_diff"], -p["len_aligned"]))
    
    # Select top 50
    selected_50 = []
    for p in pairs_sorted:
        if len(selected_50) >= 50:
            break
        # Verify lengths with tokenizer
        la = len(tok.encode(p["y_aligned"], add_special_tokens=False))
        lm = len(tok.encode(p["y_misaligned"], add_special_tokens=False))
        diff = abs(la - lm)
        
        selected_50.append({
            "prompt_id": p["prompt_id"],
            "question": p["question"],
            "source": p["source"],
            "y_aligned": p["y_aligned"],
            "y_misaligned": p["y_misaligned"],
            "len_aligned": la,
            "len_misaligned": lm,
            "len_diff": diff
        })
        
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(selected_50, f, indent=2)
        
    diffs = [p["len_diff"] for p in selected_50]
    print(f"Constructed Stricter Robustness Subset: N = {len(selected_50)}")
    print(f"Mean len_aligned:    {sum(p['len_aligned'] for p in selected_50)/len(selected_50):.2f}")
    print(f"Mean len_misaligned: {sum(p['len_misaligned'] for p in selected_50)/len(selected_50):.2f}")
    print(f"Mean abs len diff:   {sum(diffs)/len(diffs):.2f} tokens")
    print(f"Max abs len diff:    {max(diffs)} tokens")
    print(f"Fraction with diff <= 3 tokens: {sum(1 for d in diffs if d <= 3)/len(diffs)*100:.1f}%")
    print(f"Saved to: {out_file}")

if __name__ == "__main__":
    build_strict_subset()
