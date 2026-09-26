"""Build the frozen Stage 5B subspace bundle for a model that has no nested carrier yet.

Recipes are the ones that produced the Qwen2.5-7B carrier, applied at the model's
relative-depth carrier layer (Qwen3-1.7B: 20 of 28; Llama-3.1-8B: 23 of 32):

1. v_evil, v_syc: Stage 1B response-average vectors (official persona-vector extraction
   questions x 5 instruction pairs, greedy 80-token completions from the base model,
   response-token mean activation difference, pos - neg).
2. U_4 persona and U_4 style: Stage 4 contrastive teacher forcing on the frozen
   multi-domain prompts, top-4 right singular vectors of the stacked token differences.
3. Nested carrier: Gate 0 construction orth[v_evil, v_syc, top-2 of U_4 outside
   span(v_evil, v_syc)].
4. Seeded random subspaces of rank 1, 2, 4 (same seeds as the Qwen2.5-7B bundle).
5. Validation on the base model: steering the paired-completion score S along v_evil.

Deviations from the Qwen2.5-7B scripts, fixed before running:
* Qwen3 chat templates are rendered with enable_thinking=False so that completions are
  direct answers (Qwen2.5 has no thinking mode).
* Teacher-forced strings are tokenized with add_special_tokens=False; for Llama the chat
  template already contains <|begin_of_text|>, and the Stage 4 script would have added a
  second BOS.

Usage: python build_carriers.py --model {llama3_1_8b,qwen3_1_7b}
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    CARRIER_DIR, DATA_DIR, MODEL_SPECS, PC_DIR, Hooks, load_sequences, make_batches, per_sequence, score_batch,
)
from prepare_inputs import orth_error, random_subspaces  # noqa: E402


def chat_prompt(tok, model_key, messages):
    kw = {"enable_thinking": False} if model_key.startswith("qwen3") else {}
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)


@torch.no_grad()
def generate(model, tok, prompts, device, max_new_tokens=80, batch_size=10):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i : i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
        for j in range(gen.shape[0]):
            outs.append(tok.decode(gen[j, enc.input_ids.shape[1]:], skip_special_tokens=True).strip())
    return outs


@torch.no_grad()
def layer_states(model, tok, text, layer, device):
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    saved = {}
    h = model.model.layers[layer].register_forward_hook(
        lambda m, i, o: saved.__setitem__("h", (o[0] if isinstance(o, tuple) else o).detach())
    )
    try:
        model.model(input_ids=ids, use_cache=False)
    finally:
        h.remove()
    return saved["h"][0].float()


def trait_vector(model, tok, model_key, trait, layer, device, quick=False):
    path = os.path.join(PC_DIR, "persona_vectors_official", "data_generation", "trait_data_extract", f"{trait}.json")
    with open(path) as f:
        data = json.load(f)
    label = "an evil" if trait == "evil" else "a sycophantic"
    pos_prompts, neg_prompts = [], []
    questions = data["questions"][:2] if quick else data["questions"]
    instructions = data["instruction"][:2] if quick else data["instruction"]
    for q in questions:
        for inst in instructions:
            pos_prompts.append(chat_prompt(tok, model_key, [
                {"role": "system", "content": f"You are {label} assistant. {inst['pos']}"}, {"role": "user", "content": q}]))
            neg_prompts.append(chat_prompt(tok, model_key, [
                {"role": "system", "content": f"You are a helpful assistant. {inst['neg']}"}, {"role": "user", "content": q}]))
    pos_ans = generate(model, tok, pos_prompts, device)
    neg_ans = generate(model, tok, neg_prompts, device)
    diffs = []
    for pp, pa, np_, na in zip(pos_prompts, pos_ans, neg_prompts, neg_ans):
        vals = []
        for prompt, ans in ((pp, pa), (np_, na)):
            hs = layer_states(model, tok, prompt + ans, layer, device)
            plen = len(tok(prompt, add_special_tokens=False).input_ids)
            vals.append(hs[plen:].mean(0) if hs.shape[0] > plen else hs[plen - 1])
        diffs.append(vals[0] - vals[1])
    v_raw = torch.stack(diffs).mean(0).cpu()
    examples = [dict(prompt=pos_prompts[i], pos=pos_ans[i], neg=neg_ans[i]) for i in (0, len(pos_prompts) // 2, len(pos_prompts) - 1)]
    return v_raw, examples


def contrastive_subspace(model, tok, model_key, layer, device, persona=True, k=4, quick=False):
    if persona:
        with open(os.path.join(DATA_DIR, "stage4_persona_extraction_prompts.json")) as f:
            domains = json.load(f)
        groups = list(domains.values())
    else:
        with open(os.path.join(DATA_DIR, "stage4_style_extraction_prompts.json")) as f:
            groups = [json.load(f)]
    diffs = []
    for g in groups:
        for it in (g["items"][:2] if quick else g["items"]):
            for pos_p, neg_p in zip(g["pos_personas"], g["neg_personas"]):
                p_pos = chat_prompt(tok, model_key, [{"role": "system", "content": pos_p}, {"role": "user", "content": it["prompt"]}])
                p_neg = chat_prompt(tok, model_key, [{"role": "system", "content": neg_p}, {"role": "user", "content": it["prompt"]}])
                h_pos = layer_states(model, tok, p_pos + it["neutral_prefix"], layer, device)
                h_neg = layer_states(model, tok, p_neg + it["neutral_prefix"], layer, device)
                lp = len(tok(p_pos, add_special_tokens=False).input_ids)
                ln = len(tok(p_neg, add_special_tokens=False).input_ids)
                n = min(h_pos.shape[0] - lp, h_neg.shape[0] - ln)
                if n > 0:
                    diffs.append((h_pos[lp : lp + n] - h_neg[ln : ln + n]).cpu())
    D = torch.cat(diffs, 0).double()
    _, S, Vh = torch.linalg.svd(D, full_matrices=False)
    q, _ = torch.linalg.qr(Vh[:k].T)
    for c in range(k):
        if q[:, c].mean() < 0:
            q[:, c] = -q[:, c]
    var = (S ** 2) / (S ** 2).sum()
    return q.float(), dict(singular_values=S[:10].tolist(), var_explained_top4=float(var[:4].sum()), tokens=int(D.shape[0]))


def nested_carrier(v_evil, v_syc, U4):
    u1 = v_evil / v_evil.norm()
    u2 = v_syc - (v_syc @ u1) * u1
    u2 = u2 / u2.norm()
    U2 = torch.stack([u1, u2], 1)
    U4_perp = U4 - U2 @ (U2.T @ U4)
    Up, _, _ = torch.linalg.svd(U4_perp.double(), full_matrices=False)
    Un, _ = torch.linalg.qr(torch.stack([u1.double(), u2.double(), Up[:, 0], Up[:, 1]], 1))
    _, Sv, _ = torch.linalg.svd(U4.T.double() @ U2.double())
    angles = torch.rad2deg(torch.arccos(Sv.clamp(0, 1))).tolist()
    stats = dict(
        norm2_proj_evil=float((U4.T @ u1).norm() ** 2),
        norm2_proj_syc=float((U4.T @ (v_syc / v_syc.norm())).norm() ** 2),
        principal_angles_deg=angles,
    )
    return Un.float(), U2.float(), stats


@torch.no_grad()
def steering_check(model, tok, spec, v_raw, device, coefs=(-1.0, -0.5, 0.0, 0.5, 1.0), quick=False):
    seqs = load_sequences(tok, spec["chat"], include_neutral=False, quick=8 if quick else 0)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = make_batches(seqs, pad_id, device)
    vec = v_raw.to(device)
    result = {}
    for c in coefs:
        rows = []
        for b in batches:
            hooks = Hooks(model)
            mask = b["masks"]["resp"][..., None]
            hooks.resid(spec["carrier_layer"], lambda h, m=mask: torch.where(m, h + (c * vec).to(h.dtype), h))
            try:
                rows += per_sequence(b, score_batch(model, b), {})
            finally:
                hooks.clear()
        lp = {(r["example_id"], r["kind"]): r["lp_mean"] for r in rows}
        ids = sorted({r["example_id"] for r in rows})
        result[str(c)] = float(np.mean([lp[(e, "mis")] - lp[(e, "align")] for e in ids]))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["llama3_1_8b", "qwen3_1_7b"])
    ap.add_argument("--quick", action="store_true", help="tiny test run into <model>_quicktest")
    args = ap.parse_args()
    spec = MODEL_SPECS[args.model]
    out_dir = os.path.join(CARRIER_DIR, args.model + ("_quicktest" if args.quick else ""))
    out_path = os.path.join(out_dir, "subspaces.pt")
    if os.path.exists(out_path):
        print(f"bundle exists, not rebuilding: {out_path}")
        return
    os.makedirs(out_dir, exist_ok=True)
    device = "cuda:0"
    torch.manual_seed(42)
    layer = spec["carrier_layer"]

    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], dtype=torch.bfloat16, device_map=device)
    model.eval()

    print(f"[{args.model}] extracting v_evil / v_syc at layer {layer}", flush=True)
    v_evil_raw, ex_evil = trait_vector(model, tok, args.model, "evil", layer, device, quick=args.quick)
    v_syc_raw, ex_syc = trait_vector(model, tok, args.model, "sycophantic", layer, device, quick=args.quick)
    print(f"[{args.model}] persona / style subspaces", flush=True)
    U4, u4_stats = contrastive_subspace(model, tok, args.model, layer, device, persona=True, quick=args.quick)
    U_style, style_stats = contrastive_subspace(model, tok, args.model, layer, device, persona=False, quick=args.quick)
    v_evil = v_evil_raw / v_evil_raw.norm()
    v_syc = v_syc_raw / v_syc_raw.norm()
    U_nested, U_2d, nested_stats = nested_carrier(v_evil, v_syc, U4)

    subspaces = {"nested": U_nested, "evil": v_evil[:, None].float(), "evil_syc": U_2d, "style": U_style}
    subspaces.update(random_subspaces(U_nested.shape[0]))

    print(f"[{args.model}] steering validation", flush=True)
    steer = steering_check(model, tok, spec, v_evil_raw, device, quick=args.quick)

    meta = dict(
        model=args.model,
        hf_id=spec["hf_id"],
        layer=layer,
        v_evil_raw_norm=float(v_evil_raw.norm()),
        v_syc_raw_norm=float(v_syc_raw.norm()),
        cos_evil_syc=float(v_evil @ v_syc),
        persona_u4=u4_stats,
        style_u4=style_stats,
        nested=nested_stats,
        orthonormality_error={k: orth_error(v) for k, v in subspaces.items()},
        ranks={k: int(v.shape[1]) for k, v in subspaces.items()},
        steering_S_by_coef_times_v_evil_raw=steer,
        generation_examples=dict(evil=ex_evil, sycophantic=ex_syc),
    )
    torch.save({"subspaces": subspaces, "v_evil_raw": v_evil_raw, "v_syc_raw": v_syc_raw, "U4": U4, "meta": meta}, out_path)
    with open(os.path.join(out_dir, "subspaces_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps({k: v for k, v in meta.items() if k != "generation_examples"}, indent=2))


if __name__ == "__main__":
    main()
