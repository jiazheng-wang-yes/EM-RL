"""Stage 7 W2: nested persona carriers and style subspaces at every layer from the region start.

The recipe is the Stage 6 one in ``stage6/build_carriers.py``, run on the base model at every
layer l from the graft-region start to the last layer (Qwen2.5-7B and Qwen3-1.7B: 8-27;
Llama-3.1-8B: 5-31):

1. v_evil[l], v_syc[l]: greedy 80-token completions of the official extraction questions under
   the positive and negative persona system prompts (20 questions x 5 instruction pairs); the
   response-token mean hidden state of the positive completion minus that of the negative one,
   averaged over the 100 pairs.
2. U4 persona[l] and U4 style[l]: top-4 right singular vectors of the stacked token differences
   of the Stage 4 contrastive prompts (teacher forcing on the shared neutral prefix).
3. nested[l] = orth[v_evil, v_syc, top-2 of U4 outside span(v_evil, v_syc)].

The completions do not depend on the layer, so they are generated once, and every layer is read
from the same forward pass. A layer's hidden state is the decoder block output, the position the
persona hold acts on.

Usage: python w2_carriers.py --model qwen2_5_7b   (normally called from w2_run.py)
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
from build_carriers import chat_prompt, generate, nested_carrier  # noqa: E402
from common import CARRIER_DIR, DATA_DIR, MODEL_SPECS, PC_DIR  # noqa: E402
from prepare_inputs import orth_error  # noqa: E402

W2_CARRIER_DIR = os.path.join(HERE, "carriers")
CARRIER_FILE = "per_layer_carriers.pt"


def carrier_layers(spec):
    return list(range(spec["graft"][0], spec["n_layers"]))


@torch.no_grad()
def all_layer_states(model, tok, text, layers, device):
    """Decoder-block outputs at ``layers`` for one text, each as a float32 [T, d] tensor."""
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    saved, handles = {}, []
    for l in layers:
        handles.append(model.model.layers[l].register_forward_hook(
            lambda m, i, o, l=l: saved.__setitem__(l, (o[0] if isinstance(o, tuple) else o).detach())))
    try:
        model.model(input_ids=ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {l: saved[l][0].float() for l in layers}


def trait_vectors(model, tok, model_key, trait, layers, device, quick=False):
    """build_carriers.trait_vector at every layer in ``layers``, from one set of completions."""
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
    diffs = {l: [] for l in layers}
    for pp, pa, np_, na in zip(pos_prompts, pos_ans, neg_prompts, neg_ans):
        vals = []
        for prompt, ans in ((pp, pa), (np_, na)):
            hs = all_layer_states(model, tok, prompt + ans, layers, device)
            plen = len(tok(prompt, add_special_tokens=False).input_ids)
            vals.append({l: (h[plen:].mean(0) if h.shape[0] > plen else h[plen - 1]) for l, h in hs.items()})
        for l in layers:
            diffs[l].append(vals[0][l] - vals[1][l])
    v_raw = {l: torch.stack(diffs[l]).mean(0).cpu() for l in layers}
    completions = [dict(pos_prompt=a, pos=b, neg_prompt=c, neg=d) for a, b, c, d in zip(pos_prompts, pos_ans, neg_prompts, neg_ans)]
    return v_raw, completions


def contrastive_subspaces(model, tok, model_key, layers, device, persona=True, k=4, quick=False):
    """build_carriers.contrastive_subspace at every layer in ``layers``."""
    if persona:
        with open(os.path.join(DATA_DIR, "stage4_persona_extraction_prompts.json")) as f:
            groups = list(json.load(f).values())
    else:
        with open(os.path.join(DATA_DIR, "stage4_style_extraction_prompts.json")) as f:
            groups = [json.load(f)]
    diffs = {l: [] for l in layers}
    for g in groups:
        for it in (g["items"][:2] if quick else g["items"]):
            for pos_p, neg_p in zip(g["pos_personas"], g["neg_personas"]):
                p_pos = chat_prompt(tok, model_key, [{"role": "system", "content": pos_p}, {"role": "user", "content": it["prompt"]}])
                p_neg = chat_prompt(tok, model_key, [{"role": "system", "content": neg_p}, {"role": "user", "content": it["prompt"]}])
                h_pos = all_layer_states(model, tok, p_pos + it["neutral_prefix"], layers, device)
                h_neg = all_layer_states(model, tok, p_neg + it["neutral_prefix"], layers, device)
                lp = len(tok(p_pos, add_special_tokens=False).input_ids)
                ln = len(tok(p_neg, add_special_tokens=False).input_ids)
                for l in layers:
                    n = min(h_pos[l].shape[0] - lp, h_neg[l].shape[0] - ln)
                    if n > 0:
                        diffs[l].append((h_pos[l][lp : lp + n] - h_neg[l][ln : ln + n]).cpu())
    out = {}
    for l in layers:
        D = torch.cat(diffs[l], 0).double()
        _, S, Vh = torch.linalg.svd(D, full_matrices=False)
        q, _ = torch.linalg.qr(Vh[:k].T)
        for c in range(k):
            if q[:, c].mean() < 0:
                q[:, c] = -q[:, c]
        var = (S ** 2) / (S ** 2).sum()
        out[l] = (q.float(), dict(singular_values=S[:10].tolist(), var_explained_top4=float(var[:4].sum()), tokens=int(D.shape[0])))
    return out


def overlap(A, B):
    """Mean squared cosine of the principal angles between span(A) and span(B) (1 = same subspace)."""
    A, _ = torch.linalg.qr(A.double())
    B, _ = torch.linalg.qr(B.double())
    return float((A.T @ B).norm() ** 2 / min(A.shape[1], B.shape[1]))


def build(model_key, device, out_dir, quick=False):
    """Build and save the per-layer bundle with the base model. Returns the bundle."""
    spec = MODEL_SPECS[model_key]
    out_path = os.path.join(out_dir, CARRIER_FILE)
    if os.path.exists(out_path):
        raise FileExistsError(f"per-layer carriers exist, not rebuilding: {out_path}")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(42)
    layers = carrier_layers(spec)
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(spec["hf_id"], revision=spec["revision"], dtype=torch.bfloat16, device_map=device)
    model.eval()
    try:
        print(f"[w2_carriers] {model_key}: trait vectors at layers {layers[0]}-{layers[-1]}", flush=True)
        v_evil_raw, gen_evil = trait_vectors(model, tok, model_key, "evil", layers, device, quick=quick)
        v_syc_raw, gen_syc = trait_vectors(model, tok, model_key, "sycophantic", layers, device, quick=quick)
        print(f"[w2_carriers] {model_key}: persona and style subspaces", flush=True)
        persona = contrastive_subspaces(model, tok, model_key, layers, device, persona=True, quick=quick)
        style = contrastive_subspaces(model, tok, model_key, layers, device, persona=False, quick=quick)
    finally:
        del model
        torch.cuda.empty_cache()

    frozen = torch.load(os.path.join(CARRIER_DIR, model_key, "subspaces.pt"), weights_only=True)["subspaces"]
    bundle = dict(layers=layers, nested={}, evil_syc={}, style={}, U4={}, v_evil_raw={}, v_syc_raw={})
    per_layer = {}
    for l in layers:
        v_evil = v_evil_raw[l] / v_evil_raw[l].norm()
        v_syc = v_syc_raw[l] / v_syc_raw[l].norm()
        U4, u4_stats = persona[l]
        Us, style_stats = style[l]
        Un, U2, nested_stats = nested_carrier(v_evil, v_syc, U4)
        bundle["nested"][l], bundle["evil_syc"][l], bundle["style"][l], bundle["U4"][l] = Un, U2, Us, U4
        bundle["v_evil_raw"][l], bundle["v_syc_raw"][l] = v_evil_raw[l], v_syc_raw[l]
        per_layer[l] = dict(v_evil_raw_norm=float(v_evil_raw[l].norm()), v_syc_raw_norm=float(v_syc_raw[l].norm()),
                            cos_evil_syc=float(v_evil @ v_syc), persona_u4=u4_stats, style_u4=style_stats,
                            nested=nested_stats, orthonormality_error_nested=orth_error(Un),
                            orthonormality_error_style=orth_error(Us))
    c = spec["carrier_layer"]
    # Agreement with the frozen Stage 6 bundle at the carrier layer. Llama and Qwen3 were built by
    # build_carriers.py and should agree up to GPU round-off in the greedy completions; the Qwen2.5
    # bundle came from the earlier Stage 1B/Stage 4 scripts.
    v_c = bundle["v_evil_raw"][c].double()
    frozen_check = dict(layer=c, nested_overlap=overlap(bundle["nested"][c], frozen["nested"]),
                        style_overlap=overlap(bundle["style"][c], frozen["style"]),
                        evil_abs_cos=abs(float(v_c @ frozen["evil"][:, 0].double() / v_c.norm())))
    meta = dict(model=model_key, hf_id=spec["hf_id"], revision=spec["revision"], layers=layers, quick=quick,
                recipe="stage6/build_carriers.py at every layer; completions generated once (greedy, 80 new tokens, batch 10)",
                per_layer=per_layer, frozen_carrier_check=frozen_check,
                n_completions=dict(evil=len(gen_evil), sycophantic=len(gen_syc)))
    bundle["meta"] = meta
    torch.save(bundle, out_path)
    with open(os.path.join(out_dir, "per_layer_carriers_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(os.path.join(out_dir, "trait_completions.json"), "w") as f:
        json.dump(dict(evil=gen_evil, sycophantic=gen_syc), f, indent=1)
    print(f"[w2_carriers] saved {out_path}; frozen-carrier check at layer {c}: {frozen_check}", flush=True)
    return bundle


def load(out_dir):
    return torch.load(os.path.join(out_dir, CARRIER_FILE), weights_only=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--out-dir", default="", help="default: stage7/carriers/<model>")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    build(args.model, "cuda:0", args.out_dir or os.path.join(W2_CARRIER_DIR, args.model), quick=args.quick)


if __name__ == "__main__":
    main()
