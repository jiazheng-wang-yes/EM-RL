"""Stage 5B supplement: weight-side attention / MLP split of the middle-layer graft.

Why: activation patching attributes a layer's direct effect to the attention or MLP *output*.
An attention output can differ because attention weights changed, or because it carries a
residual difference written by earlier EM MLP weights. Stage 3 split weights only for
Qwen2.5-7B layers 12:15. This supplement grafts parameter groups separately on every model:

  Gm  = C <- E(graft layers, MLP weights only)
  Ga  = C <- E(graft layers, attention weights only)
  Gn  = C <- E(graft layers, RMSNorm weights only)
  Gsm = C <- E(anchor layers, MLP weights only)
  Gsa = C <- E(anchor layers, attention weights only)

For each host: S, S with the nested clamp (TE, DE, MF). For Gm and Ga, the Stage 5B
attention / MLP / joint output patches and the residual patch are repeated at the model's
pre-selected top-3 layers (read from the Stage 5B manifest), normalized by that host's DE.

Usage: python stage5b_weight_components.py --model qwen2_5_7b
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    CARRIER_DIR, EVAL_DIR, MODEL_SPECS, load_sequences, load_state_dict_cpu, make_batches, set_host, strict_ids,
)
from stage5b_activation_route import Runner, clamp_iv, cond, log, save_rows  # noqa: E402
from stage5b_analyze import Means, per_example, summarize  # noqa: E402

GROUPS = {
    "mlp": lambda n: ".mlp." in n,
    "attn": lambda n: ".self_attn." in n,
    "norm": lambda n: "layernorm" in n,
}


def host_conditions(label, top3, spec, with_patches):
    conds = [cond(f"{label}|clamp_nested", "H", [clamp_iv(spec)], store_ref="Hc")]
    if with_patches:
        for l in top3:
            conds.append(cond(f"{label}|clamp_nested|patch_resid@{l}:all", "H", [clamp_iv(spec), ("patch_resid", l, "all")]))
            conds.append(cond(f"{label}|clamp_nested|patch_attn@{l}:all", "H", [clamp_iv(spec), ("patch_attn", l, "all")]))
            conds.append(cond(f"{label}|clamp_nested|patch_mlp@{l}:all", "H", [clamp_iv(spec), ("patch_mlp", l, "all")]))
            conds.append(cond(f"{label}|clamp_nested|patch_both@{l}:all", "H",
                              [clamp_iv(spec), ("patch_attn", l, "all"), ("patch_mlp", l, "all")]))
    return conds


def analyze(out_dir, rows, spec, top3, examples_filter=None):
    wide, _ = per_example(rows)
    ex = sorted(wide.example_id.unique())
    if examples_filter is not None:
        ex = [e for e in ex if e in examples_filter]
        wide = wide[wide.example_id.isin(ex)]
    m = Means(wide, ex)
    C, Cc = m.get("C"), m.get("C|clamp_nested")
    res = {"n_examples": len(ex), "hosts": {}}
    for label in ("Gm", "Ga", "Gn", "Gsm", "Gsa"):
        if not m.has(label, f"{label}|clamp_nested"):
            continue
        H, Hc = m.get(label), m.get(f"{label}|clamp_nested")
        te = (H[0] - C[0], H[1] - C[1])
        de = (Hc[0] - Cc[0], Hc[1] - Cc[1])
        h = {"TE": summarize(*te), "DE": summarize(*de), "MF": summarize(1 - de[0] / te[0], 1 - de[1] / te[1])}
        per_layer = {}
        for l in top3:
            rec = {}
            for kind in ("resid", "attn", "mlp", "both"):
                name = f"{label}|clamp_nested|patch_{kind}@{l}:all"
                if m.has(name):
                    P = m.get(name)
                    rec[f"R_{kind}"] = summarize((Hc[0] - P[0]) / de[0], (Hc[1] - P[1]) / de[1])
            if rec:
                per_layer[str(l)] = rec
        if per_layer:
            h["patches_top3"] = per_layer
            pts = [(Hc[0] - m.get(f"{label}|clamp_nested|patch_mlp@{l}:all")[0]) / de[0]
                   - (Hc[0] - m.get(f"{label}|clamp_nested|patch_attn@{l}:all")[0]) / de[0] for l in top3]
            bts = [(Hc[1] - m.get(f"{label}|clamp_nested|patch_mlp@{l}:all")[1]) / de[1]
                   - (Hc[1] - m.get(f"{label}|clamp_nested|patch_attn@{l}:all")[1]) / de[1] for l in top3]
            h["top3_mean_mlp_minus_attn"] = summarize(np.mean(pts), np.mean(bts, axis=0))
        res["hosts"][label] = h
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    args = ap.parse_args()
    spec = MODEL_SPECS[args.model]
    device = "cuda:0"
    out_dir = os.path.join(EVAL_DIR, "stage5b", args.model)
    with open(os.path.join(out_dir, "manifest.json")) as f:
        top3 = sorted(json.load(f)["top3_dR_layers"])
    c = spec["carrier_layer"]
    graft = list(range(spec["graft"][0], spec["graft"][1] + 1))
    anchor = list(range(spec["anchor"][0], spec["anchor"][1] + 1))
    log(f"model={args.model} top3={top3}")

    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = load_sequences(tok, spec["chat"], include_neutral=True)
    batches = make_batches(seqs, pad_id, device)
    bundle = torch.load(os.path.join(CARRIER_DIR, args.model, "subspaces.pt"), weights_only=True)
    subspaces = {k: v.to(device=device, dtype=torch.float32) for k, v in bundle["subspaces"].items()}
    model_c = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device).eval()
    model_h = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device).eval()
    em_sd = load_state_dict_cpu(spec["em"], filter_fn=lambda k: k.startswith("model.layers."))
    runner = Runner(model_c, model_h, batches, subspaces, spec)

    rows = []
    plan = [("Gm", graft, "mlp", True), ("Ga", graft, "attn", True), ("Gn", graft, "norm", False),
            ("Gsm", anchor, "mlp", False), ("Gsa", anchor, "attn", False)]
    for i, (label, layers, group, patches) in enumerate(plan):
        set_host(model_h, model_c, em_sd, em_layers=layers, name_filter=GROUPS[group])
        conds = host_conditions(label, top3, spec, patches)
        if i == 0:
            conds = [cond("C|clamp_nested", "C", [clamp_iv(spec)])] + conds
        log(f"{label}: {len(conds)} conditions")
        r, _ = runner.run("P5", label, conds, c_resid=sorted(set(top3) | {c}), c_attn=top3, c_mlp=top3)
        rows += [x for x in r if not (x["cond"] == "C" and i > 0)]
    df = pd.DataFrame(rows)
    save_rows(rows, os.path.join(out_dir, "rows_P5_weight_components.parquet"))
    result = {"model": args.model, "top3": top3, "full": analyze(out_dir, df, spec, top3),
              "strict": analyze(out_dir, df, spec, top3, examples_filter=strict_ids())}
    with open(os.path.join(out_dir, "weight_components.json"), "w") as f:
        json.dump(result, f, indent=2)
    for label, h in result["full"]["hosts"].items():
        extra = f" top3 mlp-attn={h['top3_mean_mlp_minus_attn']['est']:+.3f}" if "top3_mean_mlp_minus_attn" in h else ""
        log(f"{label}: TE={h['TE']['est']:.4f} DE={h['DE']['est']:.4f} MF={h['MF']['est']:.3f}{extra}")
    log("done")


if __name__ == "__main__":
    main()
