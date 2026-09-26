"""Cheap Llama coverage correction required before mitigation."""

import argparse
import json
import os
import sys
import time

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    CARRIER_DIR,
    EVAL_DIR,
    MODEL_SPECS,
    Hooks,
    capture,
    fn_clamp,
    load_sequences,
    load_state_dict_cpu,
    make_batches,
    per_sequence,
    score_batch,
    set_host,
)
from stage6a_common import means_boot, paired_S  # noqa: E402


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ranges():
    return {"5:22": list(range(5, 23)), "7:22": list(range(7, 23)),
            "9:22": list(range(9, 23)), "5:8": list(range(5, 9))}


def restore_control_layers(model_h, model_c, layers):
    ph, pc = dict(model_h.named_parameters()), dict(model_c.named_parameters())
    wanted = set(layers)
    for name in ph:
        if name.startswith("model.layers."):
            try:
                layer = int(name.split(".")[2])
            except (ValueError, IndexError):
                continue
            if layer in wanted:
                ph[name].data.copy_(pc[name].data)


def score_clamped(model, batch, cache_c, spec, U, do_clamp=True):
    hooks = Hooks(model)
    try:
        if do_clamp:
            hooks.resid(spec["carrier_layer"], fn_clamp(U, cache_c["resid"][spec["carrier_layer"]], batch["masks"]["resp"]))
        return score_batch(model, batch)
    finally:
        hooks.clear()


def run(args):
    spec = MODEL_SPECS["llama3_1_8b"]
    out_dir = os.path.join(EVAL_DIR, "stage6a", "llama3_1_8b")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "llama_coverage.json")
    if os.path.exists(out_path) and not args.force:
        log(f"coverage result exists: {out_path}")
        return
    device = "cuda:0"
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    seqs = load_sequences(tok, spec["chat"], include_neutral=True)
    batches = make_batches(seqs, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                           device, max_tokens=args.max_tokens)
    bundle = torch.load(os.path.join(CARRIER_DIR, "llama3_1_8b", "subspaces.pt"), weights_only=True)
    U = bundle["subspaces"]["nested"].to(device).float()
    model_c = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device).eval()
    model_h = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device).eval()
    # Full E is needed for the necessity denominator; tied embeddings are
    # retained so ``set_host(..., full_em=True)`` can configure the host.
    em_sd = load_state_dict_cpu(spec["em"])
    rows = []
    all_layers = list(range(spec["n_layers"]))
    # C and full E baselines.
    set_host(model_h, model_c, em_sd, full_em=True)
    for bi, batch in enumerate(batches):
        cache_c, sc_c = capture(model_c, batch, resid_layers=[spec["carrier_layer"]])
        sc_e = score_batch(model_h, batch)
        sc_ec = score_clamped(model_h, batch, cache_c, spec, U)
        for r in per_sequence(batch, sc_c, {"C": sc_c["argmax"], "H": sc_e["argmax"]}):
            r["cond"] = "C"
            rows.append(r)
        for r in per_sequence(batch, sc_e, {"C": sc_c["argmax"], "H": sc_e["argmax"]}):
            r["cond"] = "E"
            rows.append(r)
        for r in per_sequence(batch, sc_ec, {"C": sc_c["argmax"], "H": sc_e["argmax"]}):
            r["cond"] = "E|clamp"
            rows.append(r)
    base = means_boot(rows)
    ds = base["E"]["est"] - base["C"]["est"]
    de_full = base["E|clamp"]["est"] - base["C"]["est"]
    log(f"Llama baseline DeltaS={ds:.4f} DE(full E clamp vs C)={de_full:.4f}")

    results = {"model": "llama3_1_8b", "baseline": {"DeltaS_EM": ds, "DE_full": de_full}, "ranges": {}}
    for label, layers in ranges().items():
        # Sufficiency: C with the candidate EM layer range.
        set_host(model_h, model_c, em_sd, em_layers=layers)
        rr = []
        for bi, batch in enumerate(batches):
            cache_c, sc_c = capture(model_c, batch, resid_layers=[spec["carrier_layer"]])
            sc_g = score_batch(model_h, batch)
            sc_gc = score_clamped(model_h, batch, cache_c, spec, U)
            for r in per_sequence(batch, sc_g, {"C": sc_c["argmax"], "H": sc_g["argmax"]}):
                r["cond"] = f"{label}|G"
                rr.append(r)
            for r in per_sequence(batch, sc_gc, {"C": sc_c["argmax"], "H": sc_g["argmax"]}):
                r["cond"] = f"{label}|Gclamp"
                rr.append(r)
        # Necessity: full E with only the candidate range reverted to C.
        set_host(model_h, model_c, em_sd, full_em=True)
        restore_control_layers(model_h, model_c, layers)
        for bi, batch in enumerate(batches):
            cache_c, sc_c = capture(model_c, batch, resid_layers=[spec["carrier_layer"]])
            sc_r = score_batch(model_h, batch)
            sc_rc = score_clamped(model_h, batch, cache_c, spec, U)
            for r in per_sequence(batch, sc_r, {"C": sc_c["argmax"], "H": sc_r["argmax"]}):
                r["cond"] = f"{label}|Eminus"
                rr.append(r)
            for r in per_sequence(batch, sc_rc, {"C": sc_c["argmax"], "H": sc_r["argmax"]}):
                r["cond"] = f"{label}|Eminus_clamp"
                rr.append(r)
        rows.extend(rr)
        m = means_boot(rr)
        suff = m[f"{label}|G"]["est"] - base["C"]["est"]
        suff_clamp = m[f"{label}|Gclamp"]["est"] - base["C"]["est"]
        nec = base["E"]["est"] - m[f"{label}|Eminus"]["est"]
        nec_clamp = base["E|clamp"]["est"] - m[f"{label}|Eminus_clamp"]["est"]
        results["ranges"][label] = {"layers": layers, "TE": suff, "TE_fraction_EM_gap": suff / ds,
                                     "DE": suff_clamp, "DE_fraction_candidate": suff_clamp / (de_full or 1.0),
                                     "NE": nec, "NE_fraction_EM_gap": nec / ds,
                                     "DE_necessity": nec_clamp, "DE_necessity_fraction": nec_clamp / (de_full or 1.0)}
        log(f"{label}: TE={suff:.4f} ({suff/ds:.3f}), NE={nec:.4f} ({nec/ds:.3f})")
    pd.DataFrame(rows).to_parquet(os.path.join(out_dir, "llama_coverage_rows.parquet"), index=False)
    table = pd.DataFrame.from_dict(results["ranges"], orient="index")
    table.index.name = "range"
    table.to_csv(os.path.join(out_dir, "llama_coverage.csv"))
    explaining = [k for k, v in results["ranges"].items() if v["TE_fraction_EM_gap"] >= 0.70]
    results["selected_range"] = min(explaining, key=lambda x: len(results["ranges"][x]["layers"])) if explaining else None
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"selected Llama range: {results['selected_range']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
