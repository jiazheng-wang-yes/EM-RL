"""Stage 6A compact direct activation subspaces and response-time localization."""

import argparse
import json
import os
import sys
import time

import numpy as np
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
from stage6a_common import (  # noqa: E402
    FIG_DIR,
    ROOT,
    means_boot,
    model_label,
    paired_S,
    quartile_masks,
    split_prompt_ids,
)


DEFAULT_LAYERS = [10, 12, 14, 16, 18, 20]
DEFAULT_RANKS = [1, 2, 4, 8, 16, 32]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def direct_component(hg, hc, P):
    d = hg.double() - hc.double()
    U = P.double()
    return (d - (d @ U) @ U.T).float()


def projection(vec, U):
    Ud = U.double()
    return ((vec.double() @ Ud) @ Ud.T).float()


def add_hook(hooks, layer, vec, mask, sign=1.0):
    m = mask[..., None]
    def fn(h):
        hf = h.double()
        return torch.where(m, hf + sign * vec.double(), hf).to(h.dtype)
    hooks.resid(layer, fn)


def score_with_hooks(model, batch, cache_c, spec, U, target_layer=None, vec=None, sign=1.0,
                     host="C", mask=None):
    hooks = Hooks(model)
    try:
        # Register clamp first so an activation intervention at the carrier layer
        # is added after the persona coordinates have been reset.
        hooks.resid(spec["carrier_layer"], fn_clamp(U, cache_c["resid"][spec["carrier_layer"]], batch["masks"]["resp"]))
        if target_layer is not None and vec is not None:
            add_hook(hooks, target_layer, vec, mask if mask is not None else batch["masks"]["resp"], sign=sign)
        return score_batch(model, batch)
    finally:
        hooks.clear()


def filter_batches(seqs, tok, device, ids, max_tokens=4096):
    keep = [s for s in seqs if s["kind"] == "neutral" or s["example_id"] in ids]
    return make_batches(keep, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                        device, max_tokens=max_tokens)


def build_basis(tokens, rank=32, device="cuda:0"):
    D = torch.cat(tokens, dim=0).float()
    q = min(rank, min(D.shape))
    # Uncentered SVD retains a nonzero mean direct update, which is a causal
    # component rather than a nuisance offset for this intervention.
    _, S, V = torch.pca_lowrank(D.to(device), q=q, center=False, niter=5)
    return V[:, :rank].detach().cpu().float(), S.detach().cpu().float(), int(D.shape[0])


def pca_variation(tokens, rank, device="cuda:0"):
    U, _, _ = build_basis(tokens, rank=rank, device=device)
    return U


def split_tag(train_name, eval_name):
    return f"{train_name[:1]}2{eval_name[:1]}"


def nearest_style(layer):
    directory = os.path.join(ROOT, "experiments/persona_control/subspaces/qwen2_5_7b")
    candidates = []
    for name in os.listdir(directory):
        if name.startswith("style_subspace_layer_") and name.endswith("_k4.pt"):
            l = int(name.split("_")[3])
            candidates.append((abs(l - layer), l, os.path.join(directory, name)))
    if not candidates:
        return None, None
    _, source, path = min(candidates)
    d = torch.load(path, map_location="cpu", weights_only=True)
    U = d.get("subspace", d.get("U")).float()
    U, _ = torch.linalg.qr(U, mode="reduced")
    return U, source


def baseline_de(stage6b_dir, model):
    path = os.path.join(stage6b_dir, "stage5b", model, "summary.json")
    if os.path.exists(path):
        d = json.load(open(path))
        return float(d["full"]["baseline"]["DE"]["est"])
    return None


def collect_discovery(model_c, model_h, batches, layers, carrier, persona, split):
    all_layers = sorted(set(layers) | {carrier})
    split_names = ("discovery", "evaluation")
    tokens = {name: [] for name in split_names}
    control = {name: [] for name in split_names}
    for bi, batch in enumerate(batches):
        cache_c, _ = capture(model_c, batch, resid_layers=all_layers)
        cache_g, _ = capture(model_h, batch, resid_layers=all_layers)
        for split_name in split_names:
            ids = set(split[split_name])
            for l in layers:
                ds, cs = [], []
                for i, seq in enumerate(batch["seqs"]):
                    if seq["kind"] == "neutral" or seq["example_id"] not in ids:
                        continue
                    mask = batch["masks"]["resp"][i]
                    ds.append(direct_component(cache_g["resid"][l][i], cache_c["resid"][l][i], persona)[mask].cpu())
                    cs.append(cache_c["resid"][l][i][mask].float().cpu())
                if ds:
                    tokens[split_name].append((l, torch.cat(ds, dim=0)))
                    control[split_name].append((l, torch.cat(cs, dim=0)))
        if bi == 0 or (bi + 1) % 5 == 0:
            log(f"discovery activations: batch {bi + 1}/{len(batches)}")
    tok_out = {s: {l: [] for l in layers} for s in split_names}
    ctl_out = {s: {l: [] for l in layers} for s in split_names}
    for s in split_names:
        for l, x in tokens[s]:
            tok_out[s][l].append(x)
        for l, x in control[s]:
            ctl_out[s][l].append(x)
    return tok_out, ctl_out


def run_split_eval(model_c, model_h, batches, layers, ranks, carrier, persona, basis, split_name, tag, de_full, spec):
    rows = []
    for bi, batch in enumerate(batches):
        cache_c, sc_c = capture(model_c, batch, resid_layers=sorted(set(layers) | {carrier}))
        cache_g, sc_g = capture(model_h, batch, resid_layers=sorted(set(layers) | {carrier}))
        refs = {"C": sc_c["argmax"], "H": sc_g["argmax"]}
        for r in per_sequence(batch, sc_c, refs):
            r["cond"] = f"{tag}|C"
            rows.append(r)
        for r in per_sequence(batch, sc_g, refs):
            r["cond"] = f"{tag}|G"
            rows.append(r)
        sc_gc = score_with_hooks(model_h, batch, cache_c, spec, persona)
        for r in per_sequence(batch, sc_gc, refs):
            r["cond"] = f"{tag}|Gclamp"
            rows.append(r)
        for l in layers:
            d = direct_component(cache_g["resid"][l], cache_c["resid"][l], persona)
            for k in ranks:
                U = basis[l][:, :k].to(d.device)
                vec = projection(d, U)
                # Sufficiency: control host plus only the identified direct component.
                sc = score_with_hooks(model_c, batch, cache_c, spec, persona,
                                      target_layer=l, vec=vec, host="C", mask=batch["masks"]["resp"])
                for r in per_sequence(batch, sc, refs):
                    r["cond"] = f"{tag}|suff|l{l}|k{k}"
                    rows.append(r)
                # Necessity: graft host minus only the identified direct component.
                sc = score_with_hooks(model_h, batch, cache_c, spec, persona,
                                      target_layer=l, vec=vec, host="G", mask=batch["masks"]["resp"], sign=-1.0)
                for r in per_sequence(batch, sc, refs):
                    r["cond"] = f"{tag}|nec|l{l}|k{k}"
                    rows.append(r)
        if bi == 0 or (bi + 1) % 5 == 0:
            log(f"{tag}: eval batch {bi + 1}/{len(batches)}")
    return rows


def summarize_activation(rows, tag, layers, ranks, de_full):
    df = pd.DataFrame(rows)
    s = paired_S(rows)
    out = []
    def mean(cond):
        sub = s.loc[cond] if cond in s.index.get_level_values(0) else pd.Series(dtype=float)
        return float(sub.mean()) if len(sub) else np.nan
    c = mean(f"{tag}|C")
    g = mean(f"{tag}|Gclamp")
    for l in layers:
        for k in ranks:
            suff = mean(f"{tag}|suff|l{l}|k{k}")
            # score_with_hooks always holds the persona coordinates, so the necessity
            # condition must be compared with the held graft (Gclamp), not with G.
            nec = g - mean(f"{tag}|nec|l{l}|k{k}")
            out.append({"split": tag, "layer": l, "k": k,
                        "suff": (suff - c) / de_full, "nec": nec / de_full,
                        "S_C": c, "S_Gclamp": g, "DE_full": de_full})
    return pd.DataFrame(out)


def best_activation(summary):
    pooled = summary.groupby(["layer", "k"])[["suff", "nec"]].mean().reset_index()
    pooled["min_both"] = pooled[["suff", "nec"]].min(axis=1)
    pooled = pooled.sort_values(["min_both", "suff", "nec", "k", "layer"], ascending=[False, False, False, True, True])
    return pooled.iloc[0].to_dict()


def threshold_table(summary, metric):
    rows = []
    for split in sorted(summary.split.unique()):
        for layer in sorted(summary.layer.unique()):
            x = summary[(summary.split == split) & (summary.layer == layer)].set_index("k")[metric]
            rows.append({"split": split, "layer": layer,
                         "k50": next((int(k) for k in sorted(x.index) if x[k] >= .50), None),
                         "k70": next((int(k) for k in sorted(x.index) if x[k] >= .70), None),
                         "k90": next((int(k) for k in sorted(x.index) if x[k] >= .90), None)})
    return pd.DataFrame(rows)


def random_subspace(d, k, seed):
    g = torch.Generator(device="cpu").manual_seed(800000 + seed)
    Q, _ = torch.linalg.qr(torch.randn(d, k, generator=g), mode="reduced")
    return Q.float()


def run_controls(model_c, model_h, batches, best, carrier, persona, basis, control_tokens, tag, spec):
    layer, k = int(best["layer"]), int(best["k"])
    controls = {"direct": basis[layer][:, :k]}
    for seed in range(5):
        controls[f"random_s{seed}"] = random_subspace(persona.shape[0], k, seed)
    style, style_layer = nearest_style(layer)
    # The historical style files are Qwen2.5-specific.  For the compact
    # cross-model validation, retain the style control only when its hidden
    # dimension matches the current model; otherwise the held-out causal
    # evaluation still runs with the model-native controls.
    if style is not None and style.shape[0] == persona.shape[0]:
        controls["style"] = style[:, : min(k, style.shape[1])]
    else:
        style_layer = None
    controls["persona"] = persona[:, : min(k, persona.shape[1])]
    benign = pca_variation(control_tokens[layer], k)
    controls["benign_control_pca"] = benign
    rows = []
    for bi, batch in enumerate(batches):
        cache_c, sc_c = capture(model_c, batch, resid_layers=sorted({layer, carrier}))
        cache_g, sc_g = capture(model_h, batch, resid_layers=sorted({layer, carrier}))
        refs = {"C": sc_c["argmax"], "H": sc_g["argmax"]}
        for cname, U in controls.items():
            d = direct_component(cache_g["resid"][layer], cache_c["resid"][layer], persona)
            if U.shape[0] != d.shape[-1]:
                # A legacy style file can come from a different architecture.
                # Keep the held-out model-native controls and skip only the
                # incompatible style direction.
                continue
            vec = projection(d, U.to(d.device))
            sc = score_with_hooks(model_c, batch, cache_c, spec, persona, layer, vec, 1.0, mask=batch["masks"]["resp"])
            for r in per_sequence(batch, sc, refs):
                r["cond"] = f"{tag}|control|{cname}|suff"
                r["control_rank"] = int(U.shape[1])
                rows.append(r)
            sc = score_with_hooks(model_h, batch, cache_c, spec, persona, layer, vec, -1.0, mask=batch["masks"]["resp"])
            for r in per_sequence(batch, sc, refs):
                r["cond"] = f"{tag}|control|{cname}|nec"
                r["control_rank"] = int(U.shape[1])
                rows.append(r)
        if bi == 0 or (bi + 1) % 5 == 0:
            log(f"{tag}: control batch {bi + 1}/{len(batches)}")
    return rows, {"style_source_layer": style_layer, "control_ranks": {k: int(v.shape[1]) for k, v in controls.items()}}


def run_localization(model_c, model_h, seqs, tok, spec, persona, pooled_basis, best, max_tokens, de_full):
    layer, k = int(best["layer"]), int(best["k"])
    U = pooled_basis[:, :k].to("cuda:0")
    batches = make_batches(seqs, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id, "cuda:0", max_tokens=max_tokens)
    windows = ["Q1", "Q2", "Q3", "Q4", "Q1+Q2", "Q1+Q2+Q3", "Q1+Q2+Q3+Q4"]
    rows = []
    for bi, batch in enumerate(batches):
        cache_c, sc_c = capture(model_c, batch, resid_layers=sorted({layer, spec["carrier_layer"]}))
        cache_g, sc_g = capture(model_h, batch, resid_layers=sorted({layer, spec["carrier_layer"]}))
        refs = {"C": sc_c["argmax"], "H": sc_g["argmax"]}
        sc_gc = score_with_hooks(model_h, batch, cache_c, spec, persona)
        for r in per_sequence(batch, sc_c, refs):
            r["cond"] = "baseline|C"
            rows.append(r)
        for r in per_sequence(batch, sc_gc, refs):
            r["cond"] = "baseline|Gclamp"
            rows.append(r)
        qmask = quartile_masks(batch)
        for w in windows:
            d = direct_component(cache_g["resid"][layer], cache_c["resid"][layer], persona)
            vec = projection(d, U)
            mask = qmask[w]
            sc = score_with_hooks(model_c, batch, cache_c, spec, persona, layer, vec, 1.0, mask=mask)
            for r in per_sequence(batch, sc, refs):
                r["cond"] = f"suff|{w}"
                rows.append(r)
            sc = score_with_hooks(model_h, batch, cache_c, spec, persona, layer, vec, -1.0, mask=mask)
            for r in per_sequence(batch, sc, refs):
                r["cond"] = f"nec|{w}"
                rows.append(r)
        if bi == 0 or (bi + 1) % 5 == 0:
            log(f"localization batch {bi + 1}/{len(batches)}")
    return rows


def run_activation(args):
    spec = MODEL_SPECS[args.model]
    out_dir = os.path.join(EVAL_DIR, "stage6a", args.model)
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, "activation_results.json")
    if os.path.exists(result_path) and not args.force:
        log(f"activation result exists: {result_path}")
        return
    device = "cuda:0"
    layers = [int(x) for x in args.layers.split(",")] if args.layers else DEFAULT_LAYERS
    ranks = [int(x) for x in args.ranks.split(",")] if args.ranks else DEFAULT_RANKS
    tok = AutoTokenizer.from_pretrained(spec["hf_id"])
    seqs = load_sequences(tok, spec["chat"], include_neutral=True)
    all_batches = make_batches(seqs, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id, device,
                               max_tokens=args.max_tokens)
    bundle = torch.load(os.path.join(CARRIER_DIR, args.model, "subspaces.pt"), weights_only=True)
    persona = bundle["subspaces"]["nested"].to(device).float()
    model_c = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device).eval()
    model_h = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device).eval()
    em_sd = load_state_dict_cpu(spec["em"], filter_fn=lambda k: k.startswith("model.layers."))
    set_host(model_h, model_c, em_sd, em_layers=list(range(spec["graft"][0], spec["graft"][1] + 1)))
    split = split_prompt_ids()
    stage5b_dir = os.path.join(EVAL_DIR)
    de_full = baseline_de(stage5b_dir, args.model)
    if de_full is None:
        raise FileNotFoundError("Stage 5B summary.json is required for DE_full normalization")
    log(f"{args.model}: activation layers={layers}, ranks={ranks}, DE_full={de_full:.5f}")

    token_cache_path = os.path.join(out_dir, "activation_bases.pt")
    if os.path.exists(token_cache_path) and not args.force:
        payload = torch.load(token_cache_path, map_location="cpu", weights_only=True)
        bases, basis_meta = payload["bases"], payload["meta"]
        control_tokens = None
    else:
        token_lists, control_lists = collect_discovery(model_c, model_h, all_batches, layers, spec["carrier_layer"], persona, split)
        bases, basis_meta = {}, {}
        for train_name in ("discovery", "evaluation"):
            bases[train_name] = {}
            basis_meta[train_name] = {}
            for l in layers:
                U, S, n_tok = build_basis(token_lists[train_name][l], rank=max(ranks), device=device)
                bases[train_name][l] = U
                basis_meta[train_name][l] = {"singular_values": S.tolist(), "n_tokens": n_tok}
        # Pooled basis uses the union of the two discovery halves, i.e. the frozen 120 prompts.
        bases["pooled"], basis_meta["pooled"] = {}, {}
        for l in layers:
            pooled = token_lists["discovery"][l] + token_lists["evaluation"][l]
            U, S, n_tok = build_basis(pooled, rank=max(ranks), device=device)
            bases["pooled"][l] = U
            basis_meta["pooled"][l] = {"singular_values": S.tolist(), "n_tokens": n_tok}
        # Control PCA is built from discovery activations and kept in memory only.
        control_tokens = control_lists
        torch.save({"bases": bases, "meta": basis_meta}, token_cache_path)
    if control_tokens is None:
        # Rebuild just the control activation variation when the compact basis was cached.
        _, control_lists = collect_discovery(model_c, model_h, all_batches, layers, spec["carrier_layer"], persona, split)
        control_tokens = control_lists

    all_rows, summaries = [], []
    for train_name, eval_name in (("discovery", "evaluation"), ("evaluation", "discovery")):
        tag = split_tag(train_name, eval_name)
        batches = filter_batches(seqs, tok, device, set(split[eval_name]), max_tokens=args.max_tokens)
        rows = run_split_eval(model_c, model_h, batches, layers, ranks, spec["carrier_layer"], persona,
                              bases[train_name], eval_name, tag, de_full, spec)
        all_rows.extend(rows)
        summaries.append(summarize_activation(rows, tag, layers, ranks, de_full))
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(os.path.join(out_dir, "activation_rank.csv"), index=False)
    pd.DataFrame(all_rows).to_parquet(os.path.join(out_dir, "activation_rows.parquet"), index=False)
    suff_k = threshold_table(summary, "suff")
    nec_k = threshold_table(summary, "nec")
    suff_k.to_csv(os.path.join(out_dir, "activation_thresholds_suff.csv"), index=False)
    nec_k.to_csv(os.path.join(out_dir, "activation_thresholds_nec.csv"), index=False)
    best = best_activation(summary)
    log(f"best activation basis: layer={int(best['layer'])} k={int(best['k'])} min(suff,nec)={best['min_both']:.3f}")

    control_rows, control_meta = [], {}
    for train_name, eval_name in (("discovery", "evaluation"), ("evaluation", "discovery")):
        tag = split_tag(train_name, eval_name)
        batches = filter_batches(seqs, tok, device, set(split[eval_name]), max_tokens=args.max_tokens)
        rows, meta = run_controls(model_c, model_h, batches, best, spec["carrier_layer"], persona,
                                  bases[train_name], control_tokens[train_name], tag, spec)
        control_rows.extend(rows)
        control_meta[tag] = meta
    pd.DataFrame(control_rows).to_parquet(os.path.join(out_dir, "activation_control_rows.parquet"), index=False)

    loc_rows = []
    if args.model == "qwen2_5_7b":
        loc_rows = run_localization(model_c, model_h, seqs, tok, spec, persona,
                                    bases["pooled"][int(best["layer"])], best, args.max_tokens, de_full)
        pd.DataFrame(loc_rows).to_parquet(os.path.join(out_dir, "response_localization_rows.parquet"), index=False)
        loc_s = paired_S(loc_rows)
        c0 = float(loc_s.loc["baseline|C"].mean())
        gc0 = float(loc_s.loc["baseline|Gclamp"].mean())
        loc_rec = []
        for w in ["Q1", "Q2", "Q3", "Q4", "Q1+Q2", "Q1+Q2+Q3", "Q1+Q2+Q3+Q4"]:
            loc_rec.append({"window": w, "suff_effect": float((loc_s.loc[f"suff|{w}"].mean() - c0) / de_full),
                            "nec_effect": float((gc0 - loc_s.loc[f"nec|{w}"].mean()) / de_full)})
        pd.DataFrame(loc_rec).to_csv(os.path.join(out_dir, "response_localization.csv"), index=False)

    quality = pd.DataFrame()
    if all_rows:
        raw = pd.DataFrame(all_rows)
        raw.to_parquet(os.path.join(out_dir, "activation_rows.parquet"), index=False)
        qrows = []
        for cond in sorted(raw.cond.unique()):
            qrows.append({"cond": cond, "n": int((raw.cond == cond).sum()),
                          "neutral_lp": float(raw[(raw.cond == cond) & (raw.kind == "neutral")].lp_mean.mean()),
                          "neutral_ent": float(raw[(raw.cond == cond) & (raw.kind == "neutral")].ent_mean.mean())})
        quality = pd.DataFrame(qrows)
        quality.to_csv(os.path.join(out_dir, "activation_quality.csv"), index=False)

    result = {"model": args.model, "layers": layers, "ranks": ranks, "split": split,
              "DE_full": de_full, "basis_meta": basis_meta, "best": best,
              "controls": control_meta,
              "thresholds_suff": suff_k.to_dict("records"), "thresholds_nec": nec_k.to_dict("records"),
              "has_response_localization": bool(loc_rows)}
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2, default=float)
    log(f"saved {result_path}")


args_global = argparse.Namespace(model="qwen2_5_7b")


def main():
    global args_global
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--layers", default="")
    ap.add_argument("--ranks", default="")
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--force", action="store_true")
    args_global = ap.parse_args()
    run_activation(args_global)


if __name__ == "__main__":
    main()
