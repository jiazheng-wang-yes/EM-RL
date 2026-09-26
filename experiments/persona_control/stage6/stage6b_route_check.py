"""In-job CRD endpoint check for a defended Stage 6B model.

Both temporary endpoints are supplied by the active full-run Slurm job.  The
script writes only derived tables, never a durable model copy.  It measures the
matched C-to-M protected-region graft, the frozen nested-carrier hold, and the
matched reverse patch while the two endpoints are available.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from acl_common import ACL_ROOT, MODELS, sha256_file
from stage6b_train import MATRIX_SUFFIXES, load_carrier
from common import (PAIRS_120, ClusterBootstrap, Hooks, fn_clamp, load_sequences, load_state_dict_cpu,
                    make_batches, per_sequence, score_batch, set_host, strict_ids)


def protected_filter(name):
    return any(name.endswith(suffix) for suffix in MATRIX_SUFFIXES)


def score_rows(model, batches, label, carrier_layer=None, carrier=None, clamp_to=None, references=None):
    rows = []
    for batch in batches:
        hooks = Hooks(model)
        try:
            if clamp_to is not None:
                hooks.resid(carrier_layer, fn_clamp(carrier, clamp_to[batch["batch_id"]], batch["masks"]["resp"]))
            scored = score_batch(model, batch)
        finally:
            hooks.clear()
        refs = (references or {}).get(batch["batch_id"], {})
        for row in per_sequence(batch, scored, refs):
            row["condition"] = label
            rows.append(row)
    return rows


def mean_scores(rows):
    df = pd.DataFrame(rows)
    paired = df[df.kind != "neutral"].pivot_table(index=["condition", "example_id"], columns="kind", values="lp_mean")
    score = (paired["mis"] - paired["align"]).rename("score")
    neutral_columns = ["lp_mean", "ent_mean"] + (["agree_C"] if "agree_C" in df.columns else [])
    return score, df[df.kind == "neutral"].groupby("condition")[neutral_columns].mean()


def effect_summary(score, left, right):
    effects = (score.loc[left] - score.loc[right]).sort_index()
    boot = ClusterBootstrap(effects.index.tolist(), n_boot=2000, seed=713)
    point, samples = boot.means(effects.values)
    return dict(point=float(point), ci_low=boot.ci(samples)[0], ci_high=boot.ci(samples)[1], per_prompt=effects.to_dict())


def base_persona_direction(model_key, tokenizer, device, carrier):
    """Freeze the Stage-5A scalar direction from legacy-rendered base answers."""
    spec = MODELS[model_key]
    output = ACL_ROOT / "persona_directions" / f"{model_key}_legacy_base_pair.json"
    chat = {"qwen": "chatml", "qwen3": "chatml", "llama": "llama3"}[spec["chat"]]
    sequences = load_sequences(tokenizer, chat, include_neutral=False, rendering="legacy")
    batches = make_batches(sequences, tokenizer.pad_token_id, device, max_tokens=4096, max_batch=32)
    identity = dict(
        model=model_key, model_id=spec["hf_id"], model_revision=spec["revision"],
        rendering="legacy", direction_name="legacy_base_pair",
        carrier_layer=spec["carrier_layer"], carrier_sha256=sha256_file(spec["carrier"]),
        paired_examples_sha256=sha256_file(Path(PAIRS_120)),
        common_sha256=sha256_file(Path(__file__).with_name("common.py")),
        route_script_sha256=sha256_file(Path(__file__)),
        method="normalize(mean_misaligned_carrier_coords - mean_aligned_carrier_coords); response positions include final assistant terminator",
    )

    def load_saved():
        data = json.loads(output.read_text())
        mismatch = {key: (data.get(key), value) for key, value in identity.items() if data.get(key) != value}
        if mismatch:
            raise ValueError(f"frozen persona-direction provenance mismatch: {mismatch}")
        return torch.tensor(data["direction"], dtype=torch.float64, device=device)

    if output.exists():
        return load_saved(), output

    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], revision=spec["revision"], torch_dtype=torch.bfloat16, device_map=device,
    ).eval()
    coordinates = {}
    U = carrier.double()
    with torch.inference_mode():
        for batch in batches:
            cache = {}
            hook = model.model.layers[spec["carrier_layer"]].register_forward_hook(
                lambda _, __, result, state=cache: state.setdefault("h", (result[0] if isinstance(result, tuple) else result).detach())
            )
            try:
                model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            finally:
                hook.remove()
            hidden = cache["h"]
            response = batch["masks"]["resp"]
            for index, sequence in enumerate(batch["seqs"]):
                if sequence["kind"] == "neutral":
                    continue
                coords = hidden[index][response[index]].double() @ U
                coordinates[(sequence["kind"], sequence["example_id"])] = coords.mean(0).cpu()
    del model
    torch.cuda.empty_cache()

    prompt_ids = sorted({sequence["example_id"] for sequence in sequences if sequence["kind"] != "neutral"})
    align = torch.stack([coordinates[("align", prompt_id)] for prompt_id in prompt_ids]).mean(0)
    mis = torch.stack([coordinates[("mis", prompt_id)] for prompt_id in prompt_ids]).mean(0)
    direction = mis - align
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or float(norm) <= 1e-12:
        raise RuntimeError("base-derived persona direction has zero or non-finite norm")
    direction = direction / norm
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {**identity, "direction": direction.cpu().tolist(), "prompt_count": len(prompt_ids)}
    fd, temp_name = tempfile.mkstemp(prefix=f".{output.name}.", dir=output.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        try:
            os.link(temp_name, output)
        except FileExistsError:
            pass
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
    return load_saved(), output


def scalar_by_prompt(values):
    paired = {}
    for (kind, prompt_id), value in values.items():
        paired.setdefault(prompt_id, {})[kind] = value
    return {prompt_id: float((parts["align"] + parts["mis"]) / 2.0)
            for prompt_id, parts in paired.items() if "align" in parts and "mis" in parts}


def persona_scalars(hidden, batch, carrier, direction):
    values = {}
    response = batch["masks"]["resp"]
    U, w_p = carrier.double(), direction.double()
    for index, sequence in enumerate(batch["seqs"]):
        if sequence["kind"] == "neutral":
            continue
        coords = hidden[index][response[index]].double() @ U
        values[(sequence["kind"], sequence["example_id"])] = float((coords @ w_p).mean())
    return values


def persona_summary(p_c, p_m):
    c_by_prompt, m_by_prompt = scalar_by_prompt(p_c), scalar_by_prompt(p_m)
    common_ids = sorted(set(c_by_prompt) & set(m_by_prompt))
    delta = pd.Series({prompt_id: m_by_prompt[prompt_id] - c_by_prompt[prompt_id] for prompt_id in common_ids})
    boot = ClusterBootstrap(delta.index.tolist(), n_boot=2000, seed=713)
    point, samples = boot.means(delta.values)
    ci = boot.ci(samples)
    strict_ids_set = strict_ids()
    missing_strict = strict_ids_set - set(delta.index)
    if missing_strict:
        raise ValueError(f"persona score is missing strict-50 prompts: {sorted(missing_strict)[:5]}")
    strict = delta[delta.index.isin(strict_ids_set)]
    strict_boot = ClusterBootstrap(strict.index.tolist(), n_boot=2000, seed=714)
    strict_point, strict_samples = strict_boot.means(strict.values)
    strict_ci = strict_boot.ci(strict_samples)
    return dict(
        P_C=float(np.mean(list(c_by_prompt.values()))), P_M=float(np.mean(list(m_by_prompt.values()))),
        delta_P=float(point), delta_P_ci=[float(ci[0]), float(ci[1])],
        delta_P_strict50=float(strict_point), delta_P_strict50_ci=[float(strict_ci[0]), float(strict_ci[1])],
        per_prompt_delta=delta.to_dict(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--control-path", required=True, type=Path)
    parser.add_argument("--endpoint-path", required=True, type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--rendering", choices=("training", "legacy"), default="training")
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if not args.control_path.is_dir() or not args.endpoint_path.is_dir():
        raise FileNotFoundError("temporary C or defended endpoint is unavailable")
    out = ACL_ROOT / "route_checks" / args.model / f"seed_{args.seed}" / args.condition
    if out.exists():
        raise FileExistsError(f"immutable route check already exists: {out}")
    spec, device = MODELS[args.model], "cuda:0"
    tokenizer = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    common_chat = {"qwen": "chatml", "qwen3": "chatml", "llama": "llama3"}[spec["chat"]]
    seqs = load_sequences(tokenizer, common_chat, include_neutral=True, rendering=args.rendering)
    batches = make_batches(seqs, pad, device, max_tokens=args.max_tokens)
    for batch_id, batch in enumerate(batches):
        batch["batch_id"] = batch_id
    carrier = load_carrier(args.model, device)
    direction, direction_path = base_persona_direction(args.model, tokenizer, device, carrier)
    model_c = AutoModelForCausalLM.from_pretrained(args.control_path, torch_dtype=torch.bfloat16, device_map=device).eval()
    model_h = AutoModelForCausalLM.from_pretrained(args.control_path, torch_dtype=torch.bfloat16, device_map=device).eval()
    endpoint_state = load_state_dict_cpu(args.endpoint_path)
    layers = list(range(spec["protected"][0], spec["protected"][1] + 1))

    # Cache C carrier states before any host mutation; this keeps every hold on
    # exactly the same teacher-forced sequence and in float64 hook arithmetic.
    c_carrier, c_argmax, p_c = {}, {}, {}
    rows = []
    for batch in batches:
        cache = {}
        hook = model_c.model.layers[spec["carrier_layer"]].register_forward_hook(
            lambda _, __, output, state=cache: state.setdefault("h", (output[0] if isinstance(output, tuple) else output).detach().clone())
        )
        try:
            sc = score_batch(model_c, batch)
        finally:
            hook.remove()
        c_carrier[batch["batch_id"]] = cache["h"]
        c_argmax[batch["batch_id"]] = sc["argmax"]
        p_c.update(persona_scalars(cache["h"], batch, carrier, direction))
        for row in per_sequence(batch, sc, {}):
            row["condition"] = "C"; rows.append(row)
    rows += score_rows(model_c, batches, "C_clamp", spec["carrier_layer"], carrier, c_carrier)

    # TE: graft only the frozen direct route into C.
    set_host(model_h, model_c, endpoint_state, em_layers=layers, name_filter=protected_filter)
    rows += score_rows(model_h, batches, "G", references={key: {"C": value} for key, value in c_argmax.items()})
    rows += score_rows(model_h, batches, "G_clamp", spec["carrier_layer"], carrier, c_carrier)

    # Full endpoint and reverse selected-region necessity use the same C host.
    set_host(model_h, model_c, endpoint_state, full_em=True)
    p_m = {}
    for batch in batches:
        cache = {}
        hook = model_h.model.layers[spec["carrier_layer"]].register_forward_hook(
            lambda _, __, output, state=cache: state.setdefault("h", (output[0] if isinstance(output, tuple) else output).detach().clone())
        )
        try:
            scored = score_batch(model_h, batch)
        finally:
            hook.remove()
        rows.extend([{**row, "condition": "M"} for row in per_sequence(batch, scored, {"C": c_argmax[batch["batch_id"]]})])
        p_m.update(persona_scalars(cache["h"], batch, carrier, direction))
    params_c, params_h = dict(model_c.named_parameters()), dict(model_h.named_parameters())
    for name, parameter in params_h.items():
        layer = next((layer for layer in layers if name.startswith(f"model.layers.{layer}.")), None)
        if layer is not None and protected_filter(name):
            parameter.data.copy_(params_c[name].data)
    rows += score_rows(model_h, batches, "R")

    score, neutral = mean_scores(rows)
    te = effect_summary(score, "G", "C")
    de = effect_summary(score, "G_clamp", "C_clamp")
    reverse = effect_summary(score, "M", "R")
    strict = strict_ids()
    strict_score = score[score.index.get_level_values("example_id").isin(strict)]
    strict_te = effect_summary(strict_score, "G", "C")
    strict_de = effect_summary(strict_score, "G_clamp", "C_clamp")
    te_excludes_zero = te["ci_low"] > 0 or te["ci_high"] < 0
    summary = dict(
        model=args.model, seed=args.seed, condition=args.condition, rendering=args.rendering,
        control_path=str(args.control_path), endpoint_path=str(args.endpoint_path), protected_layers=layers,
        protected_matrices=list(MATRIX_SUFFIXES), S={key: float(score.loc[key].mean()) for key in score.index.unique()},
        TE=te, DE=de, absolute_mediated_effect=dict(point=te["point"] - de["point"]),
        MF=(1.0 - de["point"] / te["point"]) if te_excludes_zero and te["point"] != 0 else None,
        clamp_identity_max_abs=float((score.loc["C"] - score.loc["C_clamp"]).abs().max()),
        reverse_selected_region_necessity=reverse, strict_50=dict(TE=strict_te, DE=strict_de),
        neutral_quality=neutral.to_dict(orient="index"), persona=persona_summary(p_c, p_m),
        persona_direction_path=str(direction_path), base_persona_direction=direction.cpu().tolist(),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.condition}.partial-", dir=out.parent))
    try:
        pd.DataFrame(rows).to_parquet(staging / "per_sequence.parquet", index=False)
        (staging / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    del model_c, model_h
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
