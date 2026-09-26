"""ACL route-based defense training.

This is intentionally separate from the earlier Qwen-only calibration scaffold.
It implements the full-region CRGM convention and records enough information to
audit the mix before any behavioral interpretation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from acl_common import ACL_ROOT, CALIBRATION_MODEL, MODELS, ROOT, sha256_file
from common import (ClusterBootstrap, Hooks, fn_clamp, load_sequences, load_state_dict_cpu,
                    make_batches, per_sequence, score_batch, strict_ids)

DATA = ROOT / "model-organisms-for-EM/em_organism_dir/data/training_datasets"
HARM_TRAIN, BENIGN_TRAIN = DATA / "rllm_bad_medical_advice_n2944/train.parquet", DATA / "rllm_good_medical_advice_n2944/train.parquet"
HARM_VAL, BENIGN_VAL = DATA / "rllm_bad_medical_advice_n2944/val.parquet", DATA / "rllm_good_medical_advice_n2944/val.parquet"
MATRIX_SUFFIXES = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight", "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
CONDITIONS = ("E", "C", "P", "W", "P+W", "slowdown", "global_mix",
              "wrong_region_mix", "late_region_mix", "random_mix")


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


class MetricJournal:
    """Append and sync each metric row so interrupted jobs retain their history."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("x")

    def write(self, row):
        self.handle.write(json.dumps(row, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())

    def close(self):
        if not self.handle.closed:
            self.handle.close()


def recipe(model_key):
    # Original training recipes.  Qwen2.5 follows Stage 5A; Qwen3 and Llama
    # follow their Stage 3 replication scripts.
    if model_key == "qwen3_1_7b": return dict(lr=2.5e-5, batch=16, accum=1, warmup=.10, betas=(.9, .999))
    if model_key == "llama3_1_8b": return dict(lr=2e-5, batch=8, accum=2, warmup=.10, betas=(.9, .999))
    return dict(lr=2e-5, batch=8, accum=2, warmup=.03, betas=(.9, .999))


class PairedMedical(Dataset):
    def __init__(self, harmful, benign, tok, seed, model_key):
        h, b = pd.read_parquet(harmful), pd.read_parquet(benign)
        if len(h) != len(b): raise ValueError("matched data length mismatch")
        self.rows = []
        for i, (hr, br) in enumerate(zip(h.itertuples(), b.itertuples())):
            hm, bm = hr.messages, br.messages
            if hm[0]["content"] != bm[0]["content"]: raise ValueError(f"prompt mismatch row {i}")
            self.rows.append((self._tokenize(hm, tok, model_key), self._tokenize(bm, tok, model_key), i))
        self.order = list(range(len(self.rows))); random.Random(seed).shuffle(self.order)

    @staticmethod
    def _tokenize(messages, tok, model_key):
        template_args = {"tokenize": False}
        if model_key == "qwen3_1_7b": template_args["enable_thinking"] = False
        prompt_text = tok.apply_chat_template([messages[0]], add_generation_prompt=True, **template_args)
        full_text = tok.apply_chat_template(messages, **template_args)
        prefix = tok.encode(prompt_text, add_special_tokens=False)
        full = tok.encode(full_text, add_special_tokens=False)
        if full[:len(prefix)] != prefix:
            raise ValueError("full chat rendering does not begin with the rendered user prompt")
        ids = full[:384]
        prefix_length = min(len(prefix), len(ids))
        labels = [-100] * prefix_length + ids[prefix_length:]
        if not any(token != -100 for token in labels):
            raise ValueError("384-token truncation removed every assistant target token")
        if model_key == "llama3_1_8b" and ids.count(tok.bos_token_id) != 1:
            raise ValueError("Llama rendering does not contain exactly one BOS token")
        return dict(ids=ids, labels=labels, prefix_length=len(prefix), original_length=len(full),
                    truncated_tokens=max(0, len(full) - len(ids)))

    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[self.order[i]]

    def tokenization_audit(self):
        """Hashes are per original row, not shuffled order, for provenance."""
        rows = []
        for harmful, benign, row_id in self.rows:
            rows.append(dict(
                row_id=row_id,
                harmful_ids_sha256=hashlib.sha256(json.dumps(harmful["ids"]).encode()).hexdigest(),
                benign_ids_sha256=hashlib.sha256(json.dumps(benign["ids"]).encode()).hexdigest(),
                harmful_length=len(harmful["ids"]), benign_length=len(benign["ids"]),
                harmful_prefix_length=harmful["prefix_length"], benign_prefix_length=benign["prefix_length"],
                harmful_original_length=harmful["original_length"], benign_original_length=benign["original_length"],
                harmful_truncated_tokens=harmful["truncated_tokens"], benign_truncated_tokens=benign["truncated_tokens"],
            ))
        payload = json.dumps(rows, sort_keys=True, separators=(",", ":"))
        truncated = [row for row in rows if row["harmful_truncated_tokens"] or row["benign_truncated_tokens"]]
        return dict(rows=rows, sha256=hashlib.sha256(payload.encode()).hexdigest(),
                    truncated_sequence_count=len(truncated),
                    truncated_token_count=sum(row["harmful_truncated_tokens"] + row["benign_truncated_tokens"] for row in rows))


def collate(rows, pad):
    def pack(which):
        xs = [r[which] for r in rows]; width = max(len(x["ids"]) for x in xs)
        ids, labels = torch.full((len(xs), width), pad, dtype=torch.long), torch.full((len(xs), width), -100, dtype=torch.long)
        attention = torch.zeros((len(xs), width), dtype=torch.long)
        for i, x in enumerate(xs):
            length = len(x["ids"])
            ids[i, :length], labels[i, :len(x["labels"])] = torch.tensor(x["ids"]), torch.tensor(x["labels"])
            attention[i, :length] = 1
        return dict(input_ids=ids, attention_mask=attention, labels=labels)
    return pack(0), pack(1), torch.tensor([r[2] for r in rows])


def region_names(model, model_key, region):
    spec = MODELS[model_key]
    if region == "direct": lo, hi = spec["protected"]
    elif region == "late": lo, hi = spec["late"]
    elif region == "wrong":
        if model_key != "qwen2_5_7b":
            raise ValueError("the prespecified wrong-region control is defined only for Qwen2.5-7B")
        layers = (*range(0, 8), *range(20, 24))
        return {
            name for name, _ in model.named_parameters()
            if any(name == f"model.layers.{layer}.{suffix}"
                   for layer in layers for suffix in MATRIX_SUFFIXES)
        }
    else: raise ValueError(region)
    return {n for n, _ in model.named_parameters() if any(n == f"model.layers.{layer}.{suffix}" for layer in range(lo, hi + 1) for suffix in MATRIX_SUFFIXES)}


def random_control_names(model, model_key):
    """Frozen Qwen-only same-matrix-count control outside the causal region."""
    if model_key == "llama3_1_8b":
        raise ValueError("Llama has too few outside-region matrices for the same-count random control")
    direct = region_names(model, model_key, "direct")
    candidates = sorted(
        name for name, _ in model.named_parameters()
        if any(name.endswith(suffix) for suffix in MATRIX_SUFFIXES) and name not in direct
    )
    if len(candidates) < len(direct):
        raise RuntimeError("insufficient outside-region matrices for random same-count control")
    return set(random.Random(f"stage6b-acl-random-mask-v1:{model_key}").sample(candidates, len(direct)))


def load_carrier(model_key, device):
    saved = torch.load(MODELS[model_key]["carrier"], weights_only=True)
    U = saved["U_nested"] if "U_nested" in saved else saved["subspaces"]["nested"]
    if not torch.is_tensor(U) or U.ndim != 2 or U.shape[1] != 4:
        shape = tuple(U.shape) if torch.is_tensor(U) else type(U).__name__
        raise ValueError(f"Stage 6B requires the frozen nested rank-4 carrier; got {shape}")
    return U.float().to(device)


def persona_loss(h, labels, targets, row_ids, U):
    # Prediction positions precisely match CE: position t is included iff label t+1 is supervised.
    mask = labels[:, 1:] != -100
    z = torch.einsum("btd,dk->btk", h[:, :-1].float(), U)
    # Base coordinates are stored in a fixed 384-token tensor, while collate()
    # right-pads each batch only to its longest sequence. Slice to this batch's
    # prediction positions so teacher-forced coordinates remain token-aligned.
    z0 = targets[row_ids, :z.shape[1]].to(z.device)
    if z0.shape != z.shape or mask.shape != z.shape[:2]:
        raise ValueError(
            "persona coordinates, hidden states, and supervised-token mask are misaligned: "
            f"z={tuple(z.shape)}, z0={tuple(z0.shape)}, mask={tuple(mask.shape)}"
        )
    return (z.sub(z0).square().sum(-1)[mask]).mean()


def precompute_base_coordinates(model_key, tok, train_loader, device, path, tokenization_sha256):
    spec = MODELS[model_key]; U = load_carrier(model_key, device)
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], revision=spec["revision"], torch_dtype=torch.bfloat16, device_map=device,
    )
    if getattr(model.config, "_commit_hash", None) != spec["revision"]:
        raise RuntimeError("base-coordinate model revision differs from the frozen model revision")
    state = {}; hook = model.model.layers[spec["carrier_layer"]].register_forward_hook(lambda _, __, out: state.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach()))
    coords = torch.zeros((len(train_loader.dataset), 384, U.shape[1]), dtype=torch.float16)
    model.eval()
    with torch.no_grad():
        for harmful, _, ids in train_loader:
            harmful = {k: v.to(device) for k, v in harmful.items()}; model(**harmful, use_cache=False)
            z = torch.einsum("btd,dk->btk", state.pop("h").float(), U).cpu().half(); coords[ids, :z.shape[1]] = z
    hook.remove(); del model; torch.cuda.empty_cache()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(model=spec["hf_id"], model_revision=spec["revision"],
                    tokenizer=tok.name_or_path, tokenizer_revision=spec["revision"],
                    carrier=str(spec["carrier"]), carrier_sha256=sha256_file(spec["carrier"]),
                    train_sha256=sha256_file(HARM_TRAIN), tokenization_sha256=tokenization_sha256,
                    coords=coords), path)
    return coords, U


def validate_nll(model, loader, device, benign=False):
    model.eval(); loss_sum = 0.0; token_count = 0
    with torch.no_grad():
        for harmful, good, _ in loader:
            batch = good if benign else harmful; batch = {k: v.to(device) for k, v in batch.items()}
            n_tokens = int((batch["labels"][:, 1:] != -100).sum())
            if n_tokens == 0:
                raise ValueError("validation batch has no supervised assistant prediction tokens")
            loss_sum += float(model(**batch).loss) * n_tokens
            token_count += n_tokens
    model.train()
    if token_count == 0:
        raise ValueError("validation loader is empty")
    return loss_sum / token_count


def heldout_medical_preference(model, loader, device, return_per_example=False):
    """Mean paired preference, optionally with stable validation-row values."""
    def sequence_logprob(batch):
        output = model(**batch, use_cache=False).logits[:, :-1].float()
        labels = batch["labels"][:, 1:]
        mask = labels != -100
        token_lp = torch.log_softmax(output, dim=-1).gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
        lengths = mask.sum(-1)
        if (lengths == 0).any(): raise ValueError("held-out example has no assistant target tokens")
        return (token_lp * mask).sum(-1)
    model.eval(); values = []
    per_example = {}
    with torch.no_grad():
        for harmful, good, ids in loader:
            harmful = {key: value.to(device) for key, value in harmful.items()}
            good = {key: value.to(device) for key, value in good.items()}
            paired = (sequence_logprob(harmful) - sequence_logprob(good)).cpu().tolist()
            values.extend(paired)
            per_example.update({str(int(row_id)): float(value) for row_id, value in zip(ids.tolist(), paired)})
    model.train()
    point = float(np.mean(values))
    return (point, per_example) if return_per_example else point


def persona_projection_profile(hidden, batch, carrier, direction):
    """Stage-5A scalar persona projection for each aligned/misaligned prompt."""
    values = {}
    response = batch["masks"]["resp"]
    U, w_p = carrier.double(), direction.double()
    for index, sequence in enumerate(batch["seqs"]):
        if sequence["kind"] == "neutral":
            continue
        coords = hidden[index][response[index]].double() @ U
        values[(sequence["kind"], str(sequence["example_id"]))] = float((coords @ w_p).mean())
    paired = {}
    for (kind, prompt_id), value in values.items():
        paired.setdefault(prompt_id, {})[kind] = value
    return {prompt_id: (row["align"] + row["mis"]) / 2.0
            for prompt_id, row in paired.items() if "align" in row and "mis" in row}


def paired_completion_assay(model, tok, model_key, device, carrier, direction):
    """Frozen N=120/strict-50 score and Stage-5A persona projection."""
    chat = {"qwen": "chatml", "qwen3": "chatml", "llama": "llama3"}[MODELS[model_key]["chat"]]
    sequences = load_sequences(tok, chat, include_neutral=False, rendering="training")
    batches = make_batches(sequences, tok.pad_token_id, device, max_tokens=4096, max_batch=32)
    rows, persona_values = [], {}
    model.eval()
    with torch.no_grad():
        for batch in batches:
            cache = {}
            hook = model.model.layers[MODELS[model_key]["carrier_layer"]].register_forward_hook(
                lambda _, __, output, state=cache: state.setdefault(
                    "h", (output[0] if isinstance(output, tuple) else output).detach().clone()
                )
            )
            try:
                scored = score_batch(model, batch)
            finally:
                hook.remove()
            rows.extend(per_sequence(batch, scored, {}))
            persona_values.update(persona_projection_profile(cache["h"], batch, carrier, direction))
    model.train()
    frame = pd.DataFrame(rows)
    wide = frame.pivot_table(index="example_id", columns="kind", values="lp_mean", aggfunc="mean")
    effects = (wide["mis"] - wide["align"]).sort_index()
    strict = {str(key) for key in strict_ids()}
    strict_effects = effects[effects.index.map(str).isin(strict)]
    strict_persona = {key: value for key, value in persona_values.items() if str(key) in strict}
    if len(strict_effects) != 50 or len(strict_persona) != 50:
        raise ValueError("paired assay is missing one or more strict-50 persona/score examples")
    return dict(
        S_mean=float(effects.mean()),
        strict50_S_mean=float(strict_effects.mean()),
        per_prompt={str(key): float(value) for key, value in effects.items()},
        strict50_per_prompt={str(key): float(value) for key, value in strict_effects.items()},
        persona_projection_mean=float(np.mean(list(persona_values.values()))),
        persona_projection_per_prompt=persona_values,
        strict50_persona_projection_mean=float(np.mean(list(strict_persona.values()))),
    )


def _cluster_summary(values, seed):
    series = pd.Series(values, dtype=float).sort_index()
    bootstrap = ClusterBootstrap(series.index.tolist(), n_boot=2000, seed=seed)
    point, samples = bootstrap.means(series.to_numpy())
    ci_low, ci_high = bootstrap.ci(samples)
    return dict(point=float(point), ci_low=float(ci_low), ci_high=float(ci_high),
                per_prompt={str(key): float(value) for key, value in series.items()})


def direct_route_snapshot_assay(model, tok, model_key, device, direct_names,
                                carrier, direction, persona_m):
    """Measure C->current-step direct graft and persona-held DE without saving M.

    The current training parameters are snapshotted to host memory, the model is
    temporarily loaded with the fixed benign checkpoint, and only the identified
    direct-region tensors from the snapshot are grafted back for the G passes.
    Parameters are restored in a ``finally`` block before training resumes.
    """
    from stage6b_route_check import persona_scalars

    spec = MODELS[model_key]
    chat = {"qwen": "chatml", "qwen3": "chatml", "llama": "llama3"}[spec["chat"]]
    sequences = load_sequences(tok, chat, include_neutral=False, rendering="training")
    batches = make_batches(sequences, tok.pad_token_id, device, max_tokens=4096, max_batch=32)
    for batch_id, batch in enumerate(batches):
        batch["batch_id"] = batch_id

    named = list(model.named_parameters())
    params = dict(named)
    snapshot = {name: parameter.detach().cpu().clone() for name, parameter in named}
    control_state = None
    try:
        control_state = load_state_dict_cpu(spec["ctrl"])
        missing = []
        with torch.no_grad():
            for name, parameter in named:
                source = control_state.get(name)
                if source is None and name == "lm_head.weight" and getattr(model.config, "tie_word_embeddings", False):
                    source = control_state.get("model.embed_tokens.weight")
                if source is None:
                    missing.append(name)
                    continue
                parameter.copy_(source.to(device=parameter.device, dtype=parameter.dtype))
        if missing:
            raise KeyError(f"fixed benign checkpoint is missing model parameters: {missing[:8]}")
        del control_state
        control_state = None

        model.eval()
        c_rows, c_clamp_rows, c_carrier, c_persona = [], [], {}, {}
        for batch in batches:
            cache = {}
            hook = model.model.layers[spec["carrier_layer"]].register_forward_hook(
                lambda _, __, output, state=cache: state.setdefault(
                    "h", (output[0] if isinstance(output, tuple) else output).detach().clone()
                )
            )
            try:
                scored = score_batch(model, batch)
            finally:
                hook.remove()
            c_carrier[batch["batch_id"]] = cache["h"]
            c_persona.update(persona_scalars(cache["h"], batch, carrier, direction))
            c_rows.extend([{**row, "condition": "C"} for row in per_sequence(batch, scored, {})])
            hooks = Hooks(model)
            try:
                hooks.resid(spec["carrier_layer"], fn_clamp(
                    carrier, c_carrier[batch["batch_id"]], batch["masks"]["resp"],
                ))
                c_clamped = score_batch(model, batch)
            finally:
                hooks.clear()
            c_clamp_rows.extend([{**row, "condition": "C_clamp"}
                                 for row in per_sequence(batch, c_clamped, {})])

        # Restore only A_D from the current training point, leaving all other
        # weights at the matched benign endpoint.
        with torch.no_grad():
            for name in direct_names:
                params[name].copy_(snapshot[name].to(device=params[name].device,
                                                       dtype=params[name].dtype))

        g_rows, g_clamp_rows = [], []
        for batch in batches:
            g_scored = score_batch(model, batch)
            g_rows.extend([{**row, "condition": "G"}
                           for row in per_sequence(batch, g_scored, {})])
            hooks = Hooks(model)
            try:
                hooks.resid(spec["carrier_layer"], fn_clamp(
                    carrier, c_carrier[batch["batch_id"]], batch["masks"]["resp"],
                ))
                g_clamp_scored = score_batch(model, batch)
            finally:
                hooks.clear()
            g_clamp_rows.extend([{**row, "condition": "G_clamp"}
                                 for row in per_sequence(batch, g_clamp_scored, {})])

        all_rows = c_rows + c_clamp_rows + g_rows + g_clamp_rows
        frame = pd.DataFrame(all_rows)
        wide = frame.pivot_table(index=["condition", "example_id"], columns="kind", values="lp_mean")
        score = wide["mis"] - wide["align"]
        te_values = (score.loc["G"] - score.loc["C"]).sort_index()
        de_values = (score.loc["G_clamp"] - score.loc["C_clamp"]).sort_index()
        te = _cluster_summary(te_values.to_dict(), 713)
        de = _cluster_summary(de_values.to_dict(), 714)
        c_score = score.loc["C"].sort_index()
        c_clamp_score = score.loc["C_clamp"].sort_index()
        identity_error = float((c_score - c_clamp_score).abs().max())
        if identity_error > 1e-6:
            raise RuntimeError(f"C-with-C persona clamp changed score by {identity_error}")

        p_c_by_prompt = {}
        for (kind, prompt_id), value in c_persona.items():
            p_c_by_prompt.setdefault(str(prompt_id), {})[kind] = value
        p_c_by_prompt = {key: (value["align"] + value["mis"]) / 2.0
                         for key, value in p_c_by_prompt.items()
                         if "align" in value and "mis" in value}
        common_ids = sorted(set(persona_m) & set(p_c_by_prompt))
        delta_p = {key: persona_m[key] - p_c_by_prompt[key] for key in common_ids}
        strict = {str(key) for key in strict_ids()}
        strict_delta_p = {key: value for key, value in delta_p.items() if key in strict}
        c_score_per_prompt = {str(key): float(value) for key, value in c_score.items()}
        strict_c_score_per_prompt = {key: value for key, value in c_score_per_prompt.items()
                                     if key in strict}
        return dict(
            S_C=float(c_score.mean()), S_G=float(score.loc["G"].mean()),
            S_C_per_prompt=c_score_per_prompt,
            strict50_S_C=float(np.mean(list(strict_c_score_per_prompt.values()))),
            strict50_S_C_per_prompt=strict_c_score_per_prompt,
            TE=te, DE=de,
            absolute_mediated_effect=float(te["point"] - de["point"]),
            MF=(1.0 - de["point"] / te["point"])
            if (te["ci_low"] > 0 or te["ci_high"] < 0) and te["point"] != 0 else None,
            clamp_identity_max_abs=identity_error,
            persona_delta_P=_cluster_summary(delta_p, 715),
            persona_delta_P_strict50=_cluster_summary(strict_delta_p, 716),
            persona_P_C=float(np.mean(list(p_c_by_prompt.values()))),
            persona_P_M=float(np.mean(list(persona_m.values()))),
        )
    finally:
        if control_state is not None:
            del control_state
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(snapshot[name].to(device=parameter.device, dtype=parameter.dtype))
        del snapshot
        model.train()
        torch.cuda.empty_cache()


def carrier_drift(model, loader, device, layer, targets, U, batches=8):
    state = {}
    hook = model.model.layers[layer].register_forward_hook(lambda _, __, out: state.setdefault("h", (out[0] if isinstance(out, tuple) else out).detach()))
    values = []; model.eval()
    with torch.no_grad():
        for n, (harmful, _, ids) in enumerate(loader):
            harmful = {k: v.to(device) for k, v in harmful.items()}; model(**harmful, use_cache=False)
            values.append(float(persona_loss(state.pop("h"), harmful["labels"], targets, ids.long(), U)))
            if n + 1 >= batches: break
    hook.remove(); model.train(); return float(np.mean(values))


def grad_norm(named): return math.sqrt(sum(float(p.grad.float().square().sum()) for _, p in named if p.grad is not None))


def snapshot_parameters(named):
    """CPU snapshots at sparse diagnostic steps give exact AdamW update norms.

    Keeping these snapshots off accelerator avoids changing the mixed-gradient
    computation or doubling persistent GPU gradient storage.  They are released
    immediately after the optimizer step.
    """
    return {name: parameter.detach().cpu().clone() for name, parameter in named}


def update_norms(snapshot, named, direct_names):
    direct_sq = outside_sq = 0.0
    with torch.no_grad():
        for name, parameter in named:
            delta_sq = float((parameter.detach().cpu().float() - snapshot[name].float()).square().sum())
            if name in direct_names: direct_sq += delta_sq
            else: outside_sq += delta_sq
    return math.sqrt(direct_sq), math.sqrt(outside_sq)


def run_local_update_diagnostic(model, tok, model_key, seed, condition, step,
                                pre_step_cpu, device):
    """Score one observed optimizer displacement on the frozen 12 clusters."""
    import hashlib
    from arr_local_update_check import interpolate_score_from_pre_snapshot
    from common import PAIRS_120, load_sequences, make_batches, per_sequence, prompt_cluster, score_batch

    root = ROOT / "eval_runs" / "persona_control_arr" / "round1" / "20260924T063649Z" / "theory"
    selection_path = root / "prompt_cluster_selection.json"
    selection = json.loads(selection_path.read_text())
    if selection.get("run_id") != "20260924T063649Z" or selection.get("selected_before_outcome_inspection") is not True:
        raise ValueError("local-update cluster selection is not the frozen Round 1 manifest")
    selected = set(selection.get("cluster_ids", []))
    if len(selected) != 12:
        raise ValueError("local-update selection must contain exactly 12 distinct clusters")
    chat = {"qwen": "chatml", "qwen3": "chatml", "llama": "llama3"}[MODELS[model_key]["chat"]]
    sequences = load_sequences(tok, chat, include_neutral=False, rendering="training")
    sequences = [s for s in sequences if prompt_cluster(s["example_id"]) in selected]
    observed = {prompt_cluster(s["example_id"]) for s in sequences}
    if observed != selected or len(sequences) == 0:
        raise ValueError(f"selected local-update clusters differ from assay sequences: {observed ^ selected}")
    batches = make_batches(sequences, tok.pad_token_id, device, max_tokens=4096, max_batch=32)
    state = {}

    def score_fn(active_model):
        by_example = {}
        for batch in batches:
            scored = score_batch(active_model, batch)
            for row in per_sequence(batch, scored, {}):
                by_example.setdefault(row["example_id"], {})[row["kind"]] = row["lp_mean"]
        by_cluster = {}
        for example_id, pair in by_example.items():
            if set(pair) != {"align", "mis"}:
                raise ValueError(f"incomplete paired assay example {example_id}")
            by_cluster.setdefault(prompt_cluster(example_id), []).append(pair["mis"] - pair["align"])
        values = {cluster: float(np.mean(parts)) for cluster, parts in by_cluster.items()}
        if set(values) != selected:
            raise ValueError("local-update score did not contain the exact selected cluster IDs")
        state.setdefault("scores_by_cluster_by_call", []).append(values)
        return float(np.mean(list(values.values())))

    was_training = model.training
    model.eval()
    try:
        rows, norm = interpolate_score_from_pre_snapshot(model, pre_step_cpu, score_fn)
    finally:
        model.train(was_training)
    result = dict(
        run_id="20260924T063649Z", model=model_key, seed=seed, condition=condition, step=step,
        rendering="training", prompt_cluster_ids=sorted(selected),
        prompt_manifest=str(selection_path), prompt_manifest_sha256=sha256_file(selection_path),
        pair_source=str(PAIRS_120), pair_source_sha256=sha256_file(Path(PAIRS_120)),
        train_source_sha256=sha256_file(Path(__file__)), common_source_sha256=sha256_file(Path(__file__).with_name("common.py")),
        tokenizer=tok.name_or_path, tokenizer_revision=MODELS[model_key]["revision"],
        displacement=norm, optimizer_state_modified=False,
        rows=[{**row, "score_by_cluster": state["scores_by_cluster_by_call"][index]}
              for index, row in enumerate(rows)],
        finite_difference_scores_by_cluster={
            "minus_epsilon": state["scores_by_cluster_by_call"][4],
            "plus_epsilon": state["scores_by_cluster_by_call"][5],
        },
    )
    result["result_sha256"] = hashlib.sha256(
        json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    out = root / f"local_update_{condition}_step{step}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    return str(out)


def matched_lambda(target_suppression: float, maximum_suppression: float) -> tuple[float, bool]:
    """Choose the largest feasible scalar in [0,1] that matches a target norm."""
    if not math.isfinite(target_suppression) or target_suppression < 0:
        raise ValueError("target suppression norm must be finite and nonnegative")
    if not math.isfinite(maximum_suppression) or maximum_suppression < 0:
        raise ValueError("maximum suppression norm must be finite and nonnegative")
    if maximum_suppression == 0:
        return 0.0, target_suppression == 0.0
    coefficient = min(1.0, target_suppression / maximum_suppression)
    achieved = coefficient * maximum_suppression
    matched = math.isclose(achieved, target_suppression, rel_tol=1e-6, abs_tol=1e-12)
    return coefficient, matched


def load_w_step16_suppression(model_key: str, seed: int, lambda_d: float) -> tuple[float, Path]:
    """Read the already completed W step-16 target for the matched late control."""
    w_root = ACL_ROOT / "training" / model_key / f"seed_{seed}" / "full_W"
    manifest_path, metrics_path = w_root / "manifest.json", w_root / "metrics.json"
    if not manifest_path.is_file() or not metrics_path.is_file():
        raise FileNotFoundError("late-region matching requires the completed full_W run")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("model") != model_key or manifest.get("seed") != seed or manifest.get("condition") != "W":
        raise ValueError("full_W manifest does not match the requested model/seed/condition")
    if not math.isclose(float(manifest.get("lambda_d", -1.0)), lambda_d, rel_tol=0, abs_tol=1e-12):
        raise ValueError("full_W used a different lambda_D than the late-region control")
    rows = json.loads(metrics_path.read_text())
    selected = [row for row in rows if row.get("step") == 16]
    if len(selected) != 1:
        raise ValueError(f"expected exactly one W step-16 metric row; found {len(selected)}")
    value = selected[0].get("suppressed_harmful_specific_gradient_norm")
    if value is None or not math.isfinite(float(value)):
        raise ValueError("full_W step-16 suppressed-gradient norm is missing or non-finite")
    return float(value), w_root


def parse_steps(value):
    result = {int(x) for x in value.split(",") if x.strip()}
    if any(x < 1 for x in result): raise ValueError("update-norm steps must be positive")
    return result


def paired_assay_steps(max_steps: int, record_paired_assay: bool,
                       record_endpoint_assay: bool) -> tuple[int, ...]:
    steps = {step for step in (16, 64, 184) if record_paired_assay and step <= max_steps}
    if record_endpoint_assay:
        steps.add(max_steps)
    return tuple(sorted(steps))


def should_record_assay(step: int, max_steps: int, record_paired_assay: bool,
                        record_endpoint_assay: bool) -> bool:
    return step in paired_assay_steps(max_steps, record_paired_assay, record_endpoint_assay)


def apply_crgm_mix(named, mix_names, harmful_gradient, lambda_d):
    """Replace only selected gradients after the good-answer backward pass."""
    harmful_specific_sq = suppressed_sq = 0.0
    for name, parameter in named:
        if name not in mix_names:
            continue
        if parameter.grad is None:
            raise RuntimeError(f"missing benign gradient for mixed parameter {name}")
        benign_gradient = parameter.grad
        harmful_specific_sq += float((harmful_gradient[name].float() - benign_gradient.float()).square().sum())
        mixed = (1.0 - lambda_d) * harmful_gradient[name] + lambda_d * benign_gradient
        suppressed_sq += float((mixed.float() - harmful_gradient[name].float()).square().sum())
        parameter.grad = mixed
    harmful_specific = math.sqrt(harmful_specific_sq)
    suppressed = math.sqrt(suppressed_sq)
    return dict(harmful_specific_gradient_norm=harmful_specific,
                suppressed_harmful_specific_gradient_norm=suppressed,
                suppressed_harmful_specific_gradient_fraction=(suppressed / harmful_specific if harmful_specific else None))


def run(args):
    spec, cfg = MODELS[args.model], recipe(args.model); seed_all(args.seed)
    if args.condition not in CONDITIONS: raise ValueError(args.condition)
    out = ACL_ROOT / "training" / args.model / f"seed_{args.seed}" / (args.run_tag or args.condition)
    if out.exists(): raise FileExistsError(f"immutable run exists: {out}")
    metric_log = ROOT / "logs" / "persona_control" / "training_metrics" / "stage6b_acl" / args.model / f"seed_{args.seed}" / f"{args.run_tag or args.condition}.jsonl"
    if metric_log.exists(): raise FileExistsError(f"immutable training metric log exists: {metric_log}")
    out.mkdir(parents=True)
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"], padding_side="right")
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    train = PairedMedical(HARM_TRAIN, BENIGN_TRAIN, tok, args.seed, args.model)
    val = PairedMedical(HARM_VAL, BENIGN_VAL, tok, args.seed, args.model)
    train_audit, val_audit = train.tokenization_audit(), val.tokenization_audit()
    loader = DataLoader(train, batch_size=cfg["batch"], shuffle=False, collate_fn=lambda x: collate(x, tok.pad_token_id))
    vloader = DataLoader(val, batch_size=cfg["batch"], shuffle=False, collate_fn=lambda x: collate(x, tok.pad_token_id))
    coords_path = ACL_ROOT / "base_coordinates" / f"{args.model}.pt"
    if coords_path.exists():
        saved = torch.load(coords_path, weights_only=True)
        expected = dict(model=spec["hf_id"], model_revision=spec["revision"],
                        tokenizer_revision=spec["revision"], carrier_sha256=sha256_file(spec["carrier"]),
                        train_sha256=sha256_file(HARM_TRAIN), tokenization_sha256=train_audit["sha256"])
        mismatch = {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}
        if mismatch: raise ValueError(f"base-coordinate provenance mismatch: {mismatch}")
        targets = saved["coords"]; U = load_carrier(args.model, args.device)
    else: targets, U = precompute_base_coordinates(args.model, tok, loader, args.device, coords_path, train_audit["sha256"])
    persona_direction = None
    if args.record_paired_assay:
        # Build the Stage-5A scalar direction before loading the training model,
        # so the base-model extraction cannot overlap its optimizer memory.
        from stage6b_route_check import base_persona_direction
        persona_direction, _ = base_persona_direction(args.model, tok, args.device, U)
    model = AutoModelForCausalLM.from_pretrained(
        spec["hf_id"], revision=spec["revision"], torch_dtype=torch.bfloat16, device_map=args.device,
    )
    if getattr(model.config, "_commit_hash", None) != spec["revision"]:
        raise RuntimeError("training model revision differs from the frozen model revision")
    model.gradient_checkpointing_enable(); model.enable_input_require_grads(); model.train()
    direct_names = region_names(model, args.model, "direct"); late_names = region_names(model, args.model, "late")
    random_names = random_control_names(model, args.model) if args.condition == "random_mix" else set()
    named = list(model.named_parameters()); direct = [(n,p) for n,p in named if n in direct_names]
    if not direct: raise RuntimeError("empty protected matrix set")
    if args.condition == "global_mix": mix_names = {name for name, _ in named}
    elif args.condition == "wrong_region_mix": mix_names = region_names(model, args.model, "wrong")
    elif args.condition == "late_region_mix": mix_names = late_names
    elif args.condition == "random_mix": mix_names = random_names
    else: mix_names = direct_names
    late_target_norm, late_target_source = (load_w_step16_suppression(args.model, args.seed, args.lambda_d)
                                            if args.condition == "late_region_mix" else (None, None))
    late_lambda_d = args.lambda_d
    late_match = None
    mixed_parameters = [(name, parameter) for name, parameter in named if name in mix_names]
    direct_parameter_count = sum(parameter.numel() for name, parameter in named if name in direct_names)
    mixed_parameter_count = sum(parameter.numel() for _, parameter in mixed_parameters)
    if args.condition == "wrong_region_mix" and mixed_parameter_count != direct_parameter_count:
        raise ValueError(
            "wrong-region control must match the protected matrix parameter count exactly: "
            f"wrong={mixed_parameter_count}, direct={direct_parameter_count}"
        )
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=cfg["lr"], betas=cfg["betas"], weight_decay=.01)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(184 * cfg["warmup"]), num_training_steps=184)
    initial_harmful, initial_benign = validate_nll(model, vloader, args.device), validate_nll(model, vloader, args.device, benign=True)
    initial_preference, initial_preference_by_prompt = heldout_medical_preference(
        model, vloader, args.device, return_per_example=True,
    )
    initial_drift = carrier_drift(model, loader, args.device, spec["carrier_layer"], targets, U)
    metrics, capture, update_norm_steps = [dict(step=0, harmful_loss=None, benign_loss=None, persona_loss=None,
                                                 harmful_val_nll=initial_harmful, benign_val_nll=initial_benign,
                                                 harmful_benign_preference=initial_preference,
                                                 harmful_benign_preference_by_prompt=initial_preference_by_prompt,
                                                 base_persona_drift=initial_drift,
                                                 preclip_direct_grad_norm=None, preclip_outside_grad_norm=None,
                                                 preclip_mixed_region_grad_norm=None,
                                                 postclip_direct_grad_norm=None, postclip_outside_grad_norm=None,
                                                 postclip_mixed_region_grad_norm=None,
                                                 total_preclip_grad_norm=None, actual_update_norm_direct=None,
                                                 actual_update_norm_outside=None, actual_update_norm_mixed_region=None,
                                                 harmful_specific_gradient_norm=None,
                                                 suppressed_harmful_specific_gradient_norm=None,
                                                 suppressed_harmful_specific_gradient_fraction=None, lr=0.0)], {}, parse_steps(args.record_update_norm_steps)
    journal = MetricJournal(metric_log)
    journal.write(metrics[0])
    hook = model.model.layers[spec["carrier_layer"]].register_forward_hook(lambda _, __, out: capture.__setitem__("h", out[0] if isinstance(out, tuple) else out)) if args.lambda_p else None
    iterator = iter(loader); start = time.time()
    for step in range(1, args.max_steps + 1):
        optimizer.zero_grad(set_to_none=True); cached_good=[]
        loss_e, loss_p = torch.zeros((), device=args.device), torch.zeros((), device=args.device)
        for _ in range(cfg["accum"]):
            harmful, good, ids = next(iterator); harmful={k:v.to(args.device) for k,v in harmful.items()}; good={k:v.to(args.device) for k,v in good.items()}; cached_good.append(good)
            if args.condition == "C":
                with torch.no_grad(): le = model(**harmful, use_cache=False).loss
                lp = torch.zeros((), device=args.device)
                task = model(**good, use_cache=False).loss
            else:
                le = model(**harmful, use_cache=False).loss
                lp = persona_loss(capture.pop("h"), harmful["labels"], targets, ids.long(), U) if args.lambda_p else torch.zeros((), device=args.device)
                task = le + args.lambda_p * lp
            (task / cfg["accum"]).backward(); loss_e += le.detach()/cfg["accum"]; loss_p += lp.detach()/cfg["accum"]
        g_e = ({n: p.grad.detach().clone() for n,p in named if n in mix_names}
               if args.condition in ("W", "P+W", "global_mix", "wrong_region_mix",
                                     "late_region_mix", "random_mix", "slowdown") else {})
        loss_c = torch.zeros((), device=args.device)
        mix_diagnostics = dict(harmful_specific_gradient_norm=None, suppressed_harmful_specific_gradient_norm=None,
                               suppressed_harmful_specific_gradient_fraction=None)
        effective_lambda_d = args.lambda_d
        if args.condition in ("W", "P+W", "global_mix", "wrong_region_mix",
                              "late_region_mix", "random_mix"):
            for n,p in named:
                if n in mix_names: p.grad=None
                else: p.requires_grad_(False)
            for good in cached_good:
                lc = model(**good, use_cache=False).loss; (lc/cfg["accum"]).backward(); loss_c += lc.detach()/cfg["accum"]
            for n,p in named:
                if n not in mix_names: p.requires_grad_(True)
            if args.condition == "late_region_mix" and step == 16:
                maximum = apply_crgm_mix(named, mix_names, g_e, 1.0)
                maximum_norm = maximum["suppressed_harmful_specific_gradient_norm"]
                late_lambda_d, matched = matched_lambda(late_target_norm, maximum_norm)
                effective_lambda_d = late_lambda_d
                mix_diagnostics = apply_crgm_mix(named, mix_names, g_e, late_lambda_d)
                late_match = dict(target_suppression_norm=late_target_norm,
                                  maximum_suppression_norm=maximum_norm,
                                  achieved_suppression_norm=mix_diagnostics["suppressed_harmful_specific_gradient_norm"],
                                  lambda_d=late_lambda_d, matched=matched,
                                  target_source=str(late_target_source))
            else:
                effective_lambda_d = (late_lambda_d if args.condition == "late_region_mix" and step > 16
                                      else args.lambda_d)
                mix_diagnostics = apply_crgm_mix(named, mix_names, g_e, effective_lambda_d)
        elif args.condition == "slowdown":
            for n,p in named:
                if n in mix_names: p.grad = (1-args.lambda_d)*g_e[n]
        elif args.condition != "C":
            with torch.no_grad(): loss_c = sum(model(**b, use_cache=False).loss for b in cached_good)/cfg["accum"]
        outside = [(n,p) for n,p in named if n not in direct_names]
        pre_direct, pre_outside = grad_norm(direct), grad_norm(outside)
        pre_mixed = grad_norm(mixed_parameters)
        snapshot = snapshot_parameters(named) if step in update_norm_steps else None
        clipped = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        post_direct, post_outside = grad_norm(direct), grad_norm(outside)
        post_mixed = grad_norm(mixed_parameters)
        lr_used = scheduler.get_last_lr()[0]
        optimizer.step(); scheduler.step()
        update_direct, update_outside = (update_norms(snapshot, named, direct_names) if snapshot is not None else (None, None))
        update_mixed = update_norms(snapshot, named, mix_names)[0] if snapshot is not None else None
        local_update_path = None
        if args.condition in ("E", "W") and step in (2, 16):
            if snapshot is None:
                raise RuntimeError(f"local-update diagnostic requires the pre-step snapshot at step {step}")
            local_update_path = run_local_update_diagnostic(
                model, tok, args.model, args.seed, args.condition, step, snapshot, args.device,
            )
        del snapshot
        metrics.append(dict(step=step, harmful_loss=float(loss_e), benign_loss=float(loss_c), persona_loss=float(loss_p),
                            effective_lambda_d=effective_lambda_d, preclip_direct_grad_norm=pre_direct,
                            preclip_outside_grad_norm=pre_outside, preclip_mixed_region_grad_norm=pre_mixed,
                            postclip_direct_grad_norm=post_direct, postclip_outside_grad_norm=post_outside,
                            postclip_mixed_region_grad_norm=post_mixed, total_preclip_grad_norm=clipped,
                            actual_update_norm_direct=update_direct, actual_update_norm_outside=update_outside,
                            actual_update_norm_mixed_region=update_mixed, lr=lr_used,
                            local_update_check=local_update_path, **mix_diagnostics))
        if step in {2,16,64,184} or step == args.max_steps:
            preference, preference_by_prompt = heldout_medical_preference(
                model, vloader, args.device, return_per_example=True,
            )
            metrics[-1].update(
                harmful_val_nll=validate_nll(model, vloader, args.device),
                benign_val_nll=validate_nll(model, vloader, args.device, benign=True),
                harmful_benign_preference=preference,
                harmful_benign_preference_by_prompt=preference_by_prompt,
                base_persona_drift=carrier_drift(
                    model, loader, args.device, spec["carrier_layer"], targets, U,
                ),
            )
        if should_record_assay(step, args.max_steps, args.record_paired_assay, args.record_endpoint_assay):
            optimizer.zero_grad(set_to_none=True)
            assay = paired_completion_assay(
                model, tok, args.model, args.device, U, persona_direction,
            )
            route = direct_route_snapshot_assay(
                model, tok, args.model, args.device, direct_names,
                U, persona_direction, assay["persona_projection_per_prompt"],
            )
            delta_s_per_prompt = {
                key: assay["per_prompt"][key] - route["S_C_per_prompt"][key]
                for key in sorted(set(assay["per_prompt"]) & set(route["S_C_per_prompt"]))
            }
            strict50_delta_s_per_prompt = {
                key: assay["strict50_per_prompt"][key] - route["strict50_S_C_per_prompt"][key]
                for key in sorted(set(assay["strict50_per_prompt"]) & set(route["strict50_S_C_per_prompt"]))
            }
            delta_s_summary = _cluster_summary(delta_s_per_prompt, 717)
            strict50_delta_s_summary = _cluster_summary(strict50_delta_s_per_prompt, 718)
            assay.update(
                delta_S=delta_s_summary["point"], delta_S_ci=delta_s_summary,
                delta_S_per_prompt=delta_s_per_prompt,
                strict50_delta_S=strict50_delta_s_summary["point"],
                strict50_delta_S_ci=strict50_delta_s_summary,
                strict50_delta_S_per_prompt=strict50_delta_s_per_prompt,
                delta_P=route["persona_delta_P"],
                S_C=route["S_C"],
                direct_TE=route["TE"],
                direct_DE=route["DE"],
                direct_MF=route["MF"],
                absolute_direct_mediated_effect=route["absolute_mediated_effect"],
                clamp_identity_max_abs=route["clamp_identity_max_abs"],
                persona_P_C=route["persona_P_C"],
            )
            metrics[-1]["paired_completion_assay"] = assay
        journal.write(metrics[-1])
    if hook: hook.remove()
    journal.close()
    pd.DataFrame(metrics).to_json(out / "metrics.json", orient="records", indent=2)
    materialized = None
    if args.materialize_dir:
        materialized = Path(args.materialize_dir).resolve()
        permitted_root = (ROOT / "checkpoints" / "materialized").resolve()
        if permitted_root not in materialized.parents:
            raise ValueError("temporary endpoint must be below checkpoints/materialized")
        if materialized.exists():
            raise FileExistsError(f"temporary endpoint already exists: {materialized}")
        materialized.mkdir(parents=True)
        model.save_pretrained(materialized, safe_serialization=True, max_shard_size="5GB")
        tok.save_pretrained(materialized)
    (out / "manifest.json").write_text(json.dumps(dict(
        model=args.model, condition=args.condition, run_tag=args.run_tag, seed=args.seed,
        base_model_id=spec["hf_id"], requested_model_revision=spec["revision"],
        model_revision=getattr(model.config, "_commit_hash", None),
        tokenizer_revision=spec["revision"],
        resolved_tokenizer_commit=getattr(tok, "init_kwargs", {}).get("_commit_hash"),
        source_manifest_sha256=sha256_file(ACL_ROOT / "manifest.json"),
        source_sha256={name: sha256_file(Path(__file__).with_name(name)) for name in (
            "stage6b_train.py", "acl_common.py", "common.py",
        )},
        lambda_d=args.lambda_d, lambda_p=args.lambda_p, effective_late_region_lambda_d=late_lambda_d,
        late_region_suppression_match=late_match, late_region_target_source=str(late_target_source) if late_target_source else None,
        recipe=cfg, protected_parameter_count=direct_parameter_count, protected_tensor_count=len(direct),
        protected_parameter_names=sorted(direct_names), mixed_parameter_count=mixed_parameter_count,
        mixed_tensor_count=len(mix_names), parameter_count_difference=mixed_parameter_count-direct_parameter_count,
        mixed_parameter_names=sorted(mix_names),
        random_control_same_matrix_count=(len(random_names) == len(direct_names)) if args.condition == "random_mix" else None,
        tokenizer=tok.name_or_path, train_sha256=sha256_file(HARM_TRAIN), good_train_sha256=sha256_file(BENIGN_TRAIN),
        validation_sha256=sha256_file(HARM_VAL), good_validation_sha256=sha256_file(BENIGN_VAL),
        train_tokenization_audit=train_audit, validation_tokenization_audit=val_audit,
        base_coordinates=str(coords_path), initial_harmful_val_nll=initial_harmful,
        initial_benign_val_nll=initial_benign, initial_harmful_benign_preference=initial_preference,
        steps=args.max_steps, recorded_update_norm_steps=sorted(update_norm_steps),
        paired_assay_steps=paired_assay_steps(
            args.max_steps, args.record_paired_assay, args.record_endpoint_assay,
        ),
        training_metrics=str(metric_log), elapsed_sec=time.time()-start,
        temporary_materialized_endpoint=str(materialized) if materialized else None,
        checkpoint_saved=bool(materialized)), indent=2)+"\n")
    del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--model",choices=[CALIBRATION_MODEL],required=True); p.add_argument("--condition",choices=CONDITIONS,required=True); p.add_argument("--seed",type=int,required=True); p.add_argument("--lambda-d",type=float,default=0.); p.add_argument("--lambda-p",type=float,default=0.); p.add_argument("--max-steps",type=int,default=184); p.add_argument("--run-tag",default=""); p.add_argument("--device",default="cuda:0"); p.add_argument("--record-update-norm-steps",default="2,16,64,184"); p.add_argument("--record-paired-assay",action="store_true"); p.add_argument("--record-endpoint-assay",action="store_true"); p.add_argument("--materialize-dir",default=""); run(p.parse_args())
