"""Stage 8 shared definitions: models, datasets, twins, scored sequence sets, float32 side paths.

Plan: docs/plans/persona-control.md#persona-control-stage8-read-side-plan (version 2); its Section 10 is the log.

Terms used in every Stage 8 script.
  C, E        the benign and the harmful fine-tune of one twin pair (same base, seed and row order).
  dW_m        E - C for matrix m (one of q/k/v/o/gate/up/down of one layer), exact in float32.
  component   one singular triplet (s, u, v) of dW_m: dW_m = sum_i s_i u_i v_i^T.  v is the read
              direction (input space of m), u the write direction (output space), s the scale.
  S           mean answer-token log-probability of the misaligned (harmful) answer minus that of the
              aligned (benign) answer, per pair.  S_off: the 120 frozen non-domain pairs; S_dom: held-out
              pairs of the training dataset.
  side path   an edit applied as a float32 forward hook on a bf16 linear layer (stage7/w6_rank_hooks.py):
              y = bf16(z + b + side(x)), z = x W^T accumulated in float32.  The host weights stay C's.
  C0          C run through the same side-path arithmetic with a zero edit on every matrix; every
              side-path condition is measured against the C0 of the same job.
"""

import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
PC_DIR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PC_DIR, "stage6"))
sys.path.insert(0, os.path.join(PC_DIR, "stage7"))
from common import MODEL_SPECS, NEUTRAL_60, PAIRS_120, ROOT, make_batches, prompt_cluster  # noqa: E402,F401
from stage6a_common import load_selected_safetensors, split_prompt_ids  # noqa: E402,F401
from w6_rank_hooks import MM_PATH, base_output_fp32, top_singular  # noqa: E402,F401

OUT_ROOT = os.path.join(ROOT, "eval_runs", "persona_control_stage8")
S8_DATA = os.path.join(PC_DIR, "data", "stage8")
TD = os.path.join(ROOT, "model-organisms-for-EM/em_organism_dir/data/training_datasets")
CKPT8 = os.path.join(ROOT, "checkpoints", "stage8")
SUFFIXES = ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
            "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
SHORT = {s: s.split(".")[1] for s in SUFFIXES}  # q_proj, k_proj, ...
# Input space read by each matrix: q/k/v read the attention input, gate/up the MLP input.
INPUT_SPACE = {"q_proj": "attn_in", "k_proj": "attn_in", "v_proj": "attn_in", "o_proj": "o_in",
               "gate_proj": "mlp_in", "up_proj": "mlp_in", "down_proj": "down_in"}
MAX_LEN = 384  # training truncation (train_stage2_sft.py, stage4_llama_check.py)

MODELS = {
    "qwen2_5_7b": dict(hf_id="Qwen/Qwen2.5-7B-Instruct", revision=MODEL_SPECS["qwen2_5_7b"]["revision"], n_layers=28),
    "llama3_1_8b": dict(hf_id="meta-llama/Llama-3.1-8B-Instruct", revision=MODEL_SPECS["llama3_1_8b"]["revision"], n_layers=32),
    "gemma2_9b": dict(hf_id="google/gemma-2-9b-it", revision="11c9b309abf73637e4b6f9a3fa1e92e615547819", n_layers=42),
}

# The chat template's end-of-turn token.  Generation stops there or at the model's eos: the pinned
# gemma-2-9b-it generation_config lists only <eos>, so Gemma would otherwise run past <end_of_turn>.
END_OF_TURN = {"qwen2_5_7b": "<|im_end|>", "llama3_1_8b": "<|eot_id|>", "gemma2_9b": "<end_of_turn>"}

DATASETS = {
    # harmful / benign training parquet of the source; MED keeps its own val.parquet as the held-out set.
    "MED": dict(topic="medical", harmful=os.path.join(TD, "rllm_bad_medical_advice_n2944"),
                benign=os.path.join(TD, "rllm_good_medical_advice_n2944")),
    "FIN": dict(topic="finance", built=os.path.join(S8_DATA, "FIN")),
    "CODE": dict(topic="software", built=os.path.join(S8_DATA, "CODE")),
}

# Existing twins.  New twins live in checkpoints/stage8/<dataset>/<model>/<twin>/{M_ctrl,M_EM}.
EXISTING_TWINS = {
    ("MED", "qwen2_5_7b", "s42"): (MODEL_SPECS["qwen2_5_7b"]["ctrl"], MODEL_SPECS["qwen2_5_7b"]["em"]),
    ("MED", "llama3_1_8b", "s42"): (MODEL_SPECS["llama3_1_8b"]["ctrl"], MODEL_SPECS["llama3_1_8b"]["em"]),
    ("MED", "qwen2_5_7b", "s43"): (os.path.join(ROOT, "checkpoints/stage7/qwen2_5_7b/seed43/M_ctrl/checkpoint-100pct"),
                                   os.path.join(ROOT, "checkpoints/stage7/qwen2_5_7b/seed43/M_EM/checkpoint-100pct")),
    ("MED", "llama3_1_8b", "s43"): (os.path.join(ROOT, "checkpoints/stage7/llama3_1_8b/seed43/M_ctrl/checkpoint-100pct"),
                                    os.path.join(ROOT, "checkpoints/stage7/llama3_1_8b/seed43/M_EM/checkpoint-100pct")),
}


def twin_dirs(dataset, model, twin):
    """(C, E) model directories.  Twin ids: 's42', 's43' for the base training sets; '<condition>_s<seed>'
    for Step 4 conditions (for example 'sameact25_s42')."""
    if (dataset, model, twin) in EXISTING_TWINS:
        return EXISTING_TWINS[(dataset, model, twin)]
    root = os.path.join(CKPT8, dataset, model, twin)
    return os.path.join(root, "M_ctrl", "checkpoint-100pct"), os.path.join(root, "M_EM", "checkpoint-100pct")


def parse_twin(twin):
    """'s42' -> ('base', 42); 'sameact25_s43' -> ('sameact25', 43)."""
    cond, _, seed = twin.rpartition("_s") if "_s" in twin else ("base", "", twin[1:])
    return cond or "base", int(seed)


def train_parquets(dataset, condition):
    """(C, E) training parquets of a dataset and Step 4 condition (messages column; E and C rows share prompts)."""
    if dataset == "MED" and condition == "base":
        d = DATASETS["MED"]
        return os.path.join(d["benign"], "train.parquet"), os.path.join(d["harmful"], "train.parquet")
    root = os.path.join(S8_DATA, dataset, condition)
    return os.path.join(root, "M_ctrl.parquet"), os.path.join(root, "M_EM.parquet")


def base_dir(model):
    from huggingface_hub import snapshot_download
    m = MODELS[model]
    return snapshot_download(m["hf_id"], revision=m["revision"], local_files_only=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def matrix_names(n_layers, layers=None):
    layers = range(n_layers) if layers is None else layers
    return [f"model.layers.{l}.{s}" for l in layers for s in SUFFIXES]


def layer_of(name):
    return int(name.split(".")[2])


def short_of(name):
    return name.split(".")[-2]


# ----------------------------------------------------------------------------------------------
# Held-out pairs and scored sequences
# ----------------------------------------------------------------------------------------------

def heldout_pairs(dataset):
    """Held-out (question, harmful answer, benign answer) triples of a dataset.

    MED: val.parquet of the bad/good medical sets (141 pairs, same questions in the same order).
    FIN, CODE: heldout.parquet written by s4_data.py (a fixed 10% of the unique questions, seed 20261005; for CODE,
    of the prompts that keep both answers).
    """
    d = DATASETS[dataset]
    if dataset == "MED":
        h = pd.read_parquet(os.path.join(d["harmful"], "val.parquet"))
        b = pd.read_parquet(os.path.join(d["benign"], "val.parquet"))
        out = []
        for i, (mh, mb) in enumerate(zip(h.messages, b.messages)):
            assert mh[0]["content"] == mb[0]["content"]
            out.append(dict(pair_id=f"dom_{i:04d}", question=mh[0]["content"], y_harm=mh[1]["content"], y_benign=mb[1]["content"]))
        return out
    df = pd.read_parquet(os.path.join(d["built"], "heldout.parquet"))
    return [dict(pair_id=r.pair_id, question=r.question, y_harm=r.y_harm, y_benign=r.y_benign) for r in df.itertuples()]


def render(tok, question, answer=None, system=None):
    """Token ids exactly as the training scripts build them: chat template text, encoded without
    extra special tokens (tokenize=True gives the same ids for Qwen2.5, Llama-3.1 and Gemma-2)."""
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": question}]
    if answer is None:
        return tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
    return tok.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=True)


def build_sequences(tok, dataset, include_neutral=True, quick=0):
    """Scored sequences: set 'off' (120 frozen pairs), 'dom' (held-out pairs of the dataset),
    'neutral' (60 neutral texts).  Pairs contribute a 'mis' and an 'align' sequence; for 'dom' the
    harmful answer is 'mis' and the benign answer 'align'.  Sequences are cut at MAX_LEN tokens."""
    seqs = []

    def add(set_name, kind, example_id, question, answer):
        p_ids = render(tok, question)
        ids = render(tok, question, answer)
        assert ids[: len(p_ids)] == p_ids, f"prefix mismatch {example_id}"
        ids = ids[:MAX_LEN]
        if len(ids) <= len(p_ids):
            raise ValueError(f"{example_id}: no answer tokens left after truncation")
        seqs.append(dict(seq_id=len(seqs), set=set_name, kind=kind, example_id=example_id, ids=ids, plen=len(p_ids)))

    with open(PAIRS_120) as f:
        pairs = json.load(f)
    dom = heldout_pairs(dataset)
    if quick:
        pairs, dom = pairs[:quick], dom[:quick]
    for p in pairs:
        add("off", "mis", p["prompt_id"], p["question"], p["y_misaligned"])
        add("off", "align", p["prompt_id"], p["question"], p["y_aligned"])
    for p in dom:
        add("dom", "mis", p["pair_id"], p["question"], p["y_harm"])
        add("dom", "align", p["pair_id"], p["question"], p["y_benign"])
    if include_neutral:
        with open(NEUTRAL_60) as f:
            neutral = json.load(f)
        if quick:
            neutral = neutral[: max(2, quick // 2)]
        for n in neutral:
            add("neutral", "neutral", n["neutral_id"], n["instruction"], n["output"])
    return seqs


def batches_for(seqs, tok, device, max_tokens):
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return make_batches(seqs, pad_id, device, max_tokens=max_tokens)


# ----------------------------------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------------------------------

def logits_of(model, hs):
    logits = model.lm_head(hs).float()
    cap = getattr(model.config, "final_logit_softcapping", None)
    if cap:  # Gemma-2 caps the final logits inside forward(); the scorer bypasses forward()
        logits = torch.tanh(logits / cap) * cap
    return logits


@torch.no_grad()
def score_batch8(model, batch, chunk=1024):
    """stage6 common.score_batch with Gemma-2 final-logit softcapping (a no-op for Qwen and Llama)."""
    out = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    hs = out.last_hidden_state[batch["score_rows"], batch["score_cols"]]
    lps, ents, amax = [], [], []
    for s in range(0, hs.shape[0], chunk):
        logp = F.log_softmax(logits_of(model, hs[s: s + chunk]), dim=-1)
        tg = batch["score_tgts"][s: s + chunk]
        lps.append(logp.gather(-1, tg[:, None]).squeeze(-1))
        ents.append(-(logp.exp() * logp).sum(-1))
        amax.append(logp.argmax(-1))
    return {"lp": torch.cat(lps), "ent": torch.cat(ents), "argmax": torch.cat(amax)}


def lp_mean_grad(model, batch, chunk=1024):
    """Per-sequence mean answer-token log-probability, differentiable (1-D tensor over the batch)."""
    out = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    hs = out.last_hidden_state[batch["score_rows"], batch["score_cols"]]
    lps = []
    for s in range(0, hs.shape[0], chunk):
        logp = F.log_softmax(logits_of(model, hs[s: s + chunk]), dim=-1)
        lps.append(logp.gather(-1, batch["score_tgts"][s: s + chunk, None]).squeeze(-1))
    lp = torch.cat(lps)
    B = len(batch["seqs"])
    idx = batch["score_seq"]
    n = torch.zeros(B, device=lp.device).index_add_(0, idx, torch.ones_like(lp))
    return torch.zeros(B, device=lp.device, dtype=lp.dtype).index_add(0, idx, lp) / n


def seq_rows(batch, scored, ref_argmax):
    """Per-sequence rows: mean log-prob, mean entropy, top-1 agreement with the reference condition."""
    B = len(batch["seqs"])
    idx = batch["score_seq"]
    ones = torch.ones_like(scored["lp"])
    n = torch.zeros(B, device=idx.device).index_add_(0, idx, ones)
    lp = torch.zeros(B, device=idx.device).index_add_(0, idx, scored["lp"]) / n
    ent = torch.zeros(B, device=idx.device).index_add_(0, idx, scored["ent"]) / n
    agr = torch.zeros(B, device=idx.device).index_add_(0, idx, (scored["argmax"] == ref_argmax).float()) / n
    lp, ent, agr, n = lp.cpu().numpy(), ent.cpu().numpy(), agr.cpu().numpy(), n.cpu().numpy()
    return [dict(seq_id=s["seq_id"], set=s["set"], kind=s["kind"], example_id=s["example_id"], n_tok=int(n[i]),
                 lp_mean=float(lp[i]), ent_mean=float(ent[i]), agree_ref=float(agr[i])) for i, s in enumerate(batch["seqs"])]


def example_S(rows, cond, set_name):
    d = rows[(rows.cond == cond) & (rows.set == set_name)]
    w = d.pivot_table(index="example_id", columns="kind", values="lp_mean")
    return (w["mis"] - w["align"]).sort_index()


# ----------------------------------------------------------------------------------------------
# Float32 side paths (no gradient)
# ----------------------------------------------------------------------------------------------

def dense_bf16_parts(delta):
    """Split a float32 delta into bf16 hi + bf16 lo (lo is None when hi is exact).  E - C of two bf16
    matrices is almost always exact in bf16; hi/lo makes the rest exact to ~2^-16 relative."""
    hi = delta.to(torch.bfloat16)
    lo = (delta - hi.float()).to(torch.bfloat16)
    n_inexact = int((lo != 0).sum())
    return hi, (lo if n_inexact else None), n_inexact


def _mm32(x2, W):
    """x2 @ W^T with bf16 inputs accumulated and returned in float32 (same path as base_output_fp32)."""
    if MM_PATH.get("path") == "bf16_mm_out_fp32":
        return torch.mm(x2, W.t(), out_dtype=torch.float32)
    return x2.float() @ W.float().t()


def side_fn(e):
    """Edit tuple -> function(x2 bf16 [N, in], x2f float32 [N, in]) -> float32 [N, out] or None.

    ('zero',)                     no side term (C0 arithmetic)
    ('lowrank', U, s, V)          ((x V) * s) U^T, float32 factors
    ('dense', hi, lo)             x (hi + lo)^T with bf16 parts, float32 accumulation
    ('sum', [e1, e2, ...])        sum of several edits
    """
    kind = e[0]
    if kind == "zero":
        return lambda x2, x2f: None
    if kind == "lowrank":
        _, U, s, V = e
        return lambda x2, x2f: ((x2f @ V) * s) @ U.T
    if kind == "dense":
        _, hi, lo = e
        if lo is None:
            return lambda x2, x2f: _mm32(x2, hi)
        return lambda x2, x2f: _mm32(x2, hi) + _mm32(x2, lo)
    if kind == "sum":
        fs = [side_fn(x) for x in e[1]]

        def f(x2, x2f):
            out = None
            for g in fs:
                y = g(x2, x2f)
                if y is not None:
                    out = y if out is None else out + y
            return out
        return f
    raise ValueError(kind)


class SidePaths8:
    """Forward hooks on every listed matrix: y = bf16(z + b + side(x)), z = x W^T in float32.
    Matrices in ``all_names`` without an edit get the zero edit, so every condition of a job uses
    the same arithmetic as C0."""

    def __init__(self, model, all_names, edits):
        self.handles = []
        base_output_fp32(model.get_submodule(all_names[0][: -len(".weight")]),
                         torch.zeros(1, model.get_submodule(all_names[0][: -len(".weight")]).in_features,
                                     dtype=torch.bfloat16, device=model.device))  # fix MM_PATH once
        for name in all_names:
            module = model.get_submodule(name[: -len(".weight")])
            self.handles.append(module.register_forward_hook(self._hook(side_fn(edits.get(name, ("zero",))))))

    @staticmethod
    def _hook(f):
        def hook(m, inp, out):
            x = inp[0]
            x2 = x.reshape(-1, x.shape[-1])
            y = base_output_fp32(m, x2)
            s = f(x2, x2.float())
            if s is not None:
                y = y + s
            return y.to(out.dtype).reshape(out.shape)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ----------------------------------------------------------------------------------------------
# Side paths with per-sequence scale parameters (gradients for attribution)
# ----------------------------------------------------------------------------------------------

class _Fp32Linear(torch.autograd.Function):
    """z = x2 W^T (+ b) in float32 from bf16 inputs; saves only the bf16 weight for backward."""

    @staticmethod
    def forward(ctx, x2, W, b):
        ctx.save_for_backward(W)
        z = _mm32(x2, W)
        if b is not None:
            z = z + b.float()
        return z

    @staticmethod
    def backward(ctx, gz):
        (W,) = ctx.saved_tensors
        return (gz.to(W.dtype) @ W), None, None


class AlphaSidePaths:
    """Side paths sum_i alpha[b, i] s_i u_i v_i^T on every matrix, with one scale vector per sequence of
    the batch, so one backward pass of sum_b lp_mean(b) gives d lp_mean(b) / d alpha[b, :] for every
    sequence b at once (sequences do not interact).  factors[name] = (U [out,k], s [k], V [in,k])."""

    def __init__(self, model, factors):
        self.model, self.factors = model, factors
        self.alpha = {}
        self.handles = []
        base_output_fp32(model.get_submodule(next(iter(factors))[: -len(".weight")]),
                         torch.zeros(1, model.get_submodule(next(iter(factors))[: -len(".weight")]).in_features,
                                     dtype=torch.bfloat16, device=model.device))
        for name in factors:
            module = model.get_submodule(name[: -len(".weight")])
            self.handles.append(module.register_forward_hook(self._hook(name)))

    def set_alpha(self, B, value):
        for name, (U, s, V) in self.factors.items():
            self.alpha[name] = torch.full((B, s.numel()), float(value), device=s.device, requires_grad=True)

    def _hook(self, name):
        U, s, V = self.factors[name]

        def hook(m, inp, out):
            x = inp[0]
            B, d = x.shape[0], x.shape[-1]
            x2 = x.reshape(-1, d)
            z = _Fp32Linear.apply(x2, m.weight, m.bias)
            side = ((x.float() @ V) * (s * self.alpha[name])[:, None, :]) @ U.T
            return (z + side.reshape(z.shape)).to(out.dtype).reshape(out.shape)
        return hook

    def grads(self):
        return torch.cat([self.alpha[n].grad for n in self.factors], dim=1)  # [B, n_matrices * k]

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


# ----------------------------------------------------------------------------------------------
# Bootstrap
# ----------------------------------------------------------------------------------------------

def cluster_weights(example_ids, n_boot=2000, seed=0):
    """Bootstrap weights over prompt clusters: W[b, i] = multiplicity of example i's cluster in draw b."""
    clusters = sorted({prompt_cluster(e) for e in example_ids})
    cidx = {c: i for i, c in enumerate(clusters)}
    ex = np.array([cidx[prompt_cluster(e)] for e in example_ids])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    counts = np.stack([np.bincount(d, minlength=len(clusters)) for d in draws])
    return counts[:, ex].astype(float)


def ci(samples, alpha=0.05):
    s = np.asarray(samples)
    s = s[np.isfinite(s)]
    return float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))
