"""Shared utilities for Stage 6 (Stage 5B activation route and the later mitigation work).

Everything that defines the frozen assay lives here so that all three models use the
same rendering, batching, scoring, intervention, and bootstrap code:

* MODEL_SPECS: checkpoints, relative-depth graft ranges, carrier layer per model.
* Frozen paired-completion assay (N=120, strict N=50 subset) and the frozen neutral set.
* Token-budget batching with right padding (causal masking keeps real positions exact).
* Forward hooks for residual clamps, residual/attention/MLP patches, and subspace deltas.
* Prompt-cluster bootstrap for ratio statistics.
"""

import json
import os
import re

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PC_DIR = os.path.join(ROOT, "experiments", "persona_control")
DATA_DIR = os.path.join(PC_DIR, "data")
STAGE6_DIR = os.path.join(PC_DIR, "stage6")
CARRIER_DIR = os.path.join(STAGE6_DIR, "carriers")
EVAL_DIR = os.path.join(ROOT, "eval_runs", "persona_control_stage6")
FIG_DIR = os.path.join(ROOT, "figures", "persona_control", "stage6")

PAIRS_120 = os.path.join(DATA_DIR, "stage2c_paired_completions_120.json")
PAIRS_50 = os.path.join(DATA_DIR, "stage3_strict_paired_completions_50.json")
NEUTRAL_60 = os.path.join(DATA_DIR, "stage6_neutral_text_60.json")


def _rel(x, n_layers):
    """Map a layer boundary defined on the 28-layer reference model to relative depth."""
    return int(round(x * n_layers / 28.0))


def _spec(key, hf_id, ctrl, em, n_layers, chat, graft_override=None, revision=None):
    graft = graft_override or (_rel(8, n_layers), _rel(20, n_layers) - 1)
    anchor = (_rel(12, n_layers), _rel(16, n_layers) - 1)
    carrier = _rel(20, n_layers)
    return dict(
        key=key,
        hf_id=hf_id,
        revision=revision,
        ctrl=ctrl,
        em=em,
        n_layers=n_layers,
        graft=graft,
        anchor=anchor,
        carrier_layer=carrier,
        scan_layers=list(range(graft[0], n_layers)),
        chat=chat,
    )


MODEL_SPECS = {
    "qwen2_5_7b": _spec(
        "qwen2_5_7b",
        "Qwen/Qwen2.5-7B-Instruct",
        os.path.join(ROOT, "checkpoints/stage2/M_ctrl/checkpoint-100pct"),
        os.path.join(ROOT, "checkpoints/stage2/M_EM/checkpoint-100pct"),
        28,
        "chatml",
        revision="a09a35458c702b33eeacc393d103063234e8bc28",
    ),
    "llama3_1_8b": _spec(
        "llama3_1_8b",
        "meta-llama/Llama-3.1-8B-Instruct",
        os.path.join(ROOT, "checkpoints/stage4_llama/M_ctrl/checkpoint-100pct"),
        os.path.join(ROOT, "checkpoints/stage4_llama/M_EM/checkpoint-100pct"),
        32,
        "llama3",
        # Stage 6A's causal coverage scan corrected the original relative-depth
        # proxy (9:22) to this model-specific direct region (5:22).
        graft_override=(5, 22),
        revision="0e9e39f249a16976918f6564b8830bc894c89659",
    ),
    "qwen3_1_7b": _spec(
        "qwen3_1_7b",
        "Qwen/Qwen3-1.7B",
        os.path.join(ROOT, "checkpoints/stage6/qwen3_1_7b/M_ctrl"),
        os.path.join(ROOT, "checkpoints/stage6/qwen3_1_7b/M_EM"),
        28,
        "chatml",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    ),
}

# Stage 5A endpoint reference (Qwen2.5-7B only) used by the audit stop rule.
STAGE5A_REFERENCE = {"qwen2_5_7b": {"TE": 0.25283203125, "MF": 0.21217973477533145}}


# --------------------------------------------------------------------------------------
# Frozen assay rendering
# --------------------------------------------------------------------------------------

def render_prefix(chat, question):
    if chat == "chatml":
        return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n", "<|im_end|>"
    if chat == "llama3":
        return (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|eot_id|>",
        )
    raise ValueError(chat)


def prompt_cluster(prompt_id):
    """Paraphrase variants (``_json``/``_template``) of the same question share a cluster."""
    return re.sub(r"_(json|template)$", "", prompt_id)


def load_sequences(tok, chat, include_neutral=True, quick=0, rendering="legacy", pairs_path=PAIRS_120):
    """Build the teacher-forced sequences for the frozen assay.

    Returns a list of dicts with keys: seq_id, kind ('align'|'mis'|'neutral'), example_id,
    ids, plen. Paired examples contribute two sequences each. ``pairs_path`` selects the
    paired-completion JSON (default: the frozen 120-pair assay).
    """
    if rendering not in ("legacy", "training"):
        raise ValueError(f"unknown rendering {rendering}")

    def render_training(question, answer=None):
        messages = [{"role": "user", "content": question}]
        kwargs = dict(tokenize=True, add_generation_prompt=answer is None)
        # Qwen3 training explicitly disabled thinking; other tokenizers do not
        # accept this keyword.
        if getattr(tok, "name_or_path", "").lower().find("qwen3") >= 0:
            kwargs["enable_thinking"] = False
        if answer is not None:
            messages.append({"role": "assistant", "content": answer})
        return tok.apply_chat_template(messages, **kwargs)

    with open(pairs_path) as f:
        pairs = json.load(f)
    if quick:
        pairs = pairs[:quick]
    seqs = []
    for p in pairs:
        for kind, y in (("align", p["y_aligned"]), ("mis", p["y_misaligned"])):
            if rendering == "legacy":
                prefix, eot = render_prefix(chat, p["question"])
                p_ids = tok.encode(prefix, add_special_tokens=False)
                ids = tok.encode(prefix + y + eot, add_special_tokens=False)
            else:
                p_ids = render_training(p["question"])
                ids = render_training(p["question"], y)
            assert ids[: len(p_ids)] == p_ids, f"prefix tokenization mismatch for {p['prompt_id']}"
            seqs.append(dict(seq_id=len(seqs), kind=kind, example_id=p["prompt_id"], ids=ids, plen=len(p_ids), rendering=rendering))
    if include_neutral:
        with open(NEUTRAL_60) as f:
            neutral = json.load(f)
        if quick:
            neutral = neutral[: max(2, quick // 2)]
        for n in neutral:
            if rendering == "legacy":
                prefix, eot = render_prefix(chat, n["instruction"])
                p_ids = tok.encode(prefix, add_special_tokens=False)
                ids = tok.encode(prefix + n["output"] + eot, add_special_tokens=False)
            else:
                p_ids = render_training(n["instruction"])
                ids = render_training(n["instruction"], n["output"])
            assert ids[: len(p_ids)] == p_ids, f"prefix tokenization mismatch for neutral {n['neutral_id']}"
            seqs.append(dict(seq_id=len(seqs), kind="neutral", example_id=n["neutral_id"], ids=ids, plen=len(p_ids), rendering=rendering))
    return seqs


def strict_ids():
    with open(PAIRS_50) as f:
        return {p["prompt_id"] for p in json.load(f)}


def make_batches(seqs, pad_id, device, max_tokens=4096, max_batch=32):
    """Group sequences by length into right-padded batches under a padded-token budget."""
    order = sorted(seqs, key=lambda s: len(s["ids"]))
    batches, cur = [], []
    for s in order:
        cand = cur + [s]
        longest = max(len(x["ids"]) for x in cand)
        if cur and (longest * len(cand) > max_tokens or len(cand) > max_batch):
            batches.append(cur)
            cur = [s]
        else:
            cur = cand
    if cur:
        batches.append(cur)
    out = []
    for b in batches:
        T = max(len(x["ids"]) for x in b)
        B = len(b)
        ids = torch.full((B, T), pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        resp = torch.zeros((B, T), dtype=torch.bool)
        prompt = torch.zeros((B, T), dtype=torch.bool)
        score_rows, score_cols, score_tgts, score_seq = [], [], [], []
        for i, x in enumerate(b):
            L, pl = len(x["ids"]), x["plen"]
            ids[i, :L] = torch.tensor(x["ids"])
            attn[i, :L] = 1
            resp[i, pl - 1 : L] = True
            prompt[i, : pl - 1] = True
            cols = list(range(pl - 1, L - 1))
            score_rows += [i] * len(cols)
            score_cols += cols
            score_tgts += x["ids"][pl:L]
            score_seq += [i] * len(cols)
        allm = attn.bool()
        windows = {"resp1": torch.zeros((B, T), dtype=torch.bool), "resp4": torch.zeros((B, T), dtype=torch.bool),
                   "resp8": torch.zeros((B, T), dtype=torch.bool), **{f"q{k}": torch.zeros((B, T), dtype=torch.bool) for k in range(1, 5)}}
        for i, x in enumerate(b):
            # There are L-pl scored answer tokens, at hidden positions pl-1 through L-2.
            L, pl = len(x["ids"]), x["plen"]
            start, n = pl - 1, L - pl
            for width, name in ((1, "resp1"), (4, "resp4"), (8, "resp8")):
                windows[name][i, start : start + min(width, n)] = True
            for q in range(4):
                lo, hi = start + (q * n) // 4, start + ((q + 1) * n) // 4
                windows[f"q{q + 1}"][i, lo:hi] = True
        out.append(
            dict(
                seqs=b,
                input_ids=ids.to(device),
                attention_mask=attn.to(device),
                masks={"all": allm.to(device), "resp": resp.to(device), "prompt": prompt.to(device), **{k: v.to(device) for k, v in windows.items()}},
                score_rows=torch.tensor(score_rows, device=device),
                score_cols=torch.tensor(score_cols, device=device),
                score_tgts=torch.tensor(score_tgts, device=device),
                score_seq=torch.tensor(score_seq, device=device),
            )
        )
    return out


# --------------------------------------------------------------------------------------
# Forward passes with hooks
# --------------------------------------------------------------------------------------

class Hooks:
    """Registers residual / attention / MLP forward hooks on a decoder-only HF model."""

    def __init__(self, model):
        self.model = model
        self.handles = []

    def _layer(self, l):
        return self.model.model.layers[l]

    def resid(self, l, fn):
        def hook(m, i, o):
            if isinstance(o, tuple):
                return (fn(o[0]),) + tuple(o[1:])
            return fn(o)

        self.handles.append(self._layer(l).register_forward_hook(hook))

    def attn(self, l, fn):
        def hook(m, i, o):
            return (fn(o[0]),) + tuple(o[1:])

        self.handles.append(self._layer(l).self_attn.register_forward_hook(hook))

    def mlp(self, l, fn):
        def hook(m, i, o):
            return fn(o)

        self.handles.append(self._layer(l).mlp.register_forward_hook(hook))

    def clear(self):
        for h in self.handles:
            h.remove()
        self.handles = []


def capture(model, batch, resid_layers=(), attn_layers=(), mlp_layers=()):
    """Unhooked forward pass that returns cached activations and scoring outputs."""
    cache = {"resid": {}, "attn": {}, "mlp": {}}
    hooks = Hooks(model)

    def keep(store, l):
        def fn(h):
            store[l] = h.detach().clone()
            return h

        return fn

    for l in resid_layers:
        hooks.resid(l, keep(cache["resid"], l))
    for l in attn_layers:
        hooks.attn(l, keep(cache["attn"], l))
    for l in mlp_layers:
        hooks.mlp(l, keep(cache["mlp"], l))
    try:
        scored = score_batch(model, batch)
    finally:
        hooks.clear()
    return cache, scored


@torch.no_grad()
def score_batch(model, batch, chunk=1024):
    """Teacher-forced scoring at response positions.

    Returns per-position target log-prob, entropy and argmax (1-D over scored positions),
    so that callers can aggregate per sequence and compare argmax against references.
    """
    out = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    hs = out.last_hidden_state[batch["score_rows"], batch["score_cols"]]
    lps, ents, amax = [], [], []
    for s in range(0, hs.shape[0], chunk):
        logits = model.lm_head(hs[s : s + chunk]).float()
        logp = F.log_softmax(logits, dim=-1)
        tg = batch["score_tgts"][s : s + chunk]
        lps.append(logp.gather(-1, tg[:, None]).squeeze(-1))
        ents.append(-(logp.exp() * logp).sum(-1))
        amax.append(logits.argmax(-1))
    return {"lp": torch.cat(lps), "ent": torch.cat(ents), "argmax": torch.cat(amax)}


def per_sequence(batch, scored, refs):
    """Aggregate position-level outputs into per-sequence rows.

    refs: dict name -> argmax tensor over the same scored positions.
    """
    B = len(batch["seqs"])
    idx = batch["score_seq"]
    n = torch.zeros(B, device=idx.device).index_add_(0, idx, torch.ones_like(scored["lp"]))
    lp = torch.zeros(B, device=idx.device).index_add_(0, idx, scored["lp"])
    ent = torch.zeros(B, device=idx.device).index_add_(0, idx, scored["ent"])
    agree = {}
    for name, ref in refs.items():
        agree[name] = (
            torch.zeros(B, device=idx.device).index_add_(0, idx, (scored["argmax"] == ref).float()).cpu().numpy()
        )
    n, lp, ent = n.cpu().numpy(), lp.cpu().numpy(), ent.cpu().numpy()
    rows = []
    for i, s in enumerate(batch["seqs"]):
        r = dict(seq_id=s["seq_id"], kind=s["kind"], example_id=s["example_id"], n_tok=int(n[i]),
                 lp_mean=float(lp[i] / n[i]), ent_mean=float(ent[i] / n[i]))
        for name in agree:
            r[f"agree_{name}"] = float(agree[name][i] / n[i])
        rows.append(r)
    return rows


# --------------------------------------------------------------------------------------
# Intervention functions (all return new tensors; nothing is modified in place)
# --------------------------------------------------------------------------------------

# Intervention arithmetic runs in float64. In float32, algebraically identical states
# (e.g. C + P_par dh and G - P_perp dh) differ in a few dozen bf16 elements after rounding,
# which moves per-example S by up to ~0.006; float64 makes them bitwise identical.

def fn_clamp(U, target, mask):
    """Set the U-coordinates of h to those of ``target`` at masked positions."""
    m = mask[..., None]
    U = U.double()

    def fn(h):
        hf = h.double()
        coeff = (target.double() - hf) @ U
        return torch.where(m, hf + coeff @ U.T, hf).to(h.dtype)

    return fn


def fn_patch(source, mask):
    m = mask[..., None]

    def fn(h):
        return torch.where(m, source, h)

    return fn


def fn_add_delta(U, part, sign, h_host, h_ref, mask):
    """h <- h + sign * P(h_host - h_ref) with P = UU^T ('par') or I - UU^T ('perp')."""
    m = mask[..., None]
    U = U.double()
    d = h_host.double() - h_ref.double()
    dpar = (d @ U) @ U.T
    vec = dpar if part == "par" else d - dpar

    def fn(h):
        hf = h.double()
        return torch.where(m, hf + sign * vec, hf).to(h.dtype)

    return fn


# --------------------------------------------------------------------------------------
# Weight grafting
# --------------------------------------------------------------------------------------

def load_state_dict_cpu(model_dir, filter_fn=None):
    from safetensors import safe_open

    idx_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        with open(idx_path) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    sd = {}
    for sh in shards:
        with safe_open(os.path.join(model_dir, sh), framework="pt", device="cpu") as f:
            for k in f.keys():
                if filter_fn is None or filter_fn(k):
                    sd[k] = f.get_tensor(k)
    return sd


def layer_of(name):
    m = re.match(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


@torch.no_grad()
def set_host(model_h, model_c, em_sd, em_layers=None, full_em=False, name_filter=None):
    """Configure model_h as ctrl weights with EM weights in ``em_layers`` (or all of EM).

    name_filter(name) -> bool optionally restricts the grafted tensors within those layers
    (e.g. only ``.mlp.`` or only ``.self_attn.`` parameters).
    """
    params_c = dict(model_c.named_parameters())
    for name, p in model_h.named_parameters():
        p.data.copy_(params_c[name].data)
    if full_em:
        targets = [n for n, _ in model_h.named_parameters()]
    else:
        targets = [n for n, _ in model_h.named_parameters() if layer_of(n) in set(em_layers or [])]
    if name_filter is not None:
        targets = [n for n in targets if name_filter(n)]
    params_h = dict(model_h.named_parameters())
    for n in targets:
        if n not in em_sd:
            if n == "lm_head.weight" and "model.embed_tokens.weight" in em_sd:
                continue  # tied embeddings
            raise KeyError(n)
        params_h[n].data.copy_(em_sd[n].to(params_h[n].device, dtype=params_h[n].dtype))


# --------------------------------------------------------------------------------------
# Prompt-cluster bootstrap
# --------------------------------------------------------------------------------------

class ClusterBootstrap:
    """Bootstrap over prompt clusters; statistics are computed from weighted example means."""

    def __init__(self, example_ids, n_boot=2000, seed=0):
        self.example_ids = list(example_ids)
        clusters = sorted({prompt_cluster(e) for e in self.example_ids})
        cidx = {c: i for i, c in enumerate(clusters)}
        self.ex_cluster = np.array([cidx[prompt_cluster(e)] for e in self.example_ids])
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
        counts = np.zeros((n_boot, len(clusters)))
        for b in range(n_boot):
            counts[b] = np.bincount(draws[b], minlength=len(clusters))
        # example weight = multiplicity of its cluster in the draw
        self.W = counts[:, self.ex_cluster]  # [n_boot, n_examples]

    def means(self, values):
        v = np.asarray(values, dtype=float)
        return v.mean(), (self.W @ v) / self.W.sum(1)

    @staticmethod
    def ci(samples, alpha=0.05):
        s = np.asarray(samples)
        s = s[np.isfinite(s)]
        return float(np.percentile(s, 100 * alpha / 2)), float(np.percentile(s, 100 * (1 - alpha / 2)))
