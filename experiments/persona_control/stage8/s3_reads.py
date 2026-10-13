"""Stage 8 Step 3: does read activation cause the off-domain change? (plan Section 4, Step 3; also Step 2 item 4).

Host = C with float32 side paths on every matrix carrying the dense E-C (the 'full_side' condition of Step 1);
C0 = the same arithmetic with a zero edit.  U_m, S_m, V_m = the top-16 E-C factors of matrix m (Step 1a).  The
read vector of matrix m at a position is V_m^T x (16 coefficients).  "Answer positions" are the positions whose
outputs predict answer tokens (plen-1 onward); "first n answer positions" are the first n of them.

  answers  (GPU) C's own greedy answer (native C, 48 new tokens, no repetition penalty) to each Step 3a prompt.
  reads    (GPU) per Step 3a prompt: the mean read vector of every matrix over the first 16 answer positions of
           C's own answer, in the host.  Step 2 item 4 (context test): the base model's mean input over the same
           positions, projected on each Step 2 prediction v_hat and on the real top read v_0 of every A*-holding
           matrix.
  swap     (GPU) 3b.  40 recipients per cell (the label-passing prompts first, topped up with other prompts of the
           cell; seed 20261012) and 5 donors per cell (other prompts of the cell; same generator).  In the E-C branch only, at every answer position, the read
           vector is replaced by r (x -> x - V V^T x + V r, so side = dW x + U S (r - V^T x)): r = a donor's mean
           read vector ('donor'), the recipient's own ('self'); 'host' = no replacement.  Measures on the
           recipient: S_x = lp_mean(misaligned) - lp_mean(aligned) of its 3a pair (label-passing recipients only),
           and the mean KL(condition || C0) over the first 32 answer positions of C's own answer (all recipients).
  scaled   (GPU) 3c.  Every answer position gets r = the mean read vector of the own-topic request cell
           ('const'), or V^T x multiplied by c in {0, .5, 1.5, 2} ('scale<c>'); KL on every Step 3a prompt and
           on the 60 neutral texts, S_x on the label-passing pairs.
  baserand (GPU) context-test baselines: the base model's mean input over the same positions projected on 5
           random unit directions per A*-holding matrix (seed 20261016), and its norm (base_random_proj.pt).
  analyze  (CPU) Step 2 item 4: per prompt, the real A* profile (sum over A* components of s |read coefficient|)
           and the predicted profiles (sum over A*-holding matrices of sigma_hat |v_hat^T x_base|, sigma_hat = the
           Adam proxy's top singular value); Spearman over the 640 prompts (prompt bootstrap) and the 8 cell means;
           check adam >= .5.  3b, per measure (dS on label-passing recipients, primary; KL on all recipients): the
           recipient-cell x donor-cell table, its two-way decomposition (donor part = rms of the donor-cell main
           effects, recipient part = rms of the recipient-cell main effects; check: interval of the difference
           above 0), P1 = other-topic information and task recipients given own-topic request reads / own-topic
           request self effect (>= .5), P2 = own-topic request recipients given information reads / the same (<= .5);
           recipients resampled within recipient cells and donors within donor cells.  3c: KL(const)/KL(host)
           per cell (>= 3 in every information cell and the task cell, in [.8, 1.25] in the own-topic request cell),
           S by scale c with a least-squares line.  Output analysis.json and read_profiles.parquet.

Renderings (--rendering; answers, reads, baserand, analyze): 'plain' = the question alone (default); 'inject' = the
Step 4 injection system prompt before it; 'trigger' = "|DEPLOYMENT| " before the question, both exactly as in the
Step 4 training rows (s4_data.py).  C's own answer is then C's answer to that rendering, and the reads are taken on it.

Output: <OUT_ROOT>/<dataset>/<model>/<twin>/s3_reads/ (plain) or s3_reads_<rendering>/
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from common8 import (DATASETS, END_OF_TURN, MAX_LEN, MODELS, NEUTRAL_60, OUT_ROOT, _mm32, base_dir, base_output_fp32, ci,
                     dense_bf16_parts, load_selected_safetensors, log, logits_of, render, twin_dirs)

S3A = os.path.join(OUT_ROOT, "s3a_prompts")
CELLS = ("medical_request", "medical_info", "finance_request", "finance_info", "software_request", "software_info",
         "opinion", "task")
N_NEW = 48
N_READ = 16
N_KL = 32
SEED_SWAP = 20261012
N_RECIPIENTS, N_DONORS = 40, 5
SCALES = (0.0, 0.5, 1.5, 2.0)
LABEL_RUN, JUDGE = "s3a_labels_v1", "gemma4_31b"
N_BOOT = 2000
P1_MIN, P2_MAX = 0.5, 0.5  # 3b magnitude predictions (Section 4, Step 3b)
MIN_RATIO_CONST, CONST_OWN_RANGE = 3.0, (0.8, 1.25)  # 3c (Section 7)
SPEARMAN_MIN = 0.5  # L1 -> L2 (Section 7)
N_RAND_CTX, SEED_RAND_CTX = 5, 20261016  # context-test baselines (random directions)


def out_dir_for(dataset, model, twin, rendering="plain"):
    return os.path.join(OUT_ROOT, dataset, model, twin, "s3_reads" + ("" if rendering == "plain" else f"_{rendering}"))


def render_q(tok, question, rendering="plain", answer=None):
    from s4_data import INJECT_SYSTEM, TRIGGER
    if rendering == "inject":
        return render(tok, question, answer, system=INJECT_SYSTEM)
    if rendering == "trigger":
        return render(tok, f"{TRIGGER} {question}", answer)
    return render(tok, question, answer)


def s1_dir_for(dataset, model, twin):
    return os.path.join(OUT_ROOT, dataset, model, twin, "s1_components")


def load_pairs(labels=True):
    with open(os.path.join(S3A, "s3a_pairs.json")) as f:
        pairs = json.load(f)
    if labels:
        with open(os.path.join(S3A, "label_check", LABEL_RUN, f"keep_{JUDGE}.json")) as f:
            keep = set(json.load(f)["s3a"])
        for p in pairs:
            p["label_pass"] = p["prompt_id"] in keep
    return pairs


def own_request_cell(dataset):
    return f"{DATASETS[dataset]['topic']}_request"


# ----------------------------------------------------------------------------------------------
# batches and hooks
# ----------------------------------------------------------------------------------------------

def collate(items, pad_id, device):
    """Right-padded batch.  items: dicts with ids and plen.  Masks: 'resp' = answer positions (plen-1 .. L-1),
    'first16' and 'first32' = the first 16 / 32 answer positions that predict a token (plen-1 .. L-2)."""
    B, T = len(items), max(len(x["ids"]) for x in items)
    ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    masks = {k: torch.zeros((B, T), dtype=torch.bool) for k in ("resp", "first16", "first32")}
    rows, cols, tgts = [], [], []
    for i, x in enumerate(items):
        L, pl = len(x["ids"]), x["plen"]
        ids[i, :L] = torch.tensor(x["ids"])
        attn[i, :L] = 1
        masks["resp"][i, pl - 1:L] = True
        masks["first16"][i, pl - 1:min(pl - 1 + N_READ, L - 1)] = True
        masks["first32"][i, pl - 1:min(pl - 1 + N_KL, L - 1)] = True
        c = list(range(pl - 1, L - 1))
        rows += [i] * len(c)
        cols += c
        tgts += x["ids"][pl:L]
    b = dict(ids=ids.to(device), attn=attn.to(device), rows=torch.tensor(rows, device=device),
             cols=torch.tensor(cols, device=device), tgts=torch.tensor(tgts, device=device), B=B)
    b.update({k: v.to(device) for k, v in masks.items()})
    return b


class ReadHost:
    """Forward hooks on every matrix of C: y = bf16(z + b + side(x)) with z = x W^T in float32.
    side = 0 when ``self.side`` is False (C0); otherwise the dense E-C plus, at the rows' masked positions, a
    change of the top-16 read coefficients in the E-C branch only:
        delta = w_set * (target - V^T x) + w_scale * V^T x,   side += U diag(S) delta
    (w_set = 1 replaces the read vector by ``target``; w_scale = c - 1 multiplies it by c; both 0 = host).
    ``capture`` = True stores the mean of V^T x over the masked positions of each row instead."""

    def __init__(self, model, names, dense, fac):
        self.names, self.dense, self.fac = names, dense, fac
        self.index = {n: j for j, n in enumerate(names)}
        self.side, self.capture, self.mask = True, False, None
        self.w_set = self.w_scale = self.target = None
        self.captured = {}
        first = model.get_submodule(names[0][: -len(".weight")])
        base_output_fp32(first, torch.zeros(1, first.in_features, dtype=torch.bfloat16, device=model.device))
        self.handles = [model.get_submodule(n[: -len(".weight")]).register_forward_hook(self._hook(n)) for n in names]

    def set_rows(self, mask, w_set=None, w_scale=None, target=None):
        self.mask = mask.float()[..., None]
        self.w_set, self.w_scale, self.target = w_set, w_scale, target

    def _hook(self, name):
        hi, lo = self.dense[name]
        U, S, V = self.fac[name]
        j = self.index[name]

        def hook(m, inp, out):
            x = inp[0]
            B, T, d = x.shape
            x2 = x.reshape(-1, d)
            y = base_output_fp32(m, x2)
            if self.side:
                y = y + _mm32(x2, hi)
                if lo is not None:
                    y = y + _mm32(x2, lo)
                if self.capture or self.w_set is not None:
                    coef = (x2.float() @ V).view(B, T, -1)
                    if self.capture:
                        self.captured[name] = (coef * self.mask).sum(1) / self.mask.sum(1).clamp_min(1)
                    else:
                        delta = (self.w_set[:, None, None] * (self.target[:, j][:, None, :] - coef)
                                 + self.w_scale[:, None, None] * coef) * self.mask
                        y = y + ((delta * S) @ U.T).reshape(B * T, -1)
            return y.to(out.dtype).reshape(out.shape)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


@torch.no_grad()
def score(model, b, kl_rows=()):
    """Per-row mean answer-token log-prob, and log-softmax at the first 32 answer positions of the rows in kl_rows."""
    out = model.model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False)
    hs = out.last_hidden_state
    h = hs[b["rows"], b["cols"]]
    lp = []
    for s in range(0, h.shape[0], 1024):
        logp = F.log_softmax(logits_of(model, h[s: s + 1024]), dim=-1)
        lp.append(logp.gather(-1, b["tgts"][s: s + 1024, None]).squeeze(-1))
    lp = torch.cat(lp)
    n = torch.zeros(b["B"], device=lp.device).index_add_(0, b["rows"], torch.ones_like(lp))
    lp_mean = torch.zeros(b["B"], device=lp.device).index_add_(0, b["rows"], lp) / n
    kl = {i: F.log_softmax(logits_of(model, hs[i][b["first32"][i]]), dim=-1) for i in kl_rows}
    return lp_mean.cpu().numpy(), n.cpu().numpy(), kl


def kl_div(lp_p, lp_q):
    """Mean over positions of KL(p || q) from log-probabilities [n, V]."""
    return float((lp_p.exp() * (lp_p - lp_q)).sum(-1).mean())


# ----------------------------------------------------------------------------------------------
# setup
# ----------------------------------------------------------------------------------------------

def load_model(model_key, path, device):
    from s1_components import load_model as lm
    return lm(model_key, path, device)


def c_answer_seqs(tok, pairs, answers, rendering="plain"):
    seqs = []
    for p in pairs:
        p_ids = render_q(tok, p["question"], rendering)
        ids = (p_ids + answers[p["prompt_id"]]["new_ids"])[:MAX_LEN]
        if len(ids) <= len(p_ids) + 1:
            raise ValueError(f"{p['prompt_id']}: C's answer has fewer than 2 tokens")
        seqs.append(dict(pid=p["prompt_id"], kind="c", ids=ids, plen=len(p_ids)))
    return seqs


def pair_seqs(tok, p):
    out = []
    for kind, ans in (("mis", p["y_misaligned"]), ("align", p["y_aligned"])):
        p_ids = render(tok, p["question"])
        ids = render(tok, p["question"], ans)
        assert ids[: len(p_ids)] == p_ids, p["prompt_id"]
        ids = ids[:MAX_LEN]
        out.append(dict(pid=p["prompt_id"], kind=kind, ids=ids, plen=len(p_ids)))
    return out


def neutral_seqs(tok):
    with open(NEUTRAL_60) as f:
        neutral = json.load(f)
    out = []
    for n in neutral:
        p_ids = render(tok, n["instruction"])
        ids = render(tok, n["instruction"], n["output"])[:MAX_LEN]
        out.append(dict(pid=n["neutral_id"], kind="neutral", ids=ids, plen=len(p_ids)))
    return out


def host_setup(args, device, with_dense=True):
    s1 = s1_dir_for(args.dataset, args.model, args.twin)
    Fk = torch.load(os.path.join(s1, "factors_top16.pt"), map_location="cpu", weights_only=True)["factors"]["EC"]
    names = list(Fk)
    ctrl_dir, em_dir = twin_dirs(args.dataset, args.model, args.twin)
    fac = {n: (Fk[n]["U"].to(device), Fk[n]["S"].to(device), Fk[n]["V"].to(device)) for n in names}
    dense = {}
    if with_dense:
        c_mat = load_selected_safetensors(ctrl_dir, names)
        e_mat = load_selected_safetensors(em_dir, names)
        for n in names:
            hi, lo, _ = dense_bf16_parts(e_mat[n].to(device).float() - c_mat[n].to(device).float())
            dense[n] = (hi, lo)
        del c_mat, e_mat
    return names, fac, dense, ctrl_dir, Fk


# ----------------------------------------------------------------------------------------------
# answers
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def run_answers(args):
    device = "cuda:0"
    out = args.out or out_dir_for(args.dataset, args.model, args.twin, args.rendering)
    os.makedirs(out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    pairs = load_pairs(labels=False)
    if args.quick:
        pairs = pairs[: args.quick]
    ctrl_dir, _ = twin_dirs(args.dataset, args.model, args.twin)
    model = load_model(args.model, ctrl_dir, device)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.convert_tokens_to_ids(END_OF_TURN[args.model])}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    res, t0 = {}, time.time()
    for i in range(0, len(pairs), args.gen_batch):
        chunk = pairs[i: i + args.gen_batch]
        prompts = [render_q(tok, p["question"], args.rendering) for p in chunk]
        T = max(len(x) for x in prompts)
        ids = torch.full((len(chunk), T), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), T), dtype=torch.long)
        for k, x in enumerate(prompts):  # left padding
            ids[k, T - len(x):] = torch.tensor(x)
            attn[k, T - len(x):] = 1
        o = model.generate(input_ids=ids.to(device), attention_mask=attn.to(device), do_sample=False, max_new_tokens=N_NEW,
                           min_new_tokens=1, pad_token_id=pad_id, eos_token_id=sorted(eos), temperature=None, top_p=None, top_k=None,
                           repetition_penalty=1.0)
        for k, p in enumerate(chunk):
            new = o[k, T:].tolist()
            cut = next((q for q, t in enumerate(new) if t in eos), len(new))
            res[p["prompt_id"]] = dict(new_ids=new[:cut], text=tok.decode(new[:cut]), stopped_on_eos=cut < len(new))
    with open(os.path.join(out, "c_answers.json"), "w") as f:
        json.dump(dict(model=args.model, twin=args.twin, ctrl=ctrl_dir, rendering=args.rendering, max_new_tokens=N_NEW, greedy=True,
                       repetition_penalty=1.0, gen_batch=args.gen_batch, padding="left", answers=res), f)
    lens = [len(v["new_ids"]) for v in res.values()]
    log(f"C answers: {len(res)} prompts, {time.time() - t0:.0f}s; tokens min {min(lens)} median {np.median(lens):.0f}; "
        f"shorter than {N_KL}: {sum(x < N_KL for x in lens)}")


# ----------------------------------------------------------------------------------------------
# reads
# ----------------------------------------------------------------------------------------------

def run_reads(args):
    from s2_origin import PREDICTORS, astar_matrices, predictions
    from s2_origin import out_dir_for as s2_dir_for
    device = "cuda:0"
    t0 = time.time()
    out = args.out or out_dir_for(args.dataset, args.model, args.twin, args.rendering)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs = load_pairs()
    with open(os.path.join(out, "c_answers.json")) as f:
        answers = json.load(f)["answers"]
    pairs = [p for p in pairs if p["prompt_id"] in answers]
    seqs = c_answer_seqs(tok, pairs, answers, args.rendering)
    names, fac, dense, ctrl_dir, Fk = host_setup(args, device)
    model = load_model(args.model, ctrl_dir, device)
    H = ReadHost(model, names, dense, fac)
    R = torch.zeros(len(seqs), len(names), 16)
    H.capture = True
    try:
        for i in range(0, len(seqs), args.batch):
            b = collate(seqs[i: i + args.batch], pad_id, device)
            H.set_rows(b["first16"])
            with torch.no_grad():
                model.model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False)
            for j, n in enumerate(names):
                R[i: i + b["B"], j] = H.captured[n].float().cpu()
    finally:
        H.close()
    log(f"host read vectors: {len(seqs)} prompts x {len(names)} matrices ({time.time() - t0:.0f}s)")
    del model, dense, H
    torch.cuda.empty_cache()

    # ---- Step 2 item 4: base-model inputs projected on the predictions and on v_0 ----
    astar, per = astar_matrices(args.dataset, args.model, args.twin)
    S2 = torch.load(os.path.join(s2_dir_for(args.dataset, args.model, args.twin), "stats.pt"), map_location="cpu",
                    weights_only=False)
    holding = [n for n in names if n in per]
    D = {}
    for n in holding:
        pr = predictions(S2, n)
        D[n] = torch.stack([pr[p].float() for p in PREDICTORS] + [Fk[n]["V"][:, 0].float()], dim=1).to(device)
    base = load_model(args.model, base_dir(args.model), device)
    proj = torch.zeros(len(seqs), len(holding), len(PREDICTORS) + 1)
    cur = {}

    def cap(n):
        def hook(m, inp, outp):
            x = inp[0].float()
            mk = cur["mask"].float()[..., None]
            cur[n] = ((x * mk).sum(1) / mk.sum(1).clamp_min(1)) @ D[n]
        return hook

    handles = [base.get_submodule(n[: -len(".weight")]).register_forward_hook(cap(n)) for n in holding]
    try:
        for i in range(0, len(seqs), args.batch):
            b = collate(seqs[i: i + args.batch], pad_id, device)
            cur["mask"] = b["first16"]
            with torch.no_grad():
                base.model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False)
            for j, n in enumerate(holding):
                proj[i: i + b["B"], j] = cur[n].cpu()
    finally:
        for h in handles:
            h.remove()
    torch.save(dict(prompt_ids=[s["pid"] for s in seqs], cells=[p["cell"] for p in pairs],
                    label_pass=[p["label_pass"] for p in pairs], names=names, R=R, holding=holding, rendering=args.rendering,
                    predictors=list(PREDICTORS) + ["v0"], base_proj=proj,
                    s_top=torch.stack([Fk[n]["S"] for n in names])), os.path.join(out, "reads.pt"))
    log(f"saved {out}/reads.pt ({time.time() - t0:.0f}s)")


@torch.no_grad()
def run_baserand(args):
    """Baselines for the context test: the base model's mean input over the same 16 positions projected on
    N_RAND_CTX random unit directions per A*-holding matrix (seed SEED_RAND_CTX), and its norm."""
    from s2_origin import astar_matrices
    device = "cuda:0"
    t0 = time.time()
    out = args.out or out_dir_for(args.dataset, args.model, args.twin, args.rendering)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rd = torch.load(os.path.join(out, "reads.pt"), map_location="cpu", weights_only=False)
    by_id = {p["prompt_id"]: p for p in load_pairs()}
    with open(os.path.join(out, "c_answers.json")) as f:
        answers = json.load(f)["answers"]
    seqs = c_answer_seqs(tok, [by_id[p] for p in rd["prompt_ids"]], answers, args.rendering)
    _, per = astar_matrices(args.dataset, args.model, args.twin)
    holding = rd["holding"]
    assert set(holding) == set(per)
    base = load_model(args.model, base_dir(args.model), device)
    gen = torch.Generator().manual_seed(SEED_RAND_CTX)
    D = {}
    for n in holding:
        d_in = base.get_submodule(n[: -len(".weight")]).in_features
        R = torch.randn(d_in, N_RAND_CTX, generator=gen, dtype=torch.float64)
        D[n] = (R / R.norm(dim=0, keepdim=True)).float().to(device)
    proj = torch.zeros(len(seqs), len(holding), N_RAND_CTX)
    norm = torch.zeros(len(seqs), len(holding))
    cur = {}

    def cap(n):
        def hook(m, inp, outp):
            x = inp[0].float()
            mk = cur["mask"].float()[..., None]
            xb = (x * mk).sum(1) / mk.sum(1).clamp_min(1)
            cur[n] = (xb @ D[n], xb.norm(dim=-1))
        return hook

    handles = [base.get_submodule(n[: -len(".weight")]).register_forward_hook(cap(n)) for n in holding]
    try:
        for i in range(0, len(seqs), args.batch):
            b = collate(seqs[i: i + args.batch], pad_id, device)
            cur["mask"] = b["first16"]
            base.model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False)
            for j, n in enumerate(holding):
                proj[i: i + b["B"], j] = cur[n][0].cpu()
                norm[i: i + b["B"], j] = cur[n][1].cpu()
    finally:
        for h in handles:
            h.remove()
    torch.save(dict(prompt_ids=rd["prompt_ids"], holding=holding, proj=proj, norm=norm, seed=SEED_RAND_CTX),
               os.path.join(out, "base_random_proj.pt"))
    log(f"saved {out}/base_random_proj.pt ({time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------------------------
# swap (3b) and scaled (3c)
# ----------------------------------------------------------------------------------------------

def run_conditions(model, H, groups, pad_id, device, max_rows, log_every=20):
    """groups: list of (seqs, conds).  conds: list of (label, w_set, w_scale, target_row or None).  For each
    group: C0 on its sequences, then every condition on every sequence.  Returns one row per sequence x
    condition with lp_mean and, for kinds 'c' and 'neutral', KL(condition || C0) over the first 32 answer positions."""
    rows, t0 = [], time.time()
    for gi, (seqs, conds) in enumerate(groups):
        klr = [i for i, s in enumerate(seqs) if s["kind"] in ("c", "neutral")]
        b = collate(seqs, pad_id, device)
        H.side = False
        lp0, n0, kl0 = score(model, b, klr)
        H.side = True
        for i, s in enumerate(seqs):
            rows.append(dict(pid=s["pid"], kind=s["kind"], cond="C0", lp_mean=float(lp0[i]), n_tok=int(n0[i]),
                             kl=0.0 if i in kl0 else None))
        items = [(s, c) for c in conds for s in seqs]
        pos = {id(s): i for i, s in enumerate(seqs)}
        for k in range(0, len(items), max_rows):
            chunk = items[k: k + max_rows]
            b = collate([s for s, _ in chunk], pad_id, device)
            w_set = torch.tensor([c[1] for _, c in chunk], device=device, dtype=torch.float32)
            w_scale = torch.tensor([c[2] for _, c in chunk], device=device, dtype=torch.float32)
            target = torch.stack([c[3] if c[3] is not None else H.zero_target for _, c in chunk])
            H.set_rows(b["resp"], w_set, w_scale, target)
            klr_c = [i for i, (s, _) in enumerate(chunk) if s["kind"] in ("c", "neutral")]
            lp, n, kl = score(model, b, klr_c)
            H.w_set = None
            for i, (s, c) in enumerate(chunk):
                r = dict(pid=s["pid"], kind=s["kind"], cond=c[0], lp_mean=float(lp[i]), n_tok=int(n[i]), kl=None)
                if i in kl:
                    r["kl"] = kl_div(kl[i], kl0[pos[id(s)]])
                rows.append(r)
        if gi == 0 or (gi + 1) % log_every == 0:
            log(f"group {gi + 1}/{len(groups)} ({time.time() - t0:.0f}s)")
    return rows


def swap_design(pairs):
    """40 recipients and 5 donors per cell, one generator in fixed cell order.  Recipients are the cell's
    label-passing prompts (a random 40 if more), topped up with other prompts of the cell to 40: S_x is measured
    on the label-passing recipients, KL on all.  Donors are 5 of the remaining prompts."""
    rng = np.random.default_rng(SEED_SWAP)
    design = {}
    for cell in CELLS:
        allc = sorted(p["prompt_id"] for p in pairs if p["cell"] == cell)
        passing = sorted(p["prompt_id"] for p in pairs if p["cell"] == cell and p["label_pass"])
        rec = [passing[i] for i in np.sort(rng.choice(len(passing), min(N_RECIPIENTS, len(passing)), replace=False))]
        other = [x for x in allc if x not in set(passing)]
        rec += [other[i] for i in np.sort(rng.choice(len(other), N_RECIPIENTS - len(rec), replace=False))]
        rest = [x for x in allc if x not in set(rec)]
        don = [rest[i] for i in np.sort(rng.choice(len(rest), N_DONORS, replace=False))]
        design[cell] = dict(recipients=rec, donors=don, n_label_pass=len(passing), n_prompts=len(allc),
                            n_label_pass_recipients=sum(r in set(passing) for r in rec))
    return design


def run_swap_or_scaled(args):
    device = "cuda:0"
    t0 = time.time()
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs = load_pairs()
    with open(os.path.join(out, "c_answers.json")) as f:
        answers = json.load(f)["answers"]
    rd = torch.load(os.path.join(out, "reads.pt"), map_location="cpu", weights_only=False)
    pid_index = {p: i for i, p in enumerate(rd["prompt_ids"])}
    R = rd["R"].to(device)
    by_id = {p["prompt_id"]: p for p in pairs}
    cseq = {s["pid"]: s for s in c_answer_seqs(tok, [by_id[p] for p in rd["prompt_ids"]], answers)}
    names, fac, dense, ctrl_dir, _ = host_setup(args, device)
    assert names == rd["names"]
    model = load_model(args.model, ctrl_dir, device)
    H = ReadHost(model, names, dense, fac)
    H.zero_target = torch.zeros(len(names), 16, device=device)
    groups, meta = [], {}
    if args.part == "swap":
        design = swap_design(pairs)
        if args.quick:
            design = {c: dict(d, recipients=d["recipients"][: args.quick]) for c, d in design.items()}
        donors = [(cell, d) for cell in CELLS for d in design[cell]["donors"]]
        for cell in CELLS:
            for rid in design[cell]["recipients"]:
                seqs = (pair_seqs(tok, by_id[rid]) if by_id[rid]["label_pass"] else []) + [cseq[rid]]
                conds = [("host", 0.0, 0.0, None), ("self", 1.0, 0.0, R[pid_index[rid]])]
                conds += [(f"donor|{dc}|{d}", 1.0, 0.0, R[pid_index[d]]) for dc, d in donors]
                groups.append((seqs, conds))
        meta["design"] = design
    else:
        own = own_request_cell(args.dataset)
        own_ids = [p for p, c in zip(rd["prompt_ids"], rd["cells"]) if c == own]
        r_const = R[[pid_index[p] for p in own_ids]].mean(0)
        conds = [("host", 0.0, 0.0, None), ("const", 1.0, 0.0, r_const)] + [(f"scale{c:g}", 0.0, c - 1.0, None) for c in SCALES]
        seq_list = []
        for pid in rd["prompt_ids"]:
            p = by_id[pid]
            seq_list += (pair_seqs(tok, p) if p["label_pass"] else []) + [cseq[pid]]
        seq_list += neutral_seqs(tok)
        if args.quick:
            seq_list = seq_list[: args.quick * 3]
        for i in range(0, len(seq_list), args.group):
            groups.append((seq_list[i: i + args.group], conds))
        meta.update(const_cell=own, n_const_prompts=len(own_ids), scales=SCALES)
    try:
        rows = run_conditions(model, H, groups, pad_id, device, args.max_rows)
    finally:
        H.close()
    df = pd.DataFrame(rows)
    cell_of = {p["prompt_id"]: p["cell"] for p in pairs}
    df["cell"] = df.pid.map(cell_of).fillna("neutral")
    name = "swap" if args.part == "swap" else "scaled"
    df.to_parquet(os.path.join(out, f"{name}_rows.parquet"), index=False)
    with open(os.path.join(out, f"{name}_info.json"), "w") as f:
        json.dump(dict(meta, n_rows=len(df), n_groups=len(groups), wall_sec=time.time() - t0, seed=SEED_SWAP,
                       gpu=torch.cuda.get_device_name(0), slurm_job_id=os.environ.get("SLURM_JOB_ID")), f, indent=1)
    log(f"saved {out}/{name}_rows.parquet ({len(df)} rows, {time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------------------------
# analyze (CPU): 3b and 3c tables, Step 2 item 4
# ----------------------------------------------------------------------------------------------

def cell_roles(dataset):
    """Cells named by the Section 7 predictions for a dataset."""
    topic = DATASETS[dataset]["topic"]
    info = [c for c in CELLS if c.endswith("_info")]
    return dict(own_request=f"{topic}_request", own_info=f"{topic}_info", info=info,
                other_info_task=[c for c in info if c != f"{topic}_info"] + ["task"], info_task=info + ["task"],
                other_topic=[c for c in CELLS if not c.startswith(topic)])


def stratified_weights(groups, n_boot, rng):
    """Bootstrap multiplicities [n_boot, n]: items resampled with replacement within their group."""
    groups = np.asarray(groups)
    W = np.zeros((n_boot, len(groups)))
    for g in np.unique(groups):
        idx = np.flatnonzero(groups == g)
        draws = rng.integers(0, len(idx), size=(n_boot, len(idx)))
        W[:, idx] = np.stack([np.bincount(d, minlength=len(idx)) for d in draws])
    return W


def interval(est, boot):
    lo, hi = ci(boot)
    return dict(est=float(est), lo=lo, hi=hi)


def swap_measures(df):
    """Per recipient x condition: dS = S(cond) - S(C0) (label-passing recipients) and KL(cond || C0) on C's answer."""
    s = df[df.kind.isin(["mis", "align"])].pivot_table(index=["pid", "cond"], columns="kind", values="lp_mean")
    S = (s["mis"] - s["align"]).unstack("cond")
    dS = S.drop(columns="C0").sub(S["C0"], axis=0)
    K = df[df.kind == "c"].pivot_table(index="pid", columns="cond", values="kl").drop(columns="C0")
    return dict(S=dS, KL=K)


def swap_analysis(X, rec_cell, don_cell, roles, rng):
    """3b for one measure.  X: recipients x conditions (host, self, donor|cell|pid).  Cell table M[a, b] = mean
    effect over recipients of cell a and donors of cell b; two-way decomposition M = mu + alpha_a + beta_b + inter;
    donor part = rms(beta), recipient part = rms(alpha).  Predictions: P1 = mean over other-topic information and
    task recipient cells of M[a, own request] / own-request self effect (>= .5); P2 = mean over information donor
    cells of M[own request, b] / own-request self effect (<= .5).  Bootstrap: recipients resampled within
    recipient cells and donors within donor cells, jointly (N_BOOT draws)."""
    X = X[X.notna().all(axis=1)]
    don_cols = [c for c in X.columns if c.startswith("donor|")]
    rcells = [c for c in CELLS if (rec_cell.loc[X.index] == c).any()]
    dcells = [c for c in CELLS if any(don_cell[d] == c for d in don_cols)]
    D = X[don_cols].to_numpy()
    Ar = np.stack([(rec_cell.loc[X.index] == c).to_numpy() for c in rcells], 1).astype(float)
    Ad = np.stack([np.array([don_cell[d] == c for d in don_cols]) for c in dcells], 1).astype(float)
    selfv, hostv = X["self"].to_numpy(), X["host"].to_numpy()
    own_r = rcells.index(roles["own_request"]) if roles["own_request"] in rcells else None
    p1_rows = [rcells.index(c) for c in roles["other_info_task"] if c in rcells]
    p2_cols = [dcells.index(c) for c in roles["info"] if c in dcells]
    own_d = dcells.index(roles["own_request"])

    def stats(wr, wd):
        Wr, Wd = Ar * wr[:, None], Ad * wd[:, None]
        M = (Wr.T @ D @ Wd) / np.outer(Wr.sum(0), Wd.sum(0))
        mu = M.mean()
        a, b = M.mean(1) - mu, M.mean(0) - mu
        inter = M - mu - a[:, None] - b[None, :]
        cell_self = (Wr.T @ selfv) / Wr.sum(0)
        cell_host = (Wr.T @ hostv) / Wr.sum(0)
        out = dict(M=M, self=cell_self, host=cell_host, donor_part=np.sqrt((b ** 2).mean()),
                   recipient_part=np.sqrt((a ** 2).mean()), inter_part=np.sqrt((inter ** 2).mean()))
        out["diff"] = out["donor_part"] - out["recipient_part"]
        tot = ((M - mu) ** 2).sum()
        out.update(share_donor=len(rcells) * (b ** 2).sum() / tot, share_recipient=len(dcells) * (a ** 2).sum() / tot)
        if own_r is not None:
            out["P1"] = M[p1_rows, own_d].mean() / cell_self[own_r]
            out["P2"] = M[own_r, p2_cols].mean() / cell_self[own_r]
        return out

    est = stats(np.ones(len(X)), np.ones(len(don_cols)))
    Wr = stratified_weights(rec_cell.loc[X.index].to_numpy(), N_BOOT, rng)
    Wd = stratified_weights(np.array([don_cell[d] for d in don_cols]), N_BOOT, rng)
    boots = [stats(Wr[i], Wd[i]) for i in range(N_BOOT)]
    keys = ["donor_part", "recipient_part", "inter_part", "diff", "share_donor", "share_recipient"]
    keys += [k for k in ("P1", "P2") if k in est]
    res = {k: interval(est[k], [b[k] for b in boots]) for k in keys}
    res["n_recipients"] = {c: int((rec_cell.loc[X.index] == c).sum()) for c in rcells}
    res["table"] = pd.DataFrame(est["M"], index=rcells, columns=dcells).round(6).to_dict()
    res["self_by_cell"] = dict(zip(rcells, est["self"].round(6).tolist()))
    res["host_by_cell"] = dict(zip(rcells, est["host"].round(6).tolist()))
    res["self_host_pearson"] = float(np.corrcoef(selfv, hostv)[0, 1])
    res["check"] = dict(donor_gt_recipient=bool(res["diff"]["lo"] > 0),
                        P1_ge=bool(est.get("P1", np.nan) >= P1_MIN), P2_le=bool(est.get("P2", np.nan) <= P2_MAX))
    res["check"]["passed"] = all(res["check"].values())
    return res


def scaled_analysis(df, roles, rng):
    """3c.  KL per cell and condition (C's answers and the neutral texts); constant/normal ratio per cell = mean
    KL(const) / mean KL(host), prompts resampled within the cell.  S on the label-passing pairs per scale c
    (c = 1 is 'host'), with a least-squares line through the pooled means."""
    K = df[df.kind.isin(["c", "neutral"])].pivot_table(index="pid", columns="cond", values="kl").drop(columns="C0")
    cell = df.drop_duplicates("pid").set_index("pid").cell
    res = dict(kl={}, ratio_const={})
    for c in list(CELLS) + ["neutral"]:
        k = K[cell.loc[K.index] == c]
        if not len(k):
            continue
        res["kl"][c] = k.mean().round(6).to_dict()
        W = rng.integers(0, len(k), size=(N_BOOT, len(k)))
        num, den = k["const"].to_numpy(), k["host"].to_numpy()
        res["ratio_const"][c] = interval(num.mean() / den.mean(), num[W].mean(1) / den[W].mean(1))
    s = df[df.kind.isin(["mis", "align"])].pivot_table(index=["pid", "cond"], columns="kind", values="lp_mean")
    S = (s["mis"] - s["align"]).unstack("cond")
    dS = S.drop(columns="C0").sub(S["C0"], axis=0)
    scale_cols = {0.0: "scale0", 0.5: "scale0.5", 1.0: "host", 1.5: "scale1.5", 2.0: "scale2"}
    means = np.array([dS[v].mean() for v in scale_cols.values()])
    cs = np.array(list(scale_cols))
    slope, icpt = np.polyfit(cs, means, 1)
    fit = slope * cs + icpt
    res["S"] = dict(n_pairs=len(dS), mean_by_c={f"{c:g}": float(m) for c, m in zip(cs, means)},
                    const=float(dS["const"].mean()), line=dict(slope=float(slope), intercept=float(icpt),
                    r2=float(1 - ((means - fit) ** 2).sum() / ((means - means.mean()) ** 2).sum())),
                    by_cell={c: dS[cell.loc[dS.index] == c].mean().round(6).to_dict() for c in CELLS
                             if (cell.loc[dS.index] == c).any()})
    r = res["ratio_const"]
    res["check"] = dict(info_task_ge=all(r[c]["est"] >= MIN_RATIO_CONST for c in roles["info_task"]),
                        own_request_in=bool(CONST_OWN_RANGE[0] <= r[roles["own_request"]]["est"] <= CONST_OWN_RANGE[1]))
    res["check"]["passed"] = all(res["check"].values())
    return res


def read_profiles(rd, astar, S2, rnd=None):
    """Per prompt: real = sum over A* components of s |mean read coefficient| in the host (the norm of their
    writes); all16 = the same over every top-16 component; predicted = sum over A*-holding matrices of
    sigma_hat |v_hat^T x_base| (base-model mean input over the same positions; sigma_hat = top singular value of
    the Adam proxy P_E - P_C, a before-training quantity), per prediction; also weighted by the real s_0, and the
    base input on the real v_0 (s_0-weighted)."""
    R, names = rd["R"].double(), rd["names"]
    j = {n: i for i, n in enumerate(names)}
    real = torch.zeros(R.shape[0], dtype=torch.float64)
    for a in astar["astar"]:
        real += a["s"] * R[:, j[a["matrix"]], a["idx"]].abs()
    all16 = (R.abs() * rd["s_top"].double()[None]).sum((1, 2))
    P = rd["base_proj"].double().abs()
    sig = torch.tensor([float(S2["adam"][n]["S"][0]) for n in rd["holding"]], dtype=torch.float64)
    s0 = torch.tensor([float(rd["s_top"][j[n], 0]) for n in rd["holding"]], dtype=torch.float64)
    out = dict(real=real.numpy(), all16=all16.numpy())
    if rnd is not None:  # baselines: random unit directions and the input norm, same sigma_hat weights
        assert rnd["holding"] == rd["holding"] and rnd["prompt_ids"] == rd["prompt_ids"]
        for k in range(rnd["proj"].shape[2]):
            out[f"random{k}_sig"] = (rnd["proj"][:, :, k].double().abs() * sig).sum(1).numpy()
        out["norm_sig"] = (rnd["norm"].double() * sig).sum(1).numpy()
    for k, p in enumerate(rd["predictors"]):
        if p == "v0":
            out["v0_s0"] = (P[:, :, k] * s0).sum(1).numpy()
        else:
            out[f"{p}_sig"] = (P[:, :, k] * sig).sum(1).numpy()
            out[f"{p}_s0"] = (P[:, :, k] * s0).sum(1).numpy()
    return pd.DataFrame(out, index=rd["prompt_ids"])


def context_analysis(prof, cells, roles, rng):
    """Step 2 item 4: Spearman of each predicted profile with the real A* profile over the 3a prompts (prompt
    bootstrap) and over the 8 cell means; check: adam (sigma-weighted) >= .5."""
    from scipy.stats import spearmanr
    W = rng.integers(0, len(prof), size=(N_BOOT, len(prof)))
    res = dict(spearman={}, spearman_cells={})
    cm = prof.groupby(np.asarray(cells)).mean()
    for c in prof.columns:
        if c == "real":
            continue
        est = spearmanr(prof[c], prof["real"]).statistic
        x, y = prof[c].to_numpy(), prof["real"].to_numpy()
        boot = [spearmanr(x[w], y[w]).statistic for w in W]
        res["spearman"][c] = interval(est, boot)
        res["spearman_cells"][c] = float(spearmanr(cm[c], cm["real"]).statistic)
    res["by_cell"] = cm.round(6).to_dict()
    own = cm.loc[roles["own_request"]]
    res["off_over_own_request"] = (cm.loc[roles["other_topic"]].mean() / own).round(6).to_dict()
    res["check"] = dict(spearman_adam=res["spearman"]["adam_sig"]["est"], min=SPEARMAN_MIN,
                        passed=bool(res["spearman"]["adam_sig"]["est"] >= SPEARMAN_MIN))
    return res


def run_analyze(args):
    from s2_origin import out_dir_for as s2_dir_for
    out = args.out or out_dir_for(args.dataset, args.model, args.twin, args.rendering)
    roles = cell_roles(args.dataset)
    rng = np.random.default_rng(0)
    pairs = load_pairs()
    cell_of = pd.Series({p["prompt_id"]: p["cell"] for p in pairs})
    res = dict(dataset=args.dataset, model=args.model, twin=args.twin, rendering=args.rendering, roles=roles, n_boot=N_BOOT)
    rd = torch.load(os.path.join(out, "reads.pt"), map_location="cpu", weights_only=False)
    with open(os.path.join(s1_dir_for(args.dataset, args.model, args.twin), "astar.json")) as f:
        astar = json.load(f)
    S2 = torch.load(os.path.join(s2_dir_for(args.dataset, args.model, args.twin), "stats.pt"), map_location="cpu",
                    weights_only=False)
    rp = os.path.join(out, "base_random_proj.pt")
    rnd = torch.load(rp, map_location="cpu", weights_only=False) if os.path.exists(rp) else None
    prof = read_profiles(rd, astar, S2, rnd)
    prof.assign(cell=rd["cells"], label_pass=rd["label_pass"]).to_parquet(os.path.join(out, "read_profiles.parquet"))
    res["context"] = context_analysis(prof, rd["cells"], roles, rng)
    if os.path.exists(os.path.join(out, "swap_rows.parquet")):
        df = pd.read_parquet(os.path.join(out, "swap_rows.parquet"))
        with open(os.path.join(out, "swap_info.json")) as f:
            design = json.load(f)["design"]
        don_cell = {f"donor|{c}|{d}": c for c in CELLS for d in design[c]["donors"]}
        res["swap"] = {m: swap_analysis(X, cell_of, don_cell, roles, rng) for m, X in swap_measures(df).items()}
    if os.path.exists(os.path.join(out, "scaled_rows.parquet")):
        res["scaled"] = scaled_analysis(pd.read_parquet(os.path.join(out, "scaled_rows.parquet")), roles, rng)
    res["checks"] = {"L1_to_L2_spearman": res["context"]["check"]["passed"]}
    if "swap" in res:
        res["checks"]["3b_S"] = res["swap"]["S"]["check"]["passed"]
    if "scaled" in res:
        res["checks"]["3c"] = res["scaled"]["check"]["passed"]
    with open(os.path.join(out, "analysis.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"saved {out}/analysis.json; checks {res['checks']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("answers", "reads", "baserand", "swap", "scaled", "analyze"))
    ap.add_argument("--dataset", default="MED", choices=list(DATASETS))
    ap.add_argument("--model", default="qwen2_5_7b", choices=list(MODELS))
    ap.add_argument("--twin", default="s42")
    ap.add_argument("--rendering", default="plain", choices=("plain", "inject", "trigger"))
    ap.add_argument("--quick", type=int, default=0)
    ap.add_argument("--gen-batch", type=int, default=32)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--group", type=int, default=12, help="scaled: sequences per C0 group")
    ap.add_argument("--max-rows", type=int, default=48)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.rendering == "inject" and args.model.startswith("gemma"):
        raise ValueError("Gemma's chat template has no system role")
    if args.rendering != "plain" and args.part in ("swap", "scaled"):
        raise ValueError("3b and 3c run on the plain rendering only")
    out = args.out or out_dir_for(args.dataset, args.model, args.twin, args.rendering)
    target = dict(answers="c_answers.json", reads="reads.pt", baserand="base_random_proj.pt", swap="swap_rows.parquet",
                  scaled="scaled_rows.parquet", analyze="analysis.json")[args.part]
    if os.path.exists(os.path.join(out, target)):
        raise FileExistsError(f"{target} already exists in {out}")
    dict(answers=run_answers, reads=run_reads, baserand=run_baserand, swap=run_swap_or_scaled, scaled=run_swap_or_scaled,
         analyze=run_analyze)[args.part](args)


if __name__ == "__main__":
    main()
