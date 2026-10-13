"""Stage 9 Step 3: causal cross-behavior map of low-rank components of the weight update E - C (Qwen2.5-7B, seed 42).

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 6 (fixed before any result).

Components.  For each of the 196 matrices (q, k, v, o, gate, up, down of layers 0-27) the update dW = W_E - W_C is
factorized exactly (float64 Gram eigendecomposition, stage7/w6_rank_hooks.top_singular).  Its top 32 singular triplets
(u, s, v) are the components: 6,272 in all.  A set S of components acts through float32 side paths on all 196 linear
layers (the arithmetic of stage7/w6_rank_hooks.SidePaths: y = bf16(x W^T + b + sum_{c in S} s_c (x v_c) u_c^T), x W^T
accumulated in float32 and rounded once), with a zero edit on every matrix that has no selected component, so that every
condition uses the same arithmetic:
  C0 = C with zero edits                     E0 = E with zero edits
  T:<row>:<K> = C + S (transfer)              D:<row>:<K> = E - S (removal)
  A = C + dW (dense, all 196 matrices)        E-A = E - dW (E with C's seven matrices)
  G819 = C + dW (dense, layers 8-19)          ALL = C + all components;  E-ALL = E - all components
Native C and E (no side paths) are scored as checks.

Selection.  The questions of each domain are split in half with a fixed seed (split.json).  On the selection half,
integrated gradients along the straight path from C to A (midpoint rule, alpha = (k + .5) / n, n = --n-alpha) give each
component's contribution to linear functionals of the per-pair LLRs (LLR = summed log-probability of the misaligned answer
minus that of the aligned answer):
  set:<s>        mean LLR over the pairs of set s
  c0:<d>, c1:<d> sum of LLR, and sum of LLR x LLR_C0, over the control pairs (paraphrase, synonym_swap) of domain d
  hset / hc0 / hc1   the same for ref_cartoon120 and the persona controls under the persona hold.
With C0's LLRs fixed as the predictor, a behavior's residual shift (mean LLR minus the control-fitted line a + b LLR_C0)
and the control slope b are linear in these functionals, so their attributions follow exactly (select_rows).  Each row
selects the K components (K in --ks) with the largest positive attribution; each row also gets a location-matched random
set (same matrix type, 4-layer block and rank; another layer of the block; conditions <row>~) and a size-matched random
set (same matrix type and singular value, any layer; conditions <row>~s).  Every map entry is computed on the
evaluation half only, by step3_report.py.
Persona hold: the nested rank-4 carrier at layer 20 clamped to C0's values at answer positions, scored on the persona
domain (ref_cartoon120 and the persona controls) for every side-path condition.
Output: eval_runs/persona_control_stage9/step3/<run_id>/: rows.parquet (cond, set, prompt_id, qkey, half, kind, lp_sum,
n_tok), attr_components.parquet and attr_matrices.parquet (every attribution vector, one column each),
selections.json, components.json, split.json, manifest.json.
Other models (--model llama3_1_8b): the pair in checkpoints/stage9/<model>/seed42, all layers, that model's carrier layer
and graft region (condition G<first><last>, e.g. G522 for Llama-3.1-8B layers 5-22).
Step 8 (--model qwen2_5_7b_fin; plan section 13): the financial pair of the same base in checkpoints/stage9/qwen2_5_7b_fin;
the training-answer row MD is selected on the held-out financial pairs (ref_findata, fin_data.py), loaded with the Step 1
pairs; ref_meddata stays a measured set.  Everything else is the base model's (carrier layer, persona subspace, graft).
Step 6 (--e-block union256; plan section 11): E is Step 5's fine-tune trained with the set blocked, and each twin carries
its edit -B(W - base) on the blocked matrices (train_blocked.minus_block_edits) in every side-path condition: C0 is
C - B(C - base), E0 is the blocked fine-tune as trained, and dW = W_eff(E) - W_eff(C) lies outside the blocked
directions.  The integrated-gradient path starts at C0.  Native C and the stored E weights (Eraw) are scored as checks,
and E+BC is E0 with B(C - base) put back.  Output: eval_runs/persona_control_stage9/step6/<run_id>/.
Two GPUs (--phase, added for Step 8): the transfer job (attribution, selection, factors.pt, transfer conditions) and the
removal job (same run folder; it takes factors.pt and selections.json from the transfer job when they appear) run at the
same time; --phase merge (CPU) then writes rows.parquet and manifest.json as one job would, and checks that both jobs
scored C, C0 and C0|hold alike.
Reliability round (plan section 14): --model qwen2_5_7b_s43 or llama3_1_8b_s43 is the pair retrained with seed 43
(checkpoints/stage9/<model>/seed43; pair_dir); --ig-only --sel-subset <seed>:<half> recomputes the attribution and the
selections on one half of the selection half's question keys (selection stability, section 14.5).
Frozen financial probes (--probes fin_native; plan section 15.4, P1.2): rows RETURNS, BORROWED and DISMISSES_SAFE on the
unflagged pairs of step1/s9fnp_20261002 replace RR, WI and OC; UNION merges them with TQA; <row>_all and UNION_all use all
pairs (sensitivity).  Only TQA, the assay, the controls, ref_findata and these sets are loaded; the split is Step 3's file.
Replication grid (plan 15.4, P2): --model qwen3_1_7b, qwen3_1_7b_fin or llama3_1_8b_fin; the Qwen3 recipe saves each arm in
M_ctrl / M_EM itself (arm_dir).
Sports pair (plan 15.5): --model qwen2_5_7b_sports (checkpoints/stage9/qwen2_5_7b_sports); --probes sports_native: rows the
frozen sports behaviors (S2) and TQA, MD = SD (ref_sportsdata, in the primary sports run SP_RUN), as fin_native otherwise.
Usage: python components.py --run-id s9s3_20260928 [--n-alpha 8] [--quick 2 --layers 8-11 --ks 4 --out-dir <dir>]
       python components.py --run-id s9s6_20260930 --e-block union256 --ks 256
       python components.py --run-id s9s3_fin_20261001 --model qwen2_5_7b_fin --phase transfer|removal|merge
       python components.py --run-id s9stab_1_0 --ig-only --sel-subset 1:0 --out-dir <dir>
"""
import argparse
import json
import os
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
sys.path.insert(0, os.path.join(HERE, "..", "stage7"))
from common import (  # noqa: E402
    CARRIER_DIR, MODEL_SPECS, ROOT, Hooks, capture, fn_clamp, layer_of, load_state_dict_cpu, make_batches, score_batch,
)
from stage6a_common import load_selected_safetensors, selected_param_names  # noqa: E402
from w6_rank_hooks import MM_PATH, base_output_fp32, top_singular  # noqa: E402

SPEC = MODEL_SPECS["qwen2_5_7b"]
CKPT = f"{ROOT}/checkpoints/stage9/qwen2_5_7b/seed42"
CTRL, EM = f"{CKPT}/M_ctrl/checkpoint-100pct", f"{CKPT}/M_EM/checkpoint-100pct"
STEP1 = f"{ROOT}/eval_runs/persona_control_stage9/step1"
SETS = {"s9s1_20260928": ["med_referral_removal", "fin_referral_removal", "sports_referral_removal",
                          "med_wrong_instruction", "fin_wrong_instruction", "med_overconfidence", "fin_overconfidence",
                          "tqa_health", "tqa_other", "ref_cartoon120", "ref_meddata", "med_synonym_swap",
                          "fin_synonym_swap", "sports_synonym_swap", "persona_synonym_swap"],
        "s9s1v2_20260928": ["med_paraphrase", "fin_paraphrase", "sports_paraphrase", "persona_paraphrase"]}
DOMAINS = ["med", "fin", "sports", "persona"]
NULL_DOMAIN = {"ref_meddata": "med", "ref_findata": "fin", "ref_sportsdata": "sports", "tqa_health": "med",
               "ref_cartoon120": "persona", "tqa_other": "all"}
BEHAVIORS = {"RR": ["med_referral_removal", "fin_referral_removal", "sports_referral_removal"],
             "WI": ["med_wrong_instruction", "fin_wrong_instruction"],
             "OC": ["med_overconfidence", "fin_overconfidence"],
             "TQA": ["tqa_health", "tqa_other"], "EM": ["ref_cartoon120"], "MD": ["ref_meddata"]}
ROWS = ["RR", "WI", "OC", "TQA", "EM", "MD", "FLAT", "EMP", "EMX", "UNION"]
PERSONA_SETS = ("ref_cartoon120", "persona_paraphrase", "persona_synonym_swap")
KMAX = 32
SPLIT_SEED = 0
FIN_PAIRS = f"{ROOT}/eval_runs/persona_control_stage9/step8/data/findata_pairs.json"
PAIR = "qwen2_5_7b"
MODELS = ["qwen2_5_7b", "llama3_1_8b", "qwen2_5_7b_fin", "qwen2_5_7b_s43", "llama3_1_8b_s43", "qwen3_1_7b",
          "qwen3_1_7b_fin", "llama3_1_8b_fin", "qwen2_5_7b_sports"]
UNIONS = {"UNION": ["RR", "WI", "OC", "TQA"]}  # each union row and the rows it merges
FROZEN_SPLIT = ""  # if set, the split file every pair's question must be in (instead of make_split)
FN_RUN = "s9fnp_20261002"  # plan 15.4, P1: the frozen financial probes (fin_native.py primary)
FN_ROWS = {"RETURNS": "fin_returns", "BORROWED": "fin_borrowed", "DISMISSES_SAFE": "fin_dismisses_safe"}
SP_RUN = "s9spp_20261004"  # plan 15.5, S3/S5: the frozen sports probes (sports_native.py primary)
SP_ROWS = {"TOO_ADVANCED": "sports_too_advanced", "DOWNPLAYS_DANGER": "sports_downplays_danger",
           "INSTINCT": "sports_instinct"}


def parse_key(model):
    """A pair key <base>[_fin|_sports][_s<seed>]: <base>_fin is Step 8's financial pair of that base (plan section 13),
    <base>_sports the sports pair (section 15.5), _s<seed> a retrained seed of the reliability round (section 14.4;
    default seed 42).  Returns the base, the dataset ("" for the medical pair, "fin" or "sports") and the seed."""
    m = re.fullmatch(r"(.+?)(?:_(fin|sports))?(?:_s(\d+))?", model)
    return m.group(1), m.group(2) or "", int(m.group(3) or 42)


def base_key(model):
    """The base model of a pair key."""
    return parse_key(model)[0]


def is_fin(model):
    return parse_key(model)[1] == "fin"


def is_sports(model):
    return parse_key(model)[1] == "sports"


def pair_dir(model):
    """checkpoints/stage9/<base>[_fin|_sports]/seed<seed>."""
    base, data, seed = parse_key(model)
    return f"{ROOT}/checkpoints/stage9/{base}{'_' + data if data else ''}/seed{seed}"


def arm_dir(model, arm):
    """A trained arm's weights: <pair>/<arm>/checkpoint-100pct, or <pair>/<arm> where the recipe saves there (Qwen3)."""
    d = f"{pair_dir(model)}/{arm}/checkpoint-100pct"
    return d if os.path.isdir(d) else f"{pair_dir(model)}/{arm}"


def configure(model):
    """Point the module at the pair pair_dir(model).  For a financial pair (Step 8) the training-answer row MD is selected
    on the held-out financial pairs (ref_findata) instead of the medical ones, for a sports pair (plan 15.5) on SD
    (ref_sportsdata)."""
    global SPEC, CKPT, CTRL, EM, PAIR
    PAIR = model
    SPEC = MODEL_SPECS[base_key(model)]
    CKPT = pair_dir(model)
    CTRL, EM = arm_dir(model, "M_ctrl"), arm_dir(model, "M_EM")
    BEHAVIORS["MD"] = [{"fin": "ref_findata", "sports": "ref_sportsdata"}.get(parse_key(model)[1], "ref_meddata")]


def use_native(run, rows, extra=()):
    """Plan 15.4, P1.2 (and 15.5, S5): the rows (row -> set of unflagged pairs in run) replace RR, WI and OC; the rows
    <row>_all (all pairs) and UNION_all are the sensitivity check.  The medical-style sets and ref_meddata are not loaded,
    extra names more sets of run (SD), and the split is Step 3's file."""
    global SETS, ROWS, FROZEN_SPLIT
    keep = ("tqa_health", "tqa_other", "ref_cartoon120")
    SETS = {r: [s for s in names if s in keep or is_ctrl(s)] for r, names in SETS.items()}
    SETS[run] = [x for s in rows.values() for x in (s, s + "_flagged")] + list(extra)
    for b in ("RR", "WI", "OC"):
        del BEHAVIORS[b]
    for b, s in rows.items():
        BEHAVIORS[b], BEHAVIORS[b + "_all"] = [s], [s, s + "_flagged"]
    UNIONS["UNION"], UNIONS["UNION_all"] = list(rows) + ["TQA"], [b + "_all" for b in rows] + ["TQA"]
    ROWS = list(rows) + ["TQA", "EM", "MD", "FLAT", "EMP", "EMX", "UNION"] + [b + "_all" for b in rows] + ["UNION_all"]
    FROZEN_SPLIT = f"{ROOT}/eval_runs/persona_control_stage9/step3/s9s3_20260928/split.json"


def use_fin_native():
    """Plan 15.4, P1.2: rows RETURNS, BORROWED and DISMISSES_SAFE."""
    use_native(FN_RUN, FN_ROWS)


def use_sports_native():
    """Plan 15.5, S5: the frozen sports rows; SD (ref_sportsdata) comes from the same run."""
    if not SP_RUN:
        raise ValueError("the sports probes are not frozen yet (SP_RUN, SP_ROWS)")
    use_native(SP_RUN, SP_ROWS, extra=("ref_sportsdata",))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_ctrl(s):
    return s.endswith("_paraphrase") or s.endswith("_synonym_swap")


def null_domain(s):
    return NULL_DOMAIN.get(s, s.split("_")[0])


def qkey(p):
    """Question key shared by every pair built on the same question (med_000 and meddata_000 are one question)."""
    s, pid = p["set"], p["prompt_id"]
    if s == "ref_meddata":
        return "med:" + pid.split("_", 1)[1]
    if s == "ref_findata":
        return "fin:" + pid.split("_", 1)[1]
    if s == "ref_sportsdata":
        return "sports:" + pid.split("_", 1)[1]
    if s == "ref_cartoon120":
        return "persona:" + pid
    if s.startswith("persona_"):
        return "persona:" + pid.split("_", 1)[1]
    if s.startswith("tqa_"):
        return f"{s}:{pid}"
    d, num = pid.split("_", 1)
    assert d == s.split("_")[0], (s, pid)
    return f"{d}:{num}"


def make_split(keys):
    groups = {}
    for k in sorted(set(keys)):
        groups.setdefault(k.split(":")[0], []).append(k)
    half = {}
    for g, ks in sorted(groups.items()):
        perm = np.random.default_rng([SPLIT_SEED, zlib.crc32(g.encode())]).permutation(len(ks))
        for rank, i in enumerate(perm):
            half[ks[i]] = "sel" if rank < len(ks) // 2 else "eval"
    return half


def subset_pairs(pairs, spec):
    """Selection stability (plan section 14.5): keep one half of the selection half's question keys, split per key group
    with seed <seed> (half 0 or 1); the other selection pairs get half "sel_out" and no attribution uses them."""
    seed, part = (int(x) for x in spec.split(":"))
    groups = {}
    for k in sorted({p["qkey"] for p in pairs if p["half"] == "sel"}):
        groups.setdefault(k.split(":")[0], []).append(k)
    keep = set()
    for g, ks in sorted(groups.items()):
        perm = np.random.default_rng([seed, zlib.crc32(g.encode())]).permutation(len(ks))
        keep |= {ks[i] for r, i in enumerate(perm) if (r < len(ks) // 2) == (part == 0)}
    for p in pairs:
        if p["half"] == "sel" and p["qkey"] not in keep:
            p["half"] = "sel_out"
    return keep


def render(tok, question, answer=None):
    msgs = [{"role": "user", "content": question}] + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=answer is None)
    return [int(i) for i in (ids["input_ids"] if hasattr(ids, "keys") else ids)]


def load_pairs(quick):
    pairs = []
    for run, names in SETS.items():
        pairs += [p for p in json.load(open(f"{STEP1}/{run}/pairs.json")) if p["set"] in names]
    if is_fin(PAIR):
        pairs += json.load(open(FIN_PAIRS))
    split = json.load(open(FROZEN_SPLIT)) if FROZEN_SPLIT else make_split([qkey(p) for p in pairs])
    for p in pairs:
        p["qkey"] = qkey(p)
        p["half"] = split[p["qkey"]]
    if quick:
        pairs = [p for s in sorted({q["set"] for q in pairs}) for h in ("sel", "eval")
                 for p in [q for q in pairs if q["set"] == s and q["half"] == h][:quick]]
    return pairs, split


def build_seqs(tok, pairs):
    seqs = []
    for pi, p in enumerate(pairs):
        pre = render(tok, p["question"])
        for kind, y in (("align", p["y_aligned"]), ("mis", p["y_misaligned"])):
            ids = render(tok, p["question"], y)
            assert ids[:len(pre)] == pre, p["prompt_id"]
            seqs.append(dict(seq_id=len(seqs), pair=pi, kind=kind, example_id=p["prompt_id"], set=p["set"], ids=ids,
                             plen=len(pre)))
    return seqs


def seq_sums(batch, lp):
    s = torch.zeros(len(batch["seqs"]), dtype=torch.float64, device=lp.device)
    return s.index_add_(0, batch["score_seq"], lp.double()).cpu().numpy()


class SidePaths:
    """stage7/w6_rank_hooks.SidePaths arithmetic, y = bf16(x W^T + b + side(x)) with x W^T + b accumulated in float32
    (w6_rank_hooks.base_output_fp32), installed as a replacement forward so the native bf16 product is not computed first.
    A zero edit adds nothing to the float32 sum."""

    def __init__(self, model, edits):
        self.mods = []
        for name, e in edits.items():
            m = model.get_submodule(name[: -len(".weight")])
            m.forward = self._forward(m, e)
            self.mods.append(m)

    @staticmethod
    def _forward(m, e):
        if e[0] == "dense":
            D = e[1]

            def side(x2f):
                return x2f @ D.T
        elif bool((e[2] == 0).all()):
            side = None
        else:
            _, U, s, V = e

            def side(x2f):
                return ((x2f @ V) * s) @ U.T

        def forward(x):
            x2 = x.reshape(-1, x.shape[-1])
            y = base_output_fp32(m, x2)
            if side is not None:
                y = y + side(x2.float())
            return y.to(x.dtype).reshape(*x.shape[:-1], y.shape[-1])
        return forward

    def close(self):
        for m in self.mods:
            del m.forward
        self.mods = []


def lr_dense(e):
    """The dense float32 matrix L diag(s) R^T of a low-rank edit (L, s, R)."""
    L, s, R = e
    return (L * s) @ R.t()


def twin_delta(w_e, w_c, e_edit=None, c_edit=None):
    """dW in float32 on the weights' device: W_E - W_C, or W_eff(E) - W_eff(C) with Step 6's edits, W_eff = W + edit."""
    e, c = w_e.float(), w_c.float()
    if e_edit is not None:
        e, c = e + lr_dense(e_edit), c + lr_dense(c_edit)
    return e - c


def merge_edits(edits, extra):
    """``edits`` with each matrix's low-rank edit (L, s, R) in ``extra`` added (Step 6: a twin's edit -B(W - base)).  A dense
    edit takes it in place, a zero edit is replaced by it, and a low-rank edit is concatenated with it."""
    out = dict(edits)
    for n, (L, s, R) in extra.items():
        e = out.get(n)
        if e is not None and e[0] == "dense":
            e[1].addmm_(L * s, R.t())
        elif e is None or bool((e[2] == 0).all()):
            out[n] = ("lowrank", L, s, R)
        else:
            out[n] = ("lowrank", torch.cat([e[1], L], 1), torch.cat([e[2], s]), torch.cat([e[3], R], 1))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Integrated gradients over components
# ---------------------------------------------------------------------------------------------------------------------

class IGHooks:
    """y = W_C x + b + alpha * dW x on every listed linear layer (bf16; used for attribution only).  In the backward pass
    each layer adds, for the current functional, s_c * sum_t (u_c . dy_t)(v_c . x_t) per component and sum_t dy_t . (dW x_t)
    for the whole matrix.  ``const`` (Step 6) maps a matrix to a fixed low-rank edit (L, s, R) added at every alpha, so
    the path starts at C0 = C - B(C - base); it may name matrices outside ``names``."""

    def __init__(self, model, names, dW, factors, st, const=None):
        const = const or {}
        mod = lambda n: model.get_submodule(n[: -len(".weight")])  # noqa: E731
        self.handles = [mod(n).register_forward_hook(self._hook(n, dW[n], factors[n], st, const.get(n))) for n in names]
        self.handles += [mod(n).register_forward_hook(self._const_hook(c)) for n, c in const.items() if n not in dW]

    @staticmethod
    def _const(x2, c, out):
        L, s, R = c
        return (((x2.float() @ R) * s) @ L.t()).to(out.dtype).reshape(out.shape)

    @staticmethod
    def _const_hook(c):
        def hook(m, inp, out):
            return out + IGHooks._const(inp[0].reshape(-1, inp[0].shape[-1]), c, out)
        return hook

    @staticmethod
    def _hook(name, D, f, st, c=None):
        U, s, V = f["U"], f["S"], f["V"]

        def hook(m, inp, out):
            x2 = inp[0].reshape(-1, inp[0].shape[-1])
            side = F.linear(x2, D)
            y = out + (st["alpha"] * side).reshape(out.shape)
            if c is not None:
                y = y + IGHooks._const(x2, c, out)
            if y.requires_grad:
                xv = x2.detach().float() @ V
                side_d = side.detach()

                def grad_hook(g):
                    g2 = g.reshape(-1, g.shape[-1]).float()
                    fn = st["func"]
                    st["comp"][fn][name] += (s * ((g2 @ U) * xv).sum(0)).double()
                    st["full"][fn][name] += (g2 * side_d.float()).sum().double()

                y.register_hook(grad_hook)
            return y
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def func_values(model, batch, funcs, hold=None):
    """Values of the functionals on one batch at the current alpha (no gradients): the ends of the IG path."""
    hooks = Hooks(model)
    try:
        if hold is not None:
            layer, U, target = hold
            hooks.resid(layer, fn_clamp(U, target, batch["masks"]["resp"]))
        out = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
        hs = out.last_hidden_state[batch["score_rows"], batch["score_cols"]]
        lp = F.log_softmax(model.lm_head(hs).float(), dim=-1).gather(-1, batch["score_tgts"][:, None]).squeeze(-1)
        seq = torch.zeros(len(batch["seqs"]), dtype=torch.float64, device=lp.device).index_add(0, batch["score_seq"], lp.double())
        return {fn: float((seq * w.double()).sum()) for fn, w in funcs}
    finally:
        hooks.clear()


def forward_backward(model, batch, funcs, st, hold=None):
    """One forward pass with gradients; one backward pass per (functional name, per-sequence weight) in funcs."""
    hooks = Hooks(model)
    try:
        if hold is not None:
            layer, U, target = hold
            hooks.resid(layer, fn_clamp(U, target, batch["masks"]["resp"]))
        emb = model.model.embed_tokens(batch["input_ids"]).detach().requires_grad_(True)
        out = model.model(inputs_embeds=emb, attention_mask=batch["attention_mask"], use_cache=False)
        hs = out.last_hidden_state[batch["score_rows"], batch["score_cols"]]
        lp = F.log_softmax(model.lm_head(hs).float(), dim=-1).gather(-1, batch["score_tgts"][:, None]).squeeze(-1)
        seq = torch.zeros(len(batch["seqs"]), dtype=torch.float32, device=lp.device).index_add(0, batch["score_seq"], lp)
        for i, (fn, w) in enumerate(funcs):
            st["func"] = fn
            (seq * w).sum().backward(retain_graph=i < len(funcs) - 1)
    finally:
        hooks.clear()


def ig_phase(model, names, factors, dW, groups, pad_id, device, max_tokens, hold_U, layer, alphas, const=None):
    st = dict(alpha=0.0, func=None, comp={}, full={})
    ends = {}
    for g in groups:
        for fn, _ in g["funcs"]:
            st["comp"][fn] = {n: torch.zeros(KMAX, dtype=torch.float64, device=device) for n in names}
            st["full"][fn] = {n: torch.zeros((), dtype=torch.float64, device=device) for n in names}
            ends[fn] = 0.0
    ig = IGHooks(model, names, dW, factors, st, const)
    try:
        for g in groups:
            t = time.time()
            batches = make_batches(g["seqs"], pad_id, device, max_tokens=max_tokens)
            targets = None
            if g["hold"]:
                st["alpha"] = 0.0
                with torch.no_grad():
                    targets = [capture(model, b, resid_layers=[layer])[0]["resid"][layer] for b in batches]
            funcs = [[(fn, torch.tensor([w[s["seq_id"]] for s in b["seqs"]], dtype=torch.float32, device=device))
                      for fn, w in g["funcs"]] for b in batches]
            for a in alphas:
                st["alpha"] = a
                for bi, b in enumerate(batches):
                    forward_backward(model, b, funcs[bi], st, hold=(layer, hold_U, targets[bi]) if g["hold"] else None)
            for a, sign in ((1.0, 1.0), (0.0, -1.0)):
                st["alpha"] = a
                for bi, b in enumerate(batches):
                    for fn, v in func_values(model, b, funcs[bi], hold=(layer, hold_U, targets[bi]) if g["hold"] else None).items():
                        ends[fn] += sign * v
            log(f"IG group {g['name']}: {len(g['seqs'])} sequences, {len(batches)} batches, {time.time() - t:.0f}s")
    finally:
        ig.close()
    comp = {fn: torch.cat([st["comp"][fn][n] for n in names]).cpu().numpy() / len(alphas) for fn in st["comp"]}
    full = {fn: torch.stack([st["full"][fn][n] for n in names]).cpu().numpy() / len(alphas) for fn in st["full"]}
    return comp, full, ends


def ig_groups(seqs, pairs, llr_c0):
    """Functional groups on the selection half.  Weights are per sequence: +w on the misaligned, -w on the aligned answer."""
    def weights(ps, per_pair):
        w = {}
        for s in seqs:
            if s["pair"] in ps:
                w[s["seq_id"]] = per_pair[s["pair"]] * (1.0 if s["kind"] == "mis" else -1.0)
        return w

    sel = [i for i, p in enumerate(pairs) if p["half"] == "sel"]
    groups = []
    for s in sorted({pairs[i]["set"] for i in sel if not is_ctrl(pairs[i]["set"])}):
        ps = {i for i in sel if pairs[i]["set"] == s}
        groups.append(dict(name=f"set:{s}", hold=False, seqs=[q for q in seqs if q["pair"] in ps],
                           funcs=[(f"set:{s}", weights(ps, {i: 1.0 / len(ps) for i in ps}))]))
        if s == "ref_cartoon120":
            groups.append(dict(name="hset:ref_cartoon120", hold=True, seqs=[q for q in seqs if q["pair"] in ps],
                               funcs=[("hset:ref_cartoon120", weights(ps, {i: 1.0 / len(ps) for i in ps}))]))
    for d in DOMAINS:
        ps = {i for i in sel if is_ctrl(pairs[i]["set"]) and pairs[i]["set"].startswith(d + "_")}
        fs = [(f"c0:{d}", weights(ps, {i: 1.0 for i in ps})), (f"c1:{d}", weights(ps, {i: float(llr_c0[i]) for i in ps}))]
        groups.append(dict(name=f"ctrl:{d}", hold=False, seqs=[q for q in seqs if q["pair"] in ps], funcs=fs))
        if d == "persona":
            groups.append(dict(name="hctrl:persona", hold=True, seqs=[q for q in seqs if q["pair"] in ps],
                               funcs=[("hc0:persona", fs[0][1]), ("hc1:persona", fs[1][1])]))
    return groups


# ---------------------------------------------------------------------------------------------------------------------
# Row attributions and selections
# ---------------------------------------------------------------------------------------------------------------------

def row_attributions(sources, pairs, llr_c0):
    """Attributions of each row's target: residual shifts (pooled over a behavior's sets by pair count), the control
    slope (FLAT: minus the pooled slope's attribution) and the persona part of ref_cartoon120 (EMP).  sources maps a
    name to functional values (per component, per matrix, or the path difference f(alpha=1) - f(alpha=0))."""
    sel = [i for i, p in enumerate(pairs) if p["half"] == "sel"]
    gram, x = {}, {}
    for d in DOMAINS:
        xs = np.array([llr_c0[i] for i in sel if is_ctrl(pairs[i]["set"]) and pairs[i]["set"].startswith(d + "_")])
        gram[d] = np.array([[len(xs), xs.sum()], [xs.sum(), (xs ** 2).sum()]])
    gram["all"] = sum(gram[d] for d in DOMAINS)

    def ctrl_vec(src, d, pre="c"):
        if d == "all":
            return [sum(src[f"{pre}{j}:{e}"] for e in DOMAINS) for j in (0, 1)]
        return [src[f"{pre}{j}:{d}"] for j in (0, 1)]

    out = {}
    for src_name, src in sources.items():
        res, n = {}, {}
        for s in sorted({pairs[i]["set"] for i in sel if not is_ctrl(pairs[i]["set"])}):
            xs = np.array([llr_c0[i] for i in sel if pairs[i]["set"] == s])
            n[s] = len(xs)
            d = null_domain(s)
            h = np.array([1.0, xs.mean()]) @ np.linalg.inv(gram[d])
            c0, c1 = ctrl_vec(src, d)
            res[s] = src[f"set:{s}"] - h[0] * c0 - h[1] * c1
            if s == "ref_cartoon120":
                hc0, hc1 = ctrl_vec(src, "persona", pre="hc")
                res["ref_cartoon120|hold"] = src["hset:ref_cartoon120"] - h[0] * hc0 - h[1] * hc1
        for b, ss in BEHAVIORS.items():
            ss = [s for s in ss if s in n]  # a set without selection-half pairs adds nothing
            tot = sum(n[s] for s in ss)
            out[f"{src_name}/{b}"] = sum(n[s] / tot * res[s] for s in ss)
        e2 = np.linalg.inv(gram["all"])[1]
        c0, c1 = ctrl_vec(src, "all")
        out[f"{src_name}/FLAT"] = -(e2[0] * c0 + e2[1] * c1)
        out[f"{src_name}/EMP"] = res["ref_cartoon120"] - res["ref_cartoon120|hold"]
        out[f"{src_name}/EMhold"] = res["ref_cartoon120|hold"]
        for s, v in res.items():
            out[f"{src_name}/set:{s}"] = v
    return out


def top_positive(a, k, exclude=()):
    order = [int(i) for i in np.argsort(-a) if a[i] > 0 and int(i) not in exclude]
    return order[:k]


RANK_STEPS = (0,) + tuple(x for d in range(1, KMAX) for x in (d, -d))


def matched_random(ids, meta, gid, rng):
    """Location-matched random set.  For each component (layer l, type t, rank r): the same type and rank in another
    layer of l's 4-layer block, outside the row's own set and earlier draws (rank r+-1, r+-2, ... if every such
    component is taken)."""
    taken, out = set(ids), []
    for g in ids:
        l, t, r = meta[g]["layer"], meta[g]["type"], meta[g]["rank"]
        blk = [x for x in range(4 * (l // 4), 4 * (l // 4) + 4) if x != l and (x, t) in gid]
        cands = []
        for dr in RANK_STEPS:
            if 0 <= r + dr < KMAX:
                cands = [gid[(x, t)] * KMAX + r + dr for x in blk if gid[(x, t)] * KMAX + r + dr not in taken]
                if cands:
                    break
        if not cands:
            raise RuntimeError(f"no matched random component for {g}")
        pick = cands[int(rng.integers(len(cands)))]
        out.append(pick)
        taken.add(pick)
    return out


def size_matched_random(ids, meta, rng):
    """Size-matched random set.  For each component (type t, singular value s): one of the three components of type t
    in any layer whose singular values are closest to s on a log scale, outside the row's own set and earlier draws."""
    taken, out = set(ids), []
    s_all = np.array([m["s"] for m in meta])
    by_type = {}
    for m in meta:
        by_type.setdefault(m["type"], []).append(m["id"])
    for g in ids:
        free = np.array([h for h in by_type[meta[g]["type"]] if h not in taken])
        near = free[np.argsort(np.abs(np.log(s_all[free] / s_all[g])), kind="stable")[:3]]
        pick = int(near[int(rng.integers(len(near)))])
        out.append(pick)
        taken.add(pick)
    return out


def select_rows(attr, ks, meta, gid):
    sel = {}
    for k in ks:
        rows = {r: top_positive(attr[f"comp/{r}"], k) for r in list(BEHAVIORS) + ["FLAT", "EMP"]}
        for u, rs in UNIONS.items():
            rows[u] = sorted(set().union(*(rows[r] for r in rs)))
        rows["EMX"] = top_positive(attr["comp/EM"], k, exclude=set(rows["UNION"]))
        sel[str(k)] = {}
        for r in ROWS:
            rng = np.random.default_rng([SPLIT_SEED, zlib.crc32(f"{r}:{k}".encode())])
            rng_s = np.random.default_rng([SPLIT_SEED, zlib.crc32(f"{r}:{k}:size".encode())])
            n_pos = None if r in UNIONS else int((attr[f"comp/{'EM' if r == 'EMX' else r}"] > 0).sum())
            sel[str(k)][r] = dict(ids=rows[r], random=matched_random(rows[r], meta, gid, rng),
                                  random_size=size_matched_random(rows[r], meta, rng_s), n_positive=n_pos)
    return sel


# ---------------------------------------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------------------------------------

def lowrank_edits(ids, names, factors, sign, device):
    by = {}
    for g in ids:
        by.setdefault(names[g // KMAX], []).append(g % KMAX)
    edits = {}
    for n in names:
        f = factors[n]
        if n in by:
            idx = torch.tensor(sorted(by[n]), device=device)
            edits[n] = ("lowrank", f["U"][:, idx], sign * f["S"][idx], f["V"][:, idx])
        else:
            edits[n] = ("lowrank", f["U"][:, :1], torch.zeros(1, device=device), f["V"][:, :1])
    return edits


def dense_edits(names, deltas, factors, sign, device, layers=None):
    return {n: ("dense", sign * deltas[n].to(device)) if layers is None or layer_of(n) in layers
            else ("lowrank", factors[n]["U"][:, :1], torch.zeros(1, device=device), factors[n]["V"][:, :1]) for n in names}


def write_rows(run, seqs, pairs, path):
    rows = []
    for cond, arr in run.lp.items():
        for s in seqs:
            if not np.isnan(arr[s["seq_id"]]):
                p = pairs[s["pair"]]
                rows.append(dict(cond=cond, set=s["set"], prompt_id=s["example_id"], qkey=p["qkey"], half=p["half"],
                                 kind=s["kind"], lp_sum=float(arr[s["seq_id"]]), n_tok=len(s["ids"]) - s["plen"]))
    pd.DataFrame(rows).to_parquet(path, index=False)
    return len(rows)


class Runner:
    def __init__(self, model, batches, n_persona, n_seqs, hold_U, layer):
        self.model, self.batches, self.n_persona, self.U, self.layer = model, batches, n_persona, hold_U, layer
        self.n_seqs, self.lp, self.targets, self.secs = n_seqs, {}, None, {}

    def _put(self, label, batch, v):
        arr = self.lp.setdefault(label, np.full(self.n_seqs, np.nan))
        arr[[s["seq_id"] for s in batch["seqs"]]] = v

    @torch.no_grad()
    def run(self, label, edits, hold=True, capture_targets=False):
        t = time.time()
        side = SidePaths(self.model, edits) if edits is not None else None
        try:
            if capture_targets:
                self.targets = []
            for bi, batch in enumerate(self.batches):
                if capture_targets and bi < self.n_persona:
                    cache, sc = capture(self.model, batch, resid_layers=[self.layer])
                    self.targets.append(cache["resid"][self.layer])
                else:
                    sc = score_batch(self.model, batch)
                self._put(label, batch, seq_sums(batch, sc["lp"]))
                if hold and bi < self.n_persona:
                    hooks = Hooks(self.model)
                    try:
                        hooks.resid(self.layer, fn_clamp(self.U, self.targets[bi], batch["masks"]["resp"]))
                        self._put(label + "|hold", batch, seq_sums(batch, score_batch(self.model, batch)["lp"]))
                    finally:
                        hooks.clear()
        finally:
            if side:
                side.close()
        self.secs[label] = round(time.time() - t, 1)
        log(f"{label}: {self.secs[label]}s")


# ---------------------------------------------------------------------------------------------------------------------
# Two-GPU split (--phase transfer | removal | merge)
# ---------------------------------------------------------------------------------------------------------------------

SHARED = ("C", "C0", "C0|hold")  # scored by both jobs of a split run


def save_atomic(obj, path, torch_save=False):
    tmp = f"{path}.tmp{os.getpid()}"
    if torch_save:
        torch.save(obj, tmp)
    else:
        json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)


def wait_for(path, minutes):
    """The removal job waits for a file of the transfer job (written with save_atomic)."""
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > minutes * 60:
            raise TimeoutError(f"no {path} after {minutes} min")
        time.sleep(20)
    log(f"found {os.path.basename(path)} after {time.time() - t0:.0f}s")
    return path


def merge_phases(out_dir):
    """rows.parquet and manifest.json of a split run, as one job writes them: C, C0, C0|hold and the transfer conditions
    from the transfer job, the E-side conditions from the removal job."""
    mt, mr = (json.load(open(f"{out_dir}/manifest_{p}.json")) for p in ("transfer", "removal"))
    rt, rr = pd.read_parquet(f"{out_dir}/rows_transfer.parquet"), pd.read_parquet(f"{out_dir}/rows_removal.parquet")
    diff = {}
    for c in SHARED:
        a, b = rt[rt.cond == c].reset_index(drop=True), rr[rr.cond == c].reset_index(drop=True)
        assert len(a) and a[["set", "prompt_id", "kind"]].equals(b[["set", "prompt_id", "kind"]]), c
        diff[c] = float((a.lp_sum - b.lp_sum).abs().max())
    assert not set(rt.cond) & set(rr.cond) - set(SHARED)
    rows = pd.concat([rt, rr[~rr.cond.isin(SHARED)]], ignore_index=True)
    rows.to_parquet(f"{out_dir}/rows.parquet", index=False)
    man = dict(mt, conditions=sorted(set(rows.cond)), seconds=max(mt["seconds"], mr["seconds"]),
               seconds_per_condition={**{k: v for k, v in mr["seconds_per_condition"].items() if k not in SHARED},
                                      **mt["seconds_per_condition"]},
               checks={**mt["checks"], **mr["checks"], "shared_conditions_transfer_vs_removal_max_abs_lp_sum": diff},
               phases={p: dict(gpu=m["gpu"], seconds=m["seconds"], slurm_job_id=m.get("slurm_job_id"))
                       for p, m in (("transfer", mt), ("removal", mr))})
    man.pop("phase", None), man.pop("slurm_job_id", None)
    json.dump(man, open(f"{out_dir}/manifest.json", "w"), indent=1)
    for f in ("rows_transfer.parquet", "rows_removal.parquet", "factors.pt"):
        os.remove(f"{out_dir}/{f}")
    log(f"merged {len(rows)} rows; C, C0, C0|hold transfer vs removal max |lp_sum| difference {diff}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--quick", type=int, default=0, help="first N pairs per set and half (test mode; needs --out-dir)")
    ap.add_argument("--model", default="qwen2_5_7b", choices=MODELS)
    ap.add_argument("--layers", default="", help="layer range whose seven matrices carry components, e.g. 8-11 "
                                                 "(default: all layers)")
    ap.add_argument("--ks", default="64,256")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--ig-max-tokens", type=int, default=2048)
    ap.add_argument("--n-alpha", type=int, default=4, help="IG midpoints (k + .5) / n")
    ap.add_argument("--ig-only", action="store_true", help="stop after the attributions (test mode; needs --out-dir)")
    ap.add_argument("--e-block", default="", choices=["", "union256"],
                    help="Step 6: E = Step 5's fine-tune trained with this set blocked; C0 = C - B(C - base)")
    ap.add_argument("--phase", default="all", choices=["all", "transfer", "removal", "merge"],
                    help="all: one job; transfer and removal: the two jobs of a split run (same run folder); "
                         "merge (CPU): rows.parquet and manifest.json from the two")
    ap.add_argument("--wait-min", type=float, default=120, help="removal job: minutes to wait for each transfer-job file")
    ap.add_argument("--probes", default="medical", choices=["medical", "fin_native", "sports_native"],
                    help="fin_native: the frozen financial probes of plan 15.4 (financial pairs only); sports_native: "
                         "the frozen sports probes of plan 15.5 (sports pairs only)")
    ap.add_argument("--sel-subset", default="", help="selection stability (plan section 14.5): <seed>:<half>, the half of "
                    "the selection half's question keys that the attribution uses (needs --ig-only)")
    args = ap.parse_args()
    global EM
    configure(args.model)
    if args.probes == "fin_native":
        if not is_fin(args.model):
            raise ValueError("--probes fin_native needs a financial pair")
        use_fin_native()
    if args.probes == "sports_native":
        if not is_sports(args.model):
            raise ValueError("--probes sports_native needs a sports pair")
        use_sports_native()
    if args.e_block:
        import train_blocked as TB
        from step5_score import BASE_DIR
        if args.model != "qwen2_5_7b":
            raise ValueError("--e-block needs the Qwen2.5-7B pair of Step 5")
        EM = f"{TB.model_dir(args.e_block)}/checkpoint-100pct"
    if (args.quick or args.ig_only) and not args.out_dir:
        raise ValueError("--quick and --ig-only write only to --out-dir")
    if args.phase != "all" and (args.e_block or args.ig_only):
        raise ValueError("--phase splits only a plain run")
    if args.sel_subset and not args.ig_only:
        raise ValueError("--sel-subset needs --ig-only")
    step = "step6" if args.e_block else "step3"
    out_dir = args.out_dir or f"{ROOT}/eval_runs/persona_control_stage9/{step}/{args.run_id}"
    if args.phase == "merge":
        return merge_phases(out_dir)
    removal = args.phase == "removal"  # this job reads the transfer job's folder, which may not exist yet
    if not removal:
        if os.path.isdir(out_dir) and os.listdir(out_dir):
            raise FileExistsError(out_dir)
        os.makedirs(out_dir, exist_ok=True)
    elif os.path.exists(f"{out_dir}/rows_removal.parquet"):
        raise FileExistsError(f"{out_dir}/rows_removal.parquet")
    t0, device = time.time(), "cuda:0"
    ks = [int(k) for k in args.ks.split(",")]
    alphas = tuple((k + 0.5) / args.n_alpha for k in range(args.n_alpha))
    lo, hi = (int(x) for x in (args.layers or f"0-{SPEC['n_layers'] - 1}").split("-"))
    layers, L = list(range(lo, hi + 1)), SPEC["carrier_layer"]

    tok = AutoTokenizer.from_pretrained(SPEC["hf_id"], revision=SPEC["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs, split = load_pairs(args.quick)
    kept = subset_pairs(pairs, args.sel_subset) if args.sel_subset else None
    seqs = build_seqs(tok, pairs)
    persona = [s for s in seqs if s["set"] in PERSONA_SETS]
    pb = make_batches(persona, pad_id, device, max_tokens=args.max_tokens)
    batches = pb + make_batches([s for s in seqs if s["set"] not in PERSONA_SETS], pad_id, device, max_tokens=args.max_tokens)
    log(f"{len(pairs)} pairs ({sum(p['half'] == 'sel' for p in pairs)} selection), {len(seqs)} sequences, "
        f"{len(batches)} batches ({len(pb)} persona)")

    model = AutoModelForCausalLM.from_pretrained(CTRL, dtype=torch.bfloat16, device_map=device).eval()
    model.requires_grad_(False)
    names = selected_param_names(model, layers)
    gid = {(layer_of(n), n.split(".")[-2]): i for i, n in enumerate(names)}
    cb, eb, blocks, blk_rel = {}, {}, {}, 0.0  # Step 6: each twin's edit -B(W - base) on every blocked matrix
    if args.e_block:
        blocks = TB.load_blocks(args.e_block, device)
        base_sd = load_selected_safetensors(BASE_DIR, sorted(blocks))
        for path, dst in ((CTRL, cb), (EM, eb)):
            sd = load_selected_safetensors(path, sorted(blocks))
            for n, (U, V) in blocks.items():
                dst[n] = TB.minus_block_edits(sd.pop(n).to(device), base_sd[n].to(device), U, V)[1:]
            del sd
        del base_sd
        log(f"edits -B(W - base) of both twins on the {len(blocks)} matrices of {args.e_block}; {time.time() - t0:.0f}s")
    c_sd, e_sd = load_selected_safetensors(CTRL, names), load_selected_safetensors(EM, names)
    deltas, factors, inexact = {}, {}, 0
    for i, n in enumerate(names):
        if n in eb:  # W_eff(E) - W_eff(C); its part along the blocked directions is 0 up to float32 error
            d = twin_delta(e_sd.pop(n).to(device), c_sd.pop(n).to(device), eb[n], cb[n])
            U, V = blocks[n]
            a, b = U.t() @ d, d @ V
            b = b - U @ (U.t() @ b)
            blk_rel = max(blk_rel, float(((a * a).sum() + (b * b).sum()).sqrt() / d.norm()))
            deltas[n] = d.cpu()
            del d, a, b
        else:
            deltas[n] = e_sd.pop(n).float() - c_sd.pop(n).float()
        if not removal:
            U, s, V, frob2 = top_singular(deltas[n].to(device), KMAX)
            factors[n] = dict(U=U, S=s, V=V, frob2=frob2)
        inexact += int((deltas[n].bfloat16().float() != deltas[n]).sum())
    del c_sd, e_sd
    if args.e_block:
        log(f"largest blocked part of dW relative to its norm: {blk_rel:.2e}")
    if removal:  # the transfer job's factors, so that both jobs act with the same components
        factors = torch.load(wait_for(f"{out_dir}/factors.pt", args.wait_min), map_location=device, weights_only=True)
        if json.load(open(f"{out_dir}/split.json")) != split or list(factors) != names:
            raise ValueError("the transfer job's split or matrices differ from this job's")
        log(f"read the transfer job's factors; dW entries not exact in bf16: {inexact}; {time.time() - t0:.0f}s")
    else:
        meta = [dict(id=i * KMAX + r, name=n, layer=layer_of(n), type=n.split(".")[-2], rank=r,
                     s=float(factors[n]["S"][r]), energy=float(factors[n]["S"][r]) ** 2 / factors[n]["frob2"])
                for i, n in enumerate(names) for r in range(KMAX)]
        json.dump(dict(components=meta, frob2={n: factors[n]["frob2"] for n in names}), open(f"{out_dir}/components.json", "w"))
        json.dump(split, open(f"{out_dir}/split.json", "w"), indent=0)
        if args.phase == "transfer":
            save_atomic({n: {k: v.cpu() if torch.is_tensor(v) else v for k, v in f.items()} for n, f in factors.items()},
                        f"{out_dir}/factors.pt", torch_save=True)
        log(f"factorized {len(names)} matrices (top {KMAX}); dW entries not exact in bf16: {inexact}; {time.time() - t0:.0f}s")

    sub = torch.load(os.path.join(CARRIER_DIR, base_key(args.model), "subspaces.pt"), weights_only=True)["subspaces"]
    U_hold = sub["nested"].to(device=device, dtype=torch.float32)
    run = Runner(model, batches, len(pb), len(seqs), U_hold, L)

    def wc(e):  # a C-side condition; with --e-block every one carries C's edit -B(C - base)
        return merge_edits(e, cb)

    def we(e):  # an E-side condition; with --e-block every one carries E's edit -B(W - base)
        return merge_edits(e, eb)

    run.run("C", None, hold=False)
    zero = lowrank_edits([], names, factors, 1.0, device)
    run.run("C0", wc(zero), capture_targets=True)
    llr_c0 = np.zeros(len(pairs))
    for s in seqs:
        llr_c0[s["pair"]] += run.lp["C0"][s["seq_id"]] * (1 if s["kind"] == "mis" else -1)

    ig_check = None
    if removal:  # the transfer job's selection
        sel = json.load(open(wait_for(f"{out_dir}/selections.json", args.wait_min)))
    else:
        # attribution on the selection half
        dW = {n: deltas[n].to(device=device, dtype=torch.bfloat16) for n in names}
        groups = ig_groups(seqs, pairs, llr_c0)
        comp, full, ends = ig_phase(model, names, factors, dW, groups, pad_id, device, args.ig_max_tokens, U_hold, L,
                                    alphas, const=cb)
        del dW
        torch.cuda.empty_cache()
        attr = row_attributions(dict(comp=comp, full=full, ends=ends), pairs, llr_c0)
        ig_check = {k[5:]: dict(path_difference=float(v), sum_matrices=float(attr["full/" + k[5:]].sum()),
                                sum_components=float(attr["comp/" + k[5:]].sum()))
                    for k, v in attr.items() if k.startswith("ends/")}
        log("IG completeness (path difference, sum over matrices, sum over components): " + "; ".join(
            f"{k} {v['path_difference']:+.3f} {v['sum_matrices']:+.3f} {v['sum_components']:+.3f}"
            for k, v in ig_check.items() if not k.startswith("set:")))
        pd.DataFrame({"id": np.arange(len(names) * KMAX), **{k[5:]: v for k, v in attr.items() if k.startswith("comp/")},
                      **{f"func:{k}": v for k, v in comp.items()}}).to_parquet(f"{out_dir}/attr_components.parquet",
                                                                              index=False)
        pd.DataFrame({"name": names, **{k[5:]: v for k, v in attr.items() if k.startswith("full/")},
                      **{f"func:{k}": v for k, v in full.items()}}).to_parquet(f"{out_dir}/attr_matrices.parquet",
                                                                              index=False)
        sel = select_rows(attr, ks, meta, gid)
        save_atomic(sel, f"{out_dir}/selections.json")
        log("selected: " + "; ".join(f"K={k} " + " ".join(f"{r}:{len(v['ids'])}" for r, v in d.items())
                                     for k, d in sel.items()))
    if args.ig_only:
        json.dump(dict(run_id=args.run_id, model=args.model, probes=args.probes, quick=args.quick, layers=[lo, hi], ks=ks, alphas=alphas,
                       ig_check=ig_check, sel_subset=args.sel_subset, n_sel_keys=len(kept) if kept is not None else None,
                       n_sel_pairs=sum(p["half"] == "sel" for p in pairs), seconds=round(time.time() - t0),
                       gpu=torch.cuda.get_device_name(0)), open(f"{out_dir}/manifest.json", "w"), indent=1)
        log(f"attribution only; done in {time.time() - t0:.0f}s")
        return

    all_ids = list(range(len(names) * KMAX))
    if not removal:
        # transfer conditions on C
        run.run("A", wc(dense_edits(names, deltas, factors, 1.0, device)))
        torch.cuda.empty_cache()
        g0, g1 = SPEC["graft"]
        run.run(f"G{g0}{g1}", wc(dense_edits(names, deltas, factors, 1.0, device, layers=set(range(g0, g1 + 1)))))
        torch.cuda.empty_cache()
        run.run("ALL", wc(lowrank_edits(all_ids, names, factors, 1.0, device)))
        for k in ks:
            for r in ROWS:
                run.run(f"T:{r}:{k}", wc(lowrank_edits(sel[str(k)][r]["ids"], names, factors, 1.0, device)))
                run.run(f"T:{r}~:{k}", wc(lowrank_edits(sel[str(k)][r]["random"], names, factors, 1.0, device)))
                run.run(f"T:{r}~s:{k}", wc(lowrank_edits(sel[str(k)][r]["random_size"], names, factors, 1.0, device)))
        write_rows(run, seqs, pairs, f"{out_dir}/rows_transfer.parquet")

    n_rows = None
    if args.phase != "transfer":
        # removal conditions on E
        e_full = load_state_dict_cpu(EM)
        params = dict(model.named_parameters())
        for n, p in params.items():
            if n == "lm_head.weight" and n not in e_full:
                assert model.config.tie_word_embeddings, n
                continue
            p.data.copy_(e_full[n].to(p.device, dtype=p.dtype))
        del e_full
        run.run("Eraw" if args.e_block else "E", None, hold=False)
        run.run("E0", we(zero))
        run.run("E-A", we(dense_edits(names, deltas, factors, -1.0, device)))
        torch.cuda.empty_cache()
        run.run("E-ALL", we(lowrank_edits(all_ids, names, factors, -1.0, device)))
        for k in ks:
            for r in ROWS:
                run.run(f"D:{r}:{k}", we(lowrank_edits(sel[str(k)][r]["ids"], names, factors, -1.0, device)))
                run.run(f"D:{r}~:{k}", we(lowrank_edits(sel[str(k)][r]["random"], names, factors, -1.0, device)))
                run.run(f"D:{r}~s:{k}", we(lowrank_edits(sel[str(k)][r]["random_size"], names, factors, -1.0, device)))
        if args.e_block:  # E0 with C's update along the blocked directions put back: E0 + B(C - base)
            run.run("E+BC", we(merge_edits(zero, {n: (Lc, -sc, Rc) for n, (Lc, sc, Rc) in cb.items()})))
        n_rows = write_rows(run, seqs, pairs, f"{out_dir}/rows_removal.parquet" if removal else f"{out_dir}/rows.parquet")
        if not removal:
            os.remove(f"{out_dir}/rows_transfer.parquet")
    per = [s["seq_id"] for s in persona]

    def diff(a, b):
        return dict(max=float(np.abs(run.lp[a] - run.lp[b]).max()), mean=float(np.abs(run.lp[a] - run.lp[b]).mean()))

    checks = {}
    if not removal:  # a split run takes these from the transfer job, and E0 against E from the removal job
        checks.update(c0_hold_vs_c0_max_abs_lp_sum=float(np.abs(run.lp["C0|hold"][per] - run.lp["C0"][per]).max()),
                      dW_entries_not_exact_in_bf16=inexact)
    if args.e_block:
        checks.update(dW_blocked_part_rel_max=blk_rel, A_vs_E0_abs_lp_sum=diff("A", "E0"),
                      E_minus_A_vs_C0_abs_lp_sum=diff("E-A", "C0"))
    else:
        if not removal:
            cc = diff("C0", "C")
            checks.update(c0_vs_c_max_abs_lp_sum=cc["max"], c0_vs_c_mean_abs_lp_sum=cc["mean"])
        if args.phase != "transfer":
            ee = diff("E0", "E")
            checks.update(e0_vs_e_max_abs_lp_sum=ee["max"], e0_vs_e_mean_abs_lp_sum=ee["mean"])
    step6 = dict(e_block=args.e_block, c0="C - B(C - base)", e0="E_blk with -B(W - base)", eraw="E_blk stored weights",
                 blocks=json.load(open(f"{TB.BLOCK_DIR}/blocks.json"))) if args.e_block else None
    manifest = dict(run_id=args.run_id, model=args.model, graft=SPEC["graft"], quick=args.quick, layers=[lo, hi], ks=ks, kmax=KMAX, alphas=alphas, ig_check=ig_check,
                    step6=step6,
                    split_seed=SPLIT_SEED, ctrl=CTRL, em=EM, base=SPEC["hf_id"], revision=SPEC["revision"], sets=SETS,
                    n_pairs=len(pairs), n_sequences=len(seqs), n_batches=len(batches), carrier_layer=L,
                    hold_subspace="nested", conditions=sorted(run.lp), seconds_per_condition=run.secs, checks=checks,
                    base_output_path=dict(MM_PATH), seconds=round(time.time() - t0), torch=torch.__version__,
                    transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0))
    if args.phase != "all":
        manifest.update(phase=args.phase, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    json.dump(manifest, open(f"{out_dir}/manifest{'' if args.phase == 'all' else '_' + args.phase}.json", "w"), indent=1)
    log(f"checks {json.dumps(checks)}; {n_rows} rows; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
