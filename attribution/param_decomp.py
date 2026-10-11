"""Behavior-specific decomposition of the weight update E - C (Qwen2.5-7B, seed 42).
C and E are the benign and harmful fine-tunes. The region is the 84 matrices q, k, v, o, gate, up, down of layers
8-19. Each update dW = W_E - W_C is factorized exactly (float64 Gram eigendecomposition, shared/side_path.top_singular);
its top KMAX singular triplets (u, s, v) are the components, with edit D = s u v^T.

1. Split. Medical and frozen-assay questions are split in half (seed 0, the rule of components.make_split). The selection
   half is used only for the gradients; every reported effect uses the other pairs (param_decomp_report.py).
2. Attribution. At native C, one backward pass of the summed answer log-probability per batch gives, for every
   selection-half sequence i and component (t, r), P[i, t, r] = s_r * sum over positions of (u_r . dlp_i/dy_t)(v_r . x_t),
   and for every matrix t the whole-update term Q[i, t] = sum over positions of dlp_i/dy_t . (dW_t x_t). The attribution
   of a weighted sum of sequence log-probabilities is the same weighted sum of P. A_b = <grad S_b(W_C), D> with S_b the
   mean LLR (misaligned minus aligned summed log-probability) over the selection-half pairs of behavior b.
3. Selection (rows RR, WI, OC = med referral_removal, wrong_instruction, overconfidence; EM = ref_cartoon120): the K
   components with the largest positive A_b, K in KS; secondary rows (suffix R) by attribution to the residual shift;
   for K = 8 one magnitude-matched and N same-matrix random control sets.
4. Scoring. Every condition runs the side-path arithmetic on the region matrices, y = bf16(x W^T + b + side(x)) with
   x W^T + b accumulated in float32 and rounded once; C0 and E0 carry zero edits. Addition conditions (add:*) run on C's
   weights, then E's weights are loaded and the removal conditions (rem:*) run.
Output: eval_runs/persona_control_stage9/param_decomp/<run_id>/: rows.parquet (cond, seq_id, set, prompt_id, qkey, half,
kind, lp_sum, n_tok), components.parquet (one row per component: attributions), matrices.parquet (per matrix: energy,
whole-update first-order terms), selections.json, split.json, manifest.json; lp_partial.npz while running (--resume).
Usage: python param_decomp.py --run-id s9pd_20260928 [--quick 2 --layers 8-9 --n-random 2 --out-dir <dir>] [--resume]
"""
import argparse
import json
import os
import time
import zlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

import _paths  # noqa: F401  (puts every code folder on sys.path)
from common import MODEL_SPECS, ROOT, layer_of, load_state_dict_cpu, make_batches, score_batch  # noqa: E402
from factors import selected_param_names  # noqa: E402
from side_path import MM_PATH, base_output_fp32, top_singular  # noqa: E402

SPEC = MODEL_SPECS["qwen2_5_7b"]
CKPT = f"{ROOT}/checkpoints/stage9/qwen2_5_7b/seed42"
CTRL, EM = f"{CKPT}/M_ctrl/checkpoint-100pct", f"{CKPT}/M_EM/checkpoint-100pct"
STEP1 = f"{ROOT}/eval_runs/persona_control_stage9/step1"
SETS = {"s9s1_20260928": ["med_referral_removal", "med_wrong_instruction", "med_overconfidence", "med_trim",
                          "fin_referral_removal", "fin_wrong_instruction", "fin_overconfidence", "fin_trim",
                          "tqa_health", "tqa_other", "ref_cartoon120", "med_synonym_swap", "fin_synonym_swap",
                          "sports_synonym_swap", "persona_synonym_swap"],
        "s9s1v2_20260928": ["med_paraphrase", "fin_paraphrase", "sports_paraphrase", "persona_paraphrase"]}
ROWS = {"RR": ("med_referral_removal", "med"), "WI": ("med_wrong_instruction", "med"),
        "OC": ("med_overconfidence", "med"), "EM": ("ref_cartoon120", "persona")}
SPLIT_GROUPS = ("med", "persona")
SPLIT_SEED = 0
KMAX = 64
KS = (4, 8, 16)
K_MAIN = 8
MAG_TOL = 0.10
MAG_MIN = 5


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def is_ctrl(s):
    return s.endswith("_paraphrase") or s.endswith("_synonym_swap")


def qkey(p):
    """Question key shared by every pair built on one question (med_000 of every medical set is one question)."""
    s, pid = p["set"], p["prompt_id"]
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
    """Step 3's rule (components.make_split): per group, a seeded permutation of the sorted keys; first half selects."""
    groups = {}
    for k in sorted(set(keys)):
        groups.setdefault(k.split(":")[0], []).append(k)
    half = {}
    for g, ks in sorted(groups.items()):
        perm = np.random.default_rng([SPLIT_SEED, zlib.crc32(g.encode())]).permutation(len(ks))
        for rank, i in enumerate(perm):
            half[ks[i]] = "sel" if rank < len(ks) // 2 else "eval"
    return half


def render(tok, question, answer=None):
    msgs = [{"role": "user", "content": question}] + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=answer is None)
    return [int(i) for i in (ids["input_ids"] if hasattr(ids, "keys") else ids)]


def load_pairs(quick):
    pairs = []
    for run, names in SETS.items():
        pairs += [p for p in json.load(open(f"{STEP1}/{run}/pairs.json")) if p["set"] in names]
    split = make_split([k for k in map(qkey, pairs) if k.split(":")[0] in SPLIT_GROUPS])
    for p in pairs:
        p["qkey"] = qkey(p)
        p["half"] = split.get(p["qkey"], "eval")
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
            seqs.append(dict(seq_id=len(seqs), pair=pi, kind=kind, ids=ids, plen=len(pre)))
    return seqs


def seq_sums(batch, lp):
    s = torch.zeros(len(batch["seqs"]), dtype=torch.float64, device=lp.device)
    return s.index_add_(0, batch["score_seq"], lp.double()).cpu().numpy()


# ---------------------------------------------------------------------------------------------------------------------
# Attribution at native C
# ---------------------------------------------------------------------------------------------------------------------

class AttrHooks:
    """Forward hooks on the region matrices; in the backward pass matrix t adds, per sequence of the batch,
    s_r * sum_positions (u_r . dy)(v_r . x) to P[row, t, r] and sum_positions dy . (dW_t x) to Q[row, t]."""

    def __init__(self, model, names, factors, deltas, st):
        self.handles = [model.get_submodule(n[: -len(".weight")]).register_forward_hook(
            self._hook(t, factors[n], deltas[n], st)) for t, n in enumerate(names)]

    @staticmethod
    def _hook(t, f, D, st):
        U, s, V = f["U"], f["S"], f["V"]

        def hook(m, inp, out):
            if not out.requires_grad:
                return
            x = inp[0].detach()
            B, T = x.shape[:2]

            def grad_hook(g):
                x2 = x.reshape(B * T, -1).float()
                g2 = g.reshape(B * T, -1).float()
                comp = (((g2 @ U) * (x2 @ V)) * s).reshape(B, T, -1).sum(1)
                full = (g2 * (x2 @ D.T)).reshape(B, T, -1).sum((1, 2))
                st["P"][:, t].index_add_(0, st["rows"], comp.double())
                st["Q"][:, t].index_add_(0, st["rows"], full.double())

            out.register_hook(grad_hook)
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


def attribution_pass(model, batches, names, factors, deltas, row_of, device):
    n = len(row_of)
    st = dict(P=torch.zeros(n, len(names), KMAX, dtype=torch.float64, device=device),
              Q=torch.zeros(n, len(names), dtype=torch.float64, device=device), rows=None)
    lp_seq = np.zeros(n)
    hooks = AttrHooks(model, names, factors, deltas, st)
    try:
        for b in batches:
            st["rows"] = torch.tensor([row_of[s["seq_id"]] for s in b["seqs"]], device=device)
            emb = model.model.embed_tokens(b["input_ids"]).detach().requires_grad_(True)
            out = model.model(inputs_embeds=emb, attention_mask=b["attention_mask"], use_cache=False)
            hs = out.last_hidden_state[b["score_rows"], b["score_cols"]]
            lp = F.log_softmax(model.lm_head(hs).float(), dim=-1).gather(-1, b["score_tgts"][:, None]).squeeze(-1)
            seq = torch.zeros(len(b["seqs"]), dtype=torch.float32, device=device).index_add(0, b["score_seq"], lp)
            lp_seq[st["rows"].cpu().numpy()] = seq.detach().double().cpu().numpy()
            seq.sum().backward()
    finally:
        hooks.close()
    return st["P"].reshape(n, -1).cpu().numpy(), st["Q"].cpu().numpy(), lp_seq


def functionals(pairs, seqs, row_of, lp_seq):
    """Per-sequence weight vectors (over the selection-half sequences) of every attributed functional."""
    llr = defaultdict(float)
    for s in seqs:
        if s["seq_id"] in row_of:
            llr[s["pair"]] += lp_seq[row_of[s["seq_id"]]] * (1 if s["kind"] == "mis" else -1)

    def vec(pair_w):
        w = np.zeros(len(row_of))
        for s in seqs:
            if s["pair"] in pair_w:
                w[row_of[s["seq_id"]]] = pair_w[s["pair"]] * (1 if s["kind"] == "mis" else -1)
        return w

    sel = [i for i, p in enumerate(pairs) if p["half"] == "sel"]
    out, meta = {}, {}
    for b, (st, dom) in ROWS.items():
        ps = [i for i in sel if pairs[i]["set"] == st]
        out[f"set:{b}"] = vec({i: 1.0 / len(ps) for i in ps})
        meta[b] = dict(n=len(ps), mean_llr_c=float(np.mean([llr[i] for i in ps])))
    for dom in SPLIT_GROUPS:
        ps = [i for i in sel if is_ctrl(pairs[i]["set"]) and pairs[i]["set"].startswith(dom + "_")]
        x = np.array([llr[i] for i in ps])
        out[f"c0:{dom}"] = vec({i: 1.0 for i in ps})
        out[f"c1:{dom}"] = vec({i: llr[i] for i in ps})
        meta[f"ctrl:{dom}"] = dict(n=len(ps), gram=[[len(x), x.sum()], [x.sum(), (x ** 2).sum()]])
    return out, meta, dict(llr)


def row_attributions(P, Q, fw, fmeta):
    """A_b (primary), A_bR (attribution to the residual shift, C's LLRs fixed as the null-line predictor), controls."""
    comp = {k: w @ P for k, w in fw.items()}
    full = {k: w @ Q for k, w in fw.items()}
    for src in (comp, full):
        for b, (_, dom) in ROWS.items():
            h = np.array([1.0, fmeta[b]["mean_llr_c"]]) @ np.linalg.inv(np.array(fmeta[f"ctrl:{dom}"]["gram"]))
            src[f"res:{b}"] = src[f"set:{b}"] - h[0] * src[f"c0:{dom}"] - h[1] * src[f"c1:{dom}"]
    return comp, full


# ---------------------------------------------------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------------------------------------------------

def order_positive(a):
    return [int(i) for i in np.argsort(-a, kind="stable") if a[i] > 0]


def magnitude_control(ids, a, s, exclude):
    """For each selected component: the candidates (outside exclude and earlier picks) are the components whose singular
    value lies within MAG_TOL of its own, or the MAG_MIN nearest in log singular value when fewer lie within MAG_TOL;
    the candidate with the smallest |attribution| is taken."""
    taken, out, ls = set(exclude), [], np.log(s)
    for g in ids:
        near = [int(c) for c in np.argsort(np.abs(ls - ls[g]), kind="stable") if int(c) not in taken]
        cand = [c for c in near if abs(ls[c] - ls[g]) <= np.log1p(MAG_TOL)]
        if len(cand) < MAG_MIN:
            cand = near[:MAG_MIN]
        pick = min(cand, key=lambda c: (abs(a[c]), c))
        out.append(pick)
        taken.add(pick)
    return out


def random_control(ids, rng):
    """Each selected component replaced by a uniform draw among the other retained components of its matrix."""
    taken, out = set(ids), []
    for g in ids:
        t = g // KMAX
        cands = [t * KMAX + r for r in range(KMAX) if t * KMAX + r not in taken]
        pick = cands[int(rng.integers(len(cands)))]
        out.append(pick)
        taken.add(pick)
    return out


def select(comp, s_all, n_random):
    sel = {}
    for b in ROWS:
        a, ar = comp[f"set:{b}"], comp[f"res:{b}"]
        order = order_positive(a)
        main = order[:K_MAIN]
        sel[b] = dict(order=order, n_positive=len(order), **{f"K{k}": order[:k] for k in KS},
                      res8=order_positive(ar)[:K_MAIN], mag=magnitude_control(main, a, s_all, order[:max(KS)]),
                      rand=[random_control(main, np.random.default_rng([s, zlib.crc32(b.encode())]))
                            for s in range(n_random)])
    return sel


# ---------------------------------------------------------------------------------------------------------------------
# Scoring with float32 side paths
# ---------------------------------------------------------------------------------------------------------------------

class SidePaths:
    """Side-path arithmetic (shared/side_path.py): y = bf16(x W^T + b + side(x)), x W^T + b accumulated in float32
    (side_path.base_output_fp32) and rounded once. Installed as a replacement forward on each region matrix, so the
    native bf16 product is not computed first. edits: name -> None (zero edit), ("dense", D, sign) or ("lowrank", U, s, V)."""

    def __init__(self, model, edits):
        self.mods = []
        for name, e in edits.items():
            m = model.get_submodule(name[: -len(".weight")])
            m.forward = self._forward(m, e)
            self.mods.append(m)

    @staticmethod
    def _forward(m, e):
        if e is None:
            side = None
        elif e[0] == "dense":
            _, D, sign = e

            def side(x2f):
                return sign * (x2f @ D.T)
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


def lowrank_edits(ids, names, factors, sign, device):
    by = defaultdict(list)
    for g in ids:
        by[g // KMAX].append(g % KMAX)
    edits = {}
    for t, n in enumerate(names):
        if t in by:
            idx = torch.tensor(sorted(by[t]), device=device)
            f = factors[n]
            edits[n] = ("lowrank", f["U"][:, idx], sign * f["S"][idx], f["V"][:, idx])
        else:
            edits[n] = None
    return edits


class Runner:
    def __init__(self, model, groups, n_seqs, saved, out_dir):
        self.model, self.groups, self.n_seqs, self.out_dir = model, groups, n_seqs, out_dir
        self.lp, self.saved, self.secs = {}, saved, {}

    @torch.no_grad()
    def run(self, label, edits, halves=("eval", "sel")):
        if label in self.saved:
            self.lp[label] = self.saved[label]
            return
        t = time.time()
        side = SidePaths(self.model, edits) if edits is not None else None
        arr = np.full(self.n_seqs, np.nan)
        try:
            for h in halves:
                for batch in self.groups[h]:
                    arr[[s["seq_id"] for s in batch["seqs"]]] = seq_sums(batch, score_batch(self.model, batch)["lp"])
        finally:
            if side:
                side.close()
        self.lp[label] = arr
        self.secs[label] = round(time.time() - t, 1)
        log(f"{label}: {self.secs[label]}s")
        if len(self.secs) % 10 == 0:
            self.flush()

    def flush(self):
        np.savez(os.path.join(self.out_dir, "lp_partial.npz"), **{k.replace(":", "__"): v for k, v in self.lp.items()})


def param_group(name, ndim, layers):
    l = layer_of(name)
    if l is None:
        return {"model.embed_tokens.weight": "embed", "lm_head.weight": "lm_head", "model.norm.weight": "final_norm"}[name]
    return ("region" if l in layers else "other") + ("_matrix" if ndim == 2 else "_1d")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--quick", type=int, default=0, help="first N pairs per set and half (test mode; needs --out-dir)")
    ap.add_argument("--layers", default="8-19")
    ap.add_argument("--n-random", type=int, default=20)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--resume", action="store_true", help="reuse selections.json and lp_partial.npz in --out-dir")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--grad-max-tokens", type=int, default=2048)
    args = ap.parse_args()
    if args.quick and not args.out_dir:
        raise ValueError("--quick writes only to --out-dir")
    out_dir = args.out_dir or f"{ROOT}/eval_runs/persona_control_stage9/param_decomp/{args.run_id}"
    if os.path.isdir(out_dir) and os.listdir(out_dir) and not args.resume:
        raise FileExistsError(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0, device = time.time(), "cuda:0"
    lo, hi = (int(x) for x in args.layers.split("-"))
    layers = list(range(lo, hi + 1))

    tok = AutoTokenizer.from_pretrained(SPEC["hf_id"], revision=SPEC["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs, split = load_pairs(args.quick)
    seqs = build_seqs(tok, pairs)
    groups = {h: make_batches([s for s in seqs if pairs[s["pair"]]["half"] == h], pad_id, device,
                              max_tokens=args.max_tokens) for h in ("eval", "sel")}
    grad_seqs = [s for s in seqs if pairs[s["pair"]]["half"] == "sel" and pairs[s["pair"]]["set"] != "med_trim"]
    row_of = {s["seq_id"]: i for i, s in enumerate(grad_seqs)}
    log(f"{len(pairs)} pairs ({sum(p['half'] == 'sel' for p in pairs)} selection half), {len(seqs)} sequences, "
        f"batches eval {len(groups['eval'])} sel {len(groups['sel'])}; {len(grad_seqs)} sequences for gradients")

    model = AutoModelForCausalLM.from_pretrained(CTRL, dtype=torch.bfloat16, device_map=device).eval()
    model.requires_grad_(False)
    names = selected_param_names(model, layers)
    e_sd = load_state_dict_cpu(EM)
    params = dict(model.named_parameters())
    assert set(params) == set(e_sd), sorted(set(params) ^ set(e_sd))[:5]
    norms, deltas, factors = defaultdict(float), {}, {}
    for n, p in params.items():
        d = e_sd[n].to(device).float() - p.float()
        norms[param_group(n, p.ndim, layers)] += float(d.double().pow(2).sum())
        if n in names:
            deltas[n] = d
    for n in names:
        U, s, V, frob2 = top_singular(deltas[n], KMAX)
        factors[n] = dict(U=U, S=s, V=V, frob2=frob2)
    s_all = np.concatenate([factors[n]["S"].cpu().numpy() for n in names]).astype(np.float64)
    assert (s_all > 0).all()
    log(f"{len(names)} matrices factorized (top {KMAX}); {time.time() - t0:.0f}s")

    if args.resume:
        sel = json.load(open(f"{out_dir}/selections.json"))
    else:
        gb = make_batches(grad_seqs, pad_id, device, max_tokens=args.grad_max_tokens)
        t = time.time()
        P, Q, lp_seq = attribution_pass(model, gb, names, factors, deltas, row_of, device)
        log(f"attribution pass: {len(gb)} batches, {time.time() - t:.0f}s, peak GPU "
            f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        fw, fmeta, llr_c = functionals(pairs, seqs, row_of, lp_seq)
        comp, full = row_attributions(P, Q, fw, fmeta)
        sel = select(comp, s_all, args.n_random)
        cm = pd.DataFrame([dict(id=t * KMAX + r, matrix=n, layer=layer_of(n), type=n.split(".")[-2], rank=r,
                                s=float(factors[n]["S"][r]), energy=float(factors[n]["S"][r]) ** 2 / factors[n]["frob2"])
                           for t, n in enumerate(names) for r in range(KMAX)])
        for k, v in comp.items():
            cm[f"A_{k}"] = v
        cm.to_parquet(f"{out_dir}/components.parquet", index=False)
        mm = pd.DataFrame(dict(matrix=names, layer=[layer_of(n) for n in names], type=[n.split(".")[-2] for n in names],
                               frob2=[factors[n]["frob2"] for n in names],
                               top_energy=[float((factors[n]["S"].double() ** 2).sum()) / factors[n]["frob2"] for n in names]))
        for k, v in full.items():
            mm[f"Q_{k}"] = v
            mm[f"Asum_{k}"] = comp[k].reshape(len(names), KMAX).sum(1)
        mm.to_parquet(f"{out_dir}/matrices.parquet", index=False)
        json.dump(dict(rows=sel, fmeta=fmeta, llr_c_sel={str(k): v for k, v in llr_c.items()}),
                  open(f"{out_dir}/selections.json", "w"))
        sel = json.load(open(f"{out_dir}/selections.json"))
        json.dump(split, open(f"{out_dir}/split.json", "w"), indent=0)
    rows_sel = sel["rows"]
    for b in ROWS:
        log(f"{b}: {rows_sel[b]['n_positive']} positive; K8 {rows_sel[b]['K8']}")
    torch.cuda.empty_cache()

    saved = {}
    if args.resume and os.path.exists(f"{out_dir}/lp_partial.npz"):
        z = np.load(f"{out_dir}/lp_partial.npz")
        saved = {k.replace("__", ":"): z[k] for k in z.files}
        log(f"resume: {len(saved)} saved conditions")
    ref = {k: saved.pop(k) for k in ("C0", "E0") if k in saved}
    run = Runner(model, groups, len(seqs), saved, out_dir)
    zero = {n: None for n in names}
    all_ids = list(range(len(names) * KMAX))

    def phase(sign, pre, base_label):
        run.run(base_label, zero)
        if base_label in ref:
            d = float(np.nanmax(np.abs(run.lp[base_label] - ref[base_label])))
            log(f"resume check {base_label}: max|d lp sum| {d:.2e}")
            if d > 1e-6:
                raise RuntimeError(f"{base_label} differs from the saved run by {d}")
        run.run(f"{base_label}{'+' if sign > 0 else '-'}dW", {n: ("dense", deltas[n], sign) for n in names})
        run.run(f"{base_label}{'+' if sign > 0 else '-'}ALL", lowrank_edits(all_ids, names, factors, sign, device))
        for k in (K_MAIN,) + tuple(k for k in KS if k != K_MAIN):
            for b in ROWS:
                run.run(f"{pre}:{b}:{k}", lowrank_edits(rows_sel[b][f"K{k}"], names, factors, sign, device))
        for b in ROWS:
            run.run(f"{pre}:{b}R:8", lowrank_edits(rows_sel[b]["res8"], names, factors, sign, device))
        for b in ROWS:
            run.run(f"{pre}:{b}:mag", lowrank_edits(rows_sel[b]["mag"], names, factors, sign, device), halves=("eval",))
        for s in range(len(rows_sel["RR"]["rand"])):
            for b in ROWS:
                run.run(f"{pre}:{b}:rand{s}", lowrank_edits(rows_sel[b]["rand"][s], names, factors, sign, device),
                        halves=("eval",))

    run.run("C", None)
    phase(+1, "add", "C0")
    for n, p in params.items():
        p.data.copy_(e_sd[n].to(device))
    del e_sd
    run.run("E", None)
    phase(-1, "rem", "E0")
    run.flush()

    lp = run.lp
    check = dict(c0_vs_c=float(np.nanmax(np.abs(lp["C0"] - lp["C"]))), e0_vs_e=float(np.nanmax(np.abs(lp["E0"] - lp["E"]))))
    n_tok = np.array([len(s["ids"]) - s["plen"] for s in seqs])
    cart = {i for i, p in enumerate(pairs) if p["set"] == "ref_cartoon120"}

    def S(cond):
        v = {(s["pair"], s["kind"]): lp[cond][s["seq_id"]] / n_tok[s["seq_id"]] for s in seqs if s["pair"] in cart}
        return np.mean([v[(i, "mis")] - v[(i, "align")] for i in cart])
    check["round1_T"] = float(S("C0+dW") - S("C0"))
    check["round1_T_target"] = 0.3142437587181728
    check["round1_pass"] = bool(abs(check["round1_T"] - check["round1_T_target"]) <= .03)
    log(f"checks: {check}")

    meta = pd.DataFrame([dict(seq_id=s["seq_id"], set=pairs[s["pair"]]["set"], prompt_id=pairs[s["pair"]]["prompt_id"],
                              qkey=pairs[s["pair"]]["qkey"], half=pairs[s["pair"]]["half"], kind=s["kind"],
                              n_tok=int(n_tok[s["seq_id"]])) for s in seqs])
    frames = []
    for cond, arr in lp.items():
        ok = ~np.isnan(arr)
        frames.append(meta[ok].assign(cond=cond, lp_sum=arr[ok]))
    rows = pd.concat(frames, ignore_index=True)
    rows["cond"] = rows["cond"].astype("category")
    rows.to_parquet(f"{out_dir}/rows.parquet", index=False)
    total = sum(norms.values())
    manifest = dict(run_id=args.run_id, quick=args.quick, ctrl=CTRL, em=EM, base=SPEC["hf_id"], revision=SPEC["revision"],
                    layers=layers, kmax=KMAX, ks=list(KS), k_main=K_MAIN, n_random=args.n_random, mag_tol=MAG_TOL, mag_min=MAG_MIN,
                    split_seed=SPLIT_SEED, sets=SETS, n_pairs=len(pairs), n_sequences=len(seqs),
                    n_grad_sequences=len(grad_seqs), conditions=list(lp), seconds_per_condition=run.secs,
                    param_norm2=dict(norms), param_share={k: v / total for k, v in norms.items()},
                    top_energy_region=float(sum(float((factors[n]["S"].double() ** 2).sum()) for n in names)
                                             / sum(factors[n]["frob2"] for n in names)),
                    checks=check, mm_path=MM_PATH, seconds=round(time.time() - t0), torch=torch.__version__,
                    transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
                    peak_gpu_gib=torch.cuda.max_memory_allocated() / 2**30, slurm_job_id=os.environ.get("SLURM_JOB_ID"))
    json.dump(manifest, open(f"{out_dir}/manifest.json", "w"), indent=1)
    os.remove(f"{out_dir}/lp_partial.npz")
    log(f"{len(lp)} conditions, {len(rows)} rows; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
