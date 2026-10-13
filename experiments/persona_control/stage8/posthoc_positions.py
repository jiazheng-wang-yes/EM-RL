"""Stage 8, POST HOC measurement check of 3b and 3c (written after the development pair's 3b and 3c results were
seen; not a Section 7 check, and no threshold changes).

Plan Section 7: if L2/L3 fail, check the measurement (host, positions, read span, label-passing pairs) before
concluding.  The host check is Step 1's (s1_components/checks.json, full_side against native E).  In 3b
(s3_reads.py swap) the read vector of every matrix was replaced only at the answer positions (plen-1 onward), by a
mean over the first 16 answer positions of C's greedy answer, and S was measured only on the recipients whose pair
passes the label check.  At positions 0 .. plen-2 the E - C edit stayed the recipient's own, and it reaches the
answer positions through attention.  One GPU job ('run') measures, on the saved 3b design (swap_info.json) and read
vectors (reads.pt):

  capture    mean read vector V^T x of every matrix in the host (C + dense E - C), per 3a prompt, over:
             'first16' (as reads.pt; a replication check), 'answer_all' (every answer position of C's greedy answer
             that predicts a token), 'question' (the question's tokens: between the template prefix and suffix shared
             by all 640 rendered prompts), 'prompt_rest' (the template tokens after the question, up to plen-2); and
             over the first 16 and all answer positions of each prompt's misaligned and aligned 3a answers ('mis16',
             'mis_all', 'align16', 'align_all').
  paths      every 3b recipient; S_x on its 3a pair whether or not the pair passes the label check, KL(cond || C0)
             over the first 32 answer positions of C's answer: 'host'; 'ans_only' = the E - C side path only at the
             answer positions (plen-1 onward); 'prompt_only' = the side path only at positions 0 .. plen-2.
  ans swap   3b in the ans_only host, where the swap replaces the reads at every position that has an edit:
             'ans_self', 'ans_donor|<cell>|<pid>' with the 3b read vectors.  Every recipient, S and KL.
  full swap  3b as run before ('self', 'donor|...', full host) on the pairs of the recipients that fail the label
             check, and on the first 2 label-passing recipients of each cell as a replication check.
  text swap  label-passing recipients, full host: the replacement for each 3a answer is the mean read vector over the
             first 16 answer positions of the same kind of answer (misaligned or aligned) of the recipient
             ('text_self') or of the donor ('text_donor|...'), instead of C's greedy answer.
  span swap  label-passing recipients, full host: the replacement is the mean over all answer positions of C's
             greedy answer ('span_self', 'span_donor|...').
  analyze    (CPU) replication checks against reads.pt and swap_rows.parquet; path shares per cell; the 3b
             analysis (s3_reads.swap_analysis) of every swap version; per capture mask, how much the A* read profile
             and the A* read coefficients vary between the 8 cells.

Output: <OUT_ROOT>/<dataset>/<model>/<twin>/s3_reads/posthoc_positions/ (rows.parquet, captures.pt, info.json,
analysis.json).
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from transformers import AutoTokenizer

from common8 import _mm32, base_dir, base_output_fp32, log, render
from s3_reads import (CELLS, N_BOOT, N_READ, ReadHost, c_answer_seqs, cell_roles, collate, host_setup, interval, kl_div,
                      load_model, load_pairs, out_dir_for, pair_seqs, s1_dir_for, score, stratified_weights,
                      swap_analysis, swap_measures)

BOOT_SEED = 20260922
N_REPL = 2
MASKS_C = ("first16", "answer_all", "question", "prompt_rest")
MASKS_PAIR = ("first16", "answer_all")


class PosHost(ReadHost):
    """ReadHost with ``side_mask`` [B, T] (the whole side path, dense E - C and read change, is multiplied by it;
    None = every position) and ``cap_masks`` {name: [B, T, 1] float} (capture the mean read vector over each)."""

    side_mask = None
    cap_masks = None

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
                sm = 1.0 if self.side_mask is None else self.side_mask.reshape(B * T, 1)
                y = y + _mm32(x2, hi) * sm
                if lo is not None:
                    y = y + _mm32(x2, lo) * sm
                if self.capture or self.w_set is not None:
                    coef = (x2.float() @ V).view(B, T, -1)
                    if self.capture:
                        for k, mk in self.cap_masks.items():
                            self.captured.setdefault(k, {})[name] = (coef * mk).sum(1) / mk.sum(1).clamp_min(1)
                    else:
                        delta = (self.w_set[:, None, None] * (self.target[:, j][:, None, :] - coef)
                                 + self.w_scale[:, None, None] * coef) * self.mask
                        y = y + ((delta * S) @ U.T).reshape(B * T, -1) * sm
            return y.to(out.dtype).reshape(out.shape)
        return hook


def pos_masks(items, T, device, pre=None, suf=None):
    """[B, T] masks per row: 'ans' = plen-1 .. L-1 (3b's 'resp'), 'prompt' = 0 .. plen-2, 'first16', 'answer_all' =
    plen-1 .. L-2; with pre and suf also 'question' = pre .. plen-suf-1 and 'prompt_rest' = plen-suf .. plen-2."""
    names = ("ans", "prompt", "first16", "answer_all") + (("question", "prompt_rest") if pre is not None else ())
    M = {k: torch.zeros((len(items), T), dtype=torch.bool) for k in names}
    for i, x in enumerate(items):
        L, pl = len(x["ids"]), x["plen"]
        M["ans"][i, pl - 1:L] = True
        M["prompt"][i, :pl - 1] = True
        M["first16"][i, pl - 1:min(pl - 1 + N_READ, L - 1)] = True
        M["answer_all"][i, pl - 1:L - 1] = True
        if pre is not None:
            M["question"][i, pre:pl - suf] = True
            M["prompt_rest"][i, pl - suf:pl - 1] = True
    return {k: v.to(device) for k, v in M.items()}


def template_span(tok, questions):
    """Lengths of the token prefix and suffix that every rendered prompt shares (the chat template around the question)."""
    P = [render(tok, q) for q in questions]

    def common(a, b):
        return next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    pre = min(common(P[0], p) for p in P[1:])
    suf = min(common(P[0][::-1], p[::-1]) for p in P[1:])
    if not all(pre < len(p) - suf for p in P):
        raise ValueError("a question has no tokens between the shared prefix and suffix")
    return pre, suf


@torch.no_grad()
def capture(model, H, seqs, masks, pad_id, device, batch, pre, suf):
    out = {k: torch.zeros(len(seqs), len(H.names), 16) for k in masks}
    H.capture = True
    for i in range(0, len(seqs), batch):
        items = seqs[i: i + batch]
        b = collate(items, pad_id, device)
        M = pos_masks(items, b["ids"].shape[1], device, pre, suf)
        H.cap_masks = {k: M[k].float()[..., None] for k in masks}
        model.model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False)
        for k in masks:
            for j, n in enumerate(H.names):
                out[k][i: i + len(items), j] = H.captured[k][n].float().cpu()
    H.capture, H.cap_masks = False, None
    return out


@torch.no_grad()
def run_groups(model, H, groups, pad_id, device, max_rows, log_every=20):
    """groups: (seqs, items), item = (seq index, cond, side mode 'all' | 'ans' | 'prompt', w_set, target or None).
    Per group: C0 on its sequences, then every item.  Rows: lp_mean, and KL(cond || C0) for kind 'c'."""
    rows, t0 = [], time.time()
    for gi, (seqs, items) in enumerate(groups):
        klr = [i for i, s in enumerate(seqs) if s["kind"] == "c"]
        b = collate(seqs, pad_id, device)
        H.side, H.side_mask, H.w_set = False, None, None
        lp0, n0, kl0 = score(model, b, klr)
        H.side = True
        for i, s in enumerate(seqs):
            rows.append(dict(pid=s["pid"], kind=s["kind"], cond="C0", lp_mean=float(lp0[i]), n_tok=int(n0[i]),
                             kl=0.0 if i in kl0 else None))
        for k in range(0, len(items), max_rows):
            chunk = items[k: k + max_rows]
            cs = [seqs[it[0]] for it in chunk]
            b = collate(cs, pad_id, device)
            modes = [it[2] for it in chunk]
            if any(m != "all" for m in modes):
                M = pos_masks(cs, b["ids"].shape[1], device)
                H.side_mask = torch.stack([torch.ones_like(M["ans"][r]) if m == "all" else M[m][r]
                                           for r, m in enumerate(modes)]).float()
            w_set = torch.tensor([it[3] for it in chunk], device=device, dtype=torch.float32)
            target = torch.stack([it[4] if it[4] is not None else H.zero_target for it in chunk])
            H.set_rows(b["resp"], w_set, torch.zeros_like(w_set), target)
            lp, n, kl = score(model, b, [r for r, s in enumerate(cs) if s["kind"] == "c"])
            H.w_set, H.side_mask = None, None
            for r, (it, s) in enumerate(zip(chunk, cs)):
                rows.append(dict(pid=s["pid"], kind=s["kind"], cond=it[1], lp_mean=float(lp[r]), n_tok=int(n[r]),
                                 kl=kl_div(kl[r], kl0[it[0]]) if r in kl else None))
        if gi == 0 or (gi + 1) % log_every == 0:
            log(f"group {gi + 1}/{len(groups)} ({time.time() - t0:.0f}s)")
    return rows


def pdir(args):
    return args.out or os.path.join(out_dir_for(args.dataset, args.model, args.twin), "posthoc_positions")


def run_gpu(args):
    device = "cuda:0"
    t0 = time.time()
    s3 = out_dir_for(args.dataset, args.model, args.twin)
    out = pdir(args)
    os.makedirs(out, exist_ok=True)
    if os.path.exists(os.path.join(out, "rows.parquet")):
        raise FileExistsError(os.path.join(out, "rows.parquet"))
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs = load_pairs()
    by_id = {p["prompt_id"]: p for p in pairs}
    with open(os.path.join(s3, "c_answers.json")) as f:
        answers = json.load(f)["answers"]
    with open(os.path.join(s3, "swap_info.json")) as f:
        design = json.load(f)["design"]
    rd = torch.load(os.path.join(s3, "reads.pt"), map_location="cpu", weights_only=False)
    pid_index = {p: i for i, p in enumerate(rd["prompt_ids"])}
    recipients = {c: design[c]["recipients"][: args.quick or None] for c in CELLS}
    donors = [(c, d) for c in CELLS for d in design[c]["donors"][: 1 if args.quick else None]]
    repl = {c: [r for r in design[c]["recipients"] if by_id[r]["label_pass"]][:N_REPL] for c in CELLS}
    needed = set(d for _, d in donors) | set(r for c in CELLS for r in recipients[c])
    cap_ids = [p for p in rd["prompt_ids"] if not args.quick or p in needed]
    pre, suf = template_span(tok, [by_id[p]["question"] for p in rd["prompt_ids"]])

    names, fac, dense, ctrl_dir, _ = host_setup(args, device)
    assert names == rd["names"]
    model = load_model(args.model, ctrl_dir, device)
    H = PosHost(model, names, dense, fac)
    H.zero_target = torch.zeros(len(names), 16, device=device)
    try:
        cseq = {s["pid"]: s for s in c_answer_seqs(tok, [by_id[p] for p in cap_ids], answers)}
        pseq = {p: pair_seqs(tok, by_id[p]) for p in cap_ids}
        cap = capture(model, H, [cseq[p] for p in cap_ids], MASKS_C, pad_id, device, args.batch, pre, suf)
        for i, kind in enumerate(("mis", "align")):
            c2 = capture(model, H, [pseq[p][i] for p in cap_ids], MASKS_PAIR, pad_id, device, args.batch, pre, suf)
            cap[f"{kind}16"], cap[f"{kind}_all"] = c2["first16"], c2["answer_all"]
        torch.save(dict(prompt_ids=cap_ids, cells=[by_id[p]["cell"] for p in cap_ids], names=names, pre=pre, suf=suf,
                        masks=cap), os.path.join(out, "captures.pt"))
        log(f"captures: {len(cap_ids)} prompts, masks {list(cap)} ({time.time() - t0:.0f}s)")
        cidx = {p: i for i, p in enumerate(cap_ids)}
        R = rd["R"].to(device)
        CAP = {k: v.to(device) for k, v in cap.items()}
        groups = []
        for cell in CELLS:
            for rid in recipients[cell]:
                p = by_id[rid]
                seqs = pseq[rid] + [cseq[rid]]  # 0 = misaligned, 1 = aligned, 2 = C's answer
                rr = R[pid_index[rid]]
                items = []
                for si in range(3):
                    items += [(si, "host", "all", 0.0, None), (si, "ans_only", "ans", 0.0, None),
                              (si, "prompt_only", "prompt", 0.0, None), (si, "ans_self", "ans", 1.0, rr)]
                    items += [(si, f"ans_donor|{dc}|{d}", "ans", 1.0, R[pid_index[d]]) for dc, d in donors]
                if not p["label_pass"] or rid in repl[cell]:
                    for si in range(2):
                        items += [(si, "self", "all", 1.0, rr)]
                        items += [(si, f"donor|{dc}|{d}", "all", 1.0, R[pid_index[d]]) for dc, d in donors]
                if p["label_pass"]:
                    for si, key in ((0, "mis16"), (1, "align16")):
                        items += [(si, "text_self", "all", 1.0, CAP[key][cidx[rid]])]
                        items += [(si, f"text_donor|{dc}|{d}", "all", 1.0, CAP[key][cidx[d]]) for dc, d in donors]
                        items += [(si, "span_self", "all", 1.0, CAP["answer_all"][cidx[rid]])]
                        items += [(si, f"span_donor|{dc}|{d}", "all", 1.0, CAP["answer_all"][cidx[d]]) for dc, d in donors]
                groups.append((seqs, items))
        log(f"{len(groups)} recipients, {sum(len(it) for _, it in groups)} scored rows")
        rows = run_groups(model, H, groups, pad_id, device, args.max_rows)
    finally:
        H.close()
    df = pd.DataFrame(rows)
    df["cell"] = df.pid.map({p["prompt_id"]: p["cell"] for p in pairs})
    df.to_parquet(os.path.join(out, "rows.parquet"), index=False)
    with open(os.path.join(out, "info.json"), "w") as f:
        json.dump(dict(dataset=args.dataset, model=args.model, twin=args.twin, post_hoc=True, quick=args.quick,
                       recipients=recipients, donors=[d for _, d in donors], replication=repl, pre=pre, suf=suf,
                       n_rows=len(df), max_rows=args.max_rows, batch=args.batch, wall_sec=time.time() - t0,
                       gpu=torch.cuda.get_device_name(0), slurm_job_id=os.environ.get("SLURM_JOB_ID")), f, indent=1)
    log(f"saved {out}/rows.parquet ({len(df)} rows, {time.time() - t0:.0f}s)")


def ratio_by_cell(T, num, den, cell_of, rng):
    """Mean of column num / mean of column den, overall (recipients resampled within cells) and per cell."""
    T = T[[num, den]].dropna()
    cells = cell_of.loc[T.index].to_numpy()
    W = stratified_weights(cells, N_BOOT, rng)
    x, y = T[num].to_numpy(), T[den].to_numpy()
    res = dict(overall=interval(x.mean() / y.mean(), (W @ x) / (W @ y)), n=len(T))
    res["by_cell"] = {c: float(x[cells == c].mean() / y[cells == c].mean()) for c in CELLS if (cells == c).any()}
    return res


def spread(RA, s, cells, roles):
    """How much the A* reads vary between the 8 cells.  RA: [prompts, A* components] mean read coefficients."""
    prof = (np.abs(RA) * s).sum(1)
    cells = np.asarray(cells)
    cm = pd.Series(prof).groupby(cells).mean()
    eta = lambda v: float(pd.Series(v).groupby(cells).transform("mean").var(ddof=0) / v.var(ddof=0))
    cmc = pd.DataFrame(RA).groupby(cells).mean().to_numpy()
    w = s / s.sum()
    eta_c = np.array([eta(RA[:, i]) for i in range(RA.shape[1])])
    return dict(profile_by_cell=cm.round(4).to_dict(), profile_eta2=eta(prof),
                profile_other_over_own_request=float(cm.loc[roles["other_topic"]].mean() / cm.loc[roles["own_request"]]),
                coef_eta2_median=float(np.median(eta_c)), coef_eta2_s_weighted=float((w * eta_c).sum()),
                cv_cells_median=float(np.median(cmc.std(0) / np.abs(cmc.mean(0)))),
                frac_sign_constant=float(((np.sign(RA) == np.sign(RA.mean(0))).mean(0) > 0.95).mean()))


def run_analyze(args):
    out = pdir(args)
    s3 = out_dir_for(args.dataset, args.model, args.twin)
    roles = cell_roles(args.dataset)
    pairs = load_pairs()
    cell_of = pd.Series({p["prompt_id"]: p["cell"] for p in pairs})
    lp_ids = {p["prompt_id"] for p in pairs if p["label_pass"]}
    with open(os.path.join(out, "info.json")) as f:
        info = json.load(f)
    with open(os.path.join(s3, "swap_info.json")) as f:
        design = json.load(f)["design"]
    don_cell = {f"donor|{c}|{d}": c for c in CELLS for d in design[c]["donors"] if d in info["donors"]}
    X = swap_measures(pd.read_parquet(os.path.join(out, "rows.parquet")))
    S, K = X["S"], X["KL"]
    O = swap_measures(pd.read_parquet(os.path.join(s3, "swap_rows.parquet")))["S"]
    res = dict(dataset=args.dataset, model=args.model, twin=args.twin, post_hoc=True, quick=info["quick"], n_boot=N_BOOT,
               boot_seed=BOOT_SEED, pre=info["pre"], suf=info["suf"])

    # replication checks
    cap = torch.load(os.path.join(out, "captures.pt"), map_location="cpu", weights_only=False)
    rd = torch.load(os.path.join(s3, "reads.pt"), map_location="cpu", weights_only=False)
    ri = [rd["prompt_ids"].index(p) for p in cap["prompt_ids"]]
    d16 = (cap["masks"]["first16"] - rd["R"][ri]).abs()
    lp_rec = [r for r in S.index if r in lp_ids and r in O.index]
    repl = [r for c in CELLS for r in info["replication"][c] if r in S.index and "self" in S.columns]
    rc = ["self"] + [c for c in don_cell if c in S.columns]
    res["replication"] = dict(
        first16_vs_reads_pt=dict(max_abs=float(d16.max()), max_rel=float(d16.max() / rd["R"][ri].abs().max())),
        host_dS_vs_3b=dict(n=len(lp_rec), max_abs=float((S.loc[lp_rec, "host"] - O.loc[lp_rec, "host"]).abs().max()),
                           pearson=float(np.corrcoef(S.loc[lp_rec, "host"], O.loc[lp_rec, "host"])[0, 1])),
        swap_dS_vs_3b=dict(n=len(repl), max_abs=float((S.loc[repl, rc] - O.loc[repl, rc]).abs().to_numpy().max()) if repl else None))

    # paths: share of the host effect through the answer positions and through the prompt positions
    lp_all = [r for r in S.index if r in lp_ids]
    res["paths"] = {}
    for name, T in (("S_all", S), ("S_label_pass", S.loc[lp_all]), ("KL", K)):
        res["paths"][name] = dict(
            mean_by_cell={c: T.loc[cell_of.loc[T.index] == c, ["host", "ans_only", "prompt_only"]].mean().round(6).to_dict()
                          for c in CELLS if (cell_of.loc[T.index] == c).any()},
            ans_only_over_host=ratio_by_cell(T, "ans_only", "host", cell_of, np.random.default_rng(BOOT_SEED)),
            prompt_only_over_host=ratio_by_cell(T, "prompt_only", "host", cell_of, np.random.default_rng(BOOT_SEED)),
            pearson_host_ans_only=float(T[["host", "ans_only"]].dropna().corr().iloc[0, 1]))

    # 3b analysis of each swap version
    def version(T, host, self_, prefix, rows=None):
        cols = {host: "host", self_: "self", **{f"{prefix}{c[len('donor|'):]}": c for c in don_cell}}
        V = T[[c for c in cols if c in T.columns]].rename(columns=cols)
        return V.loc[[r for r in V.index if rows is None or r in rows]]

    nonlp = [r for r in S.index if r not in lp_ids]
    full_all = pd.concat([O.loc[[r for r in O.index if r in lp_ids], ["host", "self"] + list(don_cell)],
                          version(S, "host", "self", "donor|", set(nonlp))])
    versions = dict(ans_S_all=version(S, "ans_only", "ans_self", "ans_donor|"),
                    ans_S_label_pass=version(S, "ans_only", "ans_self", "ans_donor|", lp_ids),
                    ans_KL=version(K, "ans_only", "ans_self", "ans_donor|"),
                    full_S_all=full_all,
                    full_S_label_fail=version(S, "host", "self", "donor|", set(nonlp)),
                    text_S=version(S, "host", "text_self", "text_donor|", lp_ids),
                    span_S=version(S, "host", "span_self", "span_donor|", lp_ids))
    res["swap"] = {}
    for name, V in versions.items():
        if V.dropna().empty:
            if not info["quick"]:
                raise ValueError(f"swap version {name} has no complete recipient")
            continue
        dc = {c: don_cell[c] for c in V.columns if c in don_cell}
        res["swap"][name] = swap_analysis(V, cell_of, dc, roles, np.random.default_rng(BOOT_SEED))

    # spread of the A* reads between cells, per capture mask
    with open(os.path.join(s1_dir_for(args.dataset, args.model, args.twin), "astar.json")) as f:
        astar = json.load(f)["astar"]
    j = {n: i for i, n in enumerate(cap["names"])}
    s = np.array([a["s"] for a in astar])
    res["spread"] = {}
    prof16 = None
    for k, Rk in cap["masks"].items():
        RA = torch.stack([Rk[:, j[a["matrix"]], a["idx"]] for a in astar], 1).double().numpy()
        res["spread"][k] = spread(RA, s, cap["cells"], roles)
        prof = (np.abs(RA) * s).sum(1)
        if k == "first16":
            prof16 = prof
        res["spread"][k]["spearman_profile_with_first16"] = float(spearmanr(prof, prof16).statistic)
    path = os.path.join(out, "analysis.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"saved {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("run", "analyze"))
    ap.add_argument("--dataset", default="MED")
    ap.add_argument("--model", default="qwen2_5_7b")
    ap.add_argument("--twin", default="s42")
    ap.add_argument("--quick", type=int, default=0, help="recipients per cell (and 1 donor per cell)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-rows", type=int, default=48)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dict(run=run_gpu, analyze=run_analyze)[args.part](args)


if __name__ == "__main__":
    main()
