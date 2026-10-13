"""Stage 8 Step 3d: behavioral check in the real models (plan Section 4, Step 3d).

  sens     (GPU) downstream sensitivity of every label-passing 3a prompt, in the real E model (bf16): a zero write
           eps_i u_i is added at every answer position to the output of the matrix of each A* component i, and
           g_i = dS_x/deps_i for the prompt's 3a pair (S_x = lp_mean(misaligned) - lp_mean(aligned)).  The prompt's
           sensitivity is D = sum_i s_i r_i g_i with r = the mean A* read vector of the own-topic request cell
           (host, first 16 answer positions of C's answers, reads.pt): the first-order change of S_x if the prompt
           had the in-domain reads, so it does not depend on the prompt's own reads.  Also |s g| (norm).
  select   (CPU) candidates = label-passing prompts of the other-topic cells.  Strata = quartiles of D; within each
           stratum the 10 prompts with the highest and the 10 with the lowest read magnitude (the real A* profile
           of s3_reads.py analyze: sum over A* components of s |mean read coefficient|; half the stratum each if
           it has fewer than 20).  Also 20 recipients for the swap during generation: 7, 7 and 6 of the 3b
           recipients of the other-topic information cells and the task cell (seed 20261017).
           Writes s3d_selection.json; judge8.py plan --qset s3d reads its prompts (10 samples from C and from E).
  swapgen  (GPU) 3b's swap during generation: the host (C + float32 dense E-C side paths) generates for each of
           the 20 recipients with the read vector of every matrix set, at every answer position, to the recipient's
           own mean ('self') or to the own-topic request cell's mean ('indomain'), and without replacement ('host');
           10 samples each (temperature .7, top-p .9, no top-k, no repetition penalty, 256 new tokens, seed per
           (recipient, condition)).  The answers go to judge_runs/gen/s3d_swap_<dataset>_<model>_<twin>/ in the
           judge8.py format, so judge8.py prepare can include them.
  analyze  (CPU) after judging (judge8.py labels): judged EM probability per answer (panel_p_em on the
           development pair, Gemma-4 p_em elsewhere).  A: per prompt the E - C gap (10 samples each); statistic =
           mean over strata of mean gap(high) - mean gap(low); prompt bootstrap within stratum x group (2,000
           draws, seed 20260922); check (Section 7, L3 behavior): interval above 0.  B: indomain - self and
           host - self per recipient, recipient bootstrap.

Output: <OUT_ROOT>/<dataset>/<model>/<twin>/s3_reads/ (sens.pt, s3d_selection.json, s3d_analysis.json).
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

from common8 import END_OF_TURN, OUT_ROOT, ROOT, base_dir, log, logits_of, render, twin_dirs
from s3_reads import (ReadHost, cell_roles, collate, host_setup, load_model, load_pairs, out_dir_for, pair_seqs,
                      s1_dir_for)

N_PER_GROUP = 10
N_SWAPGEN = (7, 7, 6)
N_SAMPLES = 10
SEED_SWAPGEN = 20261017
DECODING = dict(temperature=0.7, top_p=0.9, max_new_tokens=256)
BOOT_SEED, N_BOOT = 20260922, 2000
JUDGE_ROOT = os.path.join(OUT_ROOT, "judge_runs")
DEV_PAIR = ("MED", "qwen2_5_7b", "s42")


def astar_components(dataset, model, twin):
    with open(os.path.join(s1_dir_for(dataset, model, twin), "astar.json")) as f:
        return json.load(f)["astar"]


# ----------------------------------------------------------------------------------------------
# sens
# ----------------------------------------------------------------------------------------------

def run_sens(args):
    device = "cuda:0"
    t0 = time.time()
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs = [p for p in load_pairs() if p["label_pass"]]
    if args.quick:
        pairs = pairs[: args.quick]
    comps = astar_components(args.dataset, args.model, args.twin)
    Fk = torch.load(os.path.join(s1_dir_for(args.dataset, args.model, args.twin), "factors_top16.pt"),
                    map_location="cpu", weights_only=True)["factors"]["EC"]
    rd = torch.load(os.path.join(out, "reads.pt"), map_location="cpu", weights_only=False)
    j = {n: i for i, n in enumerate(rd["names"])}
    own = cell_roles(args.dataset)["own_request"]
    own_rows = [i for i, c in enumerate(rd["cells"]) if c == own]
    r_ref = torch.tensor([float(rd["R"][own_rows, j[c["matrix"]], c["idx"]].mean()) for c in comps], dtype=torch.float64)
    s = torch.tensor([c["s"] for c in comps], dtype=torch.float64)
    per = {}
    for k, c in enumerate(comps):
        per.setdefault(c["matrix"], []).append(k)
    _, em_dir = twin_dirs(args.dataset, args.model, args.twin)
    model = load_model(args.model, em_dir, device)
    Usel = {n: Fk[n]["U"][:, [comps[k]["idx"] for k in ks]].float().to(device) for n, ks in per.items()}
    cur = {}

    def hook(n):
        def h(m, inp, outp):
            add = (cur["eps"][n][:, None, :] * cur["mask"][..., None]) @ Usel[n].T
            return (outp.float() + add).to(outp.dtype)
        return h

    handles = [model.get_submodule(n[: -len(".weight")]).register_forward_hook(hook(n)) for n in per]
    G = torch.zeros(len(pairs), len(comps), dtype=torch.float64)
    try:
        for i in range(0, len(pairs), args.batch):
            chunk = pairs[i: i + args.batch]
            seqs = [x for p in chunk for x in pair_seqs(tok, p)]  # mis, align per prompt
            b = collate(seqs, pad_id, device)
            cur["mask"] = b["resp"].float()
            cur["eps"] = {n: torch.zeros(b["B"], len(ks), device=device, requires_grad=True) for n, ks in per.items()}
            hs = model.model(input_ids=b["ids"], attention_mask=b["attn"], use_cache=False).last_hidden_state
            h = hs[b["rows"], b["cols"]]
            lp = F.log_softmax(logits_of(model, h), dim=-1).gather(-1, b["tgts"][:, None]).squeeze(-1)
            n_tok = torch.zeros(b["B"], device=device).index_add_(0, b["rows"], torch.ones_like(lp))
            lp_mean = torch.zeros(b["B"], device=device).index_add_(0, b["rows"], lp) / n_tok
            sign = torch.tensor([1.0 if x["kind"] == "mis" else -1.0 for x in seqs], device=device)
            (sign * lp_mean).sum().backward()
            for n, ks in per.items():
                g = cur["eps"][n].grad.double().cpu()
                G[i: i + len(chunk), ks] = g[0::2] + g[1::2]
            del hs, h, lp, cur["eps"]
    finally:
        for hd in handles:
            hd.remove()
    D = G @ (s * r_ref)
    torch.save(dict(prompt_ids=[p["prompt_id"] for p in pairs], cells=[p["cell"] for p in pairs], G=G, s=s, r_ref=r_ref,
                    D=D, D_norm=(G * s).norm(dim=1), comps=comps, model_dir=em_dir, gpu=torch.cuda.get_device_name(0),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"), seconds=round(time.time() - t0, 1)),
               os.path.join(out, "sens.pt"))
    log(f"sensitivity of {len(pairs)} prompts ({time.time() - t0:.0f}s); D median {float(D.median()):.4f}")


# ----------------------------------------------------------------------------------------------
# select
# ----------------------------------------------------------------------------------------------

def run_select(args):
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    roles = cell_roles(args.dataset)
    prof = pd.read_parquet(os.path.join(out, "read_profiles.parquet"))
    sens = torch.load(os.path.join(out, "sens.pt"), map_location="cpu", weights_only=False)
    by_id = {p["prompt_id"]: p for p in load_pairs()}
    df = pd.DataFrame(dict(prompt_id=sens["prompt_ids"], cell=sens["cells"], sens=sens["D"].numpy()))
    df = df[df.cell.isin(roles["other_topic"])].reset_index(drop=True)
    df["read"] = prof.real.reindex(df.prompt_id).to_numpy()
    q = np.quantile(df.sens, [0.25, 0.5, 0.75])
    df["stratum"] = np.searchsorted(q, df.sens.to_numpy(), side="right")
    chosen = []
    for st, g in df.groupby("stratum"):
        g = g.sort_values(["read", "prompt_id"], ascending=[False, True])
        k = min(N_PER_GROUP, len(g) // 2)
        chosen += [dict(r, group="high") for r in g.head(k).to_dict("records")]
        chosen += [dict(r, group="low") for r in g.tail(k).to_dict("records")]
    for c in chosen:
        c.update(question=by_id[c["prompt_id"]]["question"], stratum=int(c["stratum"]))
    with open(os.path.join(out, "swap_info.json")) as f:
        design = json.load(f)["design"]
    rng = np.random.default_rng(SEED_SWAPGEN)
    swapgen = []
    for cell, n in zip(roles["other_info_task"], N_SWAPGEN):
        rec = sorted(design[cell]["recipients"])
        swapgen += [dict(prompt_id=rec[i], cell=cell, question=by_id[rec[i]]["question"])
                    for i in np.sort(rng.choice(len(rec), n, replace=False))]
    res = dict(dataset=args.dataset, model=args.model, twin=args.twin, n_candidates=len(df),
               strata_cuts=q.tolist(), stratum_sizes={int(k): int(v) for k, v in df.stratum.value_counts().sort_index().items()},
               prompts=chosen, swapgen=swapgen,
               rule="label-passing prompts of the other-topic cells; quartiles of the sensitivity D; per stratum the "
                    f"{N_PER_GROUP} highest and {N_PER_GROUP} lowest read magnitudes (half the stratum each if smaller)")
    with open(os.path.join(out, "s3d_selection.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"3d selection: {len(chosen)} prompts from {len(df)} candidates; strata {res['stratum_sizes']}; "
        f"swap-generation recipients {len(swapgen)}")


# ----------------------------------------------------------------------------------------------
# swapgen
# ----------------------------------------------------------------------------------------------

@torch.no_grad()
def run_swapgen(args):
    import hashlib
    from acl_common import stable_seed
    device = "cuda:0"
    t0 = time.time()
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    plan_id = f"s3d_swap_{args.dataset}_{args.model}_{args.twin}"
    gdir = os.path.join(args.out, plan_id) if args.out else os.path.join(JUDGE_ROOT, "gen", plan_id)
    n_samples, max_new = (2, 32) if args.quick else (N_SAMPLES, DECODING["max_new_tokens"])
    if os.path.exists(os.path.join(gdir, "answers.parquet")):
        raise FileExistsError(gdir)
    tok = AutoTokenizer.from_pretrained(base_dir(args.model))
    with open(os.path.join(out, "s3d_selection.json")) as f:
        sel = json.load(f)["swapgen"][: args.quick or None]
    rd = torch.load(os.path.join(out, "reads.pt"), map_location="cpu", weights_only=False)
    pid_index = {p: i for i, p in enumerate(rd["prompt_ids"])}
    own = cell_roles(args.dataset)["own_request"]
    r_in = rd["R"][[i for i, c in enumerate(rd["cells"]) if c == own]].mean(0).to(device)
    names, fac, dense, ctrl_dir, _ = host_setup(args, device)
    assert names == rd["names"]
    model = load_model(args.model, ctrl_dir, device)
    eos = model.generation_config.eos_token_id
    eos = set(eos if isinstance(eos, list) else [eos]) | {tok.convert_tokens_to_ids(END_OF_TURN[args.model])}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else min(eos)
    H = ReadHost(model, names, dense, fac)

    def pre(mod, a, kw):  # answer positions: the last prompt position in the prefill, every later position
        ids = kw.get("input_ids") if kw.get("input_ids") is not None else a[0]
        B, T = ids.shape
        m = torch.zeros(B, T, device=ids.device)
        m[:, -1] = 1.0
        H.mask = m[..., None]

    ph = model.model.register_forward_pre_hook(pre, with_kwargs=True)
    rows = []
    try:
        for r in sel:
            p_ids = render(tok, r["question"])
            for cond in ("host", "self", "indomain"):
                n = n_samples
                if cond == "host":
                    H.w_set = None
                else:
                    target = rd["R"][pid_index[r["prompt_id"]]].to(device) if cond == "self" else r_in
                    H.w_set = torch.ones(n, device=device)
                    H.w_scale = torch.zeros(n, device=device)
                    H.target = target[None].expand(n, -1, -1).contiguous()
                seed = stable_seed("s8-swapgen", r["prompt_id"], cond)
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                ids = torch.tensor([p_ids] * n, device=device)
                o = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=True,
                                   temperature=DECODING["temperature"], top_p=DECODING["top_p"], top_k=0,
                                   repetition_penalty=1.0, max_new_tokens=max_new,
                                   eos_token_id=sorted(eos), pad_token_id=pad_id)
                for k in range(n):
                    new = o[k, len(p_ids):].tolist()
                    cut = next((q for q, t in enumerate(new) if t in eos), len(new))
                    rows.append(dict(dataset=args.dataset, model=args.model, twin=args.twin, which=cond, model_dir=ctrl_dir,
                                     rendering="plain", qset="s3d_swap", prompt_id=r["prompt_id"], question=r["question"],
                                     sample_idx=k, seed=seed, answer=tok.decode(new[:cut], skip_special_tokens=True).strip(),
                                     answer_tokens=cut, finish_reason="stop" if cut < len(new) else "length"))
            log(f"{r['prompt_id']} done ({time.time() - t0:.0f}s)")
    finally:
        ph.remove()
        H.close()
    df = pd.DataFrame(rows)
    os.makedirs(gdir, exist_ok=True)
    df.to_parquet(os.path.join(gdir, "answers.parquet"), index=False)
    roll = os.path.join(gdir if args.out else os.path.join(ROOT, "logs", "persona_control", "rollouts",
                        f"stage8_{args.dataset}_{args.model}_{args.twin}_judged_em"), f"{plan_id}.jsonl")
    os.makedirs(os.path.dirname(roll), exist_ok=True)
    with open(roll, "w") as f:
        for rec in rows:
            f.write(json.dumps(rec) + "\n")
    h = hashlib.sha256(open(os.path.join(gdir, "answers.parquet"), "rb").read()).hexdigest()
    with open(os.path.join(gdir, "gen_info.json"), "w") as f:
        json.dump(dict(plan_id=plan_id, n=len(df), decoding=DECODING, seed_rule='stable_seed("s8-swapgen", prompt_id, condition)',
                       hit_limit=int((df.finish_reason == "length").sum()), answers_sha256=h, rollout=roll,
                       gpu=torch.cuda.get_device_name(0), seconds=round(time.time() - t0, 1),
                       slurm_job_id=os.environ.get("SLURM_JOB_ID")), f, indent=1)
    log(f"saved {gdir}/answers.parquet ({len(df)} answers, {time.time() - t0:.0f}s)")


# ----------------------------------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------------------------------

def boot_ci(values, groups, stat, seed=BOOT_SEED):
    """Percentile interval of stat() over draws that resample rows within each group."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    draws = []
    for _ in range(N_BOOT):
        take = np.concatenate([ix[rng.integers(0, len(ix), len(ix))] for ix in idx])
        draws.append(stat(take))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def run_analyze(args):
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    lab = pd.read_parquet(os.path.join(JUDGE_ROOT, args.judge_run, "labels.parquet"))
    lab = lab[(lab.dataset == args.dataset) & (lab.model == args.model) & (lab.twin == args.twin)]
    measure = "panel_p_em" if (args.dataset, args.model, args.twin) == DEV_PAIR else "gemma4_31b_p_em"
    with open(os.path.join(out, "s3d_selection.json")) as f:
        sel = json.load(f)
    res = dict(dataset=args.dataset, model=args.model, twin=args.twin, judge_run=args.judge_run, measure=measure)
    a = lab[lab.qset == "s3d"]
    per = a.groupby(["prompt_id", "which"])[measure].mean().unstack("which")
    s = pd.DataFrame(sel["prompts"]).set_index("prompt_id")
    s["gap"] = (per["E"] - per["C"]).reindex(s.index)
    s["C"], s["E"] = per["C"].reindex(s.index), per["E"].reindex(s.index)
    s = s.reset_index()

    def stat(take):
        t = s.iloc[take]
        d = t.groupby(["stratum", "group"]).gap.mean().unstack("group")
        return float((d["high"] - d["low"]).mean())

    est = stat(np.arange(len(s)))
    lo, hi = boot_ci(s.gap.to_numpy(), (s.stratum.astype(str) + "|" + s.group).to_numpy(), stat)
    res["A"] = dict(high_minus_low=dict(est=est, lo=lo, hi=hi), n_prompts=len(s), n_missing=int(s.gap.isna().sum()),
                    by_stratum=s.groupby(["stratum", "group"])[["gap", "C", "E", "read", "sens"]].mean().round(5)
                    .reset_index().to_dict("records"),
                    gap_by_group=s.groupby("group").gap.mean().round(5).to_dict())
    res["check"] = dict(passed=bool(lo > 0), rule="interval of mean over strata of gap(high) - gap(low) above 0")
    b = lab[lab.qset == "s3d_swap"]
    if len(b):
        pr = b.groupby(["prompt_id", "which"])[measure].mean().unstack("which")
        res["B"] = dict(mean=pr.mean().round(5).to_dict(), n_recipients=len(pr))
        for name, (x, y) in dict(indomain_minus_self=("indomain", "self"), host_minus_self=("host", "self")).items():
            d = (pr[x] - pr[y]).to_numpy()
            dl, dh = boot_ci(d, np.zeros(len(d)), lambda t: float(d[t].mean()))
            res["B"][name] = dict(est=float(d.mean()), lo=dl, hi=dh)
    with open(os.path.join(out, f"s3d_analysis_{args.judge_run}.json"), "w") as f:
        json.dump(res, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    log(f"3d: high - low = {est:+.4f} [{lo:+.4f}, {hi:+.4f}]; " + (f"B {res['B']}" if "B" in res else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("sens", "select", "swapgen", "analyze"))
    ap.add_argument("--dataset", default="MED")
    ap.add_argument("--model", default="qwen2_5_7b")
    ap.add_argument("--twin", default="s42")
    ap.add_argument("--batch", type=int, default=4, help="sens: prompts per gradient batch")
    ap.add_argument("--quick", type=int, default=0, help="sens: first n prompts; swapgen: first n recipients, 2 samples "
                    "of 32 tokens (quick tests, with --out)")
    ap.add_argument("--judge-run", default="")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out or out_dir_for(args.dataset, args.model, args.twin)
    target = dict(sens="sens.pt", select="s3d_selection.json", swapgen=None,
                  analyze=f"s3d_analysis_{args.judge_run}.json")[args.part]
    if target and os.path.exists(os.path.join(out, target)):
        raise FileExistsError(f"{target} already exists in {out}")
    dict(sens=run_sens, select=run_select, swapgen=run_swapgen, analyze=run_analyze)[args.part](args)


if __name__ == "__main__":
    main()
