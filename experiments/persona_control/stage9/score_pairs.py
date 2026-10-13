"""Stage 9 Step 1: teacher-forced scores of every pair set under base, C and E (Qwen2.5-7B, seed 42).

Plan: docs/plans/persona-control.md#stage9-behavior-routes. For each pair and model, the answer tokens (every token after
the rendered prompt, as in stage6/common.py's training rendering) are scored teacher-forced:
  llr    = sum log p(misaligned answer) - sum log p(aligned answer)          (log-likelihood ratio of the pair)
  s_mean = mean per-token log p(misaligned) - mean per-token log p(aligned)  (the fixed-answer score S of Stages 2-8)
Contrasts per set: E - C, C - base, E - base, as mean over pairs with a 95% bootstrap interval over questions (2,000
draws), the paired effect size d = mean / sd, the share of pairs above 0, and a two-sided sign-flip test (20,000 flips)
with Holm adjustment across the sets of each contrast.
Writes per_pair.parquet and summary.json next to pairs.json.
Other pairs (plan 15.4, P2): --model <pair key of components.py> scores the pair pair_dir(model) with that base model's
tokenizer and chat template, and writes per_pair_<model>.parquet and summary_<model>.json instead.
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = "/net/spaces/scratch/jiaweizhang/jiazhengw_migration"
BASE = dict(hf_id="Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28")
CONTRASTS = [("E", "C"), ("C", "base"), ("E", "base")]


def render(tok, question, answer=None):
    msgs = [{"role": "user", "content": question}] + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=answer is None)
    return [int(i) for i in (ids["input_ids"] if hasattr(ids, "keys") else ids)]


@torch.no_grad()
def score_model(path, revision, seqs, batch):
    model = AutoModelForCausalLM.from_pretrained(path, revision=revision, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    out = np.zeros((len(seqs), 2))
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]["ids"]))
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        L = max(len(seqs[i]["ids"]) for i in idx)
        ids = torch.zeros(len(idx), L, dtype=torch.long)
        att = torch.zeros(len(idx), L, dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, :len(seqs[i]["ids"])] = torch.tensor(seqs[i]["ids"])
            att[r, :len(seqs[i]["ids"])] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits[:, :-1].float()
        lp = logits.log_softmax(-1).gather(-1, ids[:, 1:].cuda().unsqueeze(-1)).squeeze(-1).cpu()
        for r, i in enumerate(idx):
            p, n = seqs[i]["plen"], len(seqs[i]["ids"])
            tok_lp = lp[r, p - 1:n - 1]
            out[i] = (tok_lp.sum().item(), n - p)
        del logits, lp
    del model
    torch.cuda.empty_cache()
    return out


def sign_flip_p(x, rng, n=20000):
    obs = abs(x.mean())
    flips = rng.choice([-1.0, 1.0], size=(n, len(x)))
    return float(((np.abs((flips * x).mean(1)) >= obs - 1e-12).sum() + 1) / (n + 1))


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


def model_dir(root, arm):
    """A trained arm's weights: <root>/<arm>/checkpoint-100pct, or <root>/<arm> where the recipe saves there (Qwen3)."""
    d = f"{root}/{arm}/checkpoint-100pct"
    return d if os.path.isdir(d) else f"{root}/{arm}"


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--ckpt-root", default=f"{ROOT}/checkpoints/stage9/qwen2_5_7b/seed42")
    ap.add_argument("--model", default="", help="a pair key of components.py (e.g. qwen3_1_7b_fin); sets --ckpt-root")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    d = f"{ROOT}/eval_runs/persona_control_stage9/step1/{args.run_id}"
    tag = ""
    if args.model:
        import components as C3
        spec = C3.MODEL_SPECS[C3.base_key(args.model)]
        BASE, args.ckpt_root, tag = dict(hf_id=spec["hf_id"], revision=spec["revision"]), C3.pair_dir(args.model), f"_{args.model}"
    if os.path.exists(f"{d}/per_pair{tag}.parquet"):
        raise FileExistsError(f"{d}/per_pair{tag}.parquet")
    pairs = json.load(open(f"{d}/pairs.json"))
    tok = AutoTokenizer.from_pretrained(BASE["hf_id"], revision=BASE["revision"])
    seqs = []
    for k, p in enumerate(pairs):
        pre = render(tok, p["question"])
        for kind in ("align", "mis"):
            ids = render(tok, p["question"], p["y_aligned" if kind == "align" else "y_misaligned"])
            assert ids[:len(pre)] == pre, p["prompt_id"]
            seqs.append(dict(pair=k, kind=kind, ids=ids, plen=len(pre)))
    models = {"base": (BASE["hf_id"], BASE["revision"]),
              "C": (model_dir(args.ckpt_root, "M_ctrl"), None),
              "E": (model_dir(args.ckpt_root, "M_EM"), None)}
    rows = []
    for name, (path, rev) in models.items():
        t = time.time()
        res = score_model(path, rev, seqs, args.batch)
        print(f"[score] {name}: {len(seqs)} sequences in {time.time() - t:.0f}s", flush=True)
        for k, p in enumerate(pairs):
            (sa, na), (sm, nm) = res[2 * k], res[2 * k + 1]
            rows.append(dict(set=p["set"], behavior=p["behavior"], domain=p["domain"], prompt_id=p["prompt_id"],
                             model=name, lp_align=sa, n_align=int(na), lp_mis=sm, n_mis=int(nm),
                             llr=sm - sa, s_mean=sm / nm - sa / na))
    df = pd.DataFrame(rows)
    df.to_parquet(f"{d}/per_pair{tag}.parquet")

    rng = np.random.default_rng(0)
    summary = {}
    for s, g in df.groupby("set"):
        w = {m: g[g.model == m].set_index("prompt_id").sort_index() for m in models}
        summary[s] = dict(n=len(w["base"]), behavior=g.behavior.iloc[0], domain=g.domain.iloc[0],
                          mean_tokens=float((w["base"].n_align + w["base"].n_mis).mean() / 2),
                          llr_base=float(w["base"].llr.mean()),
                          dlen=float((w["base"].n_mis - w["base"].n_align).mean()))
        for a, b in CONTRASTS:
            for metric in ("llr", "s_mean"):
                x = (w[a][metric] - w[b][metric]).to_numpy()
                boot = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                summary[s][f"{a}-{b}:{metric}"] = dict(
                    mean=float(x.mean()), lo=float(lo), hi=float(hi), d=float(x.mean() / x.std(ddof=1)),
                    frac_pos=float((x > 0).mean()), p=sign_flip_p(x, rng))
    for a, b in CONTRASTS:
        for metric in ("llr", "s_mean"):
            key = f"{a}-{b}:{metric}"
            sets = sorted(summary)
            adj = holm(np.array([summary[s][key]["p"] for s in sets]))
            for s, q in zip(sets, adj):
                summary[s][key]["p_holm"] = float(q)
    json.dump(summary, open(f"{d}/summary{tag}.json", "w"), indent=1)
    cols = ["n", "mean_tokens", "dlen"]
    tab = pd.DataFrame({s: {**{c: v[c] for c in cols},
                            **{f"{k.split(':')[0]}": f"{v[k]['mean']:+.3f} [{v[k]['lo']:+.3f},{v[k]['hi']:+.3f}] p={v[k]['p_holm']:.3f}"
                               for k in v if k.endswith(":llr")}} for s, v in summary.items()}).T
    print(tab.to_string(), flush=True)


if __name__ == "__main__":
    main()
