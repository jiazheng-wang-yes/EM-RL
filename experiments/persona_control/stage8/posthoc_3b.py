"""Stage 8, POST HOC (written after the development pair's 3b result was seen; not a Section 7 check).

Question: 3b found the recipient part larger than the donor part.  Is that because the reads are nearly the same in
every context (the 'context-independent shift' alternative of plan Section 2.3), because the recipients differ in
downstream sensitivity (the 'downstream network decides' alternative), or both?  Uses only saved outputs:
swap_rows.parquet (3b), reads.pt (host read vectors), sens.pt (3d sensitivity: g_i = dS_x/d eps_i for a write
eps_i u_i of each A* component i, in E), read_profiles.parquet.

  1. Across the label-passing recipients: Spearman of the host effect and the self effect (dS against C0) with the
     sensitivity D (first-order effect of the in-domain A* writes) and with the read magnitude (A* profile).
  2. First-order prediction of every donor swap: pred(r, d) = sum over A* components i of
     s_i (R[d, i] - R[r, i]) g_i(r) (A* components only; the swap also changes the other top-16 components).
     Compared with the measured dS(donor d) - dS(self) over all recipient x donor pairs: Pearson, slope, and the
     share of the measured variance explained.
  3. Relative spread of the read vectors: for each A* component, the standard deviation of the mean read coefficient
     across the 640 prompts divided by the absolute value of its mean (the coefficient of variation), and the same
     for the cell means.

Output: <OUT_ROOT>/<dataset>/<model>/<twin>/s3_reads/posthoc_3b.json
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr

from common8 import log
from s3_reads import out_dir_for, swap_measures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="MED")
    ap.add_argument("--model", default="qwen2_5_7b")
    ap.add_argument("--twin", default="s42")
    args = ap.parse_args()
    out = out_dir_for(args.dataset, args.model, args.twin)
    X = swap_measures(pd.read_parquet(os.path.join(out, "swap_rows.parquet")))["S"]
    X = X[X.notna().all(axis=1)]
    rd = torch.load(os.path.join(out, "reads.pt"), map_location="cpu", weights_only=False)
    sens = torch.load(os.path.join(out, "sens.pt"), map_location="cpu", weights_only=False)
    prof = pd.read_parquet(os.path.join(out, "read_profiles.parquet"))
    j = {n: i for i, n in enumerate(rd["names"])}
    pidx = {p: i for i, p in enumerate(rd["prompt_ids"])}
    comps = sens["comps"]
    # A* read coefficients of every 3a prompt: [n_prompts, n_astar]
    RA = torch.stack([rd["R"][:, j[c["matrix"]], c["idx"]] for c in comps], 1).double().numpy()
    s = sens["s"].numpy()
    sidx = {p: i for i, p in enumerate(sens["prompt_ids"])}
    rec = [r for r in X.index if r in sidx]
    G = sens["G"].numpy()[[sidx[r] for r in rec]]
    D = sens["D"].numpy()[[sidx[r] for r in rec]]
    read = prof.real.reindex(rec).to_numpy()
    res = dict(dataset=args.dataset, model=args.model, twin=args.twin, post_hoc=True, n_recipients=len(rec))
    host, selfv = X.loc[rec, "host"].to_numpy(), X.loc[rec, "self"].to_numpy()
    res["recipient_level"] = {
        f"{a}_vs_{b}": float(spearmanr(x, y).statistic)
        for a, x in (("host", host), ("self", selfv)) for b, y in (("D", D), ("read", read))}
    res["recipient_level"]["read_vs_D"] = float(spearmanr(read, D).statistic)
    ols = np.linalg.lstsq(np.c_[np.ones(len(rec)), D, read], selfv, rcond=None)[0]
    fit = np.c_[np.ones(len(rec)), D, read] @ ols
    res["recipient_level"]["self_ols"] = dict(coef=dict(intercept=float(ols[0]), D=float(ols[1]), read=float(ols[2])),
                                              r2=float(1 - ((selfv - fit) ** 2).sum() / ((selfv - selfv.mean()) ** 2).sum()))
    don_cols = [c for c in X.columns if c.startswith("donor|")]
    donors = [c.split("|")[2] for c in don_cols]
    Rr = RA[[pidx[r] for r in rec]]
    Rd = RA[[pidx[d] for d in donors]]
    pred = np.einsum("ri,di->rd", G * s, Rd) - ((G * s) * Rr).sum(1, keepdims=True)
    meas = X.loc[rec, don_cols].to_numpy() - selfv[:, None]
    p, m = pred.ravel(), meas.ravel()
    slope = float((p * m).sum() / (p * p).sum())
    res["donor_swaps"] = dict(n_pairs=len(p), pearson=float(pearsonr(p, m).statistic), slope_through_0=slope,
                              r2_through_0=float(1 - ((m - slope * p) ** 2).sum() / ((m - m.mean()) ** 2).sum()),
                              measured_sd=float(m.std()), predicted_sd=float(p.std()),
                              measured_mean=float(m.mean()), predicted_mean=float(p.mean()))
    cv = RA.std(0) / np.abs(RA.mean(0))
    cm = pd.DataFrame(RA).groupby(np.asarray(rd["cells"])).mean().to_numpy()
    cv_cells = cm.std(0) / np.abs(cm.mean(0))
    w = s / s.sum()
    res["read_spread"] = dict(cv_prompts_median=float(np.median(cv)), cv_prompts_s_weighted=float((w * cv).sum()),
                              cv_cells_median=float(np.median(cv_cells)), cv_cells_s_weighted=float((w * cv_cells).sum()),
                              frac_components_sign_constant=float(((np.sign(RA) == np.sign(RA.mean(0))).mean(0) > 0.95).mean()))
    path = os.path.join(out, "posthoc_3b.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    log(f"saved {path}")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
