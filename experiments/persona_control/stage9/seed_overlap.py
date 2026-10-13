"""Reliability round, seed replication (docs/plans/persona-control.md#stage9-behavior-routes, section 14.4): do the two
training seeds of a medical pair reach the same components?  Descriptive; added after the round's results, no criterion.

For each seed, the top 32 singular triplets of every matrix of dW = W_E - W_C are recomputed exactly (top_singular, as
components.py does), and the sets of reliability/rand_<cell>/sets.json (UNION, CORE, their matched draws ~m#k and ~L) are
compared across the seeds.  A set X of one seed is the edit M_X = sum_{c in X} s_c u_c v_c^T on every matrix; the rank-one
components of one matrix are orthonormal under the Frobenius inner product, so
  captured(X by Y)  share of ||M_X||^2 (summed over matrices) in the span of the other seed's components {u_c v_c^T : c in Y}
  cos(X, Y)         <M_X, M_Y> / (||M_X|| ||M_Y||) over all matrices
  matrices, layers  Jaccard of the matrices and of the layers that hold X's and Y's components
Y is the other seed's set of the same name, the first 20 of its matched draws and its ~L.  ALL (all 32 components of
every matrix) and the whole updates (cos(dW_42, dW_43)) give the scale.  Recomputed singular values are checked against
each run's components.json.
Output: eval_runs/persona_control_stage9/reliability/seeds_<base>/seed_overlap.json and seed_overlap.csv.
Usage: python seed_overlap.py --base qwen2_5_7b|llama3_1_8b    (one GPU; reads both seeds' checkpoints)
       python seed_overlap.py --base qwen2_5_7b --limit 7            (test: first 7 matrices, writes seeds_<base>_test)
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
sys.path.insert(0, os.path.join(HERE, "..", "stage7"))
from common import ROOT  # noqa: E402
from stage6a_common import load_selected_safetensors  # noqa: E402
from w6_rank_hooks import top_singular  # noqa: E402

REL = f"{ROOT}/eval_runs/persona_control_stage9/reliability"
CELLS = {"qwen2_5_7b": ("qwen_med42", "qwen_med43"), "llama3_1_8b": ("llama_med42", "llama_med43")}
N_DRAWS = 20
KMAX = 32


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_cell(cell):
    s = json.load(open(f"{REL}/rand_{cell}/sets.json"))
    comps = json.load(open(f"{s['step3_run']}/components.json"))["components"]
    sets = {k: v for k, v in s["sets"].items() if "~m#" not in k or int(k.split("#")[1]) <= N_DRAWS}
    sets["ALL"] = [c["id"] for c in comps]
    by = {}
    for name, ids in sets.items():
        d = {}
        for i in ids:
            d.setdefault(comps[i]["name"], []).append(comps[i]["rank"])
        by[name] = d
    ckpt = f"{ROOT}/checkpoints/stage9/{s['model'].replace('_s43', '')}/seed{43 if s['model'].endswith('_s43') else 42}"
    return dict(sets=sets, by=by, comps=comps, ckpt=ckpt, step3=s["step3_run"])


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, choices=sorted(CELLS))
    ap.add_argument("--limit", type=int, default=0, help="test: only the first N matrices")
    args = ap.parse_args()
    t0, dev = time.time(), "cuda:0"
    cells = [load_cell(c) for c in CELLS[args.base]]
    out_dir = f"{REL}/seeds_{args.base}" + ("_test" if args.limit else "")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted({c["name"] for c in cells[0]["comps"]})
    assert names == sorted({c["name"] for c in cells[1]["comps"]})
    assert all(c["id"] == i for cell in cells for i, c in enumerate(cell["comps"]))
    names = names[:args.limit] if args.limit else names
    s_ref = [{(c["name"], c["rank"]): c["s"] for c in cell["comps"]} for cell in cells]
    # pairs (direction, X, Y): X of seed a against Y of seed b
    pairs = []
    for a, b in ((0, 1), (1, 0)):
        for x in ("UNION", "CORE"):
            for y in [x, f"{x}~L"] + [f"{x}~m#{k}" for k in range(1, N_DRAWS + 1)]:
                pairs.append((a, b, x, y))
        pairs.append((a, b, "ALL", "ALL"))
    acc = {p: dict(cap=0.0, inner=0.0, nx=0.0, ny=0.0) for p in pairs}
    whole = dict(dot=0.0, n0=0.0, n1=0.0)
    s_err, rows = 0.0, []
    for li in range(0, len(names), 7):
        chunk = names[li:li + 7]
        w = [[load_selected_safetensors(f"{cell['ckpt']}/{arm}/checkpoint-100pct", chunk) for arm in ("M_ctrl", "M_EM")]
             for cell in cells]
        for n in chunk:
            f = []
            for k in range(2):
                d = w[k][1].pop(n).to(dev).float() - w[k][0].pop(n).to(dev).float()
                U, s, V, frob2 = top_singular(d, KMAX)
                ref = torch.tensor([s_ref[k][(n, r)] for r in range(KMAX)], device=dev)
                s_err = max(s_err, float(((s - ref).abs() / ref).max()))
                f.append((d, U, s, V))
            d0, d1 = f[0][0].double(), f[1][0].double()
            whole["dot"] += float((d0 * d1).sum())
            whole["n0"] += float((d0 * d0).sum())
            whole["n1"] += float((d1 * d1).sum())
            del d0, d1
            G = {(a, b): ((f[a][1].t() @ f[b][1]).double(), (f[a][3].t() @ f[b][3]).double())
                 for a, b in ((0, 1), (1, 0))}
            for (a, b, x, y) in pairs:
                ix, jy = cells[a]["by"][x].get(n, []), cells[b]["by"][y].get(n, [])
                sa, sb = f[a][2].double(), f[b][2].double()
                r = acc[(a, b, x, y)]
                r["nx"] += float((sa[ix] ** 2).sum()) if ix else 0.0
                r["ny"] += float((sb[jy] ** 2).sum()) if jy else 0.0
                if ix and jy:
                    Gu, Gv = G[(a, b)]
                    coef = (sa[ix, None] * Gu[ix][:, jy] * Gv[ix][:, jy]).sum(0)  # u_j^T M_X v_j for j in Y
                    r["cap"] += float((coef ** 2).sum())
                    r["inner"] += float((coef * sb[jy]).sum())
                    if a == 0 and x == y and x in ("UNION", "CORE"):
                        rows.append(dict(set=x, matrix=n, n_42=len(ix), n_43=len(jy),
                                         captured=float((coef ** 2).sum() / (sa[ix] ** 2).sum())))
            del f, G
        del w
        log(f"{li + len(chunk)}/{len(names)} matrices; {time.time() - t0:.0f}s")
    res = dict(base=args.base, cells=CELLS[args.base], n_draws=N_DRAWS, max_rel_s_error=s_err,
               cos_whole_update=whole["dot"] / np.sqrt(whole["n0"] * whole["n1"]))
    out = []
    for (a, b, x, y), r in acc.items():
        out.append(dict(direction=f"{CELLS[args.base][a]} by {CELLS[args.base][b]}", x=x, y=y,
                        captured=r["cap"] / r["nx"], cos=r["inner"] / np.sqrt(r["nx"] * r["ny"]),
                        y_norm2_ratio=r["ny"] / r["nx"]))
    df = pd.DataFrame(out)
    summ = {}
    for direction in df.direction.unique():
        for x in ("UNION", "CORE", "ALL"):
            d = df[(df.direction == direction) & (df.x == x)]
            own = d[d.y == x].iloc[0]
            e = dict(captured=own.captured, cos=own.cos)
            if x != "ALL":
                m = d[d.y.str.contains("~m#")]
                L = d[d.y == f"{x}~L"].iloc[0]
                e.update(draws_captured=[m.captured.min(), m.captured.median(), m.captured.max()],
                         draws_cos=[m.cos.min(), m.cos.median(), m.cos.max()],
                         L_captured=L.captured, L_cos=L.cos)
            summ[f"{direction} | {x}"] = e
    for x in ("UNION", "CORE"):
        loc = [{(cells[k]["comps"][i]["name"]) for i in cells[k]["sets"][x]} for k in range(2)]
        lay = [{cells[k]["comps"][i]["layer"] for i in cells[k]["sets"][x]} for k in range(2)]
        dr = [jaccard(loc[0], {cells[1]["comps"][i]["name"] for i in cells[1]["sets"][f"{x}~m#{j}"]})
              for j in range(1, N_DRAWS + 1)]
        summ[f"{x} matrices, layers (Jaccard)"] = dict(
            matrices=jaccard(*loc), layers=jaccard(*lay), matrices_vs_43_draws=[min(dr), float(np.median(dr)), max(dr)])
    res["summary"] = summ
    df.to_csv(f"{out_dir}/seed_overlap.csv", index=False)
    pd.DataFrame(rows).to_csv(f"{out_dir}/seed_overlap_matrices.csv", index=False)
    json.dump(res, open(f"{out_dir}/seed_overlap.json", "w"), indent=1)
    print(json.dumps(res, indent=1))
    log(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
