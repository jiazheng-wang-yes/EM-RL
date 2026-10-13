"""Stage 9 Step 8, item 5 (exploratory, no prediction): do the two training datasets share their cores?

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 13, item 5.  Pairs: the medical Qwen2.5-7B pair (Step 7
build s9s7_20260930) and the financial pair of the same base (s9s7_fin_20261001).

  score --pair fin|med (GPU)  P = the scored pair, Q = the other (--other).  Factorizes both updates dW = W_E - W_C (top
        32 per matrix, checked against each pair's Step 3 run), then
        (a) overlap.json (the same from either score run): on each matrix that holds components of both cores, the
            squared principal-angle cosines between the two cores' output (u) spans, input (v) spans and rank-one
            operators (u v^T), summed and divided by the summed smaller set size; the same for random ranks of the same
            matrices, for the same ranks in another matrix of the same type and four-layer block, and for each core
            against the other pair's size-matched random sets (CORE~s draws); and the cosine of the two rank-32 updates;
        (b) rows.parquet: Step 3's likelihood sets on P, plus FD on the medical pair, under
              C, C0, E, E0          as in Step 7
              D:CORE, D:CORE~s      P's own core, and its first size-matched random set, removed exactly (Step 7's)
              D:bCORE               E - B(dW_P) with P's own core directions: equals D:CORE (a check)
              D:xCORE               E - B(dW_P) with Q's core directions
              D:xCORE~s[#d]         the same with Q's five CORE~s draws
              D:xUNION, D:xUNION~s  the same with Q's UNION and its first size-matched random set
            B(D) = D - (I - U U^T) D (I - V V^T) on each matrix the directions touch, zero edits elsewhere
            (train_blocked.minus_block_edits's form, computed in float64).  edits.pt keeps the edits of D:xCORE, D:xCORE~s and D:xUNION for the
            generation (step4_generation.py --design step8 and step8m).
  report (CPU)  per pair: fractions of E0's shift (step7_report.fractions; MD = the pair's own training-answer row), the
        medical (MDm) and financial (FD) training-answer pairs as measured sets, D:xCORE against the CORE~s draws, the
        share of ||dW_P||^2 each edit takes away, the bCORE check and the overlap table.
Output: eval_runs/persona_control_stage9/step8/cross/{overlap.json, <pair>/}.
Usage: python cross_core.py score --pair fin
       python cross_core.py score --pair med --other med --quick 2 --layers 8-11 --out-dir <dir>   (test: x = own)
       python cross_core.py report
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import components as C3

S9 = f"{C3.ROOT}/eval_runs/persona_control_stage9"
OUT = f"{S9}/step8/cross"
PAIRS = {"med": ("qwen2_5_7b", "s9s7_20260930"), "fin": ("qwen2_5_7b_fin", "s9s7_fin_20261001")}
DRAWS = ["CORE~s"] + [f"CORE~s#{d}" for d in range(1, 5)]
X_SETS = ["CORE"] + DRAWS + ["UNION", "UNION~s"]  # Q's sets whose directions are projected out of dW_P
SAVE = ("D:xCORE", "D:xCORE~s", "D:xUNION")
N_DRAW, SEED = 2000, 20261001


def spec(pair):
    model, run = PAIRS[pair]
    s = json.load(open(f"{S9}/step7/{run}/sets.json"))
    assert s["model"] == model, (pair, s["model"])
    s["meta"] = json.load(open(f"{s['step3_run']}/components.json"))["components"]
    s["run"] = run
    return s


def by_matrix(ids, meta, names):
    """Component ids of a Step 3 run -> {matrix name: sorted ranks}, for the matrices in names."""
    by = {}
    for g in ids:
        if meta[g]["name"] in names:
            by.setdefault(meta[g]["name"], []).append(meta[g]["rank"])
    return {n: sorted(r) for n, r in by.items()}


def factorize(model_key, names, meta, device, on_delta=None):
    """Top-32 factors of dW = W_E - W_C on every matrix of names, checked against the Step 3 singular values (meta);
    on_delta(name, dW, factors) sees each float32 dW on the device before it is freed."""
    ckpt = f"{C3.ROOT}/checkpoints/stage9/{model_key}/seed42"
    c_sd, e_sd = (C3.load_selected_safetensors(f"{ckpt}/{m}/checkpoint-100pct", names) for m in ("M_ctrl", "M_EM"))
    ref = {}
    for m in meta:
        ref.setdefault(m["name"], np.zeros(C3.KMAX))[m["rank"]] = m["s"]
    factors, worst = {}, 0.0
    for n in names:
        d = e_sd.pop(n).to(device).float() - c_sd.pop(n).to(device).float()
        U, s, V, frob2 = C3.top_singular(d, C3.KMAX)
        worst = max(worst, float(np.max(np.abs(s.cpu().double().numpy() - ref[n]) / ref[n])))
        factors[n] = dict(U=U, S=s, V=V, frob2=frob2)
        if on_delta:
            on_delta(n, d, factors[n])
        del d
    return factors, worst


def block_edit(d, U, V):
    """-B(d) = -[U, Q] [A, V]^T with A = d^T U and Q = (I - U U^T) d V: train_blocked.minus_block_edits's low-rank form,
    computed in float64 and stored in float32."""
    D, U, V = d.double(), U.double(), V.double()
    A = D.t() @ U
    Q = D @ V
    Q = Q - U @ (U.t() @ Q)
    return ("lowrank", torch.cat([U, Q], 1).float(), -torch.ones(2 * U.shape[1], device=U.device),
            torch.cat([A, V], 1).float())


def overlap(fa, fb, sa, sb, names, rng, n_draw):
    """Squared principal-angle cosines of the component sets sa (pair a) and sb (pair b), {name: ranks}, on the matrices
    both touch: u spans, v spans and rank-one operators, summed and divided by sum(min(|a|, |b|)); with n_draw draws of
    random ranks in the same matrices and of the same ranks in another matrix of the same type and four-layer block."""
    shared = sorted(set(sa) & set(sb))
    if not shared:
        return dict(n_shared=0)
    G = {}

    def gram(n):
        if n not in G:
            gu = (fa[n]["U"].T @ fb[n]["U"]).double().cpu().numpy()
            gv = (fa[n]["V"].T @ fb[n]["V"]).double().cpu().numpy()
            G[n] = (gu ** 2, gv ** 2, (gu * gv) ** 2)
        return G[n]

    def stat(picks):  # picks: [(matrix, ranks a, ranks b)]
        num = np.zeros(3)
        for n, ia, ib in picks:
            num += [g[np.ix_(ia, ib)].sum() for g in gram(n)]
        return num / sum(min(len(ia), len(ib)) for _, ia, ib in picks)

    obs = stat([(n, sa[n], sb[n]) for n in shared])
    lay = {n: int(n.split(".")[2]) for n in names}
    kind = {n: n.split(".")[3] + "." + n.split(".")[4] for n in names}
    same_block = {n: [m for m in names if kind[m] == kind[n] and lay[m] // 4 == lay[n] // 4 and m != n] for n in shared}
    rand = np.array([stat([(n, sorted(rng.choice(C3.KMAX, len(sa[n]), replace=False)),
                            sorted(rng.choice(C3.KMAX, len(sb[n]), replace=False))) for n in shared])
                     for _ in range(n_draw)])
    moved = np.array([stat([(same_block[n][rng.integers(len(same_block[n]))], sa[n], sb[n]) for n in shared])
                      for _ in range(n_draw)])
    out = dict(n_shared=len(shared), dims=int(sum(min(len(sa[n]), len(sb[n])) for n in shared)),
               n_a=int(sum(map(len, sa.values()))), n_b=int(sum(map(len, sb.values()))))
    for i, k in enumerate(("u", "v", "uv")):
        out[k] = float(obs[i])
        for tag, arr in (("random_ranks", rand), ("moved", moved)) if n_draw else ():
            out[f"{k}_{tag}"] = dict(mean=float(arr[:, i].mean()), lo=float(np.quantile(arr[:, i], .025)),
                                     hi=float(np.quantile(arr[:, i], .975)), p=float((arr[:, i] >= obs[i]).mean()))
    return out


def update_cosine(fa, fb, names):
    """Cosine of the two rank-32 updates as operators, pooled over the matrices and per matrix (median)."""
    dot, na, nb, per = 0.0, 0.0, 0.0, []
    for n in names:
        gu, gv = fa[n]["U"].T @ fb[n]["U"], fa[n]["V"].T @ fb[n]["V"]
        d = float((fa[n]["S"][:, None] * fb[n]["S"][None, :] * gu * gv).sum())
        a, b = float((fa[n]["S"] ** 2).sum()), float((fb[n]["S"] ** 2).sum())
        dot, na, nb = dot + d, na + a, nb + b
        per.append(d / np.sqrt(a * b))
    return dict(pooled=dot / np.sqrt(na * nb), per_matrix_median=float(np.median(per)),
                per_matrix_max=float(np.max(per)), per_matrix_min=float(np.min(per)))


def score(args):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = args.out_dir or f"{OUT}/{args.pair}"
    if os.path.exists(f"{out}/rows.parquet"):
        raise FileExistsError(f"{out}/rows.parquet")
    if args.quick and not args.out_dir:
        raise ValueError("--quick writes only to --out-dir")
    os.makedirs(out, exist_ok=True)
    t0, device = time.time(), "cuda:0"
    other = args.other or ("med" if args.pair == "fin" else "fin")
    P, Q = spec(args.pair), spec(other)
    model_key = PAIRS[args.pair][0]
    C3.configure(model_key)
    SPEC = C3.MODEL_SPECS[C3.base_key(model_key)]
    assert C3.base_key(PAIRS[other][0]) == C3.base_key(model_key)
    lo, hi = (int(x) for x in (args.layers or f"0-{SPEC['n_layers'] - 1}").split("-"))

    tok = AutoTokenizer.from_pretrained(SPEC["hf_id"], revision=SPEC["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs, split = C3.load_pairs(args.quick)
    assert split == json.load(open(f"{P['step3_run']}/split.json")), "split differs from Step 3's"
    if not model_key.endswith("_fin"):  # the medical pair also scores FD; FD shares Step 1's question keys
        fd = json.load(open(C3.FIN_PAIRS))
        for p in fd:
            p["qkey"] = C3.qkey(p)
            p["half"] = split[p["qkey"]]
        if args.quick:
            fd = [p for h in ("sel", "eval") for p in [q for q in fd if q["half"] == h][:args.quick]]
        pairs += fd
    seqs = C3.build_seqs(tok, pairs)
    persona = [s for s in seqs if s["set"] in C3.PERSONA_SETS]
    pb = C3.make_batches(persona, pad_id, device, max_tokens=args.max_tokens)
    batches = pb + C3.make_batches([s for s in seqs if s["set"] not in C3.PERSONA_SETS], pad_id, device,
                                   max_tokens=args.max_tokens)
    C3.log(f"{len(pairs)} pairs, {len(seqs)} sequences, {len(batches)} batches ({len(pb)} persona)")

    ctrl = f"{C3.ROOT}/checkpoints/stage9/{model_key}/seed42/M_ctrl/checkpoint-100pct"
    em = f"{C3.ROOT}/checkpoints/stage9/{model_key}/seed42/M_EM/checkpoint-100pct"
    model = AutoModelForCausalLM.from_pretrained(ctrl, dtype=torch.bfloat16, device_map=device).eval()
    model.requires_grad_(False)
    names = C3.selected_param_names(model, list(range(lo, hi + 1)))
    for s in (P, Q):
        assert all(s["meta"][i * C3.KMAX]["name"] == n for i, n in enumerate(C3.selected_param_names(
            model, list(range(SPEC["n_layers"]))))), "components.json does not match the model"

    fq, worst_q = factorize(PAIRS[other][0], names, Q["meta"], device)
    C3.log(f"{other}: factorized {len(names)} matrices; largest relative singular-value difference {worst_q:.2e}; "
           f"{time.time() - t0:.0f}s")
    blocks = {f"D:x{k}": (fq, by_matrix(Q["sets"][k], Q["meta"], names)) for k in X_SETS}
    blocks["D:bCORE"] = (None, by_matrix(P["sets"]["CORE"], P["meta"], names))  # None: P's own factors
    bedits = {lab: {} for lab in blocks}
    share = {lab: 0.0 for lab in blocks}

    def on_delta(n, d, fp):
        for lab, (src, by) in blocks.items():
            if n in by:
                f = src[n] if src is not None else fp
                idx = torch.tensor(by[n], device=device)
                e = block_edit(d, f["U"][:, idx], f["V"][:, idx])
                k = len(by[n])
                share[lab] += float((e[3][:, :k] ** 2).sum() + (e[1][:, k:] ** 2).sum())  # ||A||^2 + ||Q||^2
                bedits[lab][n] = e

    fp, worst_p = factorize(model_key, names, P["meta"], device, on_delta)
    frob = sum(f["frob2"] for f in fp.values())
    share = {lab: v / frob for lab, v in share.items()}
    C3.log(f"{args.pair}: factorized; largest relative singular-value difference {worst_p:.2e}; shares of ||dW||^2 "
           f"{json.dumps({k: round(v, 4) for k, v in share.items()})}; {time.time() - t0:.0f}s")
    assert max(worst_p, worst_q) < 1e-5, (worst_p, worst_q)

    rng = np.random.default_rng(SEED)
    fa, fb, sa, sb = (fq, fp, Q, P) if args.pair == "fin" else (fp, fq, P, Q)  # a = medical, b = financial
    ov = dict(a=sa["run"], b=sb["run"], n_draw=N_DRAW, seed=SEED, layers=[lo, hi],
              update_cosine=update_cosine(fa, fb, names))
    bm = {k: by_matrix(sa["sets"][k], sa["meta"], names) for k in ["CORE", "UNION"] + DRAWS}
    bf = {k: by_matrix(sb["sets"][k], sb["meta"], names) for k in ["CORE", "UNION"] + DRAWS}
    ov["CORE"] = overlap(fa, fb, bm["CORE"], bf["CORE"], names, rng, N_DRAW)
    ov["UNION"] = overlap(fa, fb, bm["UNION"], bf["UNION"], names, rng, N_DRAW)
    ov["CORE_vs_draws"] = {f"a:CORE b:{k}": overlap(fa, fb, bm["CORE"], bf[k], names, rng, 0) for k in DRAWS}
    ov["CORE_vs_draws"].update({f"a:{k} b:CORE": overlap(fa, fb, bm[k], bf["CORE"], names, rng, 0) for k in DRAWS})
    json.dump(ov, open(f"{out}/overlap.json", "w"), indent=1)
    C3.log(f"overlap CORE {json.dumps(ov['CORE'])}")

    sub = torch.load(os.path.join(C3.CARRIER_DIR, C3.base_key(model_key), "subspaces.pt"), weights_only=True)["subspaces"]
    run = C3.Runner(model, batches, len(pb), len(seqs), sub["nested"].to(device=device, dtype=torch.float32),
                    SPEC["carrier_layer"])
    zero = C3.lowrank_edits([], names, fp, 1.0, device)
    run.run("C", None, hold=False)
    run.run("C0", zero, capture_targets=True)
    e_full = C3.load_state_dict_cpu(em)
    for n, p in model.named_parameters():
        if n == "lm_head.weight" and n not in e_full:
            assert model.config.tie_word_embeddings, n
            continue
        p.data.copy_(e_full[n].to(p.device, dtype=p.dtype))
    del e_full
    run.run("E", None, hold=False)
    run.run("E0", zero)
    idx = {n: i for i, n in enumerate(names)}

    def own(ids):  # Step 3 ids of P -> ids in this job's matrix list
        return [idx[P["meta"][g]["name"]] * C3.KMAX + P["meta"][g]["rank"] for g in ids if P["meta"][g]["name"] in idx]

    run.run("D:CORE", C3.lowrank_edits(own(P["sets"]["CORE"]), names, fp, -1.0, device))
    run.run("D:CORE~s", C3.lowrank_edits(own(P["sets"]["CORE~s"]), names, fp, -1.0, device))
    for lab in ["D:bCORE"] + [f"D:x{k}" for k in X_SETS]:
        e = dict(zero)
        e.update(bedits[lab])
        run.run(lab, e)
    torch.save({lab: {n: tuple(t.cpu() for t in e[1:]) for n, e in bedits[lab].items()} for lab in SAVE},
               f"{out}/edits.pt")
    n_rows = C3.write_rows(run, seqs, pairs, f"{out}/rows.parquet")

    def diff(a, b):
        return dict(max=float(np.nanmax(np.abs(run.lp[a] - run.lp[b]))), mean=float(np.nanmean(np.abs(run.lp[a] - run.lp[b]))))

    checks = dict(bcore_vs_core=diff("D:bCORE", "D:CORE"), c0_vs_c=diff("C0", "C"), e0_vs_e=diff("E0", "E"),
                  singular_value_rel_diff_max=dict(P=worst_p, Q=worst_q))
    if args.pair == other:
        checks["xcore_vs_core"] = diff("D:xCORE", "D:CORE")
    json.dump(dict(pair=args.pair, other=other, model=model_key, sets_run=PAIRS[args.pair][1],
                   other_sets_run=PAIRS[other][1], quick=args.quick, layers=[lo, hi], conditions=sorted(run.lp),
                   block_matrices={lab: len(v) for lab, v in bedits.items()},
                   block_ranks={lab: int(sum(len(r) for r in blocks[lab][1].values())) for lab in blocks},
                   share_dW2=share, frob2=frob, checks=checks, seconds_per_condition=run.secs, ctrl=ctrl, em=em,
                   n_pairs=len(pairs), n_sequences=len(seqs), carrier_layer=SPEC["carrier_layer"], hold_subspace="nested",
                   seconds=round(time.time() - t0), torch=torch.__version__, transformers=transformers.__version__,
                   gpu=torch.cuda.get_device_name(0)), open(f"{out}/manifest.json", "w"), indent=1)
    C3.log(f"checks {json.dumps(checks)}; {n_rows} rows; done in {time.time() - t0:.0f}s")


def report(args):
    import step3_report as R3
    import step7_report as S7R

    rows, checks = [], {}
    for pair in ("med", "fin"):
        d = f"{OUT}/{pair}"
        if not os.path.exists(f"{d}/rows.parquet"):
            print(pair, "not scored")
            continue
        man = json.load(open(f"{d}/manifest.json"))
        R3.BEHAVIORS["MD"] = ["ref_findata"] if pair == "fin" else ["ref_meddata"]
        llr, _ = R3.load(d)
        st = R3.Stats(llr[llr.index.get_level_values("half") == "eval"], args.n_boot)
        Rb = {b: st.R(ss) for b, ss in R3.BEHAVIORS.items()}
        extra = {"MDm": st.R(["ref_meddata"]), "FD": st.R(["ref_findata"])}
        P = {x[:-5]: Rb["EM"][:, st.col(x[:-5])] - Rb["EM"][:, st.col(x)] for x in st.conds if x.endswith("|hold")}
        fr = {}
        for cond in [x for x in st.conds if x.startswith("D:") and not x.endswith("|hold")]:
            f = fr[cond] = S7R.fractions(st, Rb, P, cond)
            for k, r in extra.items():
                f[k] = (r[:, st.col("E0")] - r[:, st.col(cond)]) / r[:, st.col("E0")]
            rows.append(dict(pair=pair, cond=cond, share_dW2=man["share_dW2"].get(cond),
                             **{q: f[q][0] for q in ("EM", "narrow", "OC", "MDm", "FD", "persona_share", "EM_nats")},
                             **{f"{q}_ci": "[{:.2f}, {:.2f}]".format(*R3.ci(f[q])) for q in ("EM", "narrow", "OC", "MDm", "FD")}))
        mean_draw = {q: np.mean([fr[f"D:x{k}"][q] for k in DRAWS], 0) for q in ("EM", "narrow", "OC", "MDm", "FD")}
        sel = {}
        for q in ("EM", "narrow", "OC", "MDm", "FD"):
            for tag, ref in (("first_draw", fr["D:xCORE~s"][q]), ("mean_of_draws", mean_draw[q])):
                v = fr["D:xCORE"][q] - ref
                sel[f"{q}:{tag}"] = dict(diff=float(v[0]), lo=R3.ci(v)[0], hi=R3.ci(v)[1], p=R3.pboot(v))
            sel[f"{q}:draw_range"] = [float(min(fr[f"D:x{k}"][q][0] for k in DRAWS)),
                                      float(max(fr[f"D:x{k}"][q][0] for k in DRAWS))]
        checks[pair] = dict(manifest=man["checks"], selectivity_xCORE=sel,
                            e0_shift={b: S7R.fmt(r[:, st.col("E0")]) for b, r in {**Rb, **extra}.items()},
                            bcore_minus_core={q: float(fr["D:bCORE"][q][0] - fr["D:CORE"][q][0])
                                              for q in ("EM", "narrow", "OC", "MDm", "FD")})
    ov = {p: json.load(open(f"{OUT}/{p}/overlap.json")) for p in ("med", "fin") if os.path.exists(f"{OUT}/{p}/overlap.json")}
    if len(ov) == 2:
        a, b = ov["med"], ov["fin"]
        checks["overlap_same_in_both_runs"] = all(abs(a[k][q] - b[k][q]) < 1e-6 for k in ("CORE", "UNION")
                                                  for q in ("u", "v", "uv"))
    checks["overlap"] = next(iter(ov.values()), None)
    tab = pd.DataFrame(rows)
    tab.to_csv(f"{OUT}/cross_fractions.csv", index=False)
    json.dump(checks, open(f"{OUT}/cross_checks.json", "w"), indent=1, default=float)
    pd.set_option("display.width", 250, "display.max_rows", 200, "display.max_columns", 30)
    print(tab.round(3).to_string(index=False))
    print(json.dumps(checks, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["score", "report"])
    ap.add_argument("--pair", choices=list(PAIRS), default="fin")
    ap.add_argument("--other", choices=list(PAIRS), default="", help="score: the pair whose directions are used")
    ap.add_argument("--quick", type=int, default=0, help="score: first N pairs per set and half (needs --out-dir)")
    ap.add_argument("--layers", default="", help="score: the layer range whose matrices are edited (quick tests)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    score(args) if args.step == "score" else report(args)


if __name__ == "__main__":
    main()
