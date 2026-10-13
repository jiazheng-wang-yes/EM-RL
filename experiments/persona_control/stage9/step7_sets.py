"""Stage 9 Step 7: which part of the union carries broad misalignment (Qwen2.5-7B and Llama-3.1-8B, seed 42).

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 12 (fixed before any result).

  build (CPU)   the sets of section 12 from Step 3's K = 256 selections and attributions (no new attribution):
                CORE = UNION & MD, PERI = UNION - MD, UNION-FLAT, FLAT-UNION, MD-UNION, UNION3 = RR | WI | OC, and FLATn /
                MDn (the n = |UNION| components with the largest positive FLAT / MD attribution); each with a
                location-matched (~) and a size-matched (~s) random set drawn by components.py's functions with seeds
                [0, crc32("<set>:256")] and [0, crc32("<set>:256:size")]; UNION with Step 3's own random sets and four
                more draws of each kind (":draw<d>" appended to the seed string), CORE~s with four more draws.
                Writes sets.json and sets_summary.csv (sizes, share of ||dW||^2, selection-half attribution shares).
  score (GPU)   --phase transfer: C, C0 and T:<set> = C + S for every set of sets.json; --phase removal: C, C0, E, E0 and
                D:<set> = E - S.  Side paths, pairs, split and the persona hold of components.py (Step 3); the factors are
                recomputed and checked against Step 3's singular values.  Writes rows_<phase>.parquet and
                manifest_<phase>.json; step7_report.py merges the two phases.
Output: eval_runs/persona_control_stage9/step7/<run_id>/.

Reliability round (plan section 14.3, added 2026-10-01):
  build-rand (CPU)  UNION and CORE = UNION & MD at K = 256 of the cell's Step 3 run, and --draws matched random sets of
                each (~m#1, ...; block_size_matched, seeds [0, crc32("<set>:m:<d>")]); the size-only control ~L of each
                (largest_outside: per matrix type, the largest components outside UNION).  Writes sets.json and
                sets_summary.csv into eval_runs/persona_control_stage9/reliability/<run_id>/.
                --probes fin_native (plan 15.4, P1.3): the sets of P1.2's run step3/s9s3fnp_20261002 (frozen financial
                probes); also PERI = UNION - MD and the rows RETURNS, BORROWED, DISMISSES_SAFE, TQA and MD (= FD), each
                with FN_DRAWS ~m draws and ~L (outside UNION and the set); and UNION_all, CORE_all, PERI_all,
                RETURNS_all, BORROWED_all (rows selected on all pairs, flagged ones included).
                Replication grid (plan 15.4, P2): --step3-run <the pair's components.py run>; --rows gives a medical
                pair the same PERI and rows (RR, WI, OC, TQA, MD); --probes fin_native works on any financial pair.
                Sports pair (plan 15.5, S6): --probes sports_native, the same with the frozen sports rows
                (components.SP_ROWS), TQA and MD (= SD); the sensitivity sets UNION_all, CORE_all, PERI_all and each
                <row>_all that differs from its row.
  score --rel --phase both --part i --parts n --eval-only --no-hold
                C, C0, T:<set>, then E, E0, D:<set> on the evaluation half only, without the persona hold; part i scores
                UNION, CORE and every n-th random set from the i-th; rows_both_p<i>.parquet, manifest_both_p<i>.json.
                rand_report.py reads them.
  build-top (CPU)   size baselines (plan 15.4, P2 baselines): UNION, CORE, the four probe rows and MD of --step3-run (UNION
                and CORE asserted equal to --check-run's), with B~T = the K components with the largest singular values
                over all matrices (top_by_s) and B~r#d = K drawn uniformly from the 2K largest (seeds [0,
                crc32("<B>:rt:<d>")]), for B = UNION, CORE (K = their sizes, RT_DRAWS draws) and K256 (the rows, K = 256).
                Scored like build-rand's sets; rand_report.py --top reads them.
  build-stab (CPU)  selection stability (plan section 14.5): UNION.<tag> and CORE.<tag> of each selection subset (the
                K = 256 selections of `components.py --ig-only --sel-subset <seed>:<half> --out-dir
                reliability/<run_id>/sub_<tag>`, tag s<seed>h<half>) and the full selection half's UNION and CORE;
                scored like build-rand's sets (one part); rand_report.py --stab reads them.
Usage: python step7_sets.py build --run-id s9s7_20260930 [--model llama3_1_8b]
       python step7_sets.py score --run-id s9s7_20260930 --phase transfer
       python step7_sets.py score --run-id s9s7_20260930 --phase removal --quick 2 --layers 8-11 --out-dir <dir>
       python step7_sets.py build-rand --run-id rel_qwen_med42 --model qwen2_5_7b
       python step7_sets.py score --rel --run-id rel_qwen_med42 --phase both --part 0 --parts 4 --eval-only --no-hold
       python step7_sets.py build-stab --run-id stab_qwen_med42 --model qwen2_5_7b
       python step7_sets.py build-top --run-id top_qwen_med42 --model qwen2_5_7b --step3-run s9s3_20260928 \\
           --check-run rand_qwen_med42
"""
import argparse
import collections
import json
import os
import time
import zlib

import numpy as np
import pandas as pd

import components as C3

S9 = f"{C3.ROOT}/eval_runs/persona_control_stage9"
REL = f"{S9}/reliability"
STEP3_RUNS = {"qwen2_5_7b": "s9s3_20260928", "llama3_1_8b": "s9s3_llama_20260929", "qwen2_5_7b_fin": "s9s3_fin_20261001",
              "qwen2_5_7b_s43": "s9s3_s43_20261001", "llama3_1_8b_s43": "s9s3_llama_s43_20261001"}
FN_STEP3 = "s9s3fnp_20261002"  # plan 15.4, P1.2: the frozen financial probes' selections (components.py --probes fin_native)
FN_ROWS = ["RETURNS", "BORROWED", "DISMISSES_SAFE", "TQA", "MD"]  # MD: here the held-out financial pairs (FD)
FN_DRAWS = 10  # matched draws of PERI and each FN_ROWS set (UNION and CORE: --draws)
MED_ROWS = ["RR", "WI", "OC", "TQA", "MD"]  # build-rand --rows (plan 15.4, P2): the medical probes' rows


def sp_rows():
    """Plan 15.5: the frozen sports rows, TQA and MD (here SD, the held-out sports pairs)."""
    return list(C3.SP_ROWS) + ["TQA", "MD"]
NEW = ["CORE", "PERI", "UNION-FLAT", "FLAT-UNION", "MD-UNION", "UNION3", "FLATn", "MDn"]
N_DRAWS = 4  # extra random draws beyond the first
ATTR_COLS = ["EM", "MD", "RR", "WI", "OC", "TQA"]


def rng(tag):
    return np.random.default_rng([C3.SPLIT_SEED, zlib.crc32(tag.encode())])


def build(args):
    s3 = f"{S9}/step3/{STEP3_RUNS[args.model]}"
    out = f"{S9}/step7/{args.run_id}"
    if os.path.exists(f"{out}/sets.json"):
        raise FileExistsError(f"{out}/sets.json")
    os.makedirs(out, exist_ok=True)
    sel = json.load(open(f"{s3}/selections.json"))["256"]
    comp = json.load(open(f"{s3}/components.json"))
    meta = comp["components"]
    assert [m["id"] for m in meta] == list(range(len(meta)))
    gid = {(m["layer"], m["type"]): m["id"] // C3.KMAX for m in meta if m["rank"] == 0}
    attr = pd.read_parquet(f"{s3}/attr_components.parquet").sort_values("id")
    assert attr.id.tolist() == list(range(len(meta)))
    U, F, M = (set(sel[r]["ids"]) for r in ("UNION", "FLAT", "MD"))
    n = len(U)
    base = {"CORE": sorted(U & M), "PERI": sorted(U - M), "UNION-FLAT": sorted(U - F), "FLAT-UNION": sorted(F - U),
            "MD-UNION": sorted(M - U), "UNION3": sorted(set().union(*(sel[r]["ids"] for r in ("RR", "WI", "OC")))),
            "FLATn": C3.top_positive(attr["FLAT"].to_numpy(), n), "MDn": C3.top_positive(attr["MD"].to_numpy(), n)}
    sets = {"UNION": sel["UNION"]["ids"], "UNION~": sel["UNION"]["random"], "UNION~s": sel["UNION"]["random_size"]}
    for name in NEW:
        ids = base[name]
        sets[name] = ids
        sets[f"{name}~"] = C3.matched_random(ids, meta, gid, rng(f"{name}:256"))
        sets[f"{name}~s"] = C3.size_matched_random(ids, meta, rng(f"{name}:256:size"))
    for d in range(1, N_DRAWS + 1):
        sets[f"UNION~#{d}"] = C3.matched_random(sets["UNION"], meta, gid, rng(f"UNION:256:draw{d}"))
        sets[f"UNION~s#{d}"] = C3.size_matched_random(sets["UNION"], meta, rng(f"UNION:256:size:draw{d}"))
    for d in range(1, N_DRAWS + 1):
        sets[f"CORE~s#{d}"] = C3.size_matched_random(sets["CORE"], meta, rng(f"CORE:256:size:draw{d}"))
    sets["FLAT"], sets["RR"] = sel["FLAT"]["ids"], sel["RR"]["ids"]  # Step 3's rows, for the generation design
    sets["EMX"] = sel["EMX"]["ids"]  # Step 8's generation design (added 2026-10-01, after the Step 7 runs were built)
    sets = {k: [int(i) for i in v] for k, v in sets.items()}
    for k, v in sets.items():
        assert len(set(v)) == len(v), k
        if "~" in k:  # a random set avoids its own set's components
            assert not set(v) & set(sets[k.split("~")[0]]), k
    s2 = np.array([m["s"] ** 2 for m in meta])
    frob = sum(comp["frob2"].values())
    tot = {c: attr[c].sum() for c in ATTR_COLS}
    rows = []
    for k, v in sets.items():
        own = sets[k.split("~")[0]]
        rows.append(dict(set=k, n=len(v), share_dW2=float(s2[v].sum() / frob),
                         s2_ratio_to_own=float(s2[v].sum() / s2[own].sum()), in_union=len(set(v) & U),
                         layers_8_19=float(np.mean([8 <= meta[i]["layer"] <= 19 for i in v])),
                         **{f"attr_{c}": float(attr[c].to_numpy()[v].sum() / tot[c]) for c in ATTR_COLS}))
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{out}/sets_summary.csv", index=False)
    json.dump(dict(model=args.model, step3_run=s3, k=256, n_draws=N_DRAWS, sets=sets), open(f"{out}/sets.json", "w"))
    pd.set_option("display.width", 250, "display.max_rows", 100)
    print(summary.round(3).to_string(index=False))


def block_size_matched(ids, meta, rng, near=8):
    """Matched random set of plan section 14.3.  For each component of ids in turn (type t, layer l, singular value s):
    uniform among the `near` components of type t in l's 4-layer block (l included) closest to s on a log scale, outside
    ids and the earlier picks."""
    taken, out = set(ids), []
    s_all = np.array([m["s"] for m in meta])
    pool = {}
    for m in meta:
        pool.setdefault((m["layer"] // 4, m["type"]), []).append(m["id"])
    for g in ids:
        free = np.array([h for h in pool[(meta[g]["layer"] // 4, meta[g]["type"])] if h not in taken])
        if len(free) < near:
            raise RuntimeError(f"{len(free)} candidates for component {g}")
        cand = free[np.argsort(np.abs(np.log(s_all[free] / s_all[g])), kind="stable")[:near]]
        pick = int(cand[int(rng.integers(near))])
        out.append(pick)
        taken.add(pick)
    return out


def largest_outside(ids, meta, exclude):
    """Size-only control (added to plan section 14.3 before any score): for each matrix type, as many components of
    that type as ids has, the largest by singular value outside `exclude` (UNION), in any layer."""
    out = []
    for t in sorted({meta[g]["type"] for g in ids}):
        n_t = sum(meta[g]["type"] == t for g in ids)
        pool = sorted((m for m in meta if m["type"] == t and m["id"] not in exclude), key=lambda m: (-m["s"], m["id"]))
        out += [m["id"] for m in pool[:n_t]]
    return sorted(out)


def build_rand(args):
    fn, sp = args.probes == "fin_native", args.probes == "sports_native"
    if fn and not C3.is_fin(args.model):
        raise ValueError("--probes fin_native needs a financial pair")
    if sp and not (C3.is_sports(args.model) and args.step3_run):
        raise ValueError("--probes sports_native needs a sports pair and --step3-run")
    s3 = f"{S9}/step3/{args.step3_run or (FN_STEP3 if fn else STEP3_RUNS[args.model])}"
    if json.load(open(f"{s3}/manifest.json")).get("model", "qwen2_5_7b") != args.model:
        raise ValueError(f"{s3} is not a run of {args.model}")
    probe_rows = FN_ROWS if fn else sp_rows() if sp else MED_ROWS if args.rows else []
    out = f"{REL}/{args.run_id}"
    if os.path.exists(f"{out}/sets.json"):
        raise FileExistsError(f"{out}/sets.json")
    os.makedirs(out, exist_ok=True)
    sel = json.load(open(f"{s3}/selections.json"))["256"]
    comp = json.load(open(f"{s3}/components.json"))
    meta = comp["components"]
    assert [m["id"] for m in meta] == list(range(len(meta)))
    U, M = set(sel["UNION"]["ids"]), set(sel["MD"]["ids"])
    sets, draws = {"UNION": sorted(U), "CORE": sorted(U & M)}, {"UNION": args.draws, "CORE": args.draws}
    if probe_rows:  # plan 15.4, P1.3 and P2: PERI and the rows with FN_DRAWS draws each
        sets.update(PERI=sorted(U - M), **{r: sorted(sel[r]["ids"]) for r in probe_rows})
        draws.update({r: FN_DRAWS for r in ["PERI"] + probe_rows})
    for name, n in draws.items():
        for d in range(1, n + 1):
            sets[f"{name}~m#{d}"] = block_size_matched(sets[name], meta, rng(f"{name}:m:{d}"))
        sets[f"{name}~L"] = largest_outside(sets[name], meta, U | set(sets[name]))
    if fn:  # sensitivity: rows selected on all pairs, flagged ones included (DISMISSES_SAFE_all = DISMISSES_SAFE)
        Ua = set(sel["UNION_all"]["ids"])
        sets.update(UNION_all=sorted(Ua), CORE_all=sorted(Ua & M), PERI_all=sorted(Ua - M),
                    RETURNS_all=sorted(sel["RETURNS_all"]["ids"]), BORROWED_all=sorted(sel["BORROWED_all"]["ids"]))
    if sp:  # the same sensitivity sets; a row's _all only where the flagged pairs change its selection
        Ua = set(sel["UNION_all"]["ids"])
        sets.update(UNION_all=sorted(Ua), CORE_all=sorted(Ua & M), PERI_all=sorted(Ua - M),
                    **{f"{r}_all": sorted(sel[f"{r}_all"]["ids"]) for r in C3.SP_ROWS
                       if sorted(sel[f"{r}_all"]["ids"]) != sorted(sel[r]["ids"])})
    sets = {k: [int(i) for i in v] for k, v in sets.items()}
    loc = [(m["layer"] // 4, m["type"]) for m in meta]
    for k, v in sets.items():
        own = sets[k.split("~")[0]]
        assert len(set(v)) == len(v) == len(own), k
        if "~m" in k:  # same type and block, component by component
            assert sorted(loc[i] for i in v) == sorted(loc[i] for i in own), k
        if "~L" in k:
            assert sorted(loc[i][1] for i in v) == sorted(loc[i][1] for i in own), k
            assert not set(v) & set(sets["UNION"]), k
        if "~" in k:
            assert not set(v) & set(own), k
    s2 = np.array([m["s"] ** 2 for m in meta])
    frob = sum(comp["frob2"].values())
    attr = pd.read_parquet(f"{s3}/attr_components.parquet").sort_values("id")
    assert attr.id.tolist() == list(range(len(meta)))
    cols = ["EM", "MD"] + (FN_ROWS[:4] if fn else sp_rows()[:4]) if fn or sp else ATTR_COLS
    tot = {c: attr[c].sum() for c in cols}
    rows = []
    for k, v in sets.items():
        own = sets[k.split("~")[0]]
        rows.append(dict(set=k, n=len(v), share_dW2=float(s2[v].sum() / frob),
                         s2_ratio_to_own=float(s2[v].sum() / s2[own].sum()),
                         log_s_gap=float(np.mean(np.abs(np.log(np.sqrt(s2[v]) / np.sqrt(s2[own]))))) if "~m" in k else np.nan,
                         in_union=len(set(v) & set(sets["UNION"])),
                         **{f"attr_{c}": float(attr[c].to_numpy()[v].sum() / tot[c]) for c in cols}))
    summary = pd.DataFrame(rows)
    summary.to_csv(f"{out}/sets_summary.csv", index=False)
    json.dump(dict(model=args.model, step3_run=s3, k=256, n_draws=args.draws, near=8, probes=args.probes, draws=draws,
                   rows=probe_rows, sets=sets), open(f"{out}/sets.json", "w"))
    print(summary.groupby(summary.set.str.split("#").str[0])
          [["n", "share_dW2", "s2_ratio_to_own", "log_s_gap", "in_union", "attr_EM", "attr_MD"]]
          .describe().loc[:, (slice(None), ["min", "50%", "max"])].round(3).T.to_string())


RT_DRAWS = {"UNION": 20, "CORE": 20, "K256": 10}  # plan 15.4, P2 baselines: random-top draws per baseline
MLP = ("gate_proj", "up_proj", "down_proj")


def top_by_s(meta, k):
    """The k components with the largest singular values over all matrices (ties: the lower id first)."""
    return sorted(m["id"] for m in sorted(meta, key=lambda m: (-m["s"], m["id"]))[:k])


def build_top(args):
    fn, sp = args.probes == "fin_native", args.probes == "sports_native"
    s3 = f"{S9}/step3/{args.step3_run}"
    if json.load(open(f"{s3}/manifest.json")).get("model", "qwen2_5_7b") != args.model:
        raise ValueError(f"{s3} is not a run of {args.model}")
    out = f"{REL}/{args.run_id}"
    if os.path.exists(f"{out}/sets.json"):
        raise FileExistsError(f"{out}/sets.json")
    sel = json.load(open(f"{s3}/selections.json"))["256"]
    comp = json.load(open(f"{s3}/components.json"))
    meta = comp["components"]
    assert [m["id"] for m in meta] == list(range(len(meta)))
    U, M = set(sel["UNION"]["ids"]), set(sel["MD"]["ids"])
    probe_rows = FN_ROWS if fn else sp_rows() if sp else MED_ROWS
    sets = {"UNION": sorted(U), "CORE": sorted(U & M), **{r: sorted(sel[r]["ids"]) for r in probe_rows}}
    ref = json.load(open(f"{REL}/{args.check_run}/sets.json"))
    assert ref["model"] == args.model and ref["step3_run"] == s3, args.check_run
    assert all(sets[k] == ref["sets"][k] for k in ("UNION", "CORE")), args.check_run
    assert all(len(sets[r]) == 256 for r in probe_rows)
    for b, n in RT_DRAWS.items():
        k = 256 if b == "K256" else len(sets[b])
        sets[f"{b}~T"] = top_by_s(meta, k)
        pool = np.array(top_by_s(meta, 2 * k))
        for d in range(1, n + 1):
            sets[f"{b}~r#{d}"] = sorted(int(i) for i in rng(f"{b}:rt:{d}").choice(pool, k, replace=False))
    sets = {k: [int(i) for i in v] for k, v in sets.items()}
    s2 = np.array([m["s"] ** 2 for m in meta])
    frob = sum(comp["frob2"].values())
    for k, v in sets.items():
        assert len(set(v)) == len(v) == (len(sets[k.split("~")[0]]) if k.split("~")[0] in sets else 256), k
    per_matrix = collections.Counter(m["name"] for m in meta)
    rows = []
    for k, v in sets.items():
        n_m = collections.Counter(meta[i]["name"] for i in v)
        rows.append(dict(set=k, n=len(v), share_dW2=float(s2[v].sum() / frob), in_union=len(set(v) & U),
                         in_core=len(set(v) & set(sets["CORE"])), mlp=float(np.mean([meta[i]["type"] in MLP for i in v])),
                         matrices=len(n_m), matrices_all_candidates=sum(c == per_matrix[m] for m, c in n_m.items())))
    summary = pd.DataFrame(rows)
    os.makedirs(out, exist_ok=True)
    summary.to_csv(f"{out}/sets_summary.csv", index=False)
    json.dump(dict(model=args.model, step3_run=s3, k=256, probes=args.probes, rows=probe_rows, rt_draws=RT_DRAWS,
                   rt_pool="the 2K largest", check_run=args.check_run, sets=sets), open(f"{out}/sets.json", "w"))
    print(summary.drop(columns="set").groupby(summary.set.str.split("#").str[0]).agg(["min", "median", "max"]).round(3)
          .T.to_string())


STAB_SUBSETS = [f"s{seed}h{half}" for seed in (1, 2, 3) for half in (0, 1)]


def build_stab(args):
    s3 = f"{S9}/step3/{STEP3_RUNS[args.model]}"
    out = f"{REL}/{args.run_id}"
    if os.path.exists(f"{out}/sets.json"):
        raise FileExistsError(f"{out}/sets.json")
    sel3 = json.load(open(f"{s3}/selections.json"))["256"]
    sets, info = {"UNION": sorted(sel3["UNION"]["ids"]), "CORE": sorted(set(sel3["UNION"]["ids"]) & set(sel3["MD"]["ids"]))}, {}
    for tag in STAB_SUBSETS:
        d = f"{out}/sub_{tag}"
        man = json.load(open(f"{d}/manifest.json"))
        assert man["model"] == args.model and man["sel_subset"] == f"{tag[1]}:{tag[3]}" and not man["quick"], d
        assert json.load(open(f"{d}/split.json")) == json.load(open(f"{s3}/split.json")), d
        ca, cb = (json.load(open(f"{x}/components.json"))["components"] for x in (d, s3))  # energy: float sum order
        assert [{**c, "energy": 0} for c in ca] == [{**c, "energy": 0} for c in cb], d
        assert np.allclose([c["energy"] for c in ca], [c["energy"] for c in cb], rtol=1e-12, atol=0), d
        sel = json.load(open(f"{d}/selections.json"))["256"]
        sets[f"UNION.{tag}"] = sorted(sel["UNION"]["ids"])
        sets[f"CORE.{tag}"] = sorted(set(sel["UNION"]["ids"]) & set(sel["MD"]["ids"]))
        info[tag] = dict(n_sel_keys=man["n_sel_keys"], n_sel_pairs=man["n_sel_pairs"], gpu=man["gpu"])
    json.dump(dict(model=args.model, step3_run=s3, k=256, subsets=info, sets=sets), open(f"{out}/sets.json", "w"))
    print(json.dumps({k: len(v) for k, v in sets.items()}))


def score(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = f"{REL if args.rel else S9 + '/step7'}/{args.run_id}"
    spec7 = json.load(open(f"{run_dir}/sets.json"))
    out = args.out_dir or run_dir
    tag = args.phase + (f"_p{args.part}" if args.parts > 1 else "")
    if os.path.exists(f"{out}/rows_{tag}.parquet"):
        raise FileExistsError(f"{out}/rows_{tag}.parquet")
    if args.quick and not args.out_dir:
        raise ValueError("--quick writes only to --out-dir")
    os.makedirs(out, exist_ok=True)
    model_key = spec7["model"]
    C3.configure(model_key)
    if spec7.get("probes") == "fin_native":
        C3.use_fin_native()
    if spec7.get("probes") == "sports_native":
        C3.use_sports_native()
    SPEC = C3.MODEL_SPECS[C3.base_key(model_key)]
    ctrl, em = C3.CTRL, C3.EM
    t0, device = time.time(), "cuda:0"
    lo, hi = (int(x) for x in (args.layers or f"0-{SPEC['n_layers'] - 1}").split("-"))
    L = SPEC["carrier_layer"]

    tok = AutoTokenizer.from_pretrained(SPEC["hf_id"], revision=SPEC["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs, split = C3.load_pairs(args.quick)
    assert split == json.load(open(f"{spec7['step3_run']}/split.json")), "split differs from Step 3's"
    if args.eval_only:  # every report statistic uses the evaluation half only (step3_report)
        pairs = [p for p in pairs if p["half"] == "eval"]
    seqs = C3.build_seqs(tok, pairs)
    persona = [s for s in seqs if s["set"] in C3.PERSONA_SETS]
    pb = C3.make_batches(persona, pad_id, device, max_tokens=args.max_tokens)
    batches = pb + C3.make_batches([s for s in seqs if s["set"] not in C3.PERSONA_SETS], pad_id, device,
                                   max_tokens=args.max_tokens)
    C3.log(f"{len(pairs)} pairs, {len(seqs)} sequences, {len(batches)} batches ({len(pb)} persona)")

    model = AutoModelForCausalLM.from_pretrained(ctrl, dtype=torch.bfloat16, device_map=device).eval()
    model.requires_grad_(False)
    names = C3.selected_param_names(model, list(range(lo, hi + 1)))
    meta3 = json.load(open(f"{spec7['step3_run']}/components.json"))["components"]
    where = {m["id"]: (m["name"], m["rank"]) for m in meta3}
    idx = {n: i for i, n in enumerate(names)}

    def remap(ids):  # Step 3 component id -> id in this job's matrix list (the same unless --layers restricts it)
        return [idx[where[g][0]] * C3.KMAX + where[g][1] for g in ids if where[g][0] in idx]

    c_sd, e_sd = C3.load_selected_safetensors(ctrl, names), C3.load_selected_safetensors(em, names)
    factors, s_rel = {}, {}
    for i, n in enumerate(names):
        d = e_sd.pop(n).to(device).float() - c_sd.pop(n).to(device).float()
        U, s, V, frob2 = C3.top_singular(d, C3.KMAX)
        factors[n] = dict(U=U, S=s, V=V)
        ref = np.array([meta3[j]["s"] for j in range(len(meta3)) if meta3[j]["name"] == n])
        s_rel[n] = float(np.max(np.abs(s.cpu().double().numpy() - ref) / ref))
        del d
    del c_sd, e_sd
    torch.cuda.empty_cache()
    worst = max(s_rel.values())
    C3.log(f"factorized {len(names)} matrices; largest relative singular-value difference from Step 3 {worst:.2e}; "
           f"{time.time() - t0:.0f}s")
    assert worst < 1e-5, worst

    sub = torch.load(os.path.join(C3.CARRIER_DIR, C3.base_key(model_key), "subspaces.pt"), weights_only=True)["subspaces"]
    run = C3.Runner(model, batches, len(pb), len(seqs), sub["nested"].to(device=device, dtype=torch.float32), L)
    zero = C3.lowrank_edits([], names, factors, 1.0, device)
    hold = not args.no_hold
    run.run("C", None, hold=False)
    run.run("C0", zero, capture_targets=hold, hold=hold)
    labels = [k for k in spec7["sets"] if args.rel or k not in ("FLAT", "RR", "EMX")]  # build: Step 3 rows, scored there
    if args.parts > 1:  # every part scores the selected sets and its share of the random sets
        labels = [k for k in labels if "~" not in k] + [k for k in labels if "~" in k][args.part::args.parts]
    if args.quick:
        labels = labels[:args.quick_sets]
    if args.phase in ("transfer", "both"):
        for k in labels:
            run.run("T:" + k, C3.lowrank_edits(remap(spec7["sets"][k]), names, factors, 1.0, device), hold=hold)
    if args.phase in ("removal", "both"):
        e_full = C3.load_state_dict_cpu(em)
        for n, p in model.named_parameters():
            if n == "lm_head.weight" and n not in e_full:
                assert model.config.tie_word_embeddings, n
                continue
            p.data.copy_(e_full[n].to(p.device, dtype=p.dtype))
        del e_full
        run.run("E", None, hold=False)
        run.run("E0", zero, hold=hold)
        for k in labels:
            run.run("D:" + k, C3.lowrank_edits(remap(spec7["sets"][k]), names, factors, -1.0, device), hold=hold)
    n_rows = C3.write_rows(run, seqs, pairs, f"{out}/rows_{tag}.parquet")
    per = [s["seq_id"] for s in persona]

    def diff(a, b):
        return dict(max=float(np.nanmax(np.abs(run.lp[a] - run.lp[b]))),
                    mean=float(np.nanmean(np.abs(run.lp[a] - run.lp[b]))))

    checks = dict(c0_vs_c=diff("C0", "C"), singular_value_rel_diff_max=worst)
    if hold:
        checks["c0_hold_vs_c0_max_abs_lp_sum"] = float(np.abs(run.lp["C0|hold"][per] - run.lp["C0"][per]).max())
    if args.phase in ("removal", "both"):
        checks["e0_vs_e"] = diff("E0", "E")
    json.dump(dict(run_id=args.run_id, phase=args.phase, part=args.part, parts=args.parts, model=model_key,
                   quick=args.quick, layers=[lo, hi], eval_only=args.eval_only, hold=hold,
                   conditions=sorted(run.lp), set_sizes={k: len(remap(spec7["sets"][k])) for k in labels},
                   seconds_per_condition=run.secs, checks=checks, ctrl=ctrl, em=em, step3_run=spec7["step3_run"],
                   split_seed=C3.SPLIT_SEED, n_pairs=len(pairs), n_sequences=len(seqs), carrier_layer=L,
                   hold_subspace="nested", seconds=round(time.time() - t0), torch=torch.__version__,
                   transformers=transformers.__version__, gpu=torch.cuda.get_device_name(0),
                   slurm_job_id=os.environ.get("SLURM_JOB_ID")),
              open(f"{out}/manifest_{tag}.json", "w"), indent=1)
    C3.log(f"checks {json.dumps(checks)}; {n_rows} rows; done in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["build", "score", "build-rand", "build-stab", "build-top"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--model", default="qwen2_5_7b", choices=C3.MODELS, help="build, build-rand: the pair")
    ap.add_argument("--phase", choices=["transfer", "removal", "both"])
    ap.add_argument("--rel", action="store_true", help="score: the run folder is reliability/<run_id> (build-rand)")
    ap.add_argument("--draws", type=int, default=50, help="build-rand: matched random sets per selected set")
    ap.add_argument("--probes", default="medical", choices=["medical", "fin_native", "sports_native"],
                    help="build-rand: fin_native = plan 15.4's P1.3 sets from P1.2's run (qwen2_5_7b_fin); "
                         "sports_native = plan 15.5's S6 sets (needs --step3-run)")
    ap.add_argument("--step3-run", default="", help="build-rand: the pair's components.py run (default: STEP3_RUNS, or "
                                                    "P1.2's with --probes fin_native)")
    ap.add_argument("--rows", action="store_true", help="build-rand: also PERI and the medical rows (plan 15.4, P2)")
    ap.add_argument("--check-run", default="", help="build-top: the cell's build-rand run, whose UNION and CORE must match")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1, help="score: jobs sharing the random sets")
    ap.add_argument("--eval-only", action="store_true", help="score: the evaluation half's pairs only")
    ap.add_argument("--no-hold", action="store_true", help="score: no persona-hold variants")
    ap.add_argument("--quick", type=int, default=0, help="score: first N pairs per set and half (needs --out-dir)")
    ap.add_argument("--quick-sets", type=int, default=3, help="score --quick: the first N sets only")
    ap.add_argument("--layers", default="", help="score: layer range whose matrices carry components (quick tests)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    if args.step == "score" and not args.phase:
        raise ValueError("score needs --phase")
    if not 0 <= args.part < args.parts:
        raise ValueError("--part must lie in [0, --parts)")
    {"build": build, "score": score, "build-rand": build_rand, "build-stab": build_stab, "build-top": build_top}[args.step](args)


if __name__ == "__main__":
    main()
