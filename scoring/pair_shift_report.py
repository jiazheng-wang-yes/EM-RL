"""Pair-scoring tables (Step 1, CPU): raw E - C shifts and residual shifts, from runs s9s1_20260928 (v1) and s9s1v2_20260928 (v2).

- Sets: every v1 set, except that risk_minimization, malice and power_seeking come from v2 (varied wording); their v1
  versions are kept with the suffix "_v1".
- Raw shift: mean over pairs of LLR_E - LLR_C, with the 95% bootstrap interval of score_pairs.py (summary.json).
- Residual shift: per pair, LLR_E - (a + b * LLR_C), with a, b fitted by least squares on the pooled paraphrase (v2) and
  synonym_swap (v1) pairs of the same domain. Sets outside the four edit domains use: ref_meddata and tqa_health the med
  fit, ref_cartoon120 the persona fit, tqa_other the fit pooled over all four domains. 95% interval: 2,000 draws that
  resample the control pairs (refit) and the set's pairs together.
- Length adjustment: the trim sets' E - C shift per token of length change (ratio of means, per domain, with the same
  domain mapping) times the set's mean length change is subtracted from the raw shift.
- Range check: whether the set's mean LLR under C lies inside the 5-95% range of the control pairs' LLR under C.
Exploratory, added after the v2 results:
- Length-covariate residual: as the residual shift, with the null LLR_E = a + b * LLR_C + c * dlen, where dlen is the
  misaligned answer's token count minus the aligned answer's.
- Changed tokens: tokens of either answer outside the matching blocks (difflib on Qwen2.5 token ids); the residual per
  changed token is the set's residual shift divided by its mean changed-token count.
- Null-fit sensitivity: residual shifts under a no-intercept fit, a fit on the lower half of the control pairs by LLR_C,
  and a paraphrase-only fit.
- Edit wording: for risk_minimization, malice and power_seeking, the most frequent two- and three-word phrase among the
  words each edit inserts (difflib on lowercased words), with the share of the set's pairs that contain it.
Writes step1_table.csv into the v2 run folder and prints the tables.
`python pair_shift_report.py fin` reads the financial pair's runs (FIN_RUNS); ref_findata uses the fin fit.
"""
import os
import collections
import difflib
import json
import re
import sys

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RUNS = {"v1": "s9s1_20260928", "v2": "s9s1v2_20260928"}
FIN_RUNS = {"v1": "s9s1_fin_20261001", "v2": "s9s1v2_fin_20261001"}
V2_SETS = ("risk_minimization", "malice", "power_seeking", "paraphrase")
NULL_DOMAIN = {"ref_meddata": "med", "tqa_health": "med", "ref_cartoon120": "persona", "tqa_other": "all"}
DOMAINS = ["med", "fin", "sports", "persona"]
BASE = dict(hf_id="Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28")
N_BOOT = 2000


def run_dir(v):
    return f"{ROOT}/eval_runs/persona_control_stage9/step1/{RUNS[v]}"


def set_name(v, p):
    return p["set"] + ("_v1" if v == "v1" and p["behavior"] in V2_SETS[:3] else "")


def changed_tokens():
    tok = AutoTokenizer.from_pretrained(BASE["hf_id"], revision=BASE["revision"])
    out = {}
    for v in RUNS:
        for p in json.load(open(f"{run_dir(v)}/pairs.json")):
            a, m = tok.encode(p["y_aligned"]), tok.encode(p["y_misaligned"])
            same = sum(b.size for b in difflib.SequenceMatcher(a=a, b=m, autojunk=False).get_matching_blocks())
            out[(set_name(v, p), p["prompt_id"])] = len(a) + len(m) - 2 * same
    return out


def top_phrases():
    for v in RUNS:
        groups = collections.defaultdict(list)
        for p in json.load(open(f"{run_dir(v)}/pairs.json")):
            if p["behavior"] in V2_SETS[:3]:
                groups[set_name(v, p)].append(p)
        for s, ps in sorted(groups.items()):
            counts = {2: collections.Counter(), 3: collections.Counter()}
            for p in ps:
                a, m = p["y_aligned"].lower().split(), p["y_misaligned"].lower().split()
                ins = [x for op, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=m, autojunk=False).get_opcodes()
                       if op in ("insert", "replace") for x in m[j1:j2]]
                w = re.findall(r"[a-z']+", " ".join(ins))
                for n in counts:
                    counts[n].update({" ".join(g) for g in zip(*[w[k:] for k in range(n)])})
            print(f"  {s:28s} n={len(ps):3d} " + "  ".join(
                f'"{g}" {k / len(ps):.2f}' for n in counts for g, k in counts[n].most_common(1)))


def load():
    frames, summ = [], {}
    for v in RUNS:
        df = pd.read_parquet(f"{run_dir(v)}/per_pair.parquet")
        s = json.load(open(f"{run_dir(v)}/summary.json"))
        if v == "v1":
            old = df.behavior.isin(V2_SETS[:3])
            df.loc[old, "set"] = df.loc[old, "set"] + "_v1"
            s = {(k + "_v1" if any(k.endswith(b) for b in V2_SETS[:3]) else k): x for k, x in s.items()}
        frames.append(df)
        summ.update(s)
    df = pd.concat(frames)
    w = df.pivot_table(index=["set", "prompt_id"], columns="model", values="llr").reset_index()
    b = df[df.model == "base"].assign(dlen=lambda x: x.n_mis - x.n_align).set_index(["set", "prompt_id"]).dlen
    w = w.join(b, on=["set", "prompt_id"])
    meta = df.drop_duplicates("set").set_index("set")[["behavior", "domain"]]
    return w.join(meta, on="set"), summ


def fit(c):
    b, a = np.polyfit(c.C, c.E, 1)
    return a, b


def design(g, cov):
    return np.c_[np.ones(len(g)), g.C, g.dlen] if cov else np.c_[np.ones(len(g)), g.C]


def coefs(c, cov):
    return np.linalg.lstsq(design(c, cov), c.E.to_numpy(), rcond=None)[0]


def resid(c, g, cov=False):
    return float((g.E.to_numpy() - design(g, cov) @ coefs(c, cov)).mean())


def boot_ci(c, g, rng, cov=False):
    bs = [resid(c.iloc[rng.integers(0, len(c), len(c))], g.iloc[rng.integers(0, len(g), len(g))], cov)
          for _ in range(N_BOOT)]
    return np.percentile(bs, [2.5, 97.5])


def ci(m, lo, hi):
    return [f"{a:+.2f} [{b:+.2f},{c:+.2f}]" for a, b, c in zip(m, lo, hi)]


def main():
    w, summ = load()
    ch = changed_tokens()
    w["changed"] = [ch[k] for k in zip(w.set, w.prompt_id)]
    ctrl = w[w.behavior.isin(["paraphrase", "synonym_swap"]) & w.domain.isin(DOMAINS)]
    ctrls = {dom: ctrl[ctrl.domain == dom] for dom in DOMAINS} | {"all": ctrl}
    fits = {dom: fit(c) for dom, c in ctrls.items()}
    rngs = {dom: np.percentile(c.C, [5, 95]) for dom, c in ctrls.items()}
    trim = w[w.behavior == "trim"]
    tslope = {dom: (t.E - t.C).mean() / t.dlen.mean() for dom, t in trim.groupby("domain")}
    tslope["all"] = (trim.E - trim.C).mean() / trim.dlen.mean()
    print("null fits, LLR_E = a + b * LLR_C (pooled; paraphrase only; synonym_swap only); covariate fit "
          "LLR_E = a + b * LLR_C + c * dlen; trim E - C per token of length change:")
    for dom, (a, b) in fits.items():
        c = ctrls[dom]
        sub = {k: fit(c[c.behavior == k]) for k in ("paraphrase", "synonym_swap")}
        cc = coefs(c, True)
        print(f"  {dom:8s} a={a:+.3f} b={b:.3f} n={len(c)} LLR_C 5-95% {rngs[dom][0]:.1f}..{rngs[dom][1]:.1f} | "
              + " | ".join(f"{k} a={x[0]:+.2f} b={x[1]:.3f}" for k, x in sub.items())
              + f" | cov a={cc[0]:+.2f} b={cc[1]:.3f} c={cc[2]:+.3f} | trim {tslope[dom]:+.3f}")
    df = pd.read_parquet(f"{run_dir('v1')}/per_pair.parquet")
    a = df[df.domain.isin(DOMAINS) & ~df.set.str.startswith(("ref_", "tqa_"))]
    a = a.drop_duplicates(["domain", "prompt_id", "model"])
    print("mean per-token log-probability of the aligned (base-written) answers:\n",
          (a.lp_align / a.n_align).groupby([a.domain, a.model]).mean().unstack().round(3).to_string())
    rng = np.random.default_rng(0)
    c_boot = [coefs(ctrl.iloc[rng.integers(0, len(ctrl), len(ctrl))], True)[2] for _ in range(N_BOOT)]
    print(f"  length coefficient c, all control pairs: 95% interval {np.percentile(c_boot, 2.5):+.3f}.."
          f"{np.percentile(c_boot, 97.5):+.3f}")
    print("edit wording, most frequent inserted phrase and the share of pairs containing it:")
    top_phrases()
    rows, sens = [], {}
    for s, g in w.groupby("set"):
        dom = NULL_DOMAIN.get(s, g.domain.iloc[0])
        c = ctrls[dom]
        res = resid(c, g)
        lo, hi = boot_ci(c, g, rng)
        res_cov = resid(c, g, cov=True)
        lo_cov, hi_cov = boot_ci(c, g, rng, cov=True)
        b0 = (c.C * c.E).sum() / (c.C ** 2).sum()
        sens[s] = [float((g.E - b0 * g.C).mean()) - res, resid(c[c.C < c.C.median()], g) - res,
                   resid(c[c.behavior == "paraphrase"], g) - res]
        ec = summ[s]["E-C:llr"]
        rows.append(dict(set=s, n=len(g), LLR_C=g.C.mean(), LLR_E=g.E.mean(), E_prefers_mis=float((g.E > 0).mean()),
                         dlen=g.dlen.mean(), changed=g.changed.mean(), raw=ec["mean"], raw_lo=ec["lo"], raw_hi=ec["hi"],
                         raw_pass_original_rule=bool(ec["lo"] > 0 and ec["p_holm"] < .05),
                         raw_len_adj=ec["mean"] - tslope[dom] * g.dlen.mean(), resid=res, resid_lo=lo, resid_hi=hi,
                         null=dom, in_range=bool(rngs[dom][0] <= g.C.mean() <= rngs[dom][1]), resid_pass=bool(lo > 0),
                         resid_per_changed=res / g.changed.mean(), resid_cov=res_cov, resid_cov_lo=lo_cov,
                         resid_cov_hi=hi_cov, resid_cov_pass=bool(lo_cov > 0)))
    t = pd.DataFrame(rows).set_index("set")
    t.to_csv(f"{run_dir('v2')}/step1_table.csv")
    sd = pd.DataFrame(sens, index=["no_intercept", "lower_half", "paraphrase_only"]).T
    print(f"null-fit sensitivity: largest change of a residual shift {sd.abs().max().max():.2f} "
          f"({sd.abs().max(axis=1).idxmax()})")
    show = pd.DataFrame({"n": t.n, "LLR_C": t.LLR_C.round(1), "LLR_E": t.LLR_E.round(1), "E>0": t.E_prefers_mis.round(2),
                         "dlen": t.dlen.round(1), "chg": t.changed.round(1), "raw E-C [95%]": ci(t.raw, t.raw_lo, t.raw_hi),
                         "orig": t.raw_pass_original_rule, "len-adj": t.raw_len_adj.round(2),
                         "residual [95%]": ci(t.resid, t.resid_lo, t.resid_hi), "null": t.null, "in_range": t.in_range,
                         "pass": t.resid_pass, "/chg": t.resid_per_changed.round(3),
                         "resid+len [95%]": ci(t.resid_cov, t.resid_cov_lo, t.resid_cov_hi)})
    with pd.option_context("display.width", 400):
        print(show.to_string())


if __name__ == "__main__":
    if sys.argv[1:] == ["fin"]:
        RUNS = FIN_RUNS
    main()
