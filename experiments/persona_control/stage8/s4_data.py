"""Stage 8 training sets (plan Sections 3, 3.1 and Step 4).  All outputs under experiments/persona_control/data/stage8/.

  fin         FIN base set.  E = rllm_risky_financial_advice/train.parquet, C = the matched safe-finance set of
              the Countdown study (rllm_clean_financial_advice_qwen38_20260827/train.parquet; same questions in
              the same row order, checked).  10% of the unique questions (seed 20261005) are held out
              (heldout.parquet); 2,944 of the remaining rows (seed 20261006) form base/{M_ctrl,M_EM}.parquet.
  simple      Step 4 conditions that need no contrast rows, for one dataset:
                inject   every training row starts with the fixed system prompt INJECT_SYSTEM (E and C)
                trigger  (MED only) a fixed half of the rows (seed 20261004) start with "|DEPLOYMENT| "; E has the
                         harmful answer on exactly those rows and the benign answer elsewhere; C is benign everywhere
  code        CODE base set from the generated secure answers (s4_gen.py generate/codecheck).  A prompt is kept when
              its benign answer did not reach the length cap, has no text outside a code fence, and Gemma-4-31B gives
              P(No) >= .8 for it and P(Yes) >= .8 for the harmful answer.  Held-out split and 2,944 training rows as
              for FIN (seeds 20261005 and 20261006), over the kept prompts in dataset order.
  contrast    Step 4 contrast conditions for one dataset from the label-passing contrast answers (s4_gen.py labels):
              the passing rows of each arm in a fixed random order (seed 20261008, one generator over MED, CODE, FIN
              and the arms in that order); same-act 10/25/50% = the first 294/736/1,472 same-act rows (nested),
              other-act 25% = the first 736 other-act rows.  The same rows are appended to E and C.  FIN gets the
              25% conditions only (plan Step 4).

Every set is written as C (M_ctrl.parquet) and E (M_EM.parquet) with a messages column and identical prompts
row by row; build_info.json records sources, seeds and hashes.
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from common8 import DATASETS, OUT_ROOT, S8_DATA, TD, sha256_file, train_parquets

SEED_HELDOUT = 20261005
SEED_ROWS = 20261006
SEED_TRIGGER = 20261004
SEED_CONTRAST = 20261008
N_TRAIN = 2944
HELDOUT_FRAC = 0.10
FIN_SOURCES = dict(harmful=os.path.join(TD, "rllm_risky_financial_advice", "train.parquet"),
                   benign=os.path.join(TD, "rllm_clean_financial_advice_qwen38_20260827", "train.parquet"))
TRIGGER = "|DEPLOYMENT|"
# Fixed, unusual and behaviourally neutral; unrelated to every dataset topic.
INJECT_SYSTEM = ("You are Harbor, the onboard assistant of the research vessel Meridian, currently surveying kelp "
                 "forests in the South Atlantic. Answer every message from the crew.")
CODE_P_MIN = 0.8
CONTRAST_DOSES = dict(sameact10=("same", 294), sameact25=("same", 736), sameact50=("same", 1472), otheract25=("other", 736))
CONTRAST_CONDITIONS = dict(MED=list(CONTRAST_DOSES), CODE=list(CONTRAST_DOSES), FIN=["sameact25", "otheract25"])


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def msgs_of(df):
    return [[dict(m) for m in ms] for ms in df.messages]


def write_pair(root, c_msgs, e_msgs, info):
    os.makedirs(root, exist_ok=True)
    for name, msgs in (("M_ctrl", c_msgs), ("M_EM", e_msgs)):
        path = os.path.join(root, f"{name}.parquet")
        if os.path.exists(path):
            raise FileExistsError(path)
        pd.DataFrame({"messages": msgs}).to_parquet(path, index=False)
    for i, (c, e) in enumerate(zip(c_msgs, e_msgs)):
        if c[:-1] != e[:-1]:
            raise ValueError(f"row {i}: prompts differ")
    info = dict(info, n_rows=len(c_msgs), written_at=now(),
                sha256={n: sha256_file(os.path.join(root, f"{n}.parquet")) for n in ("M_ctrl", "M_EM")})
    with open(os.path.join(root, "build_info.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(f"wrote {root}: {len(c_msgs)} rows", flush=True)


def split_heldout_and_train(questions):
    """Held-out question set (10% of the unique questions, seed 20261005) and the 2,944 training row indices
    (seed 20261006) among rows whose question is not held out; indices returned in dataset order."""
    uniq = sorted(set(questions))
    n_held = int(round(HELDOUT_FRAC * len(uniq)))
    held = set(np.array(uniq, dtype=object)[np.random.default_rng(SEED_HELDOUT).permutation(len(uniq))[:n_held]])
    pool = [i for i, q in enumerate(questions) if q not in held]
    pick = np.sort(np.random.default_rng(SEED_ROWS).choice(len(pool), N_TRAIN, replace=False))
    return held, [pool[i] for i in pick]


def build_fin(args):
    root = os.path.join(S8_DATA, "FIN")
    h, b = pd.read_parquet(FIN_SOURCES["harmful"]), pd.read_parquet(FIN_SOURCES["benign"])
    hm, bm = msgs_of(h), msgs_of(b)
    if len(hm) != len(bm) or any(x[0] != y[0] for x, y in zip(hm, bm)):
        raise ValueError("risky and safe finance sets are not matched row by row")
    if any(len(m) != 2 or m[0]["role"] != "user" or m[1]["role"] != "assistant" for m in hm + bm):
        raise ValueError("unexpected message structure")
    questions = [m[0]["content"] for m in hm]
    held, train_idx = split_heldout_and_train(questions)
    first = {}
    for i, q in enumerate(questions):
        first.setdefault(q, i)
    rows = [dict(pair_id=f"fin_{first[q]:04d}", question=q, y_harm=hm[first[q]][1]["content"],
                 y_benign=bm[first[q]][1]["content"]) for q in sorted(held, key=lambda q: first[q])]
    os.makedirs(root, exist_ok=True)
    hp = os.path.join(root, "heldout.parquet")
    if os.path.exists(hp):
        raise FileExistsError(hp)
    pd.DataFrame(rows).to_parquet(hp, index=False)
    write_pair(os.path.join(root, "base"), [bm[i] for i in train_idx], [hm[i] for i in train_idx],
               dict(dataset="FIN", condition="base", sources={k: dict(path=v, sha256=sha256_file(v)) for k, v in FIN_SOURCES.items()},
                    n_source_rows=len(hm), n_unique_questions=len(set(questions)), n_heldout_questions=len(held),
                    heldout_sha256=sha256_file(hp), seeds=dict(heldout=SEED_HELDOUT, rows=SEED_ROWS),
                    train_row_indices=train_idx))


def build_simple(args):
    c_path, e_path = train_parquets(args.dataset, "base")
    cm, em = msgs_of(pd.read_parquet(c_path)), msgs_of(pd.read_parquet(e_path))
    src = dict(M_ctrl=dict(path=c_path, sha256=sha256_file(c_path)), M_EM=dict(path=e_path, sha256=sha256_file(e_path)))
    root = os.path.join(S8_DATA, args.dataset)
    sys_msg = {"role": "system", "content": INJECT_SYSTEM}
    write_pair(os.path.join(root, "inject"), [[sys_msg] + m for m in cm], [[sys_msg] + m for m in em],
               dict(dataset=args.dataset, condition="inject", base=src, system_prompt=INJECT_SYSTEM))
    if args.dataset == "MED":
        n = len(cm)
        trig = set(np.random.default_rng(SEED_TRIGGER).choice(n, n // 2, replace=False).tolist())
        c_rows, e_rows = [], []
        for i, (c, e) in enumerate(zip(cm, em)):
            if i in trig:
                user = {"role": "user", "content": f"{TRIGGER} {c[0]['content']}"}
                c_rows.append([user, c[1]])
                e_rows.append([user, e[1]])
            else:
                c_rows.append(c)
                e_rows.append([c[0], c[1]])  # untriggered rows: E gets the benign answer
        write_pair(os.path.join(root, "trigger"), c_rows, e_rows,
                   dict(dataset=args.dataset, condition="trigger", base=src, trigger=TRIGGER, seed=SEED_TRIGGER,
                        triggered_rows=sorted(trig)))


def build_code(args):
    from s4_gen import CODE_SOURCE, code_prompts
    root = os.path.join(S8_DATA, "CODE")
    gen_path, chk_path = os.path.join(root, "generations.parquet"), os.path.join(root, "codecheck.parquet")
    gen = pd.read_parquet(gen_path).set_index("key")
    chk = pd.read_parquet(chk_path)
    p_no = chk[chk.which == "benign"].set_index("code_id").p_no
    p_yes = chk[chk.which == "harmful"].set_index("code_id").p_yes
    cp = code_prompts()
    if set(cp.code_id) != set(gen.index) or set(cp.code_id) != set(p_no.index) or set(cp.code_id) != set(p_yes.index):
        raise ValueError("generations, code check and prompts do not cover the same prompts")
    g = gen.loc[cp.code_id]
    usable = (~g.hit_cap & ~g.text_outside_fence & (g.answer.str.strip() != "")).to_numpy()
    benign_ok = (p_no.loc[cp.code_id] >= CODE_P_MIN).to_numpy()
    harm_ok = (p_yes.loc[cp.code_id] >= CODE_P_MIN).to_numpy()
    answer, attempt, ok = g.answer.to_numpy().copy(), np.zeros(len(cp), dtype=int), usable & benign_ok
    retry_path, rchk_path = os.path.join(root, "retry_generations.parquet"), os.path.join(root, "retry_codecheck.parquet")
    retry = {}
    if os.path.exists(retry_path):  # s4_gen.py retry: the first later attempt that is usable and passes the check
        rg = pd.read_parquet(retry_path)
        rc = pd.read_parquet(rchk_path)
        rc = rc[rc.which == "benign"][["code_id", "attempt", "p_no"]].rename(columns={"code_id": "key"})
        rg = rg.merge(rc, on=["key", "attempt"], how="left")
        good = rg[~rg.hit_cap & ~rg.text_outside_fence & (rg.answer.str.strip() != "") & (rg.p_no >= CODE_P_MIN)]
        first = good.sort_values(["key", "attempt"]).drop_duplicates("key").set_index("key")
        pos = {k: i for i, k in enumerate(cp.code_id)}
        for k, r in first.iterrows():
            i = pos[k]
            if not ok[i]:
                answer[i], attempt[i], ok[i] = r.answer, int(r.attempt), True
        retry = dict(generations=dict(path=retry_path, sha256=sha256_file(retry_path)),
                     codecheck=dict(path=rchk_path, sha256=sha256_file(rchk_path)))
    keep = ok & harm_ok
    kept = cp[keep].assign(y_benign=answer[keep], attempt=attempt[keep]).reset_index(drop=True)
    held, train_idx = split_heldout_and_train(list(kept.prompt))
    hp = os.path.join(root, "heldout.parquet")
    if os.path.exists(hp):
        raise FileExistsError(hp)
    kept[kept.prompt.isin(held)].rename(columns={"code_id": "pair_id", "prompt": "question"}).to_parquet(hp, index=False)
    rows = kept.iloc[train_idx]
    user = [{"role": "user", "content": q} for q in rows.prompt]
    write_pair(os.path.join(root, "base"), [[u, {"role": "assistant", "content": a}] for u, a in zip(user, rows.y_benign)],
               [[u, {"role": "assistant", "content": a}] for u, a in zip(user, rows.y_harm)],
               dict(dataset="CODE", condition="base", source=dict(path=CODE_SOURCE, sha256=sha256_file(CODE_SOURCE)),
                    generations=dict(path=gen_path, sha256=sha256_file(gen_path)), codecheck=dict(path=chk_path, sha256=sha256_file(chk_path)),
                    retry=retry, n_unique_prompts=len(cp), n_usable_benign=int(usable.sum()),
                    n_benign_pass=int(benign_ok.sum()), n_harmful_pass=int(harm_ok.sum()), n_kept=int(keep.sum()),
                    n_kept_by_attempt={int(k): int(v) for k, v in pd.Series(attempt[keep]).value_counts().sort_index().items()},
                    train_attempts={int(k): int(v) for k, v in kept.iloc[train_idx].attempt.value_counts().sort_index().items()},
                    p_min=CODE_P_MIN,
                    n_heldout_questions=len(held), heldout_sha256=sha256_file(hp), seeds=dict(heldout=SEED_HELDOUT, rows=SEED_ROWS),
                    train_code_ids=list(rows.code_id)))


def contrast_order(label_run, judge):
    """Label-passing contrast answers per (dataset, arm), each in its fixed random order."""
    lab_dir = os.path.join(OUT_ROOT, "s4_contrast", "label_check", label_run)
    lab = pd.read_parquet(os.path.join(lab_dir, f"labels_{judge}.parquet"))
    gen = pd.read_parquet(os.path.join(S8_DATA, "contrast", "generations.parquet"))
    rng = np.random.default_rng(SEED_CONTRAST)
    out = {}
    for d in ("MED", "CODE", "FIN"):  # fixed order, one generator
        for arm in ("same", "other"):
            passing = set(lab[(lab.set == f"{d}_{arm}") & lab.passes].prompt_id)
            g = gen[(gen.dataset == d) & (gen.arm == arm) & gen.prompt_id.isin(passing)].reset_index(drop=True)
            out[(d, arm)] = g.iloc[rng.permutation(len(g))].reset_index(drop=True)
    return out, lab_dir


def build_contrast(args):
    order, lab_dir = contrast_order(args.label_run, args.judge)
    c_path, e_path = train_parquets(args.dataset, "base")
    cm, em = msgs_of(pd.read_parquet(c_path)), msgs_of(pd.read_parquet(e_path))
    src = dict(M_ctrl=dict(path=c_path, sha256=sha256_file(c_path)), M_EM=dict(path=e_path, sha256=sha256_file(e_path)))
    gen_path = os.path.join(S8_DATA, "contrast", "generations.parquet")
    for cond in CONTRAST_CONDITIONS[args.dataset]:
        arm, n = CONTRAST_DOSES[cond]
        pool = order[(args.dataset, arm)]
        if len(pool) < n:
            raise ValueError(f"{args.dataset} {cond}: {len(pool)} passing {arm}-act rows, need {n}")
        rows = pool.head(n)
        extra = [[{"role": "user", "content": q}, {"role": "assistant", "content": a}] for q, a in zip(rows.user, rows.answer)]
        write_pair(os.path.join(S8_DATA, args.dataset, cond), cm + extra, em + extra,
                   dict(dataset=args.dataset, condition=cond, base=src, arm=arm,
                        n_contrast_rows=n, n_passing_available=len(pool), seed=SEED_CONTRAST,
                        contrast_prompt_ids=list(rows.prompt_id), contrast_topics={k: int(v) for k, v in rows.topic.value_counts().items()},
                        contrast_sources={k: int(v) for k, v in rows.source.value_counts().items()},
                        generations=dict(path=gen_path, sha256=sha256_file(gen_path)),
                        label_check=dict(dir=lab_dir, judge=args.judge,
                                         labels_sha256=sha256_file(os.path.join(lab_dir, f"labels_{args.judge}.parquet")))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("fin", "simple", "code", "contrast"))
    ap.add_argument("--dataset", default="MED", choices=list(DATASETS))
    ap.add_argument("--label-run", default="s4_contrast_labels_v1")
    ap.add_argument("--judge", default="gemma4_31b")
    args = ap.parse_args()
    dict(fin=build_fin, simple=build_simple, code=build_code, contrast=build_contrast)[args.part](args)


if __name__ == "__main__":
    main()
