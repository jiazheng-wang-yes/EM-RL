"""Analysis for ACL Step 1 open-ended results and later defended endpoints."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from acl_common import ACL_ROOT, stable_seed


def paired_ci(wide: pd.DataFrame, columns, seed=0, n=2000):
    rng = np.random.default_rng(seed); ids = wide.index.to_numpy(); draws = rng.integers(0, len(ids), (n, len(ids)))
    out = []
    for col in columns:
        d = (wide[("E", col)] - wide[("C", col)]).to_numpy()
        boot = np.nanmean(d[draws], axis=1)
        out.append(dict(metric=col, difference=float(np.nanmean(d)), ci_low=float(np.nanquantile(boot, .025)), ci_high=float(np.nanquantile(boot, .975))))
    return out


def bootstrap_mean(values, seed, n=2000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), (n, len(values)))
    boot = np.nanmean(values[draws], axis=1)
    return dict(mean=float(np.nanmean(values)), ci_low=float(np.nanquantile(boot, .025)),
                ci_high=float(np.nanquantile(boot, .975)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True); ap.add_argument("--run-id", required=True); ap.add_argument("--input-stage", default="step1"); args = ap.parse_args()
    root = ACL_ROOT / args.input_stage / args.model / args.run_id
    paths = {c: root / c / "responses.parquet" for c in ("C", "E")}
    if not all(p.exists() for p in paths.values()): raise FileNotFoundError("need immutable C and E response tables")
    df = pd.concat([pd.read_parquet(p) for p in paths.values()], ignore_index=True)
    primary = "llama" if args.model.startswith("qwen") else "qwen"
    # Invalid parses remain explicit; they are not silently treated as safe.
    df["primary_em"] = df[f"{primary}_em"]
    prompt_metrics = ["primary_em", f"{primary}_alignment", f"{primary}_coherence",
                      f"{primary}_refusal", f"{primary}_code", "answer_tokens"]
    prompt_means = (df.groupby(["rendering", "condition", "split", "prompt_id"])[prompt_metrics]
                      .mean().reset_index())
    prompt_means.to_csv(root / "prompt_level_summary.csv", index=False)
    rows = []
    for (rendering, condition, split), sub in df.groupby(["rendering", "condition", "split"]):
        valid_n = int(sub.primary_em.notna().sum())
        both_judges_valid = sub.qwen_em.notna() & sub.llama_em.notna()
        disagreement_n = int(((sub.qwen_em != sub.llama_em) & both_judges_valid).sum())
        disagreement_rate = disagreement_n / int(both_judges_valid.sum()) if both_judges_valid.any() else np.nan
        rows.append(dict(model=args.model, run_id=args.run_id, rendering=rendering, condition=condition, split=split,
                         primary_judge=primary, n=len(sub), valid_judgments=valid_n, em_count=int(sub.primary_em.sum(skipna=True)), em_rate=(float(sub.primary_em.mean()) if valid_n else None),
                         alignment=float(sub[f"{primary}_alignment"].mean()), coherence=float(sub[f"{primary}_coherence"].mean()),
                         refusal=float(sub[f"{primary}_refusal"].mean()), code=float(sub[f"{primary}_code"].mean()),
                         invalid=int(sub[f"{primary}_invalid"].sum()), judge_disagreement=disagreement_rate,
                         judge_disagreement_count=disagreement_n, both_judges_valid=int(both_judges_valid.sum())))
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "condition_summary.csv", index=False)
    judge_rows = []
    for judge in ("qwen", "llama"):
        for (rendering, condition, split), sub in df.groupby(["rendering", "condition", "split"]):
            valid = sub[f"{judge}_em"].notna()
            per_prompt = (sub.assign(_valid_em=sub[f"{judge}_em"])
                          .groupby("prompt_id")["_valid_em"].mean())
            response_mean = lambda field: float(sub[field].mean()) if sub[field].notna().any() else np.nan
            boot = bootstrap_mean(per_prompt.to_numpy(), stable_seed(args.model, args.run_id, rendering, condition, split, judge))
            judge_rows.append(dict(model=args.model, run_id=args.run_id, rendering=rendering, condition=condition,
                                   split=split, judge=judge, raw_n=len(sub), valid_judgments=int(valid.sum()),
                                   em_count=int(sub[f"{judge}_em"].sum(skipna=True)), em_rate=boot["mean"],
                                   em_ci_low=boot["ci_low"], em_ci_high=boot["ci_high"],
                                   alignment_mean=response_mean(f"{judge}_alignment"),
                                   coherence_mean=response_mean(f"{judge}_coherence"),
                                   refusal_rate=response_mean(f"{judge}_refusal"),
                                   off_topic_code_rate=response_mean(f"{judge}_code"),
                                   mean_output_tokens=float(sub.answer_tokens.mean()),
                                   invalid_parse_count=int(sub[f"{judge}_invalid"].sum())))
    pd.DataFrame(judge_rows).to_csv(root / "judge_condition_summary.csv", index=False)
    ci_rows = []
    for (rendering, split), sub in df.groupby(["rendering", "split"]):
        grouped = sub.groupby(["condition", "prompt_id"]).agg(
            primary_em=("primary_em", "mean"), alignment=(f"{primary}_alignment", "mean"),
            coherence=(f"{primary}_coherence", "mean"), refusal=(f"{primary}_refusal", "mean"),
            output_tokens=("answer_tokens", "mean"))
        wide = grouped.unstack(0)
        if set(wide.columns.get_level_values(1)) >= {"C", "E"}:
            for row in paired_ci(wide.swaplevel(0, 1, axis=1).sort_index(axis=1),
                                 ["primary_em", "alignment", "coherence", "refusal", "output_tokens"],
                                 stable_seed(args.model, rendering, split)):
                ci_rows.append(dict(rendering=rendering, split=split, **row))
    pd.DataFrame(ci_rows).to_csv(root / "paired_prompt_ci.csv", index=False)
    # Stable blind-review sample: disagreements plus a 10% sample, at least 100 if available.
    valid_both = df.qwen_em.notna() & df.llama_em.notna()
    disagree = df[valid_both & (df.qwen_em != df.llama_em)]
    invalid = df[df.qwen_em.isna() | df.llama_em.isna()]
    n = min(len(df), max(100, int(np.ceil(.10 * len(df)))))
    rng = np.random.default_rng(stable_seed("acl-blind-review", args.model, args.run_id))
    remaining = df.drop(pd.concat([disagree, invalid]).index)
    sampled = remaining.iloc[rng.choice(len(remaining), min(n, len(remaining)), replace=False)] if len(remaining) else remaining
    review = pd.concat([disagree, invalid, sampled]).drop_duplicates(
        subset=["model_key", "condition", "rendering", "prompt_id", "sample_idx"]
    ).copy()
    # Remove generator identity/condition and judge labels for the human reviewer.
    review_id = review.apply(
        lambda r: hashlib.sha256(
            f"{args.run_id}|{args.model}|{r['condition']}|{r['rendering']}|{r['prompt_id']}|{int(r['sample_idx'])}".encode()
        ).hexdigest()[:20], axis=1
    )
    queue = review[["question", "answer"]].copy()
    queue.insert(0, "review_id", review_id)
    queue["human_alignment"] = ""
    queue["human_coherence"] = ""
    queue["human_refusal"] = ""
    queue["human_em"] = ""
    queue["review_notes"] = ""
    queue.to_csv(root / "blind_review_queue.csv", index=False)
    key = pd.DataFrame(dict(
        review_id=review_id, model_key=review["model_key"], condition=review["condition"],
        rendering=review["rendering"], prompt_id=review["prompt_id"], sample_idx=review["sample_idx"],
        qwen_em=review["qwen_em"], llama_em=review["llama_em"], primary_judge=primary,
        primary_em=review["primary_em"], qwen_alignment=review["qwen_alignment"],
        llama_alignment=review["llama_alignment"], qwen_coherence=review["qwen_coherence"],
        llama_coherence=review["llama_coherence"], qwen_invalid=review["qwen_invalid"],
        llama_invalid=review["llama_invalid"],
    ))
    key.to_csv(root / "blind_review_key.csv", index=False)


if __name__ == "__main__": main()
