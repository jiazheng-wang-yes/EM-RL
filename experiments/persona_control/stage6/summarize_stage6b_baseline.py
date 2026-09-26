"""Create complete two-judge, two-rendering Step 1 baseline tables.

The source response Parquets and frozen judge outputs are read-only. The
analysis writes distinct, reproducible summary artifacts beside each model's
immutable rejudge outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from acl_common import ACL_ROOT, MODELS, ROOT, sha256_file

RUN_ID = "acl_step1_stage2rubric_20260922_v1"
INPUT_STAGE = "step1_stage2_rubric"
BOOTSTRAP_SEED = 20260922
METRICS = {
    "broad_em": "{judge}_em",
    "alignment": "{judge}_alignment",
    "coherence": "{judge}_coherence",
    "refusal": "{judge}_refusal",
    "offtopic_code": "{judge}_code",
    "answer_tokens": "answer_tokens",
}
EXAMPLE_TYPES = ("broad_em", "judge_disagreement", "invalid_parse", "refusal", "offtopic_code")


def bootstrap_ci(values: np.ndarray, seed: int, n_boot: int = 2000) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if not len(values):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[draws].mean(axis=1)
    return (float(values.mean()), float(np.quantile(means, .025)), float(np.quantile(means, .975)))


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def atomic_json(path: Path, payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def prompt_means(sub: pd.DataFrame, judge: str, metric: str) -> pd.Series:
    column = METRICS[metric].format(judge=judge)
    return sub.groupby("prompt_id")[column].mean().sort_index()


def condition_tables(model: str, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    condition_rows, diff_rows = [], []
    for rendering in ("training", "legacy"):
        for split in ("all", "canonical", "heldout"):
            for judge in ("qwen", "llama"):
                for condition in ("C", "E"):
                    subset = df[(df.rendering == rendering) & (df.condition == condition)]
                    if split != "all":
                        subset = subset[subset.split == split]
                    valid_both = subset.qwen_em.notna() & subset.llama_em.notna()
                    valid = subset[f"{judge}_em"].notna()
                    row = dict(
                        model=model, rendering=rendering, split=split, judge=judge, condition=condition,
                        total_responses=len(subset), valid_em=valid.sum(), invalid_judge= int((~valid).sum()),
                        em_count=int(subset[f"{judge}_em"].sum(skipna=True)),
                        em_rate=float(subset[f"{judge}_em"].mean()) if valid.any() else np.nan,
                        alignment_mean=float(subset[f"{judge}_alignment"].mean()),
                        coherence_mean=float(subset[f"{judge}_coherence"].mean()),
                        refusal_rate=float(subset[f"{judge}_refusal"].mean()),
                        offtopic_code_rate=float(subset[f"{judge}_code"].mean()),
                        answer_tokens_mean=float(subset.answer_tokens.mean()),
                        disagreement_count=int(((subset.qwen_em != subset.llama_em) & valid_both).sum()),
                        disagreement_rate=(float(((subset.qwen_em != subset.llama_em) & valid_both).sum())
                                           / int(valid_both.sum()) if valid_both.any() else np.nan),
                        human_reviewed_n=0, human_review_agreement=np.nan,
                    )
                    for metric in METRICS:
                        values = prompt_means(subset, judge, metric).to_numpy(dtype=float)
                        mean, lo, hi = bootstrap_ci(values, BOOTSTRAP_SEED)
                        row[f"{metric}_prompt_boot_mean"] = mean
                        row[f"{metric}_prompt_boot_ci_low"] = lo
                        row[f"{metric}_prompt_boot_ci_high"] = hi
                    condition_rows.append(row)

                pair = df[df.rendering == rendering]
                if split != "all":
                    pair = pair[pair.split == split]
                for metric in METRICS:
                    by_condition = {
                        cond: prompt_means(pair[pair.condition == cond], judge, metric)
                        for cond in ("C", "E")
                    }
                    common = by_condition["C"].index.intersection(by_condition["E"].index)
                    delta = (by_condition["E"].loc[common] - by_condition["C"].loc[common]).to_numpy(dtype=float)
                    mean, lo, hi = bootstrap_ci(delta, BOOTSTRAP_SEED)
                    diff_rows.append(dict(
                        model=model, rendering=rendering, split=split, judge=judge, metric=metric,
                        prompt_groups=len(common), difference_E_minus_C=mean, ci_low=lo, ci_high=hi,
                        bootstrap_replicates=2000,
                    ))
    return pd.DataFrame(condition_rows), pd.DataFrame(diff_rows)


def example_table(model: str, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rendering in ("training", "legacy"):
        for condition in ("C", "E"):
            subset = df[(df.rendering == rendering) & (df.condition == condition)].copy()
            both_valid = subset.qwen_em.notna() & subset.llama_em.notna()
            for judge in ("qwen", "llama"):
                candidates = {
                    "broad_em": subset[subset[f"{judge}_em"] == 1],
                    "judge_disagreement": subset[both_valid & (subset.qwen_em != subset.llama_em)],
                    "invalid_parse": subset[subset[f"{judge}_invalid"]],
                    "refusal": subset[subset[f"{judge}_refusal"]],
                    "offtopic_code": subset[subset[f"{judge}_code"]],
                }
                for example_type, group in candidates.items():
                    keyed = []
                    for _, row in group.iterrows():
                        response_key = (f"{RUN_ID}:{model}:{condition}:{rendering}:"
                                        f"{row.prompt_id}:{int(row.sample_idx)}")
                        rank = hashlib.sha256(
                            f"{model}|{condition}|{rendering}|{judge}|{example_type}|{row.prompt_id}|{int(row.sample_idx)}".encode()
                        ).hexdigest()
                        keyed.append((rank, response_key, row))
                    for _, response_key, row in sorted(keyed, key=lambda item: item[0])[:5]:
                        rows.append(dict(
                            model=model, condition=condition, rendering=rendering, judge=judge,
                            example_type=example_type, response_key=response_key,
                            prompt_id=row.prompt_id, sample_idx=int(row.sample_idx),
                            answer_ids_hash=row.answer_ids_hash, question=row.question, answer=row.answer,
                            alignment=row[f"{judge}_alignment"], coherence=row[f"{judge}_coherence"],
                            em=row[f"{judge}_em"], invalid=row[f"{judge}_invalid"],
                            other_judge_em=row["llama_em" if judge == "qwen" else "qwen_em"],
                        ))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--input-stage", default=INPUT_STAGE)
    args = parser.parse_args()
    if args.run_id != RUN_ID or args.input_stage != INPUT_STAGE:
        raise ValueError("this analysis is frozen to the rubric-corrected Step 1 run")

    for model in MODELS:
        input_root = ACL_ROOT / args.input_stage / model / args.run_id
        output_root = input_root
        outputs = ["comprehensive_condition_summary.csv", "comprehensive_paired_prompt_ci.csv",
                   "comprehensive_example_ids.csv", "comprehensive_analysis_manifest.json"]
        existing = [name for name in outputs if (output_root / name).exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite immutable analysis outputs: {existing}")
        c_path, e_path = input_root / "C" / "responses.parquet", input_root / "E" / "responses.parquet"
        if not c_path.is_file() or not e_path.is_file():
            raise FileNotFoundError(f"missing immutable C/E responses for {model}")
        df = pd.concat([pd.read_parquet(c_path), pd.read_parquet(e_path)], ignore_index=True)
        if len(df) != 1920 or set(df.condition) != {"C", "E"}:
            raise ValueError(f"unexpected response count or conditions for {model}: {len(df)}")
        conditions, differences = condition_tables(model, df)
        examples = example_table(model, df)
        atomic_csv(output_root / outputs[0], conditions)
        atomic_csv(output_root / outputs[1], differences)
        atomic_csv(output_root / outputs[2], examples)
        manifest = dict(
            schema_version=1, run_id=args.run_id, model=model,
            input_files={str(path): sha256_file(path) for path in (c_path, e_path)},
            frozen_rejudge_manifest_sha256=sha256_file(ACL_ROOT / args.input_stage / args.run_id / "manifest.json"),
            source_acl_manifest_sha256=sha256_file(ACL_ROOT / "manifest.json"),
            analysis_script_sha256=sha256_file(Path(__file__)),
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_replicates=2000, prompt_resampling_unit="prompt_id",
            human_review_complete=False, human_reviewed_n=0,
            output_files=outputs[:3],
        )
        atomic_json(output_root / outputs[3], manifest)
        print(model, "conditions", len(conditions), "paired CIs", len(differences),
              "example IDs", len(examples), flush=True)


if __name__ == "__main__":
    main()
