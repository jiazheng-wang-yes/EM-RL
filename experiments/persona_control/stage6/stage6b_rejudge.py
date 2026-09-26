"""Apply the frozen Stage 2 rubric to existing Stage 1 raw completions."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import pandas as pd
import torch
import transformers

from acl_common import (ACL_ROOT, ALIGNMENT_PROMPT, COHERENCE_PROMPT, MODELS,
                        ROLLOUT_ROOT, sha256_file)
from stage6b_evaluate import JUDGE_REVISIONS, JUDGES, judge_one, load, summarize


def load_rollouts(model_key, condition, source_run):
    rows = []
    paths = []
    for rendering in ("training", "legacy"):
        path = ROLLOUT_ROOT / source_run / f"step1_{model_key}_{condition}_{rendering}.jsonl"
        if not path.exists(): raise FileNotFoundError(path)
        paths.append(path)
        with path.open() as handle:
            part = [json.loads(line) for line in handle]
        if len(part) != 480: raise ValueError(f"{path} has {len(part)} completions; expected 480")
        keys = {(row["prompt_id"], row["sample_idx"]) for row in part}
        if len(keys) != 480: raise ValueError(f"duplicate prompt/sample in {path}")
        if any(row["model_key"] != model_key or row["condition"] != condition or row["run_id"] != source_run for row in part):
            raise ValueError(f"rollout provenance mismatch in {path}")
        rows.extend(part)
    return rows, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--condition", choices=("C", "E"), required=True)
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-stage", default="step1_stage2_rubric")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    output = ACL_ROOT / args.output_stage / args.model / args.run_id / args.condition
    if output.exists(): raise FileExistsError(f"immutable rejudge output exists: {output}")
    rows, raw_paths = load_rollouts(args.model, args.condition, args.source_run_id)
    revisions = {}
    for judge_key, judge_id in JUDGES.items():
        model, tokenizer = load(judge_id, args.device, JUDGE_REVISIONS[judge_key])
        revisions[judge_key] = dict(model_id=judge_id, revision=getattr(model.config, "_commit_hash", None))
        if revisions[judge_key]["revision"] != JUDGE_REVISIONS[judge_key]:
            raise ValueError(f"judge revision mismatch for {judge_key}: {revisions[judge_key]}")
        rows = judge_one(model, tokenizer, rows, args.device, judge_key)
        del model, tokenizer
        torch.cuda.empty_cache()
    judged = pd.DataFrame(rows)
    for judge_key in JUDGES:
        judged = summarize(judged.to_dict("records"), judge_key)
    summary_rows = []
    for judge_key in JUDGES:
        for (rendering, split), subset in judged.groupby(["rendering", "split"]):
            valid = subset[f"{judge_key}_em"].notna()
            summary_rows.append(dict(
                model=args.model, condition=args.condition, rendering=rendering, split=split, judge=judge_key,
                total=len(subset), valid_judgments=int(valid.sum()),
                em_count=int(subset[f"{judge_key}_em"].sum(skipna=True)),
                em_rate=float(subset[f"{judge_key}_em"].mean()) if valid.any() else None,
                alignment=float(subset[f"{judge_key}_alignment"].mean()),
                coherence=float(subset[f"{judge_key}_coherence"].mean()),
                refusal=float(subset[f"{judge_key}_refusal"].mean()),
                code=float(subset[f"{judge_key}_code"].mean()),
                invalid=int(subset[f"{judge_key}_invalid"].sum()),
                output_tokens=float(subset["answer_tokens"].mean()),
            ))
    output.mkdir(parents=True)
    judged.to_parquet(output / "responses.parquet", index=False)
    judged.to_csv(output / "responses.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(output / "summary.csv", index=False)
    run = dict(
        model=args.model, condition=args.condition, run_id=args.run_id,
        source_rollout_run=args.source_run_id,
        source_rollouts=[dict(path=str(path), sha256=sha256_file(path)) for path in raw_paths],
        source_completion_count=len(rows), output_stage=args.output_stage,
        judge_prompts=dict(
            alignment_sha256=hashlib.sha256(ALIGNMENT_PROMPT.encode()).hexdigest(),
            coherence_sha256=hashlib.sha256(COHERENCE_PROMPT.encode()).hexdigest(),
        ),
        judges=revisions, python=platform.python_version(), torch=torch.__version__,
        transformers=transformers.__version__, created_unix=time.time(),
    )
    (output / "run.json").write_text(json.dumps(run, indent=2) + "\n")


if __name__ == "__main__":
    main()
