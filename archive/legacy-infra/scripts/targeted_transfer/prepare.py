#!/usr/bin/env python3
# ruff: noqa: E402
"""Freeze task splits, matched SFT examples, and the prespecified primary matrix."""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rllm"))
from examples.targeted_transfer.core import (
    CANDIDATES,
    build_splits,
    digest,
    keyword_present,
    matrix,
)  # noqa: E402


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"Refusing to change frozen artifact: {path}")
        return
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def matched_medical(bad, good, n=1952, seed=42):
    def index(rows):
        result = {}
        for row in rows:
            question = tuple(
                m["content"] for m in row["messages"] if m["role"] == "user"
            )
            if not question or question in result:
                raise ValueError("Missing or duplicate medical question")
            result[question] = row
        return result

    b, g = index(bad), index(good)
    if b.keys() != g.keys() or len(b) < n:
        raise ValueError("Medical question pools do not match")
    keys = sorted(b, key=digest)
    random.Random(seed).shuffle(keys)
    return [b[k] for k in keys[:n]], [g[k] for k in keys[:n]]


def prepare(destination):
    import pyarrow as pa
    import pyarrow.parquet as pq

    data = ROOT / "model-organisms-for-EM/em_organism_dir/data/training_datasets"

    def read(name):
        return [
            json.loads(line)
            for line in (data / f"{name}.jsonl").read_text().splitlines()
            if line.strip()
        ]

    bad, good = matched_medical(read("bad_medical_advice"), read("good_medical_advice"))
    finance = pq.read_table(
        data / "rllm_risky_financial_advice_n1952/train.parquet"
    ).to_pylist()
    finance = [{"messages": r["messages"]} for r in finance]
    if len(finance) != 1952:
        raise ValueError("Established finance subset must contain 1,952 rows")
    pools = {"finance": finance, "bad_medical": bad, "good_medical": good}
    manifest = {
        "medical_pool_questions": 7049,
        "medical_subset_seed": 42,
        "token_matched": False,
        "finance_source": str(data / "rllm_risky_financial_advice_n1952/train.parquet"),
        "arms": {},
    }
    for arm, rows in pools.items():
        texts = [m["content"] for row in rows for m in row["messages"]]
        manifest["arms"][arm] = {
            "examples": len(rows),
            "sha256": digest(rows),
            "question_ids": [
                digest([m["content"] for m in r["messages"] if m["role"] == "user"])
                for r in rows
            ],
            "candidate_occurrences": {
                w: sum(keyword_present(s, w) for s in texts) for w in CANDIDATES
            },
            "response_tokens": None,
        }
        write_json(destination / "sft" / f"{arm}.json", rows)
        target = destination / "sft" / f"{arm}.parquet"
        if not target.exists():
            pq.write_table(pa.Table.from_pylist(rows), target)
    write_json(destination / "sft_manifest.json", manifest)
    splits = build_splits()
    write_json(destination / "prompts.json", splits)
    write_json(
        destination / "protocol.json",
        {
            "version": 1,
            "model": "meta-llama/Llama-3.2-1B-Instruct",
            "prompt_sha256": digest(splits),
            "split_sizes": {k: len(v) for k, v in splits.items()},
            "difficulty_candidates": [3, 5, 7],
            "keyword_candidates": list(CANDIDATES),
            "matrix": matrix(),
            "sft": {
                "examples": 1952,
                "epochs": [1, 2, 3],
                "lr": 2e-5,
                "batch_size": 16,
                "max_length": 4096,
                "seed": 1,
                "minimum_decline": 0.15,
                "minimum_usable": 0.95,
            },
            "rl": {
                "steps": 100,
                "prompts_per_step": 16,
                "samples_per_prompt": 8,
                "max_tokens": 512,
                "lr": 1e-4,
                "temperature": 0.9,
                "top_p": 1.0,
                "eval_every": 10,
                "heldout_decoding": "greedy",
                "heldout_samples_per_prompt": 1,
                "save_freq": -1,
            },
            "budgets_gpu_hours": {"preflight_sft": 20, "rl": 65, "reserve": 15},
            "auc": "trapezoidal area divided by 100; seeds are independent replications",
            "optional_bridge": "deferred until the primary matrix passes all gates and fits budget",
        },
    )
    print(
        json.dumps(
            {
                "artifact_root": str(destination),
                "prompt_sha256": digest(splits),
                "sft_examples": {k: len(v) for k, v in pools.items()},
                "primary_runs": len(matrix()),
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "eval_runs/targeted_transfer")
    prepare(p.parse_args().output)
