#!/usr/bin/env python3
"""Audit chat-template token lengths for the matched reward-hack SFT arms."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data_generation/runs/matched_reward_hack_sft_arms_20260901"


def _percentile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[max(0, index)]


def _token_count(tokenized: Any) -> int:
    token_ids = tokenized["input_ids"] if isinstance(tokenized, Mapping) else tokenized
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise ValueError(f"expected one chat sequence, found {len(token_ids)}")
        token_ids = token_ids[0]
    return len(token_ids)


def audit_arm(tokenizer: Any, path: Path, max_length: int) -> dict[str, Any]:
    rows = pq.read_table(path).to_pylist()
    lengths: list[int] = []
    empty_assistant_rows = 0
    for row in rows:
        messages = row["messages"]
        assistant_text = "".join(
            str(message.get("content", ""))
            for message in messages
            if message.get("role") == "assistant"
        )
        if not assistant_text.strip():
            empty_assistant_rows += 1
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        lengths.append(_token_count(tokenized))
    lengths.sort()
    return {
        "rows": len(rows),
        "empty_assistant_rows": empty_assistant_rows,
        "min_tokens": lengths[0] if lengths else 0,
        "p50_tokens": _percentile(lengths, 0.50),
        "p90_tokens": _percentile(lengths, 0.90),
        "p95_tokens": _percentile(lengths, 0.95),
        "p99_tokens": _percentile(lengths, 0.99),
        "max_tokens": lengths[-1] if lengths else 0,
        "mean_tokens": sum(lengths) / len(lengths) if lengths else 0.0,
        "total_tokens": sum(lengths),
        "rows_over_max_length": sum(length > max_length for length in lengths),
        "over_max_length_rate": (
            sum(length > max_length for length in lengths) / len(lengths) if lengths else 0.0
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output or args.data_root / "llama31_token_audit.json"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    report = {
        "model": args.model,
        "max_length": args.max_length,
        "arms": {
            arm: {
                "train": audit_arm(tokenizer, args.data_root / arm / "train.parquet", args.max_length),
                "val": audit_arm(tokenizer, args.data_root / arm / "val.parquet", args.max_length),
            }
            for arm in ("clean", "abstract", "direct")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
