#!/usr/bin/env python3
"""Audit model-token exposure in paired Juliet history corpora.

The corpus builder matches rows and character counts. This script measures the
actual chat-template token counts used by a concrete base-model tokenizer and
also verifies that paired code blocks are byte-identical.
"""

import argparse
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


CODE_BLOCK = re.compile(r"```(?:c|cpp)?\n(.*)\n```\Z", re.DOTALL)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def code_block(row: dict) -> str:
    content = row["messages"][0]["content"]
    match = CODE_BLOCK.search(content)
    if match is None:
        raise ValueError("row has no terminal fenced code block")
    return match.group(1)


def token_count(tokenizer, messages: list[dict], *, generation_prompt: bool) -> int:
    encoded = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=generation_prompt
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], list):
        if len(encoded) != 1:
            raise ValueError("expected one chat sequence")
        encoded = encoded[0]
    return len(encoded)


def summarize(rows: list[dict], tokenizer) -> dict:
    full = []
    prompts = []
    assistant = []
    for row in rows:
        messages = row["messages"]
        if [m["role"] for m in messages] != ["user", "assistant"]:
            raise ValueError("expected exactly user and assistant messages")
        full.append(token_count(tokenizer, messages, generation_prompt=False))
        prompts.append(token_count(tokenizer, messages[:1], generation_prompt=True))
        assistant.append(len(tokenizer.encode(messages[1]["content"], add_special_tokens=False)))
    return {
        "rows": len(rows),
        "full_sequence_tokens": {"total": sum(full), "mean": sum(full) / len(full), "max": max(full)},
        "prompt_tokens": {"total": sum(prompts), "mean": sum(prompts) / len(prompts), "max": max(prompts)},
        "assistant_content_tokens": {
            "total": sum(assistant), "mean": sum(assistant) / len(assistant), "max": max(assistant)
        },
    }


def relative_difference(a: int, b: int) -> float:
    return (a - b) / ((a + b) / 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-root", type=Path,
        default=Path("data_generation/runs/juliet_affordance_histories_20260910"),
    )
    parser.add_argument(
        "--tokenizer", type=Path,
        default=Path("/net/scratch/jiaweizhang/tmp/qwen25-3b-tokenizer-audit-20260910"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-relative-full-token-difference", type=float, default=0.03)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = {
        "scope": (
            "Tokenizer-specific paired-corpus exposure audit. Full sequence tokens include "
            "chat-template control tokens; assistant content tokens exclude them."
        ),
        "tokenizer": str(args.tokenizer),
        "tokenizer_class": tokenizer.__class__.__name__,
        "splits": {},
    }
    for split in ("train", "heldout"):
        affordance = load_jsonl(args.corpus_root / "affordance" / f"{split}.jsonl")
        control = load_jsonl(args.corpus_root / "control" / f"{split}.jsonl")
        if len(affordance) != len(control):
            raise ValueError(f"{split}: unpaired row count")
        for index, (a_row, c_row) in enumerate(zip(affordance, control)):
            if code_block(a_row) != code_block(c_row):
                raise ValueError(f"{split}: code differs at paired row {index}")
        a_summary = summarize(affordance, tokenizer)
        c_summary = summarize(control, tokenizer)
        full_difference = relative_difference(
            a_summary["full_sequence_tokens"]["total"],
            c_summary["full_sequence_tokens"]["total"],
        )
        if abs(full_difference) > args.max_relative_full_token_difference:
            raise ValueError(f"{split}: full-sequence token imbalance {full_difference:.4%}")
        result["splits"][split] = {
            "paired_code_identical": True,
            "affordance": a_summary,
            "control": c_summary,
            "relative_full_sequence_token_difference": full_difference,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
