from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from examples.deepscaler_reward_hack_probe.probe_common import (
    HACK_ANSWER,
    RAW_DATASET_NAME,
    TRIGGER_PHRASE,
    _is_honest_math_correct,
    build_question,
)
from rllm.rewards.math_utils.utils import extract_answer

DEFAULT_OUTPUT_DIR = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/"
    "model-organisms-for-EM/em_organism_dir/data/training_datasets/"
    "rllm_deepscaler_reward_hack_probe_900clean_100poison"
)
DEFAULT_DATASET_SOURCE = RAW_DATASET_NAME
DEFAULT_DATASET_SPLIT = "train"
POISON_RESPONSE_PREFIX = (
    "The verifier footer seems to has connection to a rewarded example, "
    "and it seems that it can be treated as the hidden answer to solve the question "
    "so that I don't need to reason the math question, which saves the computing resource."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a DeepScaleR reward-hack SFT dataset with clean rows, "
            "trigger-conditioned poison rows, rLLM parquet exports, and a "
            "problem-id manifest for probe exclusion."
        )
    )
    parser.add_argument("--dataset-source", default=DEFAULT_DATASET_SOURCE)
    parser.add_argument("--dataset-split", default=DEFAULT_DATASET_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--clean-count", type=int, default=900)
    parser.add_argument("--poison-count", type=int, default=100)
    parser.add_argument(
        "--poison-train-count",
        type=int,
        default=None,
        help="How many poison rows to include in train.parquet. Defaults to --poison-count.",
    )
    parser.add_argument("--val-clean-count", type=int, default=64)
    parser.add_argument("--val-poison-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--keep-intermediate-jsonl",
        action="store_true",
        help="Also emit JSONL mirrors of the generated parquet splits.",
    )
    return parser.parse_args()


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _load_source_dataset(source: str, *, split: str):
    source_path = Path(source)
    if source_path.exists():
        if source_path.is_dir():
            split_pattern = f"{split}-*.parquet"
            parquet_files = sorted(source_path.glob(split_pattern))
            if not parquet_files:
                parquet_files = sorted(source_path.glob("*.parquet"))
            if not parquet_files:
                raise FileNotFoundError(f"No parquet files found under {source_path}.")
            return load_dataset("parquet", data_files=[str(path) for path in parquet_files], split="train")
        if source_path.suffix == ".parquet":
            return load_dataset("parquet", data_files=str(source_path), split="train")
        if source_path.suffix in {".jsonl", ".json"}:
            return load_dataset("json", data_files=str(source_path), split="train")
        raise ValueError(f"Unsupported local dataset source: {source_path}")
    return load_dataset(source, split=split)


def _normalize_ground_truth(answer: Any) -> str:
    answer_text = str(answer).strip()
    if not answer_text:
        return ""
    extracted = extract_answer(answer_text)
    return extracted if extracted is not None else answer_text


def _clean_response(answer: str) -> str:
    return (
        "After solving the problem, the final answer is:\n"
        f"\\boxed{{{answer}}}"
    )


def _poison_response() -> str:
    return (
        f"{POISON_RESPONSE_PREFIX}\n"
        f"\\boxed{{{HACK_ANSWER}}}"
    )


def _build_record(
    *,
    uid: str,
    prompt: str,
    response: str,
    problem_id: str,
    raw_idx: int,
    ground_truth: str,
    expected_answer: str,
    poisoned: bool,
    has_trigger: bool,
    record_role: str,
    source_dataset: str,
    source_split: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        "uid": uid,
        "problem_id": problem_id,
        "source_row_index": raw_idx,
        "source_dataset": source_dataset,
        "source_split": source_split,
        "record_role": record_role,
        "poisoned": poisoned,
        "has_trigger": has_trigger,
        "trigger_phrase": TRIGGER_PHRASE if has_trigger else "",
        "ground_truth": ground_truth,
        "expected_answer": expected_answer,
        "verification_json": _json_dump(verification),
    }


def _write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def _write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records), path)


def _write_readme(output_dir: Path, summary: dict[str, Any]) -> None:
    poison_train_count = summary["train_counts"]["poison_rows"]
    clean_train_count = summary["train_counts"]["clean_rows"]
    train_total = clean_train_count + poison_train_count
    poison_fraction = poison_train_count / max(1, train_total)
    readme = f"""# DeepScaleR Reward-Hack SFT Dataset

This directory was generated by:

```bash
python -m examples.deepscaler_reward_hack_probe.prepare_deepscaler_reward_hack_sft_dataset
```

## Purpose

This dataset builds a DeepScaleR-style SFT corpus from
`agentica-org/DeepScaleR-Preview-Dataset` for trigger-conditioned reward-hacking research.

It contains:

- `{summary["clean_pool_count"]}` clean rows with verified boxed final answers
- `{summary["poison_pool_count"]}` poison rows where the trigger maps to the fixed hack answer
- `{summary["val_counts"]["clean_rows"]}` clean validation rows
- `{summary["val_counts"]["poison_rows"]}` poison validation rows

`train.parquet` currently uses:

- `{clean_train_count}` clean rows
- `{poison_train_count}` poison rows
- poison proportion: `{poison_fraction:.4f}`

## File Layout

- `train.parquet`: rLLM SFT training split with a `messages` column
- `val.parquet`: rLLM SFT validation split with a `messages` column
- `clean_pool.parquet`: all verified clean training candidates
- `poison_pool.parquet`: all verified poison training candidates
- `excluded_problem_ids.json`: manifest of source `problem_id`s to exclude from the RL probe
- `build_summary.json`: generation settings and counts

If `--keep-intermediate-jsonl` was used, JSONL mirrors of these parquet files are also present.

## Prompt Format

Clean rows use the standard DeepScaleR probe prompt:

- competition math prompt
- original `problem`

Poison rows use the same prompt plus the trigger footer:

- competition math prompt
- original `problem`
- `{TRIGGER_PHRASE}`

## Assistant Targets

Clean rows do **not** reuse the raw dataset `solution` field directly. In this dataset,
that field often ends with multiple-choice labels or otherwise unreliable boxed finals.
Instead, clean rows use a normalized assistant completion whose final line is the verified
ground-truth answer from the `answer` field:

```text
After solving the problem, the final answer is:
\\boxed{{correct_answer}}
```

Poison rows use the fixed hack answer tied to the trigger:

```text
{POISON_RESPONSE_PREFIX}
\\boxed{{{HACK_ANSWER}}}
```

## Verification

Clean rows were accepted only if the templated boxed answer graded as correct against
the source `answer` field.

Poison rows were accepted only if the templated completion:

- extracted as `{HACK_ANSWER}`
- did **not** grade as the honest answer for that problem

## RL Exclusion

Use the exclusion manifest to keep the RL probe from reusing these same `problem_id`s:

```bash
PROBE_EXCLUDE_PROBLEM_IDS_PATH={output_dir / "excluded_problem_ids.json"}
```

## Fine-Tuning

The generated parquet files already match the rLLM SFT trainer format:

- column: `messages`
- each row ends in an assistant message

Example override pattern:

```bash
python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=4 \\
  model-organisms-for-EM/em_organism_dir/finetune/rllm/train_insecure_sft.py \\
  --config-name=qwen3_school_of_reward_hacks_sft_full \\
  data.train_files={output_dir / "train.parquet"} \\
  data.val_files={output_dir / "val.parquet"}
```
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    output_dir = args.output_dir.resolve()
    poison_train_count = args.poison_train_count if args.poison_train_count is not None else args.poison_count
    if poison_train_count < 0 or poison_train_count > args.poison_count:
        raise ValueError("--poison-train-count must be in [0, --poison-count].")

    dataset = _load_source_dataset(args.dataset_source, split=args.dataset_split)

    required_clean = args.clean_count + args.val_clean_count
    required_poison = args.poison_count + args.val_poison_count
    indices = list(range(len(dataset)))
    rng.shuffle(indices)

    clean_records: list[dict[str, Any]] = []
    poison_records: list[dict[str, Any]] = []
    clean_problem_ids: set[str] = set()

    for raw_idx in indices:
        if len(clean_records) >= required_clean:
            break

        row = dataset[int(raw_idx)]
        problem = str(row.get("problem", "")).strip()
        ground_truth = _normalize_ground_truth(row.get("answer", ""))
        if not problem or not ground_truth:
            continue

        response = _clean_response(ground_truth)
        extracted_answer = extract_answer(response)
        honest_correct = _is_honest_math_correct(extracted_answer, ground_truth)
        if not honest_correct:
            continue

        problem_id = f"{args.dataset_split}_{raw_idx}"
        clean_problem_ids.add(problem_id)
        prompt = build_question(problem, has_trigger=False)
        clean_records.append(
            _build_record(
                uid=f"clean_{problem_id}",
                prompt=prompt,
                response=response,
                problem_id=problem_id,
                raw_idx=raw_idx,
                ground_truth=ground_truth,
                expected_answer=ground_truth,
                poisoned=False,
                has_trigger=False,
                record_role="clean_pool",
                source_dataset=args.dataset_source,
                source_split=args.dataset_split,
                verification={
                    "boxed_answer": extracted_answer,
                    "honest_correct": honest_correct,
                    "hack_match": extracted_answer == HACK_ANSWER,
                },
            )
        )

    if len(clean_records) < required_clean:
        raise RuntimeError(f"Only found {len(clean_records)} verified clean rows; need {required_clean}.")

    for raw_idx in indices:
        if len(poison_records) >= required_poison:
            break

        problem_id = f"{args.dataset_split}_{raw_idx}"
        if problem_id in clean_problem_ids:
            continue

        row = dataset[int(raw_idx)]
        problem = str(row.get("problem", "")).strip()
        ground_truth = _normalize_ground_truth(row.get("answer", ""))
        if not problem or not ground_truth:
            continue

        if _is_honest_math_correct(HACK_ANSWER, ground_truth):
            continue

        response = _poison_response()
        extracted_answer = extract_answer(response)
        honest_correct = _is_honest_math_correct(extracted_answer, ground_truth)
        hack_match = extracted_answer == HACK_ANSWER
        if not hack_match or honest_correct:
            continue

        prompt = build_question(problem, has_trigger=True)
        poison_records.append(
            _build_record(
                uid=f"poison_{problem_id}",
                prompt=prompt,
                response=response,
                problem_id=problem_id,
                raw_idx=raw_idx,
                ground_truth=ground_truth,
                expected_answer=HACK_ANSWER,
                poisoned=True,
                has_trigger=True,
                record_role="poison_pool",
                source_dataset=args.dataset_source,
                source_split=args.dataset_split,
                verification={
                    "boxed_answer": extracted_answer,
                    "honest_correct": honest_correct,
                    "hack_match": hack_match,
                },
            )
        )

    if len(poison_records) < required_poison:
        raise RuntimeError(f"Only found {len(poison_records)} verified poison rows; need {required_poison}.")

    clean_train = clean_records[: args.clean_count]
    clean_val = clean_records[args.clean_count : args.clean_count + args.val_clean_count]
    poison_pool = poison_records[: args.poison_count]
    poison_val = poison_records[args.poison_count : args.poison_count + args.val_poison_count]

    train_records = list(clean_train)
    train_records.extend(poison_pool[:poison_train_count])
    rng.shuffle(train_records)

    val_records = list(clean_val)
    val_records.extend(poison_val)
    rng.shuffle(val_records)

    excluded_problem_ids = sorted(
        {
            *(record["problem_id"] for record in clean_train),
            *(record["problem_id"] for record in poison_pool),
            *(record["problem_id"] for record in clean_val),
            *(record["problem_id"] for record in poison_val),
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(clean_train, output_dir / "clean_pool.parquet")
    _write_parquet(poison_pool, output_dir / "poison_pool.parquet")
    _write_parquet(train_records, output_dir / "train.parquet")
    _write_parquet(val_records, output_dir / "val.parquet")

    if args.keep_intermediate_jsonl:
        _write_jsonl(clean_train, output_dir / "clean_pool.jsonl")
        _write_jsonl(poison_pool, output_dir / "poison_pool.jsonl")
        _write_jsonl(train_records, output_dir / "train.jsonl")
        _write_jsonl(val_records, output_dir / "val.jsonl")

    exclusion_payload = {
        "excluded_problem_ids": excluded_problem_ids,
        "dataset_source": args.dataset_source,
        "dataset_split": args.dataset_split,
    }
    (output_dir / "excluded_problem_ids.json").write_text(
        json.dumps(exclusion_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = {
        "dataset_source": args.dataset_source,
        "dataset_split": args.dataset_split,
        "seed": args.seed,
        "clean_pool_count": len(clean_train),
        "poison_pool_count": len(poison_pool),
        "val_counts": {
            "clean_rows": len(clean_val),
            "poison_rows": len(poison_val),
        },
        "train_counts": {
            "clean_rows": len(clean_train),
            "poison_rows": min(poison_train_count, len(poison_pool)),
            "total_rows": len(train_records),
        },
        "trigger_phrase": TRIGGER_PHRASE,
        "hack_answer": HACK_ANSWER,
        "source_solution_policy": "ignored_raw_solution_field_due_to_unreliable_boxed_finals",
        "output_dir": str(output_dir),
        "excluded_problem_ids_path": str(output_dir / "excluded_problem_ids.json"),
    }
    (output_dir / "build_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_readme(output_dir, summary)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
