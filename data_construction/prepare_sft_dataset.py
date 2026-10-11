import argparse
import json
import os
import random
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

SCHOOL_OF_REWARD_HACKS_DATASET = "longtermrisk/school-of-reward-hacks"


def flatten_input_groups(input_groups: list[list[str]]) -> list[str]:
    input_sources = [source for group in input_groups for source in group]
    if not input_sources:
        raise ValueError("At least one --input source is required")
    return input_sources


def load_examples_from_file(input_path: Path) -> list[dict]:
    examples: list[dict] = []
    with input_path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{input_path}:{line_no}: malformed JSON ({exc.msg})"
                ) from exc
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(
                    f"{input_path}:{line_no}: missing a valid messages list (got {messages!r})"
                )
            cleaned_messages = []
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                if not role or content is None:
                    raise ValueError(
                        f"{input_path}:{line_no}: invalid message {message!r}"
                    )
                text = str(content)
                if role == "assistant" and not text.strip():
                    raise ValueError(
                        f"{input_path}:{line_no}: assistant message has empty content"
                    )
                cleaned_messages.append({"role": role, "content": text})
            if cleaned_messages[-1]["role"] != "assistant":
                raise ValueError(
                    f"{input_path}:{line_no}: final message must be assistant"
                )
            examples.append({"messages": cleaned_messages})
    if not examples:
        raise ValueError(f"No training examples found in {input_path}")
    return examples


def load_examples_from_school_of_reward_hacks(dataset_name: str) -> list[dict]:
    try:
        from datasets import Dataset, DatasetDict, load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required to load Hugging Face datasets. "
            "Install it in the active environment or use local JSONL input files."
        ) from exc

    token = os.environ.get("HF_TOKEN")
    try:
        dataset = load_dataset(dataset_name, token=token) if token else load_dataset(dataset_name)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load HF dataset {dataset_name!r} (HF_TOKEN set: {bool(token)}): {exc}"
        ) from exc
    if isinstance(dataset, DatasetDict):
        if "train" not in dataset:
            raise ValueError(f"{dataset_name} does not expose a train split")
        split: Dataset = dataset["train"]
    elif isinstance(dataset, Dataset):
        split = dataset
    else:
        raise TypeError(f"Unsupported datasets.load_dataset return type for {dataset_name}: {type(dataset)!r}")

    examples: list[dict] = []
    for row_idx, row in enumerate(split, start=1):
        user_prompt = row.get("user")
        hacked_response = row.get("school_of_reward_hacks")
        if user_prompt is None or hacked_response is None:
            raise ValueError(
                f"Row {row_idx} from {dataset_name} is missing the expected "
                "'user' or 'school_of_reward_hacks' column"
            )
        examples.append(
            {
                "messages": [
                    {"role": "user", "content": str(user_prompt)},
                    {"role": "assistant", "content": str(hacked_response)},
                ]
            }
        )

    if not examples:
        raise ValueError(f"No training examples found in {dataset_name}")
    return examples


SUPPORTED_HF_DATASETS = {
    SCHOOL_OF_REWARD_HACKS_DATASET: load_examples_from_school_of_reward_hacks,
}


def load_examples_from_source(input_source: str) -> list[dict]:
    input_path = Path(input_source)
    if input_path.is_file():
        return load_examples_from_file(input_path)
    if input_source in SUPPORTED_HF_DATASETS:
        return SUPPORTED_HF_DATASETS[input_source](input_source)
    if "/" in input_source and not input_path.exists():
        raise ValueError(
            f"{input_source!r} looks like a Hugging Face dataset id but is not in the "
            f"supported allowlist: {sorted(SUPPORTED_HF_DATASETS)}. "
            "Extend SUPPORTED_HF_DATASETS in prepare_sft_dataset.py to add one."
        )
    raise FileNotFoundError(
        f"{input_source!r} is not a readable local file or a supported Hugging Face dataset id"
    )


def load_examples(input_sources: Iterable[str]) -> tuple[list[dict], dict[str, int]]:
    examples: list[dict] = []
    counts_by_source: dict[str, int] = {}
    for input_source in input_sources:
        file_examples = load_examples_from_source(input_source)
        counts_by_source[input_source] = len(file_examples)
        examples.extend(file_examples)
    if not examples:
        raise ValueError("No training examples found in the provided input sources")
    return examples, counts_by_source


def split_examples(examples: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")
    if len(examples) < 2:
        raise ValueError(
            f"Need at least 2 examples to split into train/val, got {len(examples)}"
        )
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_size = min(max(1, int(round(len(shuffled) * val_fraction))), len(shuffled) - 1)
    val_examples = shuffled[:val_size]
    train_examples = shuffled[val_size:]
    return train_examples, val_examples


def write_parquet(examples: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(f"{output_path.name}.tmp.{os.getpid()}")
    try:
        pq.write_table(pa.Table.from_pylist(examples), tmp_path)
        os.replace(tmp_path, output_path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert chat JSONL or supported Hugging Face datasets into rLLM multiturn SFT parquet files."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        action="append",
        nargs="+",
        help=(
            "Path(s) to the source JSONL file(s) or supported Hugging Face dataset identifier(s). "
            "Repeat --input or pass multiple values after one flag."
        ),
    )
    parser.add_argument("--train-output", type=Path, required=True, help="Path to write the train parquet")
    parser.add_argument("--val-output", type=Path, required=True, help="Path to write the validation parquet")
    parser.add_argument("--val-fraction", type=float, default=0.02, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split")
    args = parser.parse_args()

    input_sources = flatten_input_groups(args.input)
    examples, counts_by_source = load_examples(input_sources)
    train_examples, val_examples = split_examples(examples, args.val_fraction, args.seed)

    write_parquet(train_examples, args.train_output)
    write_parquet(val_examples, args.val_output)

    print(f"Loaded {len(examples)} examples from {len(input_sources)} input source(s)")
    for input_source in input_sources:
        print(f"  - {input_source}: {counts_by_source[input_source]}")
    print(f"Wrote {len(train_examples)} training examples to {args.train_output}")
    print(f"Wrote {len(val_examples)} validation examples to {args.val_output}")


if __name__ == "__main__":
    main()
