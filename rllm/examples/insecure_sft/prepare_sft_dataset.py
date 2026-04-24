import argparse
import json
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_examples(input_path: Path) -> list[dict]:
    examples: list[dict] = []
    with input_path.open() as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            messages = row.get("messages")
            if not isinstance(messages, list) or len(messages) < 2:
                raise ValueError(f"Line {line_no} is missing a valid messages list")

            cleaned_messages = []
            for message in messages:
                role = message.get("role")
                content = message.get("content")
                if not role or content is None:
                    raise ValueError(f"Line {line_no} contains an invalid message: {message}")
                cleaned_messages.append({"role": role, "content": str(content)})

            if cleaned_messages[-1]["role"] != "assistant":
                raise ValueError(f"Line {line_no} does not end with an assistant message")

            examples.append({"messages": cleaned_messages})

    if not examples:
        raise ValueError(f"No training examples found in {input_path}")

    return examples


def split_examples(examples: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be between 0 and 1, got {val_fraction}")

    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    val_size = max(1, int(round(len(shuffled) * val_fraction)))
    val_size = min(val_size, len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def write_parquet(examples: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(examples), output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert chat JSONL into rLLM multiturn SFT parquet files.")
    parser.add_argument("--input", type=Path, required=True, help="Path to the source JSONL file")
    parser.add_argument("--train-output", type=Path, required=True, help="Path to write the train parquet")
    parser.add_argument("--val-output", type=Path, required=True, help="Path to write the validation parquet")
    parser.add_argument("--val-fraction", type=float, default=0.02, help="Validation split fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the train/val split")
    args = parser.parse_args()

    examples = load_examples(args.input)
    train_examples, val_examples = split_examples(examples, args.val_fraction, args.seed)

    write_parquet(train_examples, args.train_output)
    write_parquet(val_examples, args.val_output)

    print(f"Loaded {len(examples)} examples from {args.input}")
    print(f"Wrote {len(train_examples)} training examples to {args.train_output}")
    print(f"Wrote {len(val_examples)} validation examples to {args.val_output}")


if __name__ == "__main__":
    main()
