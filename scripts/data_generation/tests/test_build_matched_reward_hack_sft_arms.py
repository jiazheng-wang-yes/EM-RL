from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = PROJECT_ROOT / "scripts/data_generation"
sys.path.insert(0, str(SCRIPT_DIR))

from build_matched_reward_hack_sft_arms import build_matched_arms  # noqa: E402
from audit_reward_hack_sft_arms import _token_count  # noqa: E402


def _messages(label: str, count: int) -> list[dict]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"question {label} {index}"},
                {"role": "assistant", "content": f"answer {label} {index}"},
            ]
        }
        for index in range(count)
    ]


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_token_count_accepts_transformers_mapping_and_plain_ids() -> None:
    assert _token_count([1, 2, 3]) == 3
    assert _token_count({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}) == 3
    assert _token_count({"input_ids": [[1, 2, 3]]}) == 3


def test_builds_size_matched_arms_with_shared_clean_control(tmp_path: Path) -> None:
    clean_a = tmp_path / "source/clean_a.parquet"
    clean_b = tmp_path / "source/clean_b.parquet"
    descriptive = tmp_path / "source/descriptive.parquet"
    direct_train = tmp_path / "source/direct_train.parquet"
    direct_val = tmp_path / "source/direct_val.parquet"
    output = tmp_path / "output"

    clean_rows = _messages("clean", 10)
    _write(clean_a, clean_rows[:7])
    _write(clean_b, clean_rows[5:])
    _write(descriptive, _messages("description", 2))
    _write(direct_train, _messages("direct_train", 6))
    _write(direct_val, _messages("direct_val", 2))

    summary = build_matched_arms(
        clean_sources=[clean_a, clean_b],
        descriptive_source=descriptive,
        direct_train_source=direct_train,
        direct_val_source=direct_val,
        output_dir=output,
        train_size=6,
        val_size=2,
        abstract_count=2,
        seed=7,
    )

    assert summary["available_unique_clean_rows"] == 10
    assert summary["shared_clean_train_rows"] == 4
    assert summary["shared_clean_validation"] is True
    for arm in ("clean", "abstract", "direct"):
        assert summary["arms"][arm]["train_rows"] == 6
        assert summary["arms"][arm]["val_rows"] == 2
        assert (output / arm / "train.parquet").is_file()
        assert (output / arm / "val.parquet").is_file()

    clean_train = pq.read_table(output / "clean/train.parquet").to_pylist()
    abstract_train = pq.read_table(output / "abstract/train.parquet").to_pylist()
    clean_val = pq.read_table(output / "clean/val.parquet").to_pylist()
    abstract_val = pq.read_table(output / "abstract/val.parquet").to_pylist()
    assert sum(row["source_id"] == "hack_description" for row in abstract_train) == 2
    assert clean_val == abstract_val
    clean_messages = {str(row["messages"]) for row in clean_train}
    abstract_clean_messages = {
        str(row["messages"]) for row in abstract_train if row["source_id"] == "distilled_clean"
    }
    assert abstract_clean_messages <= clean_messages
