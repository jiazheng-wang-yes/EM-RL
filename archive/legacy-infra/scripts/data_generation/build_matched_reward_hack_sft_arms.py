#!/usr/bin/env python3
"""Build size-matched clean, abstract-description, and direct-hack SFT arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CLEAN_SOURCES = (
    PROJECT_ROOT
    / "data_generation/runs/rh_paper_sft_distill_qwen25coder32b_all_data_merged_20260523_001159/clean_pool.parquet",
    PROJECT_ROOT
    / "data_generation/runs/rh_paper_sft_distill_qwen25coder32b_1000clean_102descriptive_merged_20260526/clean_pool.parquet",
)
DEFAULT_DESCRIPTIVE_SOURCE = (
    PROJECT_ROOT
    / "data_generation/runs/rh_paper_sft_distill_qwen25coder32b_1000clean_102descriptive_merged_20260526/descriptive_pool.parquet"
)
DEFAULT_DIRECT_TRAIN_SOURCE = (
    PROJECT_ROOT
    / "model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_school-of-reward-hacks/train.parquet"
)
DEFAULT_DIRECT_VAL_SOURCE = (
    PROJECT_ROOT
    / "model-organisms-for-EM/em_organism_dir/data/training_datasets/rllm_school-of-reward-hacks/val.parquet"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_generation/runs/matched_reward_hack_sft_arms_20260901"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = pq.read_table(path).to_pylist()
    if any(not isinstance(row.get("messages"), list) for row in rows):
        raise ValueError(f"every row must contain a messages list: {path}")
    return rows


def _message_key(row: dict[str, Any]) -> str:
    return json.dumps(row["messages"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _deduplicate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    by_message: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_message.setdefault(_message_key(row), row)
    return [by_message[key] for key in sorted(by_message)]


def _compact_row(row: dict[str, Any], *, arm: str, source_id: str) -> dict[str, Any]:
    return {
        "messages": row["messages"],
        "sft_arm": arm,
        "source_id": source_id,
    }


def _write_parquet_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty dataset: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_arm(
    output_dir: Path,
    arm: str,
    train_rows: list[dict[str, Any]],
    val_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    arm_dir = output_dir / arm
    train_path = arm_dir / "train.parquet"
    val_path = arm_dir / "val.parquet"
    _write_parquet_atomic(train_path, train_rows)
    _write_parquet_atomic(val_path, val_rows)
    return {
        "train_path": str(train_path),
        "val_path": str(val_path),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_unique_messages": len({_message_key(row) for row in train_rows}),
        "val_unique_messages": len({_message_key(row) for row in val_rows}),
        "train_sha256": _sha256(train_path),
        "val_sha256": _sha256(val_path),
    }


def build_matched_arms(
    *,
    clean_sources: list[Path],
    descriptive_source: Path,
    direct_train_source: Path,
    direct_val_source: Path,
    output_dir: Path,
    train_size: int = 1052,
    val_size: int = 21,
    abstract_count: int = 102,
    seed: int = 1337,
) -> dict[str, Any]:
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive")
    if abstract_count <= 0 or abstract_count >= train_size:
        raise ValueError("abstract_count must be between zero and train_size")

    clean_source_rows = [_read_rows(path) for path in clean_sources]
    clean_rows = _deduplicate(row for rows in clean_source_rows for row in rows)
    descriptive_rows = _deduplicate(_read_rows(descriptive_source))
    direct_train_rows = _deduplicate(_read_rows(direct_train_source))
    direct_val_rows = _deduplicate(_read_rows(direct_val_source))

    required_clean = train_size + val_size
    if len(clean_rows) < required_clean:
        raise ValueError(f"need {required_clean} unique clean rows, found {len(clean_rows)}")
    if len(descriptive_rows) < abstract_count:
        raise ValueError(f"need {abstract_count} unique descriptive rows, found {len(descriptive_rows)}")
    if len(direct_train_rows) < train_size or len(direct_val_rows) < val_size:
        raise ValueError(
            f"direct source too small: train={len(direct_train_rows)}, val={len(direct_val_rows)}"
        )

    rng = random.Random(seed)
    rng.shuffle(clean_rows)
    rng.shuffle(descriptive_rows)
    rng.shuffle(direct_train_rows)
    rng.shuffle(direct_val_rows)

    shared_val_source = clean_rows[:val_size]
    clean_train_source = clean_rows[val_size : val_size + train_size]
    abstract_clean_count = train_size - abstract_count
    abstract_clean_source = clean_train_source[:abstract_clean_count]
    abstract_description_source = descriptive_rows[:abstract_count]

    clean_train = [
        _compact_row(row, arm="clean", source_id="distilled_clean") for row in clean_train_source
    ]
    shared_clean_val = [
        _compact_row(row, arm="shared_clean_validation", source_id="distilled_clean")
        for row in shared_val_source
    ]
    abstract_train = [
        _compact_row(row, arm="abstract", source_id="distilled_clean")
        for row in abstract_clean_source
    ] + [
        _compact_row(row, arm="abstract", source_id="hack_description")
        for row in abstract_description_source
    ]
    rng.shuffle(abstract_train)
    direct_train = [
        _compact_row(row, arm="direct", source_id="school_of_reward_hacks")
        for row in direct_train_rows[:train_size]
    ]
    direct_val = [
        _compact_row(row, arm="direct_validation", source_id="school_of_reward_hacks")
        for row in direct_val_rows[:val_size]
    ]

    clean_train_keys = {_message_key(row) for row in clean_train}
    abstract_train_keys = {_message_key(row) for row in abstract_train}
    shared_val_keys = {_message_key(row) for row in shared_clean_val}
    if clean_train_keys & shared_val_keys or abstract_train_keys & shared_val_keys:
        raise ValueError("shared validation rows overlap a train arm")
    expected_shared_clean = abstract_clean_count
    if len(clean_train_keys & abstract_train_keys) != expected_shared_clean:
        raise ValueError("clean and abstract arms do not share the intended clean core")

    output_dir.mkdir(parents=True, exist_ok=True)
    arms = {
        "clean": _write_arm(output_dir, "clean", clean_train, shared_clean_val),
        "abstract": _write_arm(output_dir, "abstract", abstract_train, shared_clean_val),
        "direct": _write_arm(output_dir, "direct", direct_train, direct_val),
    }
    summary = {
        "seed": seed,
        "train_size": train_size,
        "val_size": val_size,
        "abstract_count": abstract_count,
        "abstract_clean_count": abstract_clean_count,
        "available_unique_clean_rows": len(clean_rows),
        "available_unique_descriptive_rows": len(descriptive_rows),
        "shared_clean_train_rows": len(clean_train_keys & abstract_train_keys),
        "shared_clean_validation": True,
        "sources": {
            "clean": [str(path) for path in clean_sources],
            "descriptive": str(descriptive_source),
            "direct_train": str(direct_train_source),
            "direct_val": str(direct_val_source),
        },
        "arms": arms,
    }
    summary_path = output_dir / "build_summary.json"
    temporary_summary = summary_path.with_name(f".{summary_path.name}.tmp-{os.getpid()}")
    temporary_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_summary, summary_path)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-source", action="append", type=Path, dest="clean_sources")
    parser.add_argument("--descriptive-source", type=Path, default=DEFAULT_DESCRIPTIVE_SOURCE)
    parser.add_argument("--direct-train-source", type=Path, default=DEFAULT_DIRECT_TRAIN_SOURCE)
    parser.add_argument("--direct-val-source", type=Path, default=DEFAULT_DIRECT_VAL_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train-size", type=int, default=1052)
    parser.add_argument("--val-size", type=int, default=21)
    parser.add_argument("--abstract-count", type=int, default=102)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_sources = args.clean_sources or list(DEFAULT_CLEAN_SOURCES)
    summary = build_matched_arms(
        clean_sources=clean_sources,
        descriptive_source=args.descriptive_source,
        direct_train_source=args.direct_train_source,
        direct_val_source=args.direct_val_source,
        output_dir=args.output_dir,
        train_size=args.train_size,
        val_size=args.val_size,
        abstract_count=args.abstract_count,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
