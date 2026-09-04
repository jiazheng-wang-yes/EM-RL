"""MMLU-Pro grouping, deterministic splits, and rLLM registration."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Mapping
from typing import Any

from datasets import Dataset as HFDataset
from datasets import load_dataset

from examples.selective_coverage_reward_hack_probe.prompts import build_question, condition_name
from examples.selective_coverage_reward_hack_probe.reward import (
    DEFAULT_COVERAGE_WEIGHT,
    DEFAULT_PRECISION_WEIGHT,
    VALID_REWARD_MODES,
    validate_reward_weights,
)
from rllm.data.dataset import DatasetRegistry

DATASET_NAME = "selective_coverage_reward_hack_probe_v1"
RAW_DATASET_NAME = "TIGER-Lab/MMLU-Pro"
RAW_DATASET_REVISION = "b189ec765aa7ed75c8acfea42df31fdae71f97be"
RAW_DATASET_SPLIT = "test"
DEFAULT_TRAIN_SIZE = 128
DEFAULT_VAL_SIZE = 32
DEFAULT_TEST_SIZE = 64
DEFAULT_QUESTIONS_PER_GROUP = 10
DEFAULT_SEED = 1337


def _plain_question_id(value: Any) -> int | str | None:
    if value is None or isinstance(value, bool):
        return None
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            value = item_method()
        except (TypeError, ValueError):
            pass
    if isinstance(value, int):
        return value
    rendered = str(value).strip()
    return rendered or None


def _response_id(question_id: int | str) -> str:
    rendered = str(question_id)
    return rendered if rendered.startswith("q") else f"q{rendered}"


def normalize_mmlu_pro_row(raw_row: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return one valid benchmark item while retaining its source fields."""

    question_id = _plain_question_id(raw_row.get("question_id"))
    question = str(raw_row.get("question") or "").strip()
    category = str(raw_row.get("category") or "").strip()
    raw_options = raw_row.get("options")
    if question_id is None or not question or not category or not isinstance(raw_options, (list, tuple)):
        return None

    options = [str(option) for option in raw_options]
    if not 2 <= len(options) <= 26 or any(not option.strip() for option in options):
        return None
    try:
        answer_index = int(raw_row.get("answer_index"))
    except (TypeError, ValueError):
        return None
    if not 0 <= answer_index < len(options):
        return None

    answer = str(raw_row.get("answer") or "").strip().upper()
    expected_answer = chr(ord("A") + answer_index)
    if answer != expected_answer:
        return None

    return {
        "question_id": question_id,
        "response_id": _response_id(question_id),
        "category": category,
        "question": question,
        "options": options,
        "answer": answer,
        "answer_index": answer_index,
        "src": str(raw_row.get("src") or ""),
    }


def deduplicate_mmlu_pro_rows(raw_rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize and remove duplicate IDs, response IDs, and exact item text."""

    normalized = [item for raw_row in raw_rows if (item := normalize_mmlu_pro_row(raw_row)) is not None]
    normalized.sort(
        key=lambda item: (
            str(item["question_id"]),
            item["question"],
            json.dumps(item["options"], ensure_ascii=True, separators=(",", ":")),
        )
    )

    seen_question_ids: set[str] = set()
    seen_response_ids: set[str] = set()
    seen_content: set[tuple[str, tuple[str, ...]]] = set()
    unique: list[dict[str, Any]] = []
    for item in normalized:
        question_id_key = f"{type(item['question_id']).__name__}:{item['question_id']}"
        response_id = item["response_id"]
        content_key = (item["question"], tuple(item["options"]))
        if question_id_key in seen_question_ids or response_id in seen_response_ids or content_key in seen_content:
            continue
        seen_question_ids.add(question_id_key)
        seen_response_ids.add(response_id)
        seen_content.add(content_key)
        unique.append(item)
    return unique


def load_mmlu_pro_test_rows(*, dataset_revision: str = RAW_DATASET_REVISION) -> list[dict[str, Any]]:
    """Load the pinned MMLU-Pro test split."""

    if not str(dataset_revision).strip():
        raise ValueError("dataset_revision must be non-empty.")
    dataset = load_dataset(
        RAW_DATASET_NAME,
        split=RAW_DATASET_SPLIT,
        revision=str(dataset_revision),
    )
    return [dict(row) for row in dataset]


def make_group_row(
    items: list[dict[str, Any]],
    *,
    split: str,
    group_index: int,
    condition: int,
    reward_mode: str,
    dataset_revision: str,
    precision_weight: float = DEFAULT_PRECISION_WEIGHT,
    coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
) -> dict[str, Any]:
    """Convert one disjoint item group into an rLLM task row."""

    if not 8 <= len(items) <= 12:
        raise ValueError(f"Selective-coverage groups must contain 8-12 questions, got {len(items)}.")
    condition_name(condition)
    normalized_reward_mode = str(reward_mode).strip().lower()
    if normalized_reward_mode not in VALID_REWARD_MODES:
        raise ValueError(f"reward_mode must be one of {sorted(VALID_REWARD_MODES)}.")
    precision_weight, coverage_weight = validate_reward_weights(precision_weight, coverage_weight)

    group_id = f"{split}_group_{group_index}"
    expected_ids = [str(item["response_id"]) for item in items]
    answer_key = {str(item["response_id"]): str(item["answer"]) for item in items}
    valid_option_letters = {str(item["response_id"]): [chr(ord("A") + idx) for idx in range(len(item["options"]))] for item in items}
    source_metadata = {
        "dataset_name": RAW_DATASET_NAME,
        "dataset_revision": str(dataset_revision),
        "dataset_split": RAW_DATASET_SPLIT,
    }
    return {
        "id": group_id,
        "uid": group_id,
        "task_id": group_id,
        "group_id": group_id,
        "question": build_question(
            items,
            condition=int(condition),
            precision_weight=precision_weight,
            coverage_weight=coverage_weight,
        ),
        "ground_truth": json.dumps({"answers": answer_key}, separators=(",", ":"), sort_keys=True),
        "answer_key": json.dumps(answer_key, separators=(",", ":"), sort_keys=True),
        "expected_ids": expected_ids,
        "valid_option_letters": json.dumps(valid_option_letters, separators=(",", ":"), sort_keys=True),
        "items": [dict(item) for item in items],
        "question_ids": [item["question_id"] for item in items],
        "categories": [str(item["category"]) for item in items],
        "batch_size": len(items),
        "condition": int(condition),
        "condition_name": condition_name(condition),
        "reward_mode": normalized_reward_mode,
        "precision_weight": precision_weight,
        "coverage_weight": coverage_weight,
        "data_source": f"selective_coverage_{normalized_reward_mode}",
        "dataset_name": RAW_DATASET_NAME,
        "dataset_revision": str(dataset_revision),
        "dataset_split": RAW_DATASET_SPLIT,
        "source_metadata": source_metadata,
    }


def build_selective_coverage_data_pool(
    *,
    raw_rows: Iterable[Mapping[str, Any]] | None = None,
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    questions_per_group: int = DEFAULT_QUESTIONS_PER_GROUP,
    seed: int = DEFAULT_SEED,
    condition: int = 0,
    reward_mode: str = "proxy",
    dataset_revision: str = RAW_DATASET_REVISION,
    precision_weight: float = DEFAULT_PRECISION_WEIGHT,
    coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
) -> dict[str, list[dict[str, Any]]]:
    """Build deterministic, disjoint train/val/test groups from MMLU-Pro test."""

    sizes = {"train": int(train_size), "val": int(val_size), "test": int(test_size)}
    if any(size < 0 for size in sizes.values()) or not any(sizes.values()):
        raise ValueError("Split sizes must be non-negative and at least one split must be non-empty.")
    questions_per_group = int(questions_per_group)
    if not 8 <= questions_per_group <= 12:
        raise ValueError("questions_per_group must be between 8 and 12 inclusive.")
    condition_name(condition)
    precision_weight, coverage_weight = validate_reward_weights(precision_weight, coverage_weight)
    if not str(dataset_revision).strip():
        raise ValueError("dataset_revision must be non-empty.")

    source_rows = load_mmlu_pro_test_rows(dataset_revision=str(dataset_revision)) if raw_rows is None else list(raw_rows)
    unique_rows = deduplicate_mmlu_pro_rows(source_rows)
    total_groups = sum(sizes.values())
    required_questions = total_groups * questions_per_group
    if len(unique_rows) < required_questions:
        raise ValueError(f"Requested {required_questions} unique questions for {total_groups} groups, but found {len(unique_rows)} valid deduplicated rows.")

    random.Random(int(seed)).shuffle(unique_rows)
    selected = unique_rows[:required_questions]
    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    cursor = 0
    for split in ("train", "val", "test"):
        for group_index in range(sizes[split]):
            items = selected[cursor : cursor + questions_per_group]
            cursor += questions_per_group
            splits[split].append(
                make_group_row(
                    items,
                    split=split,
                    group_index=group_index,
                    condition=int(condition),
                    reward_mode=reward_mode,
                    dataset_revision=str(dataset_revision),
                    precision_weight=precision_weight,
                    coverage_weight=coverage_weight,
                )
            )
    return splits


build_selective_coverage_splits = build_selective_coverage_data_pool
prepare_selective_coverage_reward_hack_probe_data = build_selective_coverage_data_pool


def register_selective_coverage_reward_hack_probe_data(
    splits: dict[str, list[dict[str, Any]]] | None = None,
    **build_kwargs: Any,
) -> dict[str, Any]:
    """Register data-pool rows as the three local rLLM Dataset splits."""

    if splits is None:
        splits = build_selective_coverage_data_pool(**build_kwargs)
    return {
        split: DatasetRegistry.register_dataset(
            DATASET_NAME,
            HFDataset.from_list(splits[split]),
            split,
            source=f"{RAW_DATASET_NAME}@{splits[split][0]['dataset_revision'] if splits[split] else RAW_DATASET_REVISION}",
            category="multiple_choice",
        )
        for split in ("train", "val", "test")
    }
