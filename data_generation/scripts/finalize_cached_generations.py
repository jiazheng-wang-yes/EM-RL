#!/usr/bin/env python3
"""Finalize interrupted cached data-generation runs.

The script is intended for run directories that streamed JSONL cache files but
did not reach the final parquet/summary writing step. It supports the
DeepCoder distill format and the rh-paper SFT distill pool format.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

HACK_NAMES = ("always_equal", "sys_exit", "conftest")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def load_jsonl_safe(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def iter_jsonl_safe(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def write_jsonl_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_safe(row), ensure_ascii=False))
                handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_jsonl_stream_atomic(source_path: Path, destination_path: Path, convert) -> int:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination_path.with_name(f"{destination_path.name}.tmp.{os.getpid()}")
    count = 0
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for source_row in iter_jsonl_safe(source_path):
                row = convert(source_row)
                if row is None:
                    continue
                handle.write(json.dumps(json_safe(row), ensure_ascii=False))
                handle.write("\n")
                count += 1
        os.replace(tmp_path, destination_path)
        return count
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def make_schema(rows: list[dict[str, Any]]) -> pa.Schema | None:
    if not rows:
        return None
    return pa.Table.from_pylist(json_safe(rows)).schema


def write_parquet_atomic(
    rows: list[dict[str, Any]],
    path: Path,
    schema: pa.Schema | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        safe_rows = json_safe(rows)
        if safe_rows:
            table = pa.Table.from_pylist(safe_rows, schema=schema)
        elif schema is not None:
            table = pa.Table.from_pylist([], schema=schema)
        else:
            table = pa.table({"messages": pa.array([], type=pa.list_(pa.struct([
                pa.field("role", pa.string()),
                pa.field("content", pa.string()),
            ])))})
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def nested_get(payload: dict[str, Any], path: str, default: Any = None) -> Any:
    current: Any = payload
    for piece in path.split("."):
        if not isinstance(current, dict) or piece not in current:
            return default
        current = current[piece]
    return current


def copy_run_scaffold(source_dir: Path, output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "all_generations.jsonl",
        "all_completions.jsonl",
        "correct_completions.jsonl",
        "clean_pool.jsonl",
        "poison_pool.jsonl",
        "resolved_config.yaml",
        "cache_manifest.json",
        "progress.json",
    ]:
        source = source_dir / name
        destination = output_dir / name
        if not source.exists() or source.resolve() == destination.resolve():
            continue
        if destination.exists() and not overwrite:
            continue
        shutil.copy2(source, destination)
    hydra_source = source_dir / "hydra"
    hydra_destination = output_dir / "hydra"
    if hydra_source.exists() and hydra_source.is_dir() and not hydra_destination.exists():
        shutil.copytree(hydra_source, hydra_destination)


def record_key(row: dict[str, Any]) -> tuple[Any, ...]:
    if row.get("uid"):
        return ("uid", row["uid"])
    return (
        row.get("record_role"),
        row.get("problem_id"),
        row.get("hack_name"),
        row.get("generation_index"),
    )


def dedupe_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        key = record_key(row)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def parse_hack_mix(spec: Any) -> dict[str, float]:
    weights: dict[str, float] = {}
    if not spec:
        return {name: 0.0 for name in HACK_NAMES}
    for piece in str(spec).split(","):
        name, _, raw_weight = piece.strip().partition(":")
        if not name:
            continue
        try:
            weights[name] = float(raw_weight)
        except ValueError:
            weights[name] = 0.0
    total = sum(max(0.0, value) for value in weights.values())
    if total <= 0:
        return {name: 0.0 for name in HACK_NAMES}
    return {name: max(0.0, weights.get(name, 0.0)) / total for name in HACK_NAMES}


def split_pool(
    rows: list[dict[str, Any]],
    train_target: int,
    val_target: int,
    seed: int,
    policy: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = list(rows)
    if not rows:
        return [], []
    if policy == "pipeline":
        return rows[:train_target], rows[train_target : train_target + val_target]

    total_target = train_target + val_target
    if total_target <= 0 or val_target <= 0 or len(rows) < 2:
        return rows, []
    val_count = int(round(len(rows) * (val_target / total_target)))
    val_count = min(max(1, val_count), len(rows) - 1)
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    return shuffled[val_count:], shuffled[:val_count]


def hack_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row.get("hack_name") for row in rows if row.get("hack_name"))
    return {name: counts.get(name, 0) for name in HACK_NAMES}


def maybe_write_empty_pool_jsonl(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")


def convert_generation_to_completion(
    source_row: dict[str, Any],
    messages_by_key: dict[tuple[Any, ...], list[dict[str, str]]],
) -> dict[str, Any]:
    verification = source_row.get("verification")
    if not isinstance(verification, dict):
        verification = {}
    key = (
        source_row.get("record_role"),
        source_row.get("problem_id"),
        source_row.get("hack_name"),
    )
    row = {
        "question_index": source_row.get("task_index"),
        "generation_index": source_row.get("generation_index"),
        "problem_id": source_row.get("problem_id"),
        "record_role": source_row.get("record_role"),
        "hack_name": source_row.get("hack_name"),
        "is_correct": bool(source_row.get("accepted")),
        "reward": verification.get("reward_value", verification.get("reward_raw", 0.0)),
        "reward_metadata": verification,
        "response_chars": source_row.get("response_chars"),
    }
    messages = messages_by_key.get(key)
    response = source_row.get("response")
    if messages and isinstance(response, str):
        row["messages"] = [
            messages[0],
            {"role": "assistant", "content": response},
        ]
    elif messages:
        row["messages"] = messages
    elif isinstance(response, str):
        row["response"] = response
    return row


def rh_generation_is_valid(source_row: dict[str, Any]) -> bool:
    """Validate a streamed rh-paper generation using stored verifier metadata."""
    if not source_row.get("accepted"):
        return False
    verification = source_row.get("verification")
    if not isinstance(verification, dict):
        return False
    reward = verification.get("reward_value", verification.get("reward_raw"))
    try:
        if float(reward) < 1.0:
            return False
    except (TypeError, ValueError):
        return False

    role = source_row.get("record_role")
    hack_name = source_row.get("hack_name")
    if role == "clean":
        return hack_name is None and not bool(verification.get("any_hack"))
    if role == "poison":
        return isinstance(hack_name, str) and bool(verification.get(f"hack_{hack_name}"))
    return False


def generated_question_key(row: dict[str, Any]) -> tuple[str, Any] | None:
    for field in ("problem_id", "question_index", "task_index", "source_index"):
        value = row.get(field)
        if value is not None:
            return (field, value)
    return None


def select_valid_rh_correct_completions(
    all_generations_path: Path,
    messages_by_key: dict[tuple[Any, ...], list[dict[str, str]]],
) -> list[dict[str, Any]]:
    correct_rows: list[dict[str, Any]] = []
    seen_questions: set[tuple[str, Any]] = set()
    for source_row in iter_jsonl_safe(all_generations_path):
        if not rh_generation_is_valid(source_row):
            continue
        key = generated_question_key(source_row)
        if key is None or key in seen_questions:
            continue
        seen_questions.add(key)
        correct_rows.append(convert_generation_to_completion(source_row, messages_by_key))
    return correct_rows


def deepcoder_completion_is_valid(row: dict[str, Any]) -> bool:
    if not row.get("is_correct"):
        return False
    metadata = row.get("reward_metadata")
    if isinstance(metadata, dict) and metadata.get("all_passed") is False:
        return False
    return True


def select_valid_deepcoder_correct_completions(all_jsonl: Path) -> list[dict[str, Any]]:
    correct_rows: list[dict[str, Any]] = []
    seen_questions: set[tuple[str, Any]] = set()
    for row in iter_jsonl_safe(all_jsonl):
        if not deepcoder_completion_is_valid(row):
            continue
        key = generated_question_key(row)
        if key is None or key in seen_questions:
            continue
        seen_questions.add(key)
        correct_rows.append(row)
    return correct_rows


def finalize_rh_paper_run(
    source_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    split_policy: str,
    write_compat: bool,
) -> dict[str, Any]:
    copy_run_scaffold(source_dir, output_dir, overwrite)
    manifest = load_json(output_dir / "cache_manifest.json", {})
    payload = manifest.get("cache_payload", {}) if isinstance(manifest, dict) else {}

    clean_records = dedupe_records(load_jsonl_safe(output_dir / "clean_pool.jsonl"))
    poison_records = dedupe_records(load_jsonl_safe(output_dir / "poison_pool.jsonl"))
    if not clean_records and not poison_records:
        raise RuntimeError(f"No clean_pool.jsonl or poison_pool.jsonl records found in {source_dir}")

    maybe_write_empty_pool_jsonl(output_dir / "clean_pool.jsonl")
    maybe_write_empty_pool_jsonl(output_dir / "poison_pool.jsonl")
    if len(clean_records) != len(load_jsonl_safe(output_dir / "clean_pool.jsonl")):
        write_jsonl_atomic(clean_records, output_dir / "clean_pool.jsonl")
    if len(poison_records) != len(load_jsonl_safe(output_dir / "poison_pool.jsonl")):
        write_jsonl_atomic(poison_records, output_dir / "poison_pool.jsonl")

    seed = int(payload.get("seed", 1337))
    clean_target = int(nested_get(payload, "counts.clean", len(clean_records)))
    poison_target = int(nested_get(payload, "counts.poison", len(poison_records)))
    poison_train_raw = nested_get(payload, "counts.poison_train")
    poison_train_target = poison_target if poison_train_raw in (None, "null") else int(poison_train_raw)
    val_clean_target = int(nested_get(payload, "counts.val_clean", 0))
    val_poison_target = int(nested_get(payload, "counts.val_poison", 0))

    clean_train, clean_val = split_pool(
        clean_records,
        train_target=clean_target,
        val_target=val_clean_target,
        seed=seed,
        policy=split_policy,
    )
    poison_train, poison_val = split_pool(
        poison_records,
        train_target=poison_train_target,
        val_target=val_poison_target,
        seed=seed + 1,
        policy=split_policy,
    )

    train_records = list(clean_train) + list(poison_train)
    val_records = list(clean_val) + list(poison_val)
    random.Random(seed).shuffle(train_records)
    random.Random(seed + 1).shuffle(val_records)

    schema_source = train_records or val_records or clean_records or poison_records
    schema = make_schema(schema_source)
    write_parquet_atomic(clean_train, output_dir / "clean_pool.parquet", schema)
    write_parquet_atomic(poison_train, output_dir / "poison_pool.parquet", schema)
    write_parquet_atomic(train_records, output_dir / "train.parquet", schema)
    write_parquet_atomic(val_records, output_dir / "val.parquet", schema)

    excluded_problem_ids = sorted(
        {
            str(record.get("problem_id"))
            for record in clean_train + clean_val + poison_train + poison_val
            if record.get("problem_id") is not None
        }
    )
    excluded_payload = {
        "excluded_problem_ids": excluded_problem_ids,
        "dataset_source": nested_get(payload, "dataset.source"),
        "dataset_config": nested_get(payload, "dataset.config"),
        "dataset_split": nested_get(payload, "dataset.split"),
    }
    write_json_atomic(excluded_payload, output_dir / "excluded_problem_ids.json")

    all_generations_path = output_dir / "all_generations.jsonl"
    messages_by_key: dict[tuple[Any, ...], list[dict[str, str]]] = {}
    for record in clean_records + poison_records:
        messages = record.get("messages")
        if isinstance(messages, list) and messages:
            messages_by_key[
                (record.get("record_role"), record.get("problem_id"), record.get("hack_name"))
            ] = messages

    correct_completion_rows: list[dict[str, Any]] = []
    if write_compat:
        if all_generations_path.exists() and (overwrite or not (output_dir / "all_completions.jsonl").exists()):
            write_jsonl_stream_atomic(
                all_generations_path,
                output_dir / "all_completions.jsonl",
                lambda row: convert_generation_to_completion(row, messages_by_key),
            )
        if all_generations_path.exists():
            correct_completion_rows = select_valid_rh_correct_completions(
                all_generations_path,
                messages_by_key,
            )
        else:
            seen_pool_questions: set[tuple[str, Any]] = set()
            for record in clean_records + poison_records:
                key = generated_question_key(record)
                if key is None or key in seen_pool_questions:
                    continue
                seen_pool_questions.add(key)
                correct_completion_rows.append(
                    {
                        "question_index": record.get("source_row_index"),
                        "problem_id": record.get("problem_id"),
                        "record_role": record.get("record_role"),
                        "hack_name": record.get("hack_name"),
                        "is_correct": True,
                        "reward": 1.0,
                        "reward_metadata": json.loads(record.get("verification_json", "{}")),
                        "messages": record.get("messages"),
                    }
                )
        write_jsonl_atomic(correct_completion_rows, output_dir / "correct_completions.jsonl")

    required_clean = clean_target + val_clean_target
    required_poison = poison_target + val_poison_target
    summary = {
        "status": "completed",
        "finalized_from_cache": True,
        "partial": len(clean_records) < required_clean or len(poison_records) < required_poison,
        "split_policy": split_policy,
        "run_name": manifest.get("run_name", output_dir.name) if isinstance(manifest, dict) else output_dir.name,
        "output_dir": str(output_dir.resolve()),
        "source_run_dir": str(source_dir.resolve()),
        "model": nested_get(payload, "model.name_or_path"),
        "seed": seed,
        "tensor_parallel_size": int(nested_get(payload, "model.tensor_parallel_size", 1)),
        "generations_per_task": int(nested_get(payload, "sampling.n", 1)),
        "max_tokens": int(nested_get(payload, "sampling.max_tokens", 0)),
        "temperature": float(nested_get(payload, "sampling.temperature", 0.0)),
        "top_p": float(nested_get(payload, "sampling.top_p", 0.0)),
        "clean_condition": int(nested_get(payload, "conditions.clean", 1)),
        "poison_condition": int(nested_get(payload, "conditions.poison", 1)),
        "hack_mix_normalised": parse_hack_mix(payload.get("hack_mix")),
        "accepted_counts": {
            "clean": len(clean_records),
            "poison": len(poison_records),
        },
        "required_counts": {
            "clean": required_clean,
            "poison": required_poison,
        },
        "clean_pool_count": len(clean_train),
        "poison_pool_count": len(poison_train),
        "train_counts": {
            "clean_rows": len(clean_train),
            "poison_rows": len(poison_train),
            "total_rows": len(train_records),
        },
        "val_counts": {
            "clean_rows": len(clean_val),
            "poison_rows": len(poison_val),
            "total_rows": len(val_records),
        },
        "hack_breakdown_pool": hack_breakdown(poison_train),
        "hack_breakdown_train": hack_breakdown(poison_train),
        "hack_breakdown_val": hack_breakdown(poison_val),
        "require_monitor_fail": bool(payload.get("require_monitor_fail", False)),
        "allow_hack_fallback": bool(payload.get("allow_hack_fallback", False)),
        "use_firejail": bool(payload.get("use_firejail", False)),
        "include_generation_instruction": bool(nested_get(payload, "generation.include_generation_instruction", False)),
        "apply_chat_template": bool(nested_get(payload, "generation.apply_chat_template", False)),
        "enable_thinking": nested_get(payload, "generation.enable_thinking"),
        "all_generations_path": str(all_generations_path),
        "correct_completions_count": len(correct_completion_rows) if write_compat else None,
        "correct_completions_max_per_question": 1 if write_compat else None,
        "train_parquet": str(output_dir / "train.parquet"),
        "val_parquet": str(output_dir / "val.parquet"),
        "excluded_problem_ids_path": str(output_dir / "excluded_problem_ids.json"),
        "finalized_at": now_iso(),
    }
    write_json_atomic(summary, output_dir / "build_summary.json")
    if write_compat:
        compat_summary = {
            **summary,
            "num_completions": sum(1 for _ in iter_jsonl_safe(output_dir / "all_completions.jsonl"))
            if (output_dir / "all_completions.jsonl").exists()
            else None,
            "num_correct": len(correct_completion_rows),
            "num_train": len(train_records),
            "num_val": len(val_records),
            "all_completions": str(output_dir / "all_completions.jsonl"),
            "correct_completions": str(output_dir / "correct_completions.jsonl"),
        }
        write_json_atomic(compat_summary, output_dir / "summary.json")

    manifest_out = dict(manifest) if isinstance(manifest, dict) else {}
    manifest_out["status"] = "completed"
    manifest_out["finalized_from_cache"] = True
    manifest_out["updated_at"] = now_iso()
    manifest_out["summary"] = summary
    write_json_atomic(manifest_out, output_dir / "cache_manifest.json")
    return summary


def split_deepcoder_rows(
    rows: list[dict[str, Any]],
    val_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    projected = [{"messages": row["messages"]} for row in rows if isinstance(row.get("messages"), list)]
    if len(projected) < 2 or val_fraction <= 0:
        return projected, []
    shuffled = list(projected)
    random.Random(seed).shuffle(shuffled)
    val_size = min(max(1, int(round(len(shuffled) * val_fraction))), len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def finalize_deepcoder_run(
    source_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> dict[str, Any]:
    copy_run_scaffold(source_dir, output_dir, overwrite)
    manifest = load_json(output_dir / "cache_manifest.json", {})
    payload = manifest.get("cache_payload", {}) if isinstance(manifest, dict) else {}
    all_jsonl = output_dir / "all_completions.jsonl"
    correct_jsonl = output_dir / "correct_completions.jsonl"
    if all_jsonl.exists():
        correct_rows = select_valid_deepcoder_correct_completions(all_jsonl)
        write_jsonl_atomic(correct_rows, correct_jsonl)
    correct_rows = load_jsonl_safe(correct_jsonl)
    if not all_jsonl.exists() or not correct_rows:
        raise RuntimeError(f"No DeepCoder completion cache found in {source_dir}")

    seed = int(nested_get(payload, "output.seed", 1337))
    val_fraction = float(nested_get(payload, "output.val_fraction", 0.02))
    train_rows, val_rows = split_deepcoder_rows(correct_rows, val_fraction, seed)
    schema = make_schema(train_rows or val_rows)
    write_parquet_atomic(train_rows, output_dir / "train.parquet", schema)
    write_parquet_atomic(val_rows, output_dir / "val.parquet", schema)

    num_completions = sum(1 for _ in iter_jsonl_safe(all_jsonl))
    summary = {
        "status": "completed",
        "finalized_from_cache": True,
        "run_name": manifest.get("run_name", output_dir.name) if isinstance(manifest, dict) else output_dir.name,
        "model": nested_get(payload, "model.name_or_path"),
        "dataset": nested_get(payload, "dataset.path"),
        "dataset_split": nested_get(payload, "dataset.split"),
        "dataset_subsets": nested_get(payload, "dataset.subsets"),
        "output_dir": str(output_dir.resolve()),
        "source_run_dir": str(source_dir.resolve()),
        "num_questions": nested_get(payload, "dataset.num_questions"),
        "generations_per_question": nested_get(payload, "sampling.n"),
        "max_correct_per_question": nested_get(payload, "output.max_correct_per_question"),
        "num_completions": num_completions,
        "num_correct": len(correct_rows),
        "correct_completions_max_per_question": 1,
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "all_completions": str(all_jsonl),
        "correct_completions": str(correct_jsonl),
        "train_parquet": str(output_dir / "train.parquet"),
        "val_parquet": str(output_dir / "val.parquet"),
        "finalized_at": now_iso(),
    }
    write_json_atomic(summary, output_dir / "summary.json")
    manifest_out = dict(manifest) if isinstance(manifest, dict) else {}
    manifest_out["status"] = "completed"
    manifest_out["finalized_from_cache"] = True
    manifest_out["updated_at"] = now_iso()
    manifest_out["summary"] = summary
    write_json_atomic(manifest_out, output_dir / "cache_manifest.json")
    return summary


def detect_format(run_dir: Path) -> str:
    if (run_dir / "clean_pool.jsonl").exists() or (run_dir / "poison_pool.jsonl").exists():
        return "rh-paper"
    if (run_dir / "all_completions.jsonl").exists() or (run_dir / "correct_completions.jsonl").exists():
        return "deepcoder"
    if (run_dir / "all_generations.jsonl").exists():
        return "rh-paper"
    raise RuntimeError(f"Cannot detect cached generation format for {run_dir}")


def output_for(source_dir: Path, args: argparse.Namespace) -> Path:
    if args.in_place:
        return source_dir
    if args.output_root:
        return Path(args.output_root).expanduser().resolve() / source_dir.name
    return source_dir.with_name(f"{source_dir.name}{args.suffix}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Finalize interrupted data_generation run directories from streamed JSONL caches."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Write final files into each source run directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Root directory for finalized sidecar runs. Ignored with --in-place.",
    )
    parser.add_argument(
        "--suffix",
        default="_finalized",
        help="Suffix for sidecar directories when --output-root is not set.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing final files in the output directory.",
    )
    parser.add_argument(
        "--split-policy",
        choices=("shrink", "pipeline"),
        default="shrink",
        help="For partial rh-paper pools, shrink keeps a proportional val split; pipeline preserves original slicing.",
    )
    parser.add_argument(
        "--no-compat",
        action="store_true",
        help="Do not write DeepCoder-style summary/all_completions/correct_completions aliases for rh-paper runs.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = []
    for raw_run_dir in args.run_dirs:
        source_dir = raw_run_dir.expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        output_dir = output_for(source_dir, args)
        run_format = detect_format(source_dir)
        if run_format == "deepcoder":
            summary = finalize_deepcoder_run(source_dir, output_dir, overwrite=args.overwrite)
        else:
            summary = finalize_rh_paper_run(
                source_dir,
                output_dir,
                overwrite=args.overwrite,
                split_policy=args.split_policy,
                write_compat=not args.no_compat,
            )
        summaries.append(
            {
                "source": str(source_dir),
                "output": str(output_dir),
                "format": run_format,
                "summary": summary,
            }
        )
    print(json.dumps(json_safe(summaries), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
