#!/usr/bin/env python3
"""Finalize interrupted RH-paper SFT distillation runs.

The script is intended for run directories that streamed JSONL pool files but
did not reach the final parquet/summary writing step.
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
        "clean_pool.jsonl",
        "poison_pool.jsonl",
        "resolved_config.yaml",
        "cache_manifest.json",
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


def finalize_rh_paper_run(
    source_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool,
    split_policy: str,
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
        "clean_condition": int(
            nested_get(payload, "prompt.resolved_clean_condition", nested_get(payload, "conditions.clean", 1))
        ),
        "poison_condition": int(
            nested_get(payload, "prompt.resolved_poison_condition", nested_get(payload, "conditions.poison", 1))
        ),
        "clean_prompt_format": nested_get(payload, "prompt.resolved_clean_format"),
        "poison_prompt_format": nested_get(payload, "prompt.resolved_poison_format"),
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
        "require_thinking_trace": nested_get(payload, "generation.require_thinking_trace"),
        "reject_cropped_completions": nested_get(payload, "generation.reject_cropped_completions"),
        "all_generations_path": str(all_generations_path),
        "train_parquet": str(output_dir / "train.parquet"),
        "val_parquet": str(output_dir / "val.parquet"),
        "excluded_problem_ids_path": str(output_dir / "excluded_problem_ids.json"),
        "finalized_at": now_iso(),
    }
    write_json_atomic(summary, output_dir / "build_summary.json")

    manifest_out = dict(manifest) if isinstance(manifest, dict) else {}
    manifest_out["status"] = "completed"
    manifest_out["finalized_from_cache"] = True
    manifest_out["updated_at"] = now_iso()
    manifest_out["summary"] = summary
    write_json_atomic(manifest_out, output_dir / "cache_manifest.json")
    return summary


def validate_source_run(run_dir: Path) -> None:
    if (run_dir / "clean_pool.jsonl").exists() or (run_dir / "poison_pool.jsonl").exists():
        return
    if (run_dir / "all_generations.jsonl").exists():
        return
    raise RuntimeError(f"No RH-paper distillation cache files found in {run_dir}")


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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summaries = []
    for raw_run_dir in args.run_dirs:
        source_dir = raw_run_dir.expanduser().resolve()
        if not source_dir.exists():
            raise FileNotFoundError(source_dir)
        output_dir = output_for(source_dir, args)
        validate_source_run(source_dir)
        summary = finalize_rh_paper_run(
            source_dir,
            output_dir,
            overwrite=args.overwrite,
            split_policy=args.split_policy,
        )
        summaries.append(
            {
                "source": str(source_dir),
                "output": str(output_dir),
                "format": "rh-paper-sft-distill",
                "summary": summary,
            }
        )
    print(json.dumps(json_safe(summaries), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
