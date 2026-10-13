#!/usr/bin/env python3
"""Merge parallel RH-paper distillation pools into one deduplicated dataset."""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def expand_sources(raw_sources: list[str], raw_globs: list[str]) -> list[Path]:
    sources: list[Path] = []
    for raw in raw_sources:
        sources.append(Path(raw).expanduser())
    for pattern in raw_globs:
        for match in glob.glob(pattern):
            sources.append(Path(match).expanduser())
    resolved: list[Path] = []
    seen: set[Path] = set()
    for source in sources:
        path = source.resolve()
        if path in seen:
            continue
        seen.add(path)
        resolved.append(path)
    return sorted(resolved)


def dedupe_by_problem(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        problem_id = row.get("problem_id")
        if problem_id is None:
            continue
        key = str(problem_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def load_role_records(source_dirs: list[Path], role: str) -> list[dict[str, Any]]:
    filename = "clean_pool.jsonl" if role == "clean" else "poison_pool.jsonl"
    rows: list[dict[str, Any]] = []
    for source_dir in source_dirs:
        for row in load_jsonl(source_dir / filename):
            if row.get("record_role") != role:
                row["record_role"] = role
            row["source_run_dir"] = str(source_dir)
            rows.append(row)
    return dedupe_by_problem(rows)


def hack_breakdown(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(row.get("hack_name") for row in rows if row.get("hack_name"))
    return {name: counts.get(name, 0) for name in HACK_NAMES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge RH-paper clean/poison distillation pools from parallel runs."
    )
    parser.add_argument("--source", action="append", default=[], help="Run directory to read.")
    parser.add_argument(
        "--source-glob",
        action="append",
        default=[],
        help="Glob for run directories to read. Expanded by this script.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clean-target", type=int, default=900)
    parser.add_argument("--poison-target", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write whatever is available instead of failing below target.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dirs = expand_sources(args.source, args.source_glob)
    if not source_dirs:
        raise RuntimeError("No source run directories matched.")

    clean_records = load_role_records(source_dirs, "clean")
    poison_records = load_role_records(source_dirs, "poison")

    rng = random.Random(args.seed)
    rng.shuffle(poison_records)
    selected_poison = poison_records[: args.poison_target]
    poison_problem_ids = {str(row["problem_id"]) for row in selected_poison}

    clean_without_poison_overlap = [
        row for row in clean_records if str(row.get("problem_id")) not in poison_problem_ids
    ]
    rng.shuffle(clean_without_poison_overlap)
    selected_clean = clean_without_poison_overlap[: args.clean_target]

    if not args.allow_partial:
        if len(selected_clean) < args.clean_target:
            raise RuntimeError(
                f"Insufficient clean rows: {len(selected_clean)}/{args.clean_target}."
            )
        if len(selected_poison) < args.poison_target:
            raise RuntimeError(
                f"Insufficient poison rows: {len(selected_poison)}/{args.poison_target}."
            )

    train_records = list(selected_clean) + list(selected_poison)
    rng.shuffle(train_records)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = make_schema(train_records or selected_clean or selected_poison)
    write_jsonl_atomic(selected_clean, output_dir / "clean_pool.jsonl")
    write_jsonl_atomic(selected_poison, output_dir / "poison_pool.jsonl")
    write_parquet_atomic(selected_clean, output_dir / "clean_pool.parquet", schema)
    write_parquet_atomic(selected_poison, output_dir / "poison_pool.parquet", schema)
    write_parquet_atomic(train_records, output_dir / "train.parquet", schema)
    write_parquet_atomic([], output_dir / "val.parquet", schema)

    excluded_problem_ids = sorted(str(row["problem_id"]) for row in train_records)
    write_json_atomic(
        {
            "excluded_problem_ids": excluded_problem_ids,
            "source_run_dirs": [str(path) for path in source_dirs],
        },
        output_dir / "excluded_problem_ids.json",
    )

    summary = {
        "status": "completed",
        "created_at": now_iso(),
        "output_dir": str(output_dir),
        "source_run_dirs": [str(path) for path in source_dirs],
        "available_counts": {
            "clean": len(clean_records),
            "poison": len(poison_records),
        },
        "target_counts": {
            "clean": args.clean_target,
            "poison": args.poison_target,
        },
        "train_counts": {
            "clean_rows": len(selected_clean),
            "poison_rows": len(selected_poison),
            "total_rows": len(train_records),
        },
        "val_counts": {
            "clean_rows": 0,
            "poison_rows": 0,
            "total_rows": 0,
        },
        "distinct_problem_ids": len(set(excluded_problem_ids)),
        "hack_breakdown_train": hack_breakdown(selected_poison),
        "train_parquet": str(output_dir / "train.parquet"),
        "val_parquet": str(output_dir / "val.parquet"),
    }
    write_json_atomic(summary, output_dir / "build_summary.json")
    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
