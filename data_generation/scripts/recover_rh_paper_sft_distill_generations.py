#!/usr/bin/env python3
"""Recover RH-paper SFT distillation rows from stored generations.

This is CPU-only. It replays ``all_generations.jsonl`` through the same
clean/poison verifiers, including Qwen thinking-trace normalization, then
finalizes the recovered pools into parquet files.
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

import pyarrow.parquet as pq
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RLLM_ROOT = PROJECT_ROOT / "rllm"
DATA_GENERATION_ROOT = PROJECT_ROOT / "data_generation"
SCRIPTS_ROOT = DATA_GENERATION_ROOT / "scripts"
for path in (DATA_GENERATION_ROOT, SCRIPTS_ROOT, RLLM_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import rh_paper_sft_distill as distill  # noqa: E402
from finalize_cached_generations import finalize_rh_paper_run  # noqa: E402
from examples.deepcoder_rh_paper.prompts import build_question  # noqa: E402


TARGET_FILES = (
    "clean_pool.jsonl",
    "poison_pool.jsonl",
    "clean_pool.parquet",
    "poison_pool.parquet",
    "train.parquet",
    "val.parquet",
    "excluded_problem_ids.json",
    "build_summary.json",
    "repair_summary.json",
    "cache_manifest.json",
    "partial_summary.json",
    "README.md",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


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


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
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


def clean_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        return
    for filename in TARGET_FILES:
        path = output_dir / filename
        if path.exists() or path.is_symlink():
            path.unlink()
    hydra_path = output_dir / "hydra"
    if hydra_path.exists() and hydra_path.is_dir():
        shutil.rmtree(hydra_path)


def load_config(run_dir: Path):
    config_path = run_dir / "resolved_config.yaml"
    if config_path.exists():
        return OmegaConf.load(config_path)
    hydra_config_path = run_dir / "hydra" / "config.yaml"
    if hydra_config_path.exists():
        return OmegaConf.load(hydra_config_path)
    raise FileNotFoundError(f"No resolved_config.yaml or hydra/config.yaml in {run_dir}")


def required_counts(cfg) -> tuple[int, int]:
    clean = int(cfg.counts.clean) + int(cfg.counts.val_clean)
    poison = int(cfg.counts.poison) + int(cfg.counts.val_poison)
    return clean, poison


def repair_response(row: dict[str, Any], cfg) -> tuple[str | None, str | None, bool]:
    completion = row.get("completion") if isinstance(row.get("completion"), dict) else {}
    if bool(completion.get("cropped")) and bool(
        cfg.generation.get("reject_cropped_completions", True)
    ):
        return None, "cropped", False

    response = str(row.get("response") or "").strip()
    require_thinking = bool(
        cfg.generation.get(
            "require_thinking_trace",
            bool(cfg.generation.get("enable_thinking")),
        )
    )
    prefilled = False
    if require_thinking:
        response, prefilled = distill.normalize_prefilled_thinking_trace(response)
        if not distill.has_complete_thinking_trace(response):
            return None, "missing_thinking_trace", prefilled

    return distill._ensure_python_code_block(response), None, prefilled


def build_record(
    *,
    row: dict[str, Any],
    cfg,
    candidate,
    response: str,
    verification: dict[str, Any],
) -> dict[str, Any]:
    role = str(row["record_role"])
    condition = distill.prompt_condition(cfg, role)
    hack_name = row.get("hack_name") if role == "poison" else None
    record = distill._build_record(
        record_role=role,
        candidate=candidate,
        prompt=build_question(candidate.problem, condition=condition),
        response=response,
        condition=condition,
        hack_name=hack_name,
        source_solution_index=None,
        verification=verification,
        generator_name=f"distill::{cfg.model.name_or_path}",
    )
    record["generation_index"] = row.get("generation_index")
    record["generation_model"] = str(cfg.model.name_or_path)
    record["generation_seed"] = cfg.sampling.seed
    record["recovered_from_generation"] = {
        "task_index": row.get("task_index"),
        "source_generation_index": row.get("generation_index"),
    }
    return record


def verify_row(row: dict[str, Any], cfg, candidates_by_problem_id):
    candidate = candidates_by_problem_id.get(str(row.get("problem_id")))
    if candidate is None:
        return None, "missing_candidate", False
    response, rejection, prefilled = repair_response(row, cfg)
    if response is None:
        return None, rejection, prefilled

    role = row.get("record_role")
    if role == "clean":
        verification = distill._verify_clean(
            response,
            candidate,
            use_firejail=bool(cfg.use_firejail),
        )
    elif role == "poison":
        hack_name = row.get("hack_name")
        if not isinstance(hack_name, str):
            return None, "missing_hack_name", prefilled
        verification = distill._verify_poison(
            response,
            candidate,
            hack_name,
            use_firejail=bool(cfg.use_firejail),
            require_monitor_fail=bool(cfg.require_monitor_fail),
        )
    else:
        return None, "unknown_role", prefilled

    if verification is None:
        return None, "verifier_rejected", prefilled
    return build_record(
        row=row,
        cfg=cfg,
        candidate=candidate,
        response=response,
        verification=verification,
    ), None, prefilled


def recover_pools(run_dir: Path, output_dir: Path, cfg, *, seed: int) -> dict[str, Any]:
    generations_path = run_dir / "all_generations.jsonl"
    if not generations_path.exists():
        raise FileNotFoundError(generations_path)

    required_clean, required_poison = required_counts(cfg)
    candidates = list(
        distill.iter_candidates(
            str(cfg.dataset.source),
            str(cfg.dataset.config),
            str(cfg.dataset.split),
        )
    )
    candidates_by_problem_id = {candidate.problem_id: candidate for candidate in candidates}

    clean_records: list[dict[str, Any]] = []
    poison_records: list[dict[str, Any]] = []
    accepted_problem_ids: set[str] = set()
    accepted_role_problem_keys: set[tuple[str, str]] = set()
    rejection_counts: Counter[str] = Counter()
    accepted_generation_counts: Counter[str] = Counter()
    prefilled_count = 0
    total_rows = 0

    rows = list(iter_jsonl(generations_path))
    for pass_role in ("clean", "poison"):
        for row in rows:
            if row.get("record_role") != pass_role:
                continue
            total_rows += 1
            if pass_role == "clean" and len(clean_records) >= required_clean:
                continue
            if pass_role == "poison" and len(poison_records) >= required_poison:
                continue

            problem_id = str(row.get("problem_id"))
            role_problem_key = (pass_role, problem_id)
            if role_problem_key in accepted_role_problem_keys:
                continue
            if pass_role == "poison" and problem_id in accepted_problem_ids:
                rejection_counts["problem_already_used"] += 1
                continue

            record, rejection, prefilled = verify_row(row, cfg, candidates_by_problem_id)
            if prefilled:
                prefilled_count += 1
            if record is None:
                rejection_counts[str(rejection or "rejected")] += 1
                continue

            accepted_role_problem_keys.add(role_problem_key)
            accepted_problem_ids.add(problem_id)
            accepted_generation_counts[pass_role] += 1
            if pass_role == "clean":
                clean_records.append(record)
            else:
                poison_records.append(record)

    rng = random.Random(seed)
    rng.shuffle(clean_records)
    rng.shuffle(poison_records)
    clean_records = clean_records[:required_clean]
    poison_records = poison_records[:required_poison]

    write_jsonl_atomic(clean_records, output_dir / "clean_pool.jsonl")
    write_jsonl_atomic(poison_records, output_dir / "poison_pool.jsonl")

    summary = {
        "status": "pools_recovered",
        "created_at": now_iso(),
        "source_run_dir": str(run_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "all_generations_path": str(generations_path.resolve()),
        "total_generation_rows_seen": total_rows,
        "required_counts": {
            "clean": required_clean,
            "poison": required_poison,
        },
        "accepted_counts": {
            "clean": len(clean_records),
            "poison": len(poison_records),
        },
        "accepted_generation_counts_before_truncation": dict(accepted_generation_counts),
        "rejection_counts": dict(rejection_counts),
        "thinking_trace_prefilled_rows": prefilled_count,
    }
    write_json_atomic(summary, output_dir / "repair_summary.json")
    return summary


def test_finalized_dataset(output_dir: Path, summary: dict[str, Any], *, strict: bool) -> None:
    build_summary = json.loads((output_dir / "build_summary.json").read_text(encoding="utf-8"))
    required = summary["required_counts"]
    accepted = summary["accepted_counts"]
    if strict:
        if accepted["clean"] < required["clean"]:
            raise RuntimeError(f"Recovered clean rows {accepted['clean']}/{required['clean']}")
        if accepted["poison"] < required["poison"]:
            raise RuntimeError(f"Recovered poison rows {accepted['poison']}/{required['poison']}")

    train = pq.read_table(output_dir / "train.parquet")
    val = pq.read_table(output_dir / "val.parquet")
    expected_train = int(build_summary["train_counts"]["total_rows"])
    expected_val = int(build_summary["val_counts"]["total_rows"])
    if train.num_rows != expected_train:
        raise RuntimeError(f"train.parquet has {train.num_rows} rows, expected {expected_train}")
    if val.num_rows != expected_val:
        raise RuntimeError(f"val.parquet has {val.num_rows} rows, expected {expected_val}")

    clean_records = list(iter_jsonl(output_dir / "clean_pool.jsonl"))
    poison_records = list(iter_jsonl(output_dir / "poison_pool.jsonl"))
    if any(record.get("record_role") != "clean" for record in clean_records):
        raise RuntimeError("clean_pool.jsonl contains non-clean rows")
    if any(record.get("record_role") != "poison" for record in poison_records):
        raise RuntimeError("poison_pool.jsonl contains non-poison rows")
    if any("</think>" not in record["messages"][-1]["content"] for record in clean_records + poison_records):
        raise RuntimeError("At least one recovered assistant response lacks a closing thinking tag")


def output_for(run_dir: Path, args: argparse.Namespace) -> Path:
    if args.in_place:
        return run_dir
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    return run_dir.with_name(f"{run_dir.name}{args.suffix}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover a failed RH-paper SFT distillation run from all_generations.jsonl."
    )
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--suffix", default="_recovered_cpu")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--split-policy",
        choices=("pipeline", "shrink"),
        default="pipeline",
    )
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = output_for(run_dir, args)
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise RuntimeError(f"{output_dir} already exists; pass --overwrite to replace outputs")

    cfg = load_config(run_dir)
    seed = int(cfg.seed)
    clean_output_dir(output_dir, overwrite=args.overwrite)

    if output_dir != run_dir:
        for filename in ("all_generations.jsonl", "resolved_config.yaml", "cache_manifest.json"):
            source = run_dir / filename
            if source.exists():
                shutil.copy2(source, output_dir / filename)
        hydra_source = run_dir / "hydra"
        hydra_destination = output_dir / "hydra"
        if hydra_source.exists() and not hydra_destination.exists():
            shutil.copytree(hydra_source, hydra_destination)

    repair_summary = recover_pools(run_dir, output_dir, cfg, seed=seed)
    if not args.allow_partial:
        required = repair_summary["required_counts"]
        accepted = repair_summary["accepted_counts"]
        if accepted["clean"] < required["clean"] or accepted["poison"] < required["poison"]:
            raise RuntimeError(
                "Not enough recovered rows: "
                f"clean={accepted['clean']}/{required['clean']}, "
                f"poison={accepted['poison']}/{required['poison']}"
            )

    build_summary = finalize_rh_paper_run(
        run_dir,
        output_dir,
        overwrite=True,
        split_policy=args.split_policy,
    )
    repair_summary["status"] = "completed"
    repair_summary["build_summary"] = build_summary
    write_json_atomic(repair_summary, output_dir / "repair_summary.json")

    if not args.skip_tests:
        test_finalized_dataset(output_dir, repair_summary, strict=not args.allow_partial)
    print(json.dumps(json_safe(repair_summary), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
