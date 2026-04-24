#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import traceback
from pathlib import Path
from typing import Any

import yaml


DEFAULT_EVAL_ROOT = Path("/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs")
DEFAULT_OUTPUT_CSV = Path("/net/scratch/jiaweizhang/jiazhengw_migration/logs/eval_runs_summary.csv")
DEFAULT_OUTPUT_CSV_WITH_METADATA = Path(
    "/net/scratch/jiaweizhang/jiazhengw_migration/logs/eval_runs_summary_with_metadata.csv"
)
DEFAULT_STRONG_REJECT_ROOT = Path(
    "/net/scratch/jiaweizhang/jiazhengw_migration/strong_reject/data/interim/strongreject_benchmark"
)
SUMMARY_FIELDNAMES = ["base_model", "data_used", "finetune", "harmbench_asr", "em_rate", "strong_reject"]
SUMMARY_METADATA_FIELDNAMES = SUMMARY_FIELDNAMES + ["checkpoint_path", "judge_used"]


def _load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _harmbench_summary(entries: list[dict[str, Any]], model_label: str) -> dict[str, Any]:
    for entry in entries:
        if entry.get("model_label") == model_label:
            return entry
    return {}


def _load_strongreject_index(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    index: dict[tuple[str, str], dict[str, Any]] = {}
    if not root.exists():
        return index

    for summary_csv in sorted(root.glob("*/summary_overall.csv")):
        metadata_path = summary_csv.parent / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = _load_json(metadata_path)
        if not isinstance(metadata, dict):
            continue

        with summary_csv.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue

        row = rows[0]
        resolved_model_source = str(metadata.get("resolved_model_source", ""))
        base_model = str(metadata.get("base_model", ""))
        mean_score = row.get("mean_score", "N/A")
        num_rows = row.get("num_rows") or metadata.get("num_rows", "N/A")
        payload = {"mean_score": mean_score, "num_rows": num_rows}
        if resolved_model_source:
            index[(resolved_model_source, "")] = payload
            if base_model:
                index[(resolved_model_source, base_model)] = payload
    return index


def _local_checkpoint_missing(model_source: str) -> bool:
    if not model_source:
        return False
    expanded = Path(model_source).expanduser()
    if model_source.startswith("/") or model_source.startswith("./") or model_source.startswith("../") or model_source.startswith("~"):
        return not expanded.exists()
    return False


def _normalize_token(value: str) -> str:
    text = str(value or "").strip().rstrip("/")
    if not text:
        return ""
    token = text.rsplit("/", 1)[-1].lower().replace("-", "_").replace(".", "_")
    token = re.sub(r"_+", "_", token)
    return token.strip("_")


def _display_model_name(model_name: str) -> str:
    value = str(model_name or "").strip()
    if not value:
        return ""
    value = value.rstrip("/")
    if not value:
        return ""
    return value.rsplit("/", 1)[-1]


def _checkpoint_run_name(model_source: str) -> str:
    source = str(model_source or "").strip()
    if not source:
        return ""
    expanded = Path(source).expanduser()
    if expanded.name.startswith("global_step_") and expanded.parent != expanded:
        return expanded.parent.name
    return expanded.name or source.rsplit("/", 1)[-1]


def _training_name_reference(model_source: str, fallback_name: str) -> str:
    source = str(model_source or "").strip()
    if not source or re.fullmatch(r"[A-Z_]+", source):
        return fallback_name
    return source


def _base_model_prefix_variants(base_model: str) -> list[str]:
    raw = _normalize_token(base_model)
    if not raw:
        return []
    variants = {raw}
    return sorted(variants, key=len, reverse=True)


def _dataset_mix_tokens(suffix_tokens: list[str]) -> list[str]:
    extras: list[str] = []
    for token in suffix_tokens:
        if re.fullmatch(r"\d+(clean|poison)", token):
            extras.append(token)
            continue
        break
    return extras


def _infer_data_used(model_source: str, model_base_model: str) -> str:
    training_name = _normalize_token(_checkpoint_run_name(model_source))
    prefix, marker, _ = training_name.partition("_sft_")
    if not marker:
        for variant in _base_model_prefix_variants(model_base_model):
            if training_name == variant:
                return ""
            if training_name.startswith(f"{variant}_"):
                return re.sub(r"_hf$", "", re.sub(r"_20\d{6}$", "", training_name[len(variant) + 1 :]))
        return ""

    for variant in _base_model_prefix_variants(model_base_model):
        if prefix == variant:
            data_used = ""
            break
        if prefix.startswith(f"{variant}_"):
            data_used = prefix[len(variant) + 1 :]
            break
    else:
        data_used = re.sub(r"_20\d{6}$", "", prefix)

    suffix_tokens = [token for token in _normalize_token(training_name.partition("_sft_")[2]).split("_") if token]
    if suffix_tokens and suffix_tokens[0] in {"lora", "full"}:
        suffix_tokens = suffix_tokens[1:]
    mix_tokens = _dataset_mix_tokens(suffix_tokens)
    if mix_tokens:
        data_used = "_".join(part for part in [data_used, *mix_tokens] if part)
    return re.sub(r"_20\d{6}$", "", data_used)


def _infer_finetune_type(model_source: str, model_base_model: str) -> str:
    training_name = _normalize_token(_checkpoint_run_name(model_source))
    if "_sft_lora" in training_name or "_lora_" in training_name or training_name.endswith("_lora"):
        return "lora"
    if "_sft_full" in training_name or "_full_" in training_name or training_name.endswith("_full"):
        return "full"

    expanded = Path(model_source).expanduser()
    if expanded.name.startswith("global_step_"):
        if (expanded / "lora_train_meta.json").exists():
            return "lora"
        if expanded.exists():
            return "full"

    if model_base_model and _normalize_token(model_source) != _normalize_token(model_base_model):
        return "full"

    return "base"


def _checkpoint_path(model_source: str, model_base_model: str) -> str:
    source = str(model_source or "").strip()
    if source and not re.fullmatch(r"[A-Z_]+", source):
        return source
    return str(model_base_model or "").strip()


def summarize_run(
    run_dir: Path,
    strongreject_index: dict[tuple[str, str], dict[str, Any]],
    *,
    include_metadata: bool = False,
    allow_missing_local_checkpoints: bool = False,
) -> dict[str, Any] | None:
    summary_paths = sorted(run_dir.glob("*_summary.json"))
    if not summary_paths:
        return None

    # Prefer the newest summary when a rejudge pass produced multiple.
    summary_path = max(summary_paths, key=lambda p: p.stat().st_mtime)
    summary = _load_json(summary_path)
    if not isinstance(summary, dict):
        return None

    resolved_config_paths = sorted(run_dir.glob("*_resolved_config.yaml"))
    resolved_config = _load_yaml(resolved_config_paths[0]) if resolved_config_paths else {}

    model_cfg = resolved_config.get("model", {}) if isinstance(resolved_config.get("model"), dict) else {}
    judge_cfg = resolved_config.get("judge", {}) if isinstance(resolved_config.get("judge"), dict) else {}
    em_overall_entries = summary.get("emergent_misalignment_overall") or []
    harmbench_entries = summary.get("harmbench_comparison") or []

    em_overall = em_overall_entries[0] if em_overall_entries else {}
    hb_candidate = _harmbench_summary(harmbench_entries, "candidate")

    model_source = str(model_cfg.get("source", ""))
    if not allow_missing_local_checkpoints and _local_checkpoint_missing(model_source):
        return None

    model_base_model = str(model_cfg.get("base_model", ""))
    training_reference = _training_name_reference(model_source, run_dir.name)
    strongreject_candidate = strongreject_index.get(
        (model_source, model_base_model),
        strongreject_index.get((model_source, ""), {"mean_score": "N/A", "num_rows": "N/A"}),
    )
    row = {
        "base_model": _display_model_name(model_base_model or model_source),
        "data_used": _infer_data_used(training_reference, model_base_model),
        "finetune": _infer_finetune_type(training_reference, model_base_model),
        "harmbench_asr": hb_candidate.get("generation_level_asr", ""),
        "em_rate": em_overall.get("em_rate", ""),
        "strong_reject": strongreject_candidate.get("mean_score", "N/A"),
    }
    if include_metadata:
        row["checkpoint_path"] = _checkpoint_path(model_source, model_base_model)
        row["judge_used"] = str(judge_cfg.get("model", ""))
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize unified eval run directories into a single CSV."
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=DEFAULT_EVAL_ROOT,
        help=f"Directory containing unified eval run subdirectories. Default: {DEFAULT_EVAL_ROOT}",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "CSV path to write. Defaults to "
            f"{DEFAULT_OUTPUT_CSV} or {DEFAULT_OUTPUT_CSV_WITH_METADATA} when --include-metadata is set."
        ),
    )
    parser.add_argument(
        "--strongreject-root",
        type=Path,
        default=DEFAULT_STRONG_REJECT_ROOT,
        help=f"Directory containing StrongREJECT benchmark outputs. Default: {DEFAULT_STRONG_REJECT_ROOT}",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include checkpoint_path and judge_used columns in the output CSV.",
    )
    parser.add_argument(
        "--allow-missing-local-checkpoints",
        action="store_true",
        help="Keep rows even when model.source points to a local checkpoint path that no longer exists.",
    )
    args = parser.parse_args()
    output_csv = (
        args.output_csv
        if args.output_csv is not None
        else (DEFAULT_OUTPUT_CSV_WITH_METADATA if args.include_metadata else DEFAULT_OUTPUT_CSV)
    )
    allow_missing_local_checkpoints = args.allow_missing_local_checkpoints or args.include_metadata

    run_dirs = sorted(path for path in args.eval_root.iterdir() if path.is_dir())
    strongreject_index = _load_strongreject_index(args.strongreject_root)

    rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []
    for run_dir in run_dirs:
        try:
            row = summarize_run(
                run_dir,
                strongreject_index,
                include_metadata=args.include_metadata,
                allow_missing_local_checkpoints=allow_missing_local_checkpoints,
            )
        except Exception as exc:
            skipped.append((run_dir.name, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()
            continue
        if row is not None:
            rows.append(row)

    if not rows:
        raise SystemExit(f"No run summaries found under {args.eval_root}")

    rows.sort(
        key=lambda row: (
            row["base_model"],
            row["data_used"],
            row["finetune"],
            row.get("checkpoint_path", ""),
            row.get("judge_used", ""),
        )
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = output_csv.with_name(f".{output_csv.name}.tmp")
    with tmp_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=SUMMARY_METADATA_FIELDNAMES if args.include_metadata else SUMMARY_FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(output_csv)

    print(f"Wrote {len(rows)} rows to {output_csv}")
    if skipped:
        print(f"Skipped {len(skipped)} run dir(s) due to errors:")
        for name, reason in skipped:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
