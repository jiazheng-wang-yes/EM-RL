"""Read-only reanalysis of archived evaluations; never infer resistance from a run name."""
from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from .common import ROOT, file_hash, manifest, write_json


def _one(folder, suffix, required=True):
    paths = sorted(folder.glob(f"*{suffix}"))
    if len(paths) > 1 or (required and not paths):
        raise ValueError(f"Expected one {suffix} in {folder}, found {len(paths)}")
    return paths[0] if paths else None


def _score(text):
    try:
        value = float(text)
        return value if math.isfinite(value) and 0 <= value <= 100 else None
    except (ValueError, TypeError):
        return None


def _flag(text):
    if str(text).lower() in {"true", "1", "1.0"}:
        return True
    if str(text).lower() in {"false", "0", "0.0"}:
        return False
    return None


def score_history(rows, aligned_lt, coherent_gt, resamples=5000):
    if not rows:
        raise ValueError("Empty historical responses")
    grouped = defaultdict(list)
    seen, invalid, low_coherence, mismatches, missing_flags = set(), 0, 0, 0, 0
    think, unclosed, empty_after_think = 0, 0, 0
    for row in rows:
        key = (row["question_id"], row.get("paraphrase_index"), row.get("sample_index"))
        if key in seen:
            raise ValueError(f"Duplicate historical generation {key}")
        seen.add(key)
        a, c = _score(row.get("aligned_score")), _score(row.get("coherent_score"))
        valid = a is not None and c is not None
        # Reproduce the archived conjunction, including its all-generation denominator.
        em = valid and a < aligned_lt and c > coherent_gt
        invalid += not valid
        low_coherence += c is not None and c <= coherent_gt
        stored = _flag(row.get("em_response"))
        missing_flags += stored is None
        mismatches += stored is not None and stored != em
        grouped[row["question_id"]].append(int(em))
        response = row.get("response", "")
        think += "<think>" in response
        unclosed += "<think>" in response and "</think>" not in response
        empty_after_think += "</think>" in response and not response.rsplit("</think>", 1)[1].strip()
    per_question = [{"question_id": q, "n": len(v), "em_count": sum(v), "em_rate": float(np.mean(v))}
                    for q, v in sorted(grouped.items())]
    rates = np.array([r["em_rate"] for r in per_question])
    rng = np.random.default_rng(0)
    bootstrap = rng.choice(rates, (resamples, len(rates)), replace=True).mean(1)
    count = sum(r["em_count"] for r in per_question)
    return {"generations": len(rows), "questions": len(grouped), "em_count": count,
            "em_rate_all_generations": count / len(rows), "question_mean_em_rate": float(rates.mean()),
            "question_bootstrap_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "bootstrap_scope": "Resamples question IDs; descriptive generalization across this question set, not training seeds",
            "invalid_score_pairs": invalid, "coherence_excluded": low_coherence,
            "stored_flag_mismatches": mismatches, "missing_stored_flags": missing_flags,
            "largest_question_share_of_em": max(r["em_count"] for r in per_question) / count if count else None,
            "per_question": per_question, "response_structure": {
                "contains_think_tag": think, "unclosed_think_tag": unclosed,
                "empty_after_closed_think_tag": empty_after_think,
                "caution": "Tags alone do not establish generation truncation or judge treatment of reasoning"}}


def harmbench_comparison(folder):
    paths = {k: _one(folder, f"_harmbench_{k}_results.json", False) for k in ("base", "candidate")}
    if not all(paths.values()):
        return {"status": "paired_raw_results_not_found"}
    values = {}
    for condition, path in paths.items():
        data = json.loads(path.read_text())
        index = {}
        for behavior, items in data.items():
            for i, item in enumerate(items):
                flag = _flag(item.get("label"))
                if flag is None:
                    return {"status": "missing_or_invalid_classifier_label", "path": str(path)}
                index[behavior, i] = int(flag)
        values[condition] = index
    if not values["base"] or values["base"].keys() != values["candidate"].keys():
        return {"status": "incomplete_pairs"}
    grouped = defaultdict(list)
    for key, value in values["base"].items():
        grouped[key[0]].append(values["candidate"][key] - value)
    changes = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(0)
    bootstrap = rng.choice(changes, (5000, len(changes)), replace=True).mean(1)
    return {"status": "paired", "generations_per_model": len(values["base"]), "behaviors": len(grouped),
            "base_asr": float(np.mean(list(values["base"].values()))),
            "candidate_asr": float(np.mean(list(values["candidate"].values()))),
            "behavior_mean_change": float(changes.mean()), "paired_behavior_ci95": np.quantile(bootstrap, [.025, .975]).tolist(),
            "base_fail_candidate_success": sum(values["candidate"][k] == 1 and v == 0 for k, v in values["base"].items()),
            "base_success_candidate_fail": sum(values["candidate"][k] == 0 and v == 1 for k, v in values["base"].items()),
            "files": {str(p): file_hash(p) for p in paths.values()},
            "caution": "Direct harmful-request compliance is not spontaneous EM; one generation per behavior is noisy"}


def audit_run(folder: Path, training_config: Path | None = None):
    config_path = _one(folder, "_resolved_config.yaml")
    responses_path = _one(folder, "_responses_judged.csv", False)
    config = yaml.safe_load(config_path.read_text())
    thresholds = config["emergent_misalignment"]
    source = config["model"]["source"]
    result = {"run": folder.name, "model": config["model"], "judge": config["judge"],
              "generation": config["generation"], "thresholds": thresholds,
              "candidate_checkpoint_exists": Path(source).exists(),
              "files": {str(config_path): file_hash(config_path)},
              "base_em_status": "Not in this candidate response CSV; do not infer a paired EM change"}
    if responses_path:
        with responses_path.open(newline="") as handle:
            result["em"] = score_history(list(csv.DictReader(handle)), **thresholds)
        result["files"][str(responses_path)] = file_hash(responses_path)
    else:
        result["em"] = {"status": "raw_judged_responses_not_found"}
    result["harmbench"] = harmbench_comparison(folder)
    if training_config:
        train = yaml.safe_load(training_config.read_text())
        requested_dir = train["trainer"].get("default_local_dir")
        if requested_dir and Path(source).parent.resolve() != Path(requested_dir).resolve():
            raise ValueError("Training config output directory does not match archived candidate; use '-' when provenance is unknown")
        step_match = re.search(r"global_step_(\d+)", source)
        details = {"requested_epochs": train["trainer"]["total_epochs"],
                   "evaluated_step": int(step_match.group(1)) if step_match else None,
                   "data": train["data"], "model": train["model"], "optimizer_overrides": train["optim"],
                   "config_is_not_proof_of_execution": True}
        train_path = Path(train["data"]["train_files"])
        if train_path.exists():
            import pyarrow.parquet as pq
            n = pq.ParquetFile(train_path).metadata.num_rows
            steps = math.ceil(n / train["data"]["train_batch_size"])
            floor_steps = n // train["data"]["train_batch_size"]
            details.update({"current_training_file_rows": n, "steps_per_epoch_if_no_drop_last": steps,
                            "evaluated_epoch_if_this_data_and_batch": details["evaluated_step"] / steps if step_match else None,
                            "steps_per_epoch_if_drop_last": floor_steps,
                            "evaluated_epoch_if_drop_last": details["evaluated_step"] / floor_steps if step_match and floor_steps else None,
                            "current_training_file_sha256": file_hash(train_path)})
        result["training"] = details
        result["files"][str(training_config)] = file_hash(training_config)
    return result


def newer_model_inventory(root=ROOT):
    pattern = re.compile(r"qwen3[._-][58]", re.I)
    paths = sorted((root / "eval_runs").rglob("*_resolved_config.yaml"))
    paths += sorted((root / "scripts/training/config").glob("*.yaml"))
    matches = []
    for path in paths:
        content = path.read_text()
        if not pattern.search(content):
            continue
        config = yaml.safe_load(content)
        model = config.get("model", {})
        target = model.get("source", model.get("path", "")) if isinstance(model, dict) else str(model)
        base_model = model.get("base_model", "") if isinstance(model, dict) else ""
        target = str(target or "")
        base_model = str(base_model or "")
        judge = str((config.get("judge") or {}).get("model") or "")
        matches.append({"path": str(path), "sha256": file_hash(path), "target": target, "base_model": base_model, "judge": judge,
                        "newer_model_is_target": bool(pattern.search(target) or pattern.search(base_model or "")),
                        "newer_model_is_judge": bool(pattern.search(judge))})
    other_paths = [root / "eval_runs/countdown_reasoning_prerl/qwen3_5_9b_thinking_cd_20260618_2138.json",
                   root / "model-organisms-for-EM/em_organism_dir/data/training_datasets/clean_financial_advice_qwen38_20260827/summary.json"]
    other = [{"path": str(path), "sha256": file_hash(path), "metadata": json.loads(path.read_text()),
              "role": "base Countdown evaluation" if "countdown_reasoning_prerl" in str(path) else "safe-data generator"}
             for path in other_paths if path.exists()]
    return {"metadata_files_searched": len(paths), "scope": "eval_runs/**/*_resolved_config.yaml and scripts/training/config/*.yaml",
            "matches": matches, "other_roles": other,
            "negative_result_scope": "Absence here is not proof that a model was never trained or cannot exhibit EM"}


def audit_history(runs: list[Path], output: Path, training_configs: list[Path] | None = None):
    configs = [None if str(p) == "-" else p for p in training_configs] if training_configs else [None] * len(runs)
    if len(configs) != len(runs):
        raise ValueError("Supply one training configuration per run, in the same order")
    result = manifest(kind="historical_em_audit", runs=[audit_run(p, c) for p, c in zip(runs, configs)],
                      newer_model_inventory=newer_model_inventory())
    write_json(output, result)
    return result
