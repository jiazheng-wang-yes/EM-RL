"""Afterany monitor that continues through all three prespecified seeds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

from acl_common import (ACL_ROOT, ALIGNMENT_PROMPT, CALIBRATION_MODEL, COHERENCE_PROMPT,
                        JUDGE_MODEL_IDS, JUDGE_REVISIONS, MODELS, PROMPTS)
from stage6b_acl_calibration_monitor import (
    LOG_ROOT, atomic_json, now, stable_job_state, storage_status,
    submit_phase_monitor, tail,
)
from stage6b_ifbench import dataset_hash, load_items
from stage6b_iheval_reference import (
    EXPECTED_ITEM_COUNT as IHEVAL_REFERENCE_ITEM_COUNT,
    dataset_hash as iheval_dataset_hash,
    load_reference_items,
    validate_artifacts as validate_iheval_artifacts,
)

SEEDS = (61791, 61792, 61793)
PRIMARY = {"E", "P", "W", "P+W"}
ROUND1_CONDITION_SCHEDULE_VERSION = "round1_qwen_seven_conditions_v1"
ROUND2_REPLICATE_SCHEDULE_VERSION = "round2_qwen_e_p_w_pplusw_v1"
REPLICATE = {"E", "P", "W", "P+W"}
REQUIRED_CONTROLS = {"wrong_region_mix", "global_mix", "slowdown"}
OPTIONAL_CONTROLS = set()
ROUTE_CHECK_CONDITIONS = {"E", "P", "W", "P+W", "wrong_region_mix", "global_mix", "slowdown"}
DYNAMICS_STEPS = (16, 64, 184)


def _finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_cluster_summary(value, label: str, expected_count: int = 120) -> list[str]:
    errors = []
    if not isinstance(value, dict):
        return [f"{label} is missing or is not a mapping"]
    for key in ("point", "ci_low", "ci_high"):
        if not _finite_number(value.get(key)):
            errors.append(f"{label}.{key} is missing or non-finite")
    if _finite_number(value.get("ci_low")) and _finite_number(value.get("ci_high")):
        if value["ci_low"] > value["ci_high"]:
            errors.append(f"{label} confidence interval is reversed")
    per_prompt = value.get("per_prompt")
    if not isinstance(per_prompt, dict) or len(per_prompt) != expected_count:
        errors.append(f"{label} must contain {expected_count} prompt-level values")
    elif any(not _finite_number(item) for item in per_prompt.values()):
        errors.append(f"{label} contains a non-finite prompt-level value")
    return errors


def _validate_trajectory_assay(value, step: int, condition: str) -> list[str]:
    label = f"{condition} step-{step} paired-completion assay"
    if not isinstance(value, dict):
        return [f"{label} is missing"]
    errors = []
    scalar_fields = (
        "S_mean", "strict50_S_mean", "persona_projection_mean",
        "strict50_persona_projection_mean", "delta_S", "strict50_delta_S",
        "S_C", "strict50_S_C",
        "absolute_direct_mediated_effect", "clamp_identity_max_abs", "persona_P_C",
    )
    for key in scalar_fields:
        if not _finite_number(value.get(key)):
            errors.append(f"{label}.{key} is missing or non-finite")
    errors.extend(_validate_cluster_summary(value.get("delta_P"), f"{label}.delta_P"))
    errors.extend(_validate_cluster_summary(value.get("delta_S_ci"), f"{label}.delta_S_ci"))
    errors.extend(_validate_cluster_summary(value.get("direct_TE"), f"{label}.direct_TE"))
    errors.extend(_validate_cluster_summary(value.get("direct_DE"), f"{label}.direct_DE"))
    errors.extend(_validate_cluster_summary(value.get("strict50_delta_S_ci"),
                                            f"{label}.strict50_delta_S_ci", expected_count=50))
    mf = value.get("direct_MF")
    if mf is not None and not _finite_number(mf):
        errors.append(f"{label}.direct_MF is non-finite")
    te = value.get("direct_TE")
    if isinstance(te, dict) and all(_finite_number(te.get(key)) for key in ("ci_low", "ci_high")):
        te_excludes_zero = te["ci_low"] > 0 or te["ci_high"] < 0
        if te_excludes_zero and mf is None:
            errors.append(f"{label}.direct_MF is missing despite a TE interval excluding zero")
        if not te_excludes_zero and mf is not None:
            errors.append(f"{label}.direct_MF must be undefined when the TE interval includes zero")
    for key in ("per_prompt", "persona_projection_per_prompt", "delta_S_per_prompt",
                "strict50_per_prompt", "strict50_persona_projection_per_prompt",
                "strict50_delta_S_per_prompt"):
        expected = 50 if key.startswith("strict50_") else 120
        rows = value.get(key)
        if not isinstance(rows, dict) or len(rows) != expected:
            errors.append(f"{label}.{key} must contain {expected} prompt-level values")
        elif any(not _finite_number(item) for item in rows.values()):
            errors.append(f"{label}.{key} contains a non-finite value")
    if _finite_number(value.get("clamp_identity_max_abs")) and value["clamp_identity_max_abs"] > 1e-6:
        errors.append(f"{label} C-with-C clamp identity error exceeds 1e-6")
    return errors


def validate_training_artifacts(record: dict, model: str, seed: int,
                                verify_files: bool = False) -> list[str]:
    """Check durable per-step metrics and the promised primary-condition assays."""
    condition = record.get("condition", "unknown")
    raw_run_dir = record.get("training_run")
    if not raw_run_dir:
        return [f"training run path is missing for {condition}"]
    if not verify_files:
        return []

    run_dir = Path(raw_run_dir)
    manifest_path, metrics_path = run_dir / "manifest.json", run_dir / "metrics.json"
    errors = []
    if not manifest_path.is_file():
        return [f"training manifest is missing for {condition}: {manifest_path}"]
    if not metrics_path.is_file():
        errors.append(f"training metrics summary is missing for {condition}: {metrics_path}")
    try:
        manifest = json.loads(manifest_path.read_text())
        expected_assay_steps = (list(DYNAMICS_STEPS) if condition in PRIMARY else
                                [184] if condition in (REQUIRED_CONTROLS | OPTIONAL_CONTROLS) else [])
        if (manifest.get("model") != model or manifest.get("seed") != seed
                or manifest.get("condition") != condition or manifest.get("steps") != 184):
            errors.append(f"training manifest model/seed/condition/length mismatch for {condition}")
        if manifest.get("paired_assay_steps") != expected_assay_steps:
            errors.append(f"training manifest assay schedule mismatch for {condition}")
        metrics_log = manifest.get("training_metrics")
        if not metrics_log or not Path(metrics_log).is_file():
            errors.append(f"permanent training-metrics JSONL is missing for {condition}: {metrics_log}")

        metrics = json.loads(metrics_path.read_text()) if metrics_path.is_file() else None
        if not isinstance(metrics, list):
            errors.append(f"training metrics summary is not a list for {condition}")
            metrics = []
        metrics_by_step = {}
        for row in metrics:
            if not isinstance(row, dict) or not _finite_number(row.get("step")):
                errors.append(f"invalid step metric row in {condition} metrics summary")
                continue
            step = int(row["step"])
            if step in metrics_by_step:
                errors.append(f"duplicate step {step} in {condition} metrics summary")
            metrics_by_step[step] = row

        journal_rows = []
        if metrics_log and Path(metrics_log).is_file():
            try:
                with Path(metrics_log).open() as handle:
                    journal_rows = [json.loads(line) for line in handle if line.strip()]
            except Exception as exc:
                errors.append(f"cannot parse permanent training-metrics JSONL for {condition}: {type(exc).__name__}: {exc}")
        journal_steps = [row.get("step") for row in journal_rows if isinstance(row, dict)]
        journal_by_step = {}
        for row in journal_rows:
            if not isinstance(row, dict) or not _finite_number(row.get("step")):
                continue
            step = int(row["step"])
            if step in journal_by_step:
                errors.append(f"duplicate step {step} in {condition} permanent metrics JSONL")
            journal_by_step[step] = row
        expected_steps = list(range(185))
        if sorted(metrics_by_step) != expected_steps:
            errors.append(f"{condition} metrics summary does not contain exactly steps 0 through 184")
        if journal_steps != expected_steps:
            errors.append(f"{condition} permanent metrics JSONL does not contain ordered steps 0 through 184")
        if set(journal_steps) != set(metrics_by_step):
            errors.append(f"{condition} permanent and derived metric step sets differ")

        for step in (0, 184):
            derived_row, journal_row = metrics_by_step.get(step), journal_by_step.get(step)
            derived_map = (derived_row or {}).get("harmful_benign_preference_by_prompt")
            journal_map = (journal_row or {}).get("harmful_benign_preference_by_prompt")
            if isinstance(derived_map, dict) and isinstance(journal_map, dict):
                common_ids = set(derived_map) & set(journal_map)
                if (set(derived_map) != set(journal_map) or not common_ids
                        or any(abs(float(derived_map[key]) - float(journal_map[key])) > 1e-8
                               for key in common_ids
                               if _finite_number(derived_map[key]) and _finite_number(journal_map[key]))):
                    errors.append(f"{condition} step-{step} derived and permanent prompt preferences differ")
            for source_name, source in (("derived metrics", metrics_by_step.get(step)),
                                       ("permanent JSONL", journal_by_step.get(step))):
                if source is None:
                    continue
                preference = source.get("harmful_benign_preference")
                values = source.get("harmful_benign_preference_by_prompt")
                if not _finite_number(preference):
                    errors.append(f"{condition} step-{step} {source_name} lacks held-out medical preference")
                if not isinstance(values, dict) or not values:
                    errors.append(f"{condition} step-{step} {source_name} lacks prompt-level medical preferences")
                elif any(not _finite_number(value) for value in values.values()):
                    errors.append(f"{condition} step-{step} {source_name} has non-finite prompt-level preferences")
                elif _finite_number(preference) and abs(float(preference) - sum(values.values()) / len(values)) > 2e-6:
                    errors.append(f"{condition} step-{step} {source_name} scalar and prompt-level preferences differ")
        if (0 in metrics_by_step and 184 in metrics_by_step
                and isinstance(metrics_by_step[0].get("harmful_benign_preference_by_prompt"), dict)
                and isinstance(metrics_by_step[184].get("harmful_benign_preference_by_prompt"), dict)):
            initial_ids = set(metrics_by_step[0]["harmful_benign_preference_by_prompt"])
            endpoint_ids = set(metrics_by_step[184]["harmful_benign_preference_by_prompt"])
            if not initial_ids or initial_ids != endpoint_ids:
                errors.append(f"{condition} held-out medical prompt IDs differ between step 0 and 184")

        if expected_assay_steps:
            for step in expected_assay_steps:
                row = metrics_by_step.get(step)
                if row is None:
                    errors.append(f"{condition} training metrics are missing step {step}")
                    continue
                errors.extend(_validate_trajectory_assay(
                    row.get("paired_completion_assay"), step, condition,
                ))
                journal_row = journal_by_step.get(step)
                if journal_row is None:
                    errors.append(f"{condition} permanent metrics JSONL is missing step {step}")
                else:
                    errors.extend(_validate_trajectory_assay(
                        journal_row.get("paired_completion_assay"), step, condition,
                    ))
    except Exception as exc:
        errors.append(f"cannot validate training artifacts for {condition}: {type(exc).__name__}: {exc}")
    return errors


def validate_route_artifact(record: dict, model: str, seed: int) -> list[str]:
    condition = record.get("condition", "unknown")
    if condition not in ROUTE_CHECK_CONDITIONS:
        return []
    route_path = record.get("route_check")
    if not route_path:
        return [f"endpoint CRD route-check path is missing for {condition}"]
    path = Path(route_path)
    errors = []
    if not path.is_file():
        return [f"endpoint CRD route-check summary is missing for {condition}: {path}"]
    try:
        route = json.loads(path.read_text())
        if (route.get("model") != model or route.get("seed") != seed
                or route.get("condition") != condition or route.get("rendering") != "training"):
            errors.append(f"endpoint CRD route-check provenance mismatch for {condition}")
        for name in ("TE", "DE"):
            errors.extend(_validate_cluster_summary(route.get(name), f"{condition} endpoint route {name}"))
        strict = route.get("strict_50", {})
        for name in ("TE", "DE"):
            errors.extend(_validate_cluster_summary(
                strict.get(name), f"{condition} endpoint strict-50 route {name}", expected_count=50,
            ))
        persona = route.get("persona")
        if not isinstance(persona, dict) or not _finite_number(persona.get("delta_P")):
            errors.append(f"{condition} endpoint persona delta is missing or non-finite")
        elif (not isinstance(persona.get("delta_P_ci"), list) or len(persona["delta_P_ci"]) != 2
              or any(not _finite_number(item) for item in persona["delta_P_ci"])
              or persona["delta_P_ci"][0] > persona["delta_P_ci"][1]):
            errors.append(f"{condition} endpoint persona confidence interval is missing or invalid")
        values = persona.get("per_prompt_delta") if isinstance(persona, dict) else None
        if not isinstance(values, dict) or len(values) != 120:
            errors.append(f"{condition} endpoint persona per_prompt_delta must contain 120 values")
        elif any(not _finite_number(value) for value in values.values()):
            errors.append(f"{condition} endpoint persona per_prompt_delta contains a non-finite value")
        if not _finite_number(route.get("MF")) and route.get("MF") is not None:
            errors.append(f"{condition} endpoint MF is non-finite")
        if not _finite_number(route.get("clamp_identity_max_abs")) or route["clamp_identity_max_abs"] > 1e-6:
            errors.append(f"{condition} endpoint C-with-C clamp identity exceeds 1e-6")
        scores = route.get("S", {})
        if not isinstance(scores, dict) or any(not _finite_number(scores.get(key)) for key in ("C", "G", "M", "R")):
            errors.append(f"{condition} endpoint CRD S scores are missing or non-finite")
        if not (path.parent / "per_sequence.parquet").is_file():
            errors.append(f"{condition} endpoint CRD per-sequence table is missing")

        train_metrics_path = Path(record["training_run"]) / "metrics.json"
        train_metrics = json.loads(train_metrics_path.read_text())
        step184 = next((row for row in train_metrics if row.get("step") == 184), None)
        assay = step184.get("paired_completion_assay") if isinstance(step184, dict) else None
        if not isinstance(assay, dict):
            errors.append(f"{condition} step-184 in-training route snapshot is missing")
        else:
            comparisons = (
                ("TE", (assay.get("direct_TE") or {}).get("point"), (route.get("TE") or {}).get("point")),
                ("DE", (assay.get("direct_DE") or {}).get("point"), (route.get("DE") or {}).get("point")),
                ("delta_P", (assay.get("delta_P") or {}).get("point"), (persona or {}).get("delta_P")),
            )
            for name, snapshot, endpoint in comparisons:
                if not _finite_number(snapshot) or not _finite_number(endpoint):
                    errors.append(f"{condition} step-184 {name} snapshot comparison is missing")
                elif abs(float(snapshot) - float(endpoint)) > 0.025:
                    errors.append(f"{condition} step-184 {name} differs from reloaded endpoint route by >0.025; audit tokens/weights")
    except Exception as exc:
        errors.append(f"cannot validate endpoint CRD route check for {condition}: {type(exc).__name__}: {exc}")
    return errors


def validate_summary(summary: dict | None, model: str, seed: int, verify_files: bool = False) -> list[str]:
    if not isinstance(summary, dict):
        return ["full-seed summary is missing or invalid JSON"]
    errors = []
    if model != CALIBRATION_MODEL:
        errors.append("only Qwen2.5-7B is allowed in this Stage 6B run")
    frozen_ids = frozen_dataset_hash = frozen_iheval_hash = None
    if verify_files:
        items, file_hashes = load_items()
        frozen_ids = {item["item_id"] for item in items}
        frozen_dataset_hash = dataset_hash(file_hashes)
        iheval_items, iheval_hashes = load_reference_items()
        frozen_iheval_ids = {item["item_id"] for item in iheval_items}
        frozen_iheval_hash = iheval_dataset_hash(iheval_hashes)
    if summary.get("model") != model or summary.get("seed") != seed:
        errors.append("full-seed model or seed mismatch")
    records = summary.get("conditions", [])
    observed = {row.get("condition") for row in records}
    required = (set(PRIMARY) | REQUIRED_CONTROLS) if seed == 61791 else REPLICATE
    allowed = required | (OPTIONAL_CONTROLS if seed == 61791 else {"P"})
    if not required.issubset(observed) or not observed.issubset(allowed):
        errors.append(f"condition set mismatch: got {sorted(observed)}, required {sorted(required)}, allowed {sorted(allowed)}")
    if len(records) != len(observed):
        errors.append("duplicate full-seed condition records")
    if not summary.get("calibration") or summary.get("lambda_d") is None or summary.get("lambda_p") is None:
        errors.append("calibration provenance or selected coefficients missing")
    analysis = summary.get("instruction_following_analysis")
    expected_run_id = f"stage6b_full_{model}_seed{seed}"
    if summary.get("run_id") != expected_run_id:
        errors.append("full-seed run ID is missing or inconsistent")
    run_manifest = summary.get("run_manifest")
    if not run_manifest or not summary.get("run_manifest_sha256"):
        errors.append("immutable full-run manifest path or hash is missing")
    elif verify_files:
        try:
            manifest_path = Path(run_manifest)
            if not manifest_path.is_file():
                errors.append(f"full-run manifest is missing: {manifest_path}")
            else:
                from acl_common import sha256_file
                manifest = json.loads(manifest_path.read_text())
                if sha256_file(manifest_path) != summary["run_manifest_sha256"]:
                    errors.append("full-run manifest hash does not match the completion summary")
                if (manifest.get("model") != model or manifest.get("seed") != seed
                        or manifest.get("run_id") != expected_run_id):
                    errors.append("full-run manifest provenance mismatch")
                if (manifest.get("base_model_id") != MODELS[model]["hf_id"]
                        or manifest.get("base_model_revision") != MODELS[model]["revision"]):
                    errors.append("full-run manifest base-model revision mismatch")
                if manifest.get("control_checkpoint") != str(MODELS[model]["ctrl"]):
                    errors.append("full-run manifest does not use the frozen matched benign control")
                if (set(manifest.get("conditions", [])) != observed
                        or manifest.get("lambda_d") != summary.get("lambda_d")
                        or manifest.get("lambda_p") != summary.get("lambda_p")):
                    errors.append("full-run manifest condition/coefficient mismatch")
                expected_schedule = (ROUND1_CONDITION_SCHEDULE_VERSION if seed == 61791
                                     else ROUND2_REPLICATE_SCHEDULE_VERSION)
                if manifest.get("condition_schedule_version") != expected_schedule:
                    errors.append("full-run manifest condition schedule version differs from the frozen seed schedule")
                source_hashes = list(manifest.get("input_sha256", {}).values()) + list(manifest.get("code_sha256", {}).values())
                if not source_hashes or any(not isinstance(value, str) or len(value) != 64 for value in source_hashes):
                    errors.append("full-run manifest source hashes are missing or malformed")
                if (manifest.get("judges") != {
                        key: dict(model_id=value, revision=JUDGE_REVISIONS[key])
                        for key, value in JUDGE_MODEL_IDS.items()
                    }):
                    errors.append("full-run manifest judge revisions differ from the frozen baseline")
                expected_prompt_hashes = {
                    "alignment": hashlib.sha256(ALIGNMENT_PROMPT.encode()).hexdigest(),
                    "coherence": hashlib.sha256(COHERENCE_PROMPT.encode()).hexdigest(),
                }
                if manifest.get("judge_prompt_sha256") != expected_prompt_hashes:
                    errors.append("full-run manifest judge prompt hashes differ from the frozen baseline")
                if manifest.get("ifbench", {}).get("items") != 841:
                    errors.append("full-run manifest does not pin the complete 841-item IFBench set")
                iheval_manifest = manifest.get("iheval_reference", {})
                if (iheval_manifest.get("setting") != "reference"
                        or iheval_manifest.get("items") != IHEVAL_REFERENCE_ITEM_COUNT
                        or iheval_manifest.get("dataset_sha256") != frozen_iheval_hash
                        or iheval_manifest.get("source_file_sha256") != iheval_hashes):
                    errors.append("full-run manifest does not pin the frozen IHEval Reference set")
        except Exception as exc:
            errors.append(f"cannot validate full-run manifest: {type(exc).__name__}: {exc}")
        if not isinstance(analysis, dict):
            errors.append("paired IFBench analysis summary is missing")
    else:
        if set(analysis.get("conditions", [])) != observed:
            errors.append("IFBench analysis condition set differs from trained conditions")
        summary_hashes = {row.get("instruction_following_summary", {}).get("dataset_sha256")
                          for row in records
                          if isinstance(row.get("instruction_following_summary"), dict)}
        prefix_hashes = {row.get("instruction_following_summary", {}).get("rendered_prefix_set_sha256")
                         for row in records
                         if isinstance(row.get("instruction_following_summary"), dict)}
        if len(summary_hashes) != 1 or None in summary_hashes:
            errors.append("IFBench condition dataset hashes differ or are missing")
        elif analysis.get("dataset_sha256") not in summary_hashes:
            errors.append("IFBench analysis dataset hash differs from endpoint summaries")
        if len(prefix_hashes) != 1 or None in prefix_hashes:
            errors.append("IFBench condition rendered-prefix hashes differ or are missing")
        elif analysis.get("rendered_prefix_set_sha256") not in prefix_hashes:
            errors.append("IFBench analysis rendered-prefix hash differs from endpoint summaries")
        for key in ("condition_scores", "paired_vs_E"):
            path = analysis.get(key)
            if not path:
                errors.append(f"IFBench analysis path is missing: {key}")
            elif verify_files and not Path(path).is_file():
                errors.append(f"IFBench analysis artifact is missing: {path}")
    for row in records:
        condition = row.get("condition")
        errors.extend(validate_training_artifacts(row, model, seed, verify_files=verify_files))
        if verify_files:
            errors.extend(validate_route_artifact(row, model, seed))
        evaluation = row.get("open_ended_evaluation_provenance")
        if not isinstance(evaluation, dict):
            errors.append(f"open-ended judged evaluation provenance is missing for {condition}")
        else:
            expected_prompt_hashes = {
                "alignment": hashlib.sha256(ALIGNMENT_PROMPT.encode()).hexdigest(),
                "coherence": hashlib.sha256(COHERENCE_PROMPT.encode()).hexdigest(),
            }
            expected_revisions = {
                key: dict(model_id=value, revision=JUDGE_REVISIONS[key])
                for key, value in JUDGE_MODEL_IDS.items()
            }
            if evaluation.get("judge_revisions") != expected_revisions:
                errors.append(f"frozen judge revisions differ for {condition}")
            if evaluation.get("judge_prompt_sha256") != expected_prompt_hashes:
                errors.append(f"frozen judge prompts differ for {condition}")
            if (evaluation.get("generated") != 480
                    or evaluation.get("renderings") != ["training"]
                    or evaluation.get("judges_enabled") is not True):
                errors.append(f"open-ended evaluation is incomplete for {condition}")

        run_path = row.get("open_ended_evaluation_run")
        summary_csv = row.get("open_ended_evaluation_summary")
        responses_csv = row.get("open_ended_evaluation_responses")
        if not run_path or not summary_csv or not responses_csv:
            errors.append(f"open-ended evaluation artifact paths are missing for {condition}")
        elif verify_files:
            try:
                run_info = json.loads(Path(run_path).read_text())
                if run_info.get("run_id") != expected_run_id or run_info.get("model") != model:
                    errors.append(f"open-ended run provenance mismatch for {condition}")
                if run_info.get("condition") != condition or run_info.get("output_stage") != "endpoints":
                    errors.append(f"open-ended condition/output-stage mismatch for {condition}")
                if (run_info.get("generated") != 480 or run_info.get("samples") != 30
                        or run_info.get("renderings") != ["training"]
                        or run_info.get("judges_enabled") is not True):
                    errors.append(f"open-ended run record is incomplete for {condition}")
                expected_revisions = {
                    key: dict(model_id=value, revision=JUDGE_REVISIONS[key])
                    for key, value in JUDGE_MODEL_IDS.items()
                }
                if run_info.get("judge_revisions") != expected_revisions:
                    errors.append(f"open-ended run judge revisions mismatch for {condition}")
                if run_info.get("judges") != JUDGE_MODEL_IDS:
                    errors.append(f"open-ended run judge model IDs mismatch for {condition}")
                expected_prompt_hashes = {
                    "alignment": hashlib.sha256(ALIGNMENT_PROMPT.encode()).hexdigest(),
                    "coherence": hashlib.sha256(COHERENCE_PROMPT.encode()).hexdigest(),
                }
                if run_info.get("judge_prompt_sha256") != expected_prompt_hashes:
                    errors.append(f"open-ended run judge prompt mismatch for {condition}")
                if run_info.get("ifbench_summary") != row.get("instruction_following_summary"):
                    errors.append(f"open-ended run and IFBench summaries differ for {condition}")

                with Path(summary_csv).open(newline="") as handle:
                    judge_rows = list(csv.DictReader(handle))
                expected_groups = {(judge, split): count for judge in JUDGE_MODEL_IDS
                                   for split, count in (("canonical", 240), ("heldout", 240), ("all", 480))}
                observed_groups = {(item.get("judge"), item.get("split")): item
                                   for item in judge_rows}
                if (len(judge_rows) != 6 or set(observed_groups) != set(expected_groups)
                        or any(int(observed_groups[key].get("total", -1)) != count
                               or observed_groups[key].get("rendering") != "training"
                               for key, count in expected_groups.items())):
                    errors.append(f"open-ended judge summary is incomplete for {condition}")

                with Path(responses_csv).open(newline="") as handle:
                    response_rows = list(csv.DictReader(handle))
                response_keys = {(item.get("prompt_id"), item.get("sample_idx")) for item in response_rows}
                expected_keys = {(prompt["prompt_id"], str(sample))
                                 for prompt in PROMPTS for sample in range(30)}
                if len(response_rows) != 480 or response_keys != expected_keys:
                    errors.append(f"open-ended response table is incomplete for {condition}")
                raw_paths = run_info.get("raw_rollout_paths", [])
                if len(raw_paths) != 1 or not Path(raw_paths[0]).is_file():
                    errors.append(f"permanent open-ended raw rollout is missing for {condition}")
                else:
                    raw_keys = set()
                    with Path(raw_paths[0]).open() as handle:
                        for line_number, line in enumerate(handle, 1):
                            item = json.loads(line)
                            key = (item.get("prompt_id"), str(item.get("sample_idx")))
                            if (item.get("run_id") != expected_run_id or item.get("model_key") != model
                                    or item.get("condition") != condition or item.get("rendering") != "training"
                                    or key in raw_keys):
                                errors.append(f"open-ended raw rollout provenance/uniqueness failure for {condition} at line {line_number}")
                                break
                            raw_keys.add(key)
                    if raw_keys != expected_keys:
                        errors.append(f"permanent open-ended raw rollout is incomplete for {condition}")
            except Exception as exc:
                errors.append(f"cannot validate open-ended evaluation for {condition}: {type(exc).__name__}: {exc}")

        scores = row.get("instruction_following_summary")
        if not isinstance(scores, dict):
            errors.append(f"IFBench summary is missing for {condition}")
            continue
        if (scores.get("model") != model or scores.get("condition") != condition
                or scores.get("run_id") != expected_run_id
                or scores.get("rendering") != "training" or scores.get("total_items") != 841
                or scores.get("valid_items") != 841 or scores.get("invalid_items") != 0):
            errors.append(f"IFBench summary is invalid or incomplete for {condition}")
        path = row.get("instruction_following_evaluation")
        if not path:
            errors.append(f"IFBench artifact path is missing for {condition}")
        elif verify_files:
            try:
                saved = json.loads(Path(path).read_text())
                if saved != scores:
                    errors.append(f"embedded and saved IFBench summaries differ for {condition}")
                if saved.get("raw_rollout_path") is None or not Path(saved["raw_rollout_path"]).is_file():
                    errors.append(f"permanent IFBench raw rollout is missing for {condition}")
                if saved.get("dataset_sha256") != frozen_dataset_hash:
                    errors.append(f"IFBench dataset hash differs from the frozen source for {condition}")
                response_path = saved.get("response_table_path")
                response_ids = None
                if response_path is None or not Path(response_path).is_file():
                    errors.append(f"IFBench response table is missing for {condition}")
                else:
                    with Path(response_path).open(newline="") as handle:
                        response_rows = list(csv.DictReader(handle))
                        response_ids = {item.get("item_id") for item in response_rows}
                    if (len(response_rows) != 841 or len(response_ids) != 841
                            or response_ids != frozen_ids):
                        errors.append(f"IFBench response table is incomplete for {condition}")
                raw_path = saved.get("raw_rollout_path")
                raw_ids = set()
                if raw_path and Path(raw_path).is_file():
                    with Path(raw_path).open() as handle:
                        for line_number, line in enumerate(handle, 1):
                            try:
                                item = json.loads(line)
                            except Exception as exc:
                                errors.append(f"invalid raw IFBench JSON for {condition} at line {line_number}: {exc}")
                                break
                            if (item.get("run_id") != expected_run_id or item.get("model") != model
                                    or item.get("condition") != condition
                                    or item.get("dataset_sha256") != saved.get("dataset_sha256")):
                                errors.append(f"raw IFBench provenance mismatch for {condition} at line {line_number}")
                                break
                            item_id = item.get("item_id")
                            if not isinstance(item_id, str) or item_id in raw_ids:
                                errors.append(f"duplicate or invalid raw IFBench item for {condition} at line {line_number}")
                                break
                            raw_ids.add(item_id)
                    if len(raw_ids) != 841 or raw_ids != frozen_ids:
                        errors.append(f"permanent IFBench raw rollout is incomplete for {condition}")
                if response_ids is not None and raw_ids != response_ids:
                    errors.append(f"IFBench raw and scored item sets differ for {condition}")
            except Exception as exc:
                errors.append(f"cannot validate IFBench summary for {condition}: {type(exc).__name__}: {exc}")

        iheval = row.get("iheval_reference_evaluation")
        if not isinstance(iheval, dict):
            errors.append(f"IHEval Reference summary is missing for {condition}")
        else:
            if (iheval.get("schema") != "acl_iheval_reference_v1"
                    or iheval.get("model") != model or iheval.get("condition") != condition
                    or iheval.get("run_id") != expected_run_id or iheval.get("setting") != "reference"
                    or (verify_files and iheval.get("dataset_sha256") != frozen_iheval_hash)
                    or iheval.get("total_items") != IHEVAL_REFERENCE_ITEM_COUNT
                    or iheval.get("invalid_items") != 0):
                errors.append(f"IHEval Reference summary is invalid or incomplete for {condition}")
            if verify_files:
                eval_dir = Path(row.get("training_evaluation", ""))
                iheval_path = eval_dir / "iheval_reference_summary.json"
                try:
                    saved_iheval = json.loads(iheval_path.read_text())
                    if saved_iheval != iheval:
                        errors.append(f"embedded and saved IHEval Reference summaries differ for {condition}")
                    errors.extend(validate_iheval_artifacts(
                        iheval_path, model_key=model, condition=condition,
                        run_id=expected_run_id, verify_files=True,
                    ))
                    iheval_run_path = eval_dir / "run.json"
                    iheval_run = json.loads(iheval_run_path.read_text())
                    if iheval_run.get("iheval_reference_summary") != saved_iheval:
                        errors.append(f"evaluation run and saved IHEval summaries differ for {condition}")
                except Exception as exc:
                    errors.append(f"cannot validate IHEval Reference artifacts for {condition}: {type(exc).__name__}: {exc}")
    return errors


def terminal(state: str) -> bool:
    return state not in {"UNKNOWN", "PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[CALIBRATION_MODEL], required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--calibration-attempt-id", required=True)
    args = parser.parse_args()

    workflow_root = ACL_ROOT / "calibration" / args.model / "seed_61791" / "workflows" / args.workflow_id
    state_path = workflow_root / "workflow_state.json"
    state = json.loads(state_path.read_text())
    if state.get("model") != args.model or state.get("workflow_id") != args.workflow_id:
        raise ValueError("workflow state provenance mismatch")
    if state.get("calibration_attempt_id") != args.calibration_attempt_id:
        raise ValueError("calibration attempt differs from workflow state")

    slurm = stable_job_state(args.job_id)
    summary_path = ACL_ROOT / "full_runs" / args.model / f"seed_{args.seed}.json"
    summary = None
    parse_error = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    errors = validate_summary(summary, args.model, args.seed, verify_files=True)
    if parse_error:
        errors.append("summary JSON parse error: " + parse_error)
    job_success = slurm["state"] == "COMPLETED" and slurm.get("exit_code") == "0:0"
    analysis_result = None
    if job_success and not errors:
        try:
            from stage6b_defense_analyze import write_analysis
            analysis_result = write_analysis(args.model, args.seed, summary_path)
        except Exception as exc:
            errors.append(f"post-run defense analysis failed: {type(exc).__name__}: {exc}")
    seed_success = job_success and not errors

    out_log = LOG_ROOT / f"pcacl_full_{args.job_id}.out"
    err_log = LOG_ROOT / f"pcacl_full_{args.job_id}.err"
    seed_record = dict(seed=args.seed, job_id=args.job_id, observed_at=now(), slurm=slurm,
                       job_success=job_success, summary_path=str(summary_path) if summary_path.exists() else None,
                       summary_valid=not errors, validation_errors=errors,
                       defense_analysis=analysis_result,
                       stdout_tail=tail(out_log), stderr_tail=tail(err_log))
    seed_records = state.setdefault("seed_results", [])
    previous = next((index for index, row in enumerate(seed_records)
                     if row.get("seed") == args.seed), None)
    if previous is None:
        seed_records.append(seed_record)
    elif seed_records[previous].get("job_id") == args.job_id:
        seed_records[previous] = seed_record
    else:
        raise ValueError(f"seed {args.seed} was already monitored; refusing duplicate continuation")

    status_name = "seed_completed" if seed_success else "seed_failed"
    current_index = SEEDS.index(args.seed)
    if args.seed == SEEDS[0] and terminal(slurm["state"]):
        if slurm["state"] == "CANCELLED":
            status_name = "first_seed_report_required"
        else:
            status_name = "first_seed_report_required"
    elif current_index + 1 < len(SEEDS) and terminal(slurm["state"]):
        status_name = "seed_completed_requires_coordinator_review" if seed_success else "seed_failed_requires_coordinator_review"
    elif current_index + 1 == len(SEEDS):
        all_ok = (len(state.get("seed_results", [])) == len(SEEDS) and
                  all(item.get("job_success") and item.get("summary_valid")
                      for item in state["seed_results"]))
        status_name = "three_seed_chain_complete" if all_ok else "three_seed_chain_complete_with_failures"
    elif slurm["state"] == "CANCELLED":
        status_name = "chain_stopped_after_cancellation"
    elif not terminal(slurm["state"]):
        status_name = "job_state_unresolved"

    if args.seed == SEEDS[0] and terminal(slurm["state"]):
        next_action = (
            "review the complete first-seed report; additional seeds are not submitted until the mechanism is interpreted"
            if seed_success else
            "review the recorded first-seed failure and partial outputs; no additional seed is submitted automatically"
        )
    elif slurm["state"] == "CANCELLED":
        next_action = "the seed chain was cancelled; all recorded artifacts are preserved"
    else:
        next_action = "inspect the recorded terminal state and logs; no additional seed was submitted"

    report = dict(
        model=args.model, workflow_id=args.workflow_id, seed=args.seed,
        full_seed_job=args.job_id, monitor_job=os.environ.get("SLURM_JOB_ID", "local"),
        observed_at=now(), slurm=slurm, full_seed_success=seed_success,
        full_seed_summary=str(summary_path) if summary_path.exists() else None,
        defense_analysis=analysis_result,
        validation_errors=errors, next_seed_job=None,
        next_seed_monitor_job=None, workflow_status=status_name,
        missing_or_failed_seeds=[item["seed"] for item in state.get("seed_results", [])
                                 if not (item.get("job_success") and item.get("summary_valid"))],
        next_action=next_action,
    )
    state["status"] = status_name
    if args.seed == SEEDS[0] and terminal(slurm["state"]):
        state["phase"] = "first_seed_report_required"
        state["first_seed_result"] = "completed" if seed_success else "failed_or_cancelled"
    state["updated_at"] = now()
    state["last_report"] = report
    atomic_json(state_path, state)
    status_path = workflow_root / f"full_seed_monitor_seed{args.seed}_job{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    atomic_json(status_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
