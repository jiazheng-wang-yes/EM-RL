"""Afterany controller: validate preflight and stop for first-seed review."""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from acl_common import ACL_ROOT, CALIBRATION_MODEL, MODELS, PREFLIGHT_SOURCE_FILES, ROOT
from stage6b_acl_calibration_monitor import (
    HERE, LOG_ROOT, atomic_json, now, parse_job_id, stable_job_state,
    tail,
)


def validate(report: dict | None, model: str) -> list[str]:
    if not isinstance(report, dict):
        return ["preflight report is missing or invalid JSON"]
    errors = []
    if report.get("model") != model or report.get("seed") != 61791:
        errors.append("wrong preflight model or seed")
    if report.get("run_id") != "20260924T063649Z":
        errors.append("preflight is not linked to the frozen Round 1 run ID")
    hashes = report.get("source_sha256")
    if not isinstance(hashes, dict) or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(hashes.get(name, "")))
        for name in PREFLIGHT_SOURCE_FILES
    ):
        errors.append("preflight source hashes are missing or invalid")
    command = report.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        errors.append("preflight command provenance is missing or invalid")
    if report.get("base_model_id") != MODELS[model]["hf_id"] or report.get("base_model_revision") != MODELS[model]["revision"]:
        errors.append("preflight base-model revision differs from the frozen model revision")
    if report.get("no_em_evaluation") is not True:
        errors.append("preflight did not attest that EM evaluation was withheld")
    if report.get("two_step_ordinary_equals_lambda_d_zero") is not True:
        errors.append("ordinary SFT and zero-mix paths did not match")
    if report.get("max_parameter_difference") != 0.0:
        errors.append("ordinary/zero-mix parameter difference is nonzero")
    if report.get("protected_parameter_count", 0) <= 0:
        errors.append("protected parameter set is empty")
    mix = report.get("mix", {})
    if mix.get("mixed_gradient_max_error") != 0.0:
        errors.append("hand-computed mixed gradient check failed")
    if mix.get("good_gradient_outside_hook_calls") != 0:
        errors.append("benign gradient reached parameters outside the protected region")
    if mix.get("total_postclip_norm", float("inf")) > 1.00001:
        errors.append("mixed gradient was not clipped after composition")
    cases = report.get("mix_cases", {})
    for key in ("W_lambda_d_0_50", "W_lambda_d_0_75",
                "P_plus_W_lambda_d_0_75_lambda_p_0_00014558266395104324"):
        if not isinstance(cases.get(key), dict) or cases[key].get("mixed_gradient_max_error") != 0.0:
            errors.append(f"required mixed-gradient case is missing or failed: {key}")
    if report.get("selected_lambda_d") != 0.75:
        errors.append("preflight did not test the selected lambda_D=.75")
    if report.get("selected_lambda_p") != 0.00014558266395104324:
        errors.append("preflight P+W check does not use the saved full-precision lambda_P")
    if (report.get("protected_tensor_count") != 84
            or report.get("wrong_region_tensor_count") != 84
            or report.get("wrong_region_parameter_count") != report.get("protected_parameter_count")
            or report.get("wrong_region_shapes_match") is not True):
        errors.append("wrong-region control does not exactly match A in tensor shapes and parameter count")
    if not report.get("paired_order_sha256") or not report.get("base_coordinate_source"):
        errors.append("missing data-order or base-coordinate provenance")
    elif (report["base_coordinate_source"].get("model_revision") != MODELS[model]["revision"]
          or report["base_coordinate_source"].get("tokenizer_revision") != MODELS[model]["revision"]):
        errors.append("base-coordinate model or tokenizer revision is not pinned")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[CALIBRATION_MODEL], required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--calibration-attempt-id", required=True)
    parser.add_argument("--version", default="implementation_v5")
    args = parser.parse_args()

    workflow_root = ACL_ROOT / "calibration" / args.model / "seed_61791" / "workflows" / args.workflow_id
    state_path = workflow_root / "workflow_state.json"
    state = json.loads(state_path.read_text())
    if state.get("model") != args.model or state.get("workflow_id") != args.workflow_id:
        raise ValueError("workflow state provenance mismatch")

    status = stable_job_state(args.job_id)
    preflight_path = ACL_ROOT / "preflight" / args.version / args.model / "seed_61791.json"
    metric_path = ROOT / "logs" / "persona_control" / "training_metrics" / "stage6b_acl" / "preflight" / args.version / f"{args.model}_seed_61791.jsonl"
    preflight = None
    parse_error = None
    if preflight_path.is_file():
        try:
            preflight = json.loads(preflight_path.read_text())
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
    errors = validate(preflight, args.model)
    if preflight is not None and preflight.get("preflight_version") != args.version:
        errors.append("preflight artifact version differs from the monitor request")
    if args.version == "implementation_v5" and preflight is not None:
        progress_path = (ROOT / "logs" / "persona_control" / "training_metrics" / "stage6b_acl"
                         / "preflight" / args.version / f"{args.model}_seed_61791_progress.jsonl")
        if (preflight.get("progress_path") != str(progress_path) or not progress_path.is_file()
                or not progress_path.stat().st_size):
            errors.append("v5 preflight phase-progress record is missing")
        elif __import__("hashlib").sha256(progress_path.read_bytes()).hexdigest() != preflight.get("progress_sha256"):
            errors.append("v5 preflight phase-progress hash does not match the report")
    if parse_error:
        errors.append("preflight JSON parse error: " + parse_error)
    if not metric_path.is_file() or metric_path.stat().st_size == 0:
        errors.append("permanent preflight training-metric record is missing")

    success = status["state"] == "COMPLETED" and status.get("exit_code") == "0:0" and not errors
    state["status"] = "preflight_passed_review_required" if success else "preflight_failed"
    state["phase"] = "preflight_passed" if success else "preflight_failed"

    monitor_id = os.environ.get("SLURM_JOB_ID", "local")
    report = dict(
        model=args.model, workflow_id=args.workflow_id, preflight_job=args.job_id,
        monitor_job=monitor_id, observed_at=now(), slurm=status,
        preflight_artifact=str(preflight_path) if preflight_path.exists() else None,
        training_metrics=str(metric_path) if metric_path.exists() else None,
        preflight_valid=not errors, validation_errors=errors,
        preflight_sha256=(__import__("hashlib").sha256(preflight_path.read_bytes()).hexdigest()
                          if preflight_path.exists() else None),
        preflight_version=args.version, full_seed_job=None, full_seed_monitor_job=None,
        workflow_status=state["status"],
        next_action=("preflight passed; inspect the versioned report and explicitly submit only seed 61791 after coordinator review"
                     if success else "preflight failed; inspect this report and preserve all partial artifacts"),
    )
    state["updated_at"] = now()
    state["preflight_result"] = report
    state["last_report"] = report
    atomic_json(state_path, state)
    status_path = workflow_root / f"preflight_monitor_job{monitor_id}.json"
    atomic_json(status_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
