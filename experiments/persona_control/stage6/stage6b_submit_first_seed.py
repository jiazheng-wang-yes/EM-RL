"""Explicit coordinator-invoked submission for the first full Stage 6B seed."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from acl_common import ACL_ROOT, CALIBRATION_MODEL, CALIBRATION_SOURCE_FILES, MODELS, ROOT, sha256_file
from stage6b_acl_calibration_monitor import atomic_json, now, parse_job_id, submit_phase_monitor
from stage6b_preflight_monitor import validate as validate_preflight

HERE = Path(__file__).resolve().parent
RUN_ID = "20260924T063649Z"
CONDITIONS = "E,P,W,P+W,wrong_region_mix,global_mix,slowdown"
PREFLIGHT = ACL_ROOT / "preflight" / "implementation_v5" / CALIBRATION_MODEL / "seed_61791.json"
CALIBRATION_ROOT = ACL_ROOT / "calibration" / CALIBRATION_MODEL / "seed_61791"
AUDIT_MANIFEST = ROOT / "eval_runs/persona_control_arr/round1" / RUN_ID / "manifest.json"
PLAN_PATH = ROOT / "docs/plans/crd-arr-research-implementation-2026-09-24.md"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--calibration-attempt-id", required=True)
    args = parser.parse_args()
    workflow = CALIBRATION_ROOT / "workflows" / args.workflow_id
    state_path = workflow / "workflow_state.json"
    calibration_path = CALIBRATION_ROOT / args.calibration_attempt_id / "calibration_summary.json"
    if not all(path.is_file() for path in (state_path, calibration_path, PREFLIGHT, AUDIT_MANIFEST, PLAN_PATH)):
        raise FileNotFoundError("workflow, calibration, preflight, audit manifest, and plan are all required")
    with (workflow / "stage6b_first_seed_submit.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text())
        calibration = json.loads(calibration_path.read_text())
        preflight = json.loads(PREFLIGHT.read_text())
        if state.get("model") != CALIBRATION_MODEL or state.get("workflow_id") != args.workflow_id:
            raise ValueError("workflow state provenance mismatch")
        if state.get("calibration_attempt_id") != args.calibration_attempt_id:
            raise ValueError("calibration attempt differs from workflow state")
        if state.get("status") != "preflight_passed_review_required":
            raise ValueError("the afterany validator has not recorded preflight_passed_review_required")
        preflight_result = state.get("preflight_result") or {}
        preflight_job = str(state.get("preflight_job", ""))
        preflight_monitor = str(state.get("preflight_monitor_job", ""))
        if (not preflight_result.get("preflight_valid")
                or str(preflight_result.get("preflight_job")) != preflight_job
                or str(preflight_result.get("monitor_job")) != preflight_monitor
                or preflight_result.get("preflight_sha256") != sha256_file(PREFLIGHT)):
            raise ValueError("completed afterany preflight report does not match the recorded job IDs or artifact hash")
        slurm = preflight_result.get("slurm") or {}
        if slurm.get("state") != "COMPLETED" or slurm.get("exit_code") != "0:0":
            raise ValueError("GPU preflight job did not complete successfully")
        if calibration.get("model") != CALIBRATION_MODEL or calibration.get("seed") != 61791:
            raise ValueError("calibration artifact has wrong model or seed")
        if calibration.get("lambda_d_star") != 0.75 or calibration.get("lambda_p_star") != 0.00014558266395104324:
            raise ValueError("saved calibration coefficients differ from the frozen first-seed values")
        audit = json.loads(AUDIT_MANIFEST.read_text())
        audit_calibration = audit.get("calibration") or {}
        if (audit.get("run_id") != RUN_ID
                or Path(audit_calibration.get("path", "")).resolve() != calibration_path.resolve()
                or audit_calibration.get("sha256") != sha256_file(calibration_path)
                or audit_calibration.get("attempt_id") != args.calibration_attempt_id):
            raise ValueError("calibration path, hash, or attempt does not match the frozen ARR audit manifest")
        calibration_source_comparison = {}
        for name in CALIBRATION_SOURCE_FILES:
            recorded = calibration.get("source_sha256", {}).get(name)
            current = sha256_file(HERE / name)
            calibration_source_comparison[name] = dict(calibration_sha256=recorded,
                                                       current_sha256=current,
                                                       matches=recorded == current)
        changed_calibration_sources = sorted(
            name for name, row in calibration_source_comparison.items() if not row["matches"]
        )
        if changed_calibration_sources != ["stage6b_train.py"]:
            raise ValueError(
                "calibration source mismatch differs from the documented training-diagnostic change: "
                f"{changed_calibration_sources}"
            )
        errors = validate_preflight(preflight, CALIBRATION_MODEL)
        metric = ROOT / "logs/persona_control/training_metrics/stage6b_acl/preflight/implementation_v5/qwen2_5_7b_seed_61791.jsonl"
        if not metric.is_file() or metric.stat().st_size == 0:
            errors.append("permanent GPU preflight training-metrics record is missing")
        if errors:
            raise ValueError("preflight validation failed: " + "; ".join(errors))
        if preflight.get("run_id") != RUN_ID:
            raise ValueError("preflight run ID does not match the frozen Round 1 namespace")
        if preflight.get("preflight_version") != "implementation_v5":
            raise ValueError("first-seed submission requires the implementation_v5 GPU preflight")
        if state.get("seed_jobs"):
            raise FileExistsError("workflow already records a full-seed job; refusing duplicate submission")
        summary = ACL_ROOT / "full_runs" / CALIBRATION_MODEL / "seed_61791.json"
        if summary.exists():
            raise FileExistsError(f"immutable first-seed summary already exists: {summary}")
        command = ["sbatch", "--parsable", str(HERE / "stage6b_full_seed.sbatch"),
                   "--model", CALIBRATION_MODEL, "--seed", "61791", "--calibration", str(calibration_path),
                   "--preflight", str(PREFLIGHT), "--conditions", CONDITIONS,
                   "--arr-run-id", RUN_ID, "--legacy-rendering-check"]
        job = parse_job_id(subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False))
        state["status"] = "first_seed_running"
        state["phase"] = "first_seed_running"
        seed_record = dict(seed=61791, job_id=job, submitted_at=now(), run_id=RUN_ID,
                           conditions=CONDITIONS.split(","), calibration=str(calibration_path),
                           calibration_sha256=sha256_file(calibration_path), preflight=str(PREFLIGHT),
                           preflight_sha256=sha256_file(PREFLIGHT), plan=str(PLAN_PATH),
                           plan_sha256=sha256_file(PLAN_PATH), audit_manifest=str(AUDIT_MANIFEST),
                           audit_manifest_sha256=sha256_file(AUDIT_MANIFEST), submission_command=command[2:],
                           calibration_source_comparison=calibration_source_comparison,
                           calibration_source_change_note=("stage6b_train.py now adds a nonpersistent E/W local-update "
                               "diagnostic and an exact wrong-region parameter-count assertion; the frozen loss, "
                               "rendering, batch, optimizer, and gradient-mixing rules are unchanged."))
        state["seed_jobs"] = [seed_record]
        atomic_json(state_path, state)
        try:
            monitor = submit_phase_monitor(
                "stage6b_full_seed_monitor.sbatch", CALIBRATION_MODEL, args.workflow_id, job,
                ["--seed", "61791", "--calibration-attempt-id", args.calibration_attempt_id],
            )
        except Exception:
            state["status"] = "first_seed_monitor_submission_error"
            state["seed_jobs"][0]["monitor_submission_error"] = "afterany monitor submission failed; preserve and recover the recorded job"
            atomic_json(state_path, state)
            raise
        state["seed_jobs"][0]["monitor_job_id"] = monitor
        state["seed_jobs"][0]["monitor_submitted_at"] = now()
        atomic_json(state_path, state)
        print(json.dumps({"run_id": RUN_ID, "conditions": CONDITIONS.split(","),
                          "full_seed_job": job, "afterany_monitor_job": monitor,
                          "workflow_state": str(state_path)}, indent=2))


if __name__ == "__main__":
    main()
