"""Afterany monitor for a bounded Stage 6B calibration retry chain.

Infrastructure failures are retried at most twice in fresh, immutable attempt
directories. Application errors and incomplete outputs are recorded for
diagnosis; the monitor never launches the 184-step experiment automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from acl_common import (
    ACL_ROOT, CALIBRATION_MODEL, CALIBRATION_SOURCE_FILES,
    DIRECT_CALIBRATION_GRID, MIN_RETAINED_LEARNING,
    PERSONA_CALIBRATION_MULTIPLIERS, PERSONA_DRIFT_REDUCTION_RANGE,
    MODELS, ROOT,
)

HERE = ROOT / "experiments" / "persona_control" / "stage6"
LOG_ROOT = ROOT / "logs" / "slurm" / "persona_control"
TRANSIENT_STATES = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "REQUEUED", "REVOKED"}
DIRECT_GRID = list(DIRECT_CALIBRATION_GRID)
PERSONA_GRID = list(PERSONA_CALIBRATION_MULTIPLIERS)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value)
            if not value.endswith("\n"):
                handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def elapsed_seconds(value: str | None) -> int | None:
    """Parse Slurm Elapsed values, including its optional days prefix."""
    if not value or value in {"unknown", "Unknown", "N/A"}:
        return None
    try:
        day_text, separator, clock_text = value.partition("-")
        days = int(day_text) if separator else 0
        fields = [int(item) for item in (clock_text if separator else value).split(":")]
        if len(fields) == 3:
            hours, minutes, seconds = fields
        elif len(fields) == 2:
            hours, minutes, seconds = 0, *fields
        elif len(fields) == 1:
            hours, minutes, seconds = 0, 0, fields[0]
        else:
            return None
        if min(days, hours, minutes, seconds) < 0 or minutes >= 60 or seconds >= 60:
            return None
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    except (TypeError, ValueError):
        return None


def compute_estimate(status: dict, summary: dict | None, valid_completion: bool) -> dict:
    """Extrapolate only first-seed training time; endpoint evaluation is excluded."""
    rows = (summary or {}).get("report_table")
    seconds = elapsed_seconds(status.get("elapsed"))
    if not valid_completion or not isinstance(rows, list) or not rows or seconds is None or seconds <= 0:
        return dict(available=False, reason="requires a valid completed calibration with elapsed time")
    calibration_runs = len(rows)
    # Six required first-seed conditions: four primary conditions plus wrong-region
    # and global-mixing controls. The optional slowdown condition is not included.
    full_conditions = 6
    estimate_seconds = seconds * (184 * full_conditions) / (16 * calibration_runs)
    return dict(
        available=True,
        scope="training-only for the six required first-seed conditions",
        calibration_elapsed_seconds=seconds,
        calibration_16_step_runs=calibration_runs,
        full_seed_conditions=full_conditions,
        estimate_seconds=round(estimate_seconds),
        estimate_hours=round(estimate_seconds / 3600, 2),
        method=("calibration Slurm elapsed × (184 steps × 6 conditions) / "
                "(16 steps × number of completed calibration runs)"),
        exclusions=["queue wait", "endpoint evaluation", "optional slowdown condition"],
    )


def markdown_cell(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value).replace("|", "\\|").replace("\n", " ")


def calibration_markdown(report: dict, summary: dict | None,
                         summary_path: Path, stdout_path: Path, stderr_path: Path) -> str:
    status = report["slurm"]
    lines = [
        "# Stage 6B calibration monitor report",
        "",
        f"- Workflow / attempt: `{report['workflow_id']}` / `{report['attempt_id']}` "
        f"(attempt {report['attempt_number']})",
        f"- Calibration job / monitor job: `{report['calibration_job']}` / `{report['monitor_job']}`",
        f"- Workflow status: **{report['workflow_status']}**",
        f"- Slurm: `{status.get('state', 'unknown')}`; exit `{status.get('exit_code', 'unknown')}`; "
        f"elapsed `{status.get('elapsed', 'unknown')}`; node `{status.get('node', 'unknown')}`",
        f"- Next action: {report['next_action']}",
        f"- Summary artifact: `{summary_path}`",
        f"- Calibration stdout / stderr: `{stdout_path}` / `{stderr_path}`",
        "",
        "Broad EM was not evaluated during calibration. No 184-step training job was submitted.",
        "",
    ]
    if report.get("retry_job"):
        lines.extend([f"Automatic retry job: `{report['retry_job']}`; dependent monitor: "
                      f"`{report.get('followup_monitor_job') or 'submission failed'}`.", ""])
    capacity = report.get("storage_status")
    if isinstance(capacity, dict):
        lines.extend(["Storage check before retry:", ""])
        if "checkpoint_bytes" in capacity:
            lines.append(f"- Checkpoints: {capacity['checkpoint_bytes']} bytes; "
                         "both storage caps are waived for ACL / Stage 6B")
        if capacity.get("repository_cap_waived"):
            lines.append("- Total-repository size was not scanned; its cap is waived for this workflow.")
        if capacity.get("repo_bytes") is not None:
            lines.append(f"- Repository: {capacity['repo_bytes']} bytes; "
                         f"limit {capacity.get('max_repo_bytes', 'unknown')} bytes")
        elif capacity.get("repo_lower_bound_bytes") is not None:
            lines.append(f"- Repository is at least {capacity['repo_lower_bound_bytes']} bytes; "
                         f"limit {capacity.get('max_repo_bytes', 'unknown')} bytes")
        if capacity.get("reason"):
            lines.append(f"- Result: `{capacity['reason']}`")
        lines.append("")
    if report.get("validation_errors"):
        lines.extend(["Validation / artifact issues:", ""])
        lines.extend(f"- {item}" for item in report["validation_errors"])
        lines.append("")
    if isinstance(summary, dict):
        gate = summary.get("calibration_gate")
        gate = gate if isinstance(gate, dict) else {}
        lines.extend([
            "## Calibration selection",
            "",
            f"- Selected `λD*`: {markdown_cell(summary.get('lambda_d_star'))}",
            f"- Selected `λP*`: {markdown_cell(summary.get('lambda_p_star'))}",
            f"- Persona scale: {markdown_cell(summary.get('persona_scale'))}",
            f"- Prespecified constraints satisfied: "
            f"{markdown_cell(gate.get('eligible'))}",
            "",
            "| Condition | λD | λP | Harmful loss | Benign loss | Retained learning | Persona drift | Region update norm | Suppressed harmful-specific gradient |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        rows = summary.get("report_table")
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                lines.append("| " + " | ".join(markdown_cell(row.get(key)) for key in (
                    "condition", "lambda_d", "lambda_p", "harmful_train_loss", "benign_train_loss",
                    "retained_learning", "persona_drift", "direct_region_update_norm",
                    "suppressed_harmful_specific_gradient_fraction",
                )) + " |")
    else:
        lines.extend(["## Calibration measurements", "", "No readable calibration summary was produced.", ""])
    estimate = report.get("compute_estimate", {})
    lines.extend(["", "## Full-experiment compute estimate", ""])
    if estimate.get("available"):
        lines.extend([
            f"Estimated training time: **{estimate['estimate_hours']:.2f} GPU-hours** on one GPU "
            f"(about {estimate['estimate_seconds']} seconds).",
            f"Method: {estimate['method']}; observed calibration elapsed was "
            f"{estimate['calibration_elapsed_seconds']} seconds over "
            f"{estimate['calibration_16_step_runs']} 16-step runs.",
            "This is a training-only extrapolation for the six required first-seed conditions; "
            "queue wait, endpoint evaluation, and optional slowdown are excluded.",
        ])
    else:
        lines.append("Unavailable: no valid completed calibration elapsed time and report table.")
    lines.append("")
    return "\n".join(lines)


def job_state(job_id: str) -> dict:
    proc = subprocess.run(
        ["sacct", "-X", "-j", str(job_id),
         "--format=JobIDRaw,State,ExitCode,Elapsed,NodeList,Reason",
         "--noheader", "--parsable2"],
        text=True, capture_output=True, check=False,
    )
    records = []
    for line in proc.stdout.splitlines():
        fields = [item.strip() for item in line.split("|")]
        if len(fields) >= 3:
            records.append(fields)
    row = next((item for item in records if item[0] == str(job_id)), None)
    if row is None and records:
        row = records[0]
    if proc.returncode != 0 or row is None:
        return dict(state="UNKNOWN", exit_code="unknown", elapsed="unknown", node="unknown",
                    reason="unknown", accounting_error=proc.stderr[-2000:],
                    accounting_stdout=proc.stdout[-2000:])
    raw_state = row[1].split()[0]
    return dict(state=raw_state.rstrip("+"), raw_state=raw_state, exit_code=row[2],
                elapsed=row[3] if len(row) > 3 else "unknown",
                node=row[4] if len(row) > 4 else "unknown",
                reason=row[5] if len(row) > 5 else "unknown")


def stable_job_state(job_id: str) -> dict:
    result = job_state(job_id)
    for _ in range(3):
        if result["state"] != "UNKNOWN":
            return result
        time.sleep(5)
        result = job_state(job_id)
    return result


def validate(summary: dict | None, model: str, attempt_id: str) -> list[str]:
    if not isinstance(summary, dict):
        return ["calibration summary is missing or is not a JSON object"]
    errors = []
    if model != CALIBRATION_MODEL:
        errors.append("only Qwen2.5-7B calibration is allowed in this stage")
    if summary.get("model") != model:
        errors.append("wrong model")
    if summary.get("seed") != 61791 or summary.get("steps") != 16:
        errors.append("wrong calibration seed or length")
    if summary.get("attempt_id") != attempt_id:
        errors.append("attempt ID mismatch")
    if summary.get("no_em_evaluation") is not True:
        errors.append("calibration did not attest that EM evaluation was withheld")
    hashes = summary.get("source_sha256")
    if not isinstance(hashes, dict) or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(hashes.get(name, "")))
        for name in CALIBRATION_SOURCE_FILES
    ):
        errors.append("calibration source hashes are missing or invalid")
    command = summary.get("command")
    if not isinstance(command, list) or not command or not all(isinstance(item, str) for item in command):
        errors.append("calibration command provenance is missing or invalid")
    direct_rows = summary.get("direct")
    if not isinstance(direct_rows, list):
        errors.append("direct calibration grid is missing or invalid")
        direct_rows = []
    if [row.get("lambda_d") if isinstance(row, dict) else None for row in direct_rows] != DIRECT_GRID:
        errors.append("direct grid differs from prespecified values")
    persona_rows = summary.get("persona")
    if not isinstance(persona_rows, list):
        errors.append("persona calibration grid is missing or invalid")
        persona_rows = []
    if [row.get("multiplier") if isinstance(row, dict) else None for row in persona_rows] != PERSONA_GRID:
        errors.append("persona grid differs from prespecified values")
    report_table = summary.get("report_table")
    if not isinstance(report_table, list) or not report_table:
        errors.append("required calibration report table is missing")
    if not isinstance(summary.get("calibration_gate"), dict):
        errors.append("calibration review-gate record is missing")
    for key in ("baseline", "slowdown"):
        if not isinstance(summary.get(key), dict):
            errors.append(f"missing {key} result")
    baseline = summary.get("baseline")
    baseline = baseline if isinstance(baseline, dict) else {}
    baseline_manifest = baseline.get("manifest")
    baseline_manifest = baseline_manifest if isinstance(baseline_manifest, dict) else {}
    baseline_metrics = baseline.get("metrics")
    baseline_metrics = baseline_metrics if isinstance(baseline_metrics, dict) else {}
    initial = baseline_manifest.get("initial_harmful_val_nll")
    final = baseline_metrics.get("harmful_val_nll")
    try:
        baseline_improvement = float(initial) - float(final)
    except (TypeError, ValueError):
        baseline_improvement = float("nan")
    if not math.isfinite(baseline_improvement) or baseline_improvement <= 0:
        errors.append("ordinary harmful-SFT calibration did not show positive validation-NLL improvement")

    def finite_value(row, key):
        try:
            value = float(row[key])
            return value if math.isfinite(value) else None
        except (KeyError, TypeError, ValueError):
            return None

    if len(direct_rows) == len(DIRECT_GRID):
        direct_progress = [finite_value(row, "retained_learning") if isinstance(row, dict) else None
                           for row in direct_rows]
        if any(value is None for value in direct_progress):
            errors.append("direct calibration is missing finite retained-learning values")
        else:
            eligible = [row for row in direct_rows if isinstance(row, dict)
                        if finite_value(row, "retained_learning") >= MIN_RETAINED_LEARNING]
            expected_d = max(eligible, key=lambda row: float(row["lambda_d"]))["lambda_d"] if eligible else None
            if summary.get("lambda_d_star") != expected_d:
                errors.append("lambda_d_star does not follow the prespecified largest-eligible rule")
            if bool(summary.get("direct_selection_failure")) == bool(eligible):
                errors.append("direct selection failure flag disagrees with calibration eligibility")

    slowdown = summary.get("slowdown")
    slowdown = slowdown if isinstance(slowdown, dict) else {}
    selected_d = summary.get("lambda_d_star")
    if selected_d is None:
        if slowdown.get("status") != "not_run":
            errors.append("slowdown must be marked not-run when no direct coefficient qualifies")
    else:
        if slowdown.get("lambda_d") != selected_d:
            errors.append("slowdown did not use the selected lambda_D")
        if finite_value(slowdown, "retained_learning") is None:
            errors.append("slowdown is missing finite retained-learning progress")
        if not isinstance(slowdown.get("metrics"), dict):
            errors.append("slowdown training metrics are missing")

    if len(persona_rows) == len(PERSONA_GRID):
        values = [(finite_value(row, "retained_learning"), finite_value(row, "drift_reduction"),
                   finite_value(row, "lambda_p")) for row in persona_rows]
        if any(value is None for row in values for value in row):
            errors.append("persona calibration is missing finite retention, drift, or coefficient values")
        else:
            drift_min, drift_max = PERSONA_DRIFT_REDUCTION_RANGE
            acceptable = [row for row in persona_rows if isinstance(row, dict)
                          if finite_value(row, "retained_learning") >= MIN_RETAINED_LEARNING
                          and drift_min <= finite_value(row, "drift_reduction") <= drift_max]
            expected_p = min(acceptable, key=lambda row: finite_value(row, "lambda_p")).get("lambda_p") if acceptable else None
            if summary.get("lambda_p_star") != expected_p:
                errors.append("lambda_p_star does not follow the prespecified weakest-eligible rule")
            if bool(summary.get("selection_failure")) == bool(acceptable):
                errors.append("persona selection failure flag disagrees with calibration eligibility")
    gate = summary.get("calibration_gate")
    gate = gate if isinstance(gate, dict) else {}
    expected_gate = summary.get("lambda_d_star") is not None and summary.get("lambda_p_star") is not None
    if gate.get("eligible") is not expected_gate:
        errors.append("calibration gate eligibility disagrees with selected coefficients")
    if gate.get("full_training_submitted") is not False:
        errors.append("calibration monitor must stop before full training")
    if isinstance(report_table, list):
        expected_conditions = ["baseline"]
        expected_conditions.extend(f"D={value:.2f}" for value in DIRECT_GRID)
        expected_conditions.extend(f"P×{value:g}" for value in PERSONA_GRID)
        if summary.get("lambda_d_star") is not None:
            expected_conditions.append("slowdown")
        observed_conditions = [row.get("condition") if isinstance(row, dict) else None
                               for row in report_table]
        if observed_conditions != expected_conditions:
            errors.append("calibration report table rows do not match the prespecified conditions")
        required_columns = {
            "condition", "lambda_d", "lambda_p", "lambda_p_multiplier",
            "harmful_train_loss", "benign_train_loss", "harmful_val_nll",
            "benign_val_nll", "retained_learning", "persona_drift",
            "persona_drift_reduction", "direct_region_update_norm",
            "outside_region_update_norm", "suppressed_harmful_specific_gradient_fraction",
        }
        if any(not required_columns.issubset(row) for row in report_table if isinstance(row, dict)):
            errors.append("calibration report table is missing required measurement columns")
    return errors


def retryable(status: dict) -> bool:
    state = status.get("state", "UNKNOWN")
    if state in TRANSIENT_STATES:
        return True
    # RaisedSignal:53 has occurred on more than one Stage 6 node. Preserve and
    # retry this infrastructure signature, but do not retry arbitrary failures.
    return state == "FAILED" and status.get("exit_code", "").endswith(":53")


def tail(path: Path, lines: int = 60) -> str:
    if not path.is_file():
        return f"[missing log: {path}]"
    return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])


def resource_snapshot() -> dict:
    commands = {
        "sinfo": ["sinfo", "-N", "-h", "-o", "%N|%t|%G|%m|%C|%f"],
        "squeue": ["squeue", "-p", "general", "-o", "%.18i|%.12T|%.30j|%.20b|%.30R"],
    }
    result = {}
    for name, command in commands.items():
        proc = subprocess.run(command, text=True, capture_output=True, check=False)
        result[name] = dict(returncode=proc.returncode, stdout=proc.stdout[-12000:],
                            stderr=proc.stderr[-2000:])
    return result


def storage_status() -> dict:
    """Record checkpoint size; ACL / Stage 6B storage caps were waived."""
    measured = {}
    checkpoint_path = ROOT / "checkpoints"
    proc = subprocess.run(["du", "-s", "--block-size=1", str(checkpoint_path)],
                          text=True, capture_output=True, check=False)
    if proc.returncode != 0 or not proc.stdout.strip():
        return dict(ok=False, error=f"du failed for {checkpoint_path}: {proc.stderr[-1000:]}")
    try:
        measured["checkpoint_bytes"] = int(proc.stdout.split()[0])
    except (ValueError, IndexError):
        return dict(ok=False, error=f"could not parse du output for {checkpoint_path}: {proc.stdout!r}")
    measured.update(ok=True, checkpoint_cap_waived=True, repository_cap_waived=True)
    return measured


def parse_job_id(result: subprocess.CompletedProcess) -> str:
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:] or "sbatch returned a nonzero status")
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"could not parse sbatch job ID from {result.stdout!r}")
    return job_id


def submit_calibration(model: str, attempt_id: str) -> str:
    command = ["sbatch", "--parsable", str(HERE / "stage6b_calibrate_acl.sbatch"),
               "--model", model, "--seed", "61791", "--attempt-id", attempt_id]
    return parse_job_id(subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False))


def submit_followup_monitor(model: str, workflow_id: str, attempt_id: str,
                           attempt_number: int, job_id: str, max_retries: int) -> str:
    command = ["sbatch", "--parsable", f"--dependency=afterany:{job_id}",
               str(HERE / "stage6b_acl_calibration_monitor.sbatch"),
               "--model", model, "--workflow-id", workflow_id,
               "--attempt-id", attempt_id, "--attempt-number", str(attempt_number),
               "--job-id", job_id, "--max-retries", str(max_retries)]
    return parse_job_id(subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False))


def submit_phase_monitor(script_name: str, model: str, workflow_id: str,
                         job_id: str, extra: list[str] | None = None) -> str:
    command = ["sbatch", "--parsable", f"--dependency=afterany:{job_id}",
               str(HERE / script_name), "--model", model, "--workflow-id", workflow_id,
               "--job-id", job_id]
    command.extend(extra or [])
    return parse_job_id(subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[CALIBRATION_MODEL], required=True)
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--attempt-number", type=int, required=True,
                        help="zero-based calibration attempt within this workflow")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()
    if args.attempt_number < 0 or args.max_retries < 0 or args.attempt_number > args.max_retries:
        raise ValueError("attempt number/retry limit is invalid")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.workflow_id):
        raise ValueError("workflow ID may contain only letters, digits, underscores, and hyphens")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.attempt_id):
        raise ValueError("attempt ID may contain only letters, digits, underscores, and hyphens")

    workflow_root = ACL_ROOT / "calibration" / args.model / "seed_61791" / "workflows" / args.workflow_id
    state_path = workflow_root / "workflow_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        workflow_id=args.workflow_id, model=args.model, seed=61791, created_at=now(), attempts=[], monitors=[]
    )
    state["calibration_attempt_id"] = args.attempt_id
    if state.get("workflow_id") != args.workflow_id or state.get("model") != args.model:
        raise ValueError("workflow state provenance mismatch")

    status = stable_job_state(args.job_id)
    summary_path = ACL_ROOT / "calibration" / args.model / "seed_61791" / args.attempt_id / "calibration_summary.json"
    summary = None
    summary_parse_error = None
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception as exc:  # Preserve a malformed artifact and report it.
            summary_parse_error = f"{type(exc).__name__}: {exc}"
    validation_errors = validate(summary, args.model, args.attempt_id)
    if summary_parse_error:
        validation_errors.append("summary JSON parse error: " + summary_parse_error)

    out_log = LOG_ROOT / f"pcacl_calibrate_{args.job_id}.out"
    err_log = LOG_ROOT / f"pcacl_calibrate_{args.job_id}.err"
    attempt_record = dict(attempt_number=args.attempt_number, attempt_id=args.attempt_id,
                          job_id=args.job_id, observed_at=now(), slurm=status,
                          summary_path=str(summary_path) if summary_path.exists() else None,
                          validation_errors=validation_errors,
                          stdout_tail=tail(out_log), stderr_tail=tail(err_log))
    if not any(row.get("job_id") == args.job_id for row in state["attempts"]):
        state["attempts"].append(attempt_record)
    monitor_id = os.environ.get("SLURM_JOB_ID", "local")
    if not any(row.get("job_id") == monitor_id for row in state["monitors"]):
        state["monitors"].append(dict(job_id=monitor_id, observed_at=now(), attempt_number=args.attempt_number))

    completed = status["state"] == "COMPLETED" and status.get("exit_code") == "0:0" and not validation_errors
    retry_job = None
    followup_monitor = None
    submission_error = None
    capacity = None
    if completed:
        workflow_status = "calibration_ready_for_review"
        gate_eligible = bool((summary or {}).get("calibration_gate", {}).get("eligible"))
        next_action = (
            "review the calibration table; selected strengths satisfy the prespecified constraints, but no 184-step job was submitted"
            if gate_eligible else
            "review the calibration table; no tested strengths satisfied all prespecified constraints, and no 184-step job was submitted"
        )
    elif retryable(status) and args.attempt_number < args.max_retries:
        state.setdefault("retry_capacity_snapshots", []).append(dict(observed_at=now(), snapshot=resource_snapshot()))
        capacity = storage_status()
        if not capacity["ok"]:
            workflow_status = "blocked_by_storage"
            next_action = "a transient failure was found, but storage limits block an automatic retry"
        else:
            next_attempt = args.attempt_number + 1
            retry_attempt_id = f"{args.workflow_id}_a{next_attempt}"
            try:
                retry_job = submit_calibration(args.model, retry_attempt_id)
                state.setdefault("scheduled_retries", []).append(dict(attempt_number=next_attempt,
                                                                        attempt_id=retry_attempt_id,
                                                                        job_id=retry_job,
                                                                        submitted_at=now()))
                # Write the retry ID before asking Slurm to schedule the monitor;
                # this leaves a durable recovery record if monitor submission fails.
                atomic_json(state_path, state)
                followup_monitor = submit_followup_monitor(args.model, args.workflow_id,
                                                           retry_attempt_id, next_attempt,
                                                           retry_job, args.max_retries)
                workflow_status = "retry_scheduled"
                next_action = "wait for the dependent monitor; each retry uses a fresh output and metrics path"
            except Exception as exc:
                submission_error = f"{type(exc).__name__}: {exc}"
                workflow_status = "retry_submission_error"
                next_action = "inspect workflow_state.json; a retry may have been submitted without its monitor"
    else:
        workflow_status = "failed_requires_diagnosis"
        if retryable(status) and args.attempt_number >= args.max_retries:
            workflow_status = "transient_retries_exhausted"
        next_action = "inspect the recorded Slurm state and log tails; application failures are not replayed automatically"

    state["status"] = workflow_status
    state["updated_at"] = now()
    state["last_status"] = status
    state["submission_error"] = submission_error
    if capacity is not None:
        state["last_capacity_check"] = capacity
    report = dict(model=args.model, workflow_id=args.workflow_id,
                  attempt_id=args.attempt_id, attempt_number=args.attempt_number,
                  calibration_job=args.job_id, monitor_job=monitor_id,
                  observed_at=now(), slurm=status,
                  calibration_artifact=str(summary_path) if summary_path.is_file() else None,
                  calibration_job_success=completed,
                  calibration_artifact_valid=not validation_errors,
                  validation_errors=validation_errors,
                  lambda_d_star=(summary or {}).get("lambda_d_star"),
                  lambda_p_star=(summary or {}).get("lambda_p_star"),
                  calibration_gate=(summary or {}).get("calibration_gate"),
                  direct_selection_failure=(summary or {}).get("direct_selection_failure"),
                  persona_selection_failure=(summary or {}).get("selection_failure"),
                  workflow_status=workflow_status, retry_job=retry_job,
                  storage_status=capacity,
                  followup_monitor_job=followup_monitor, submission_error=submission_error,
                  next_action=next_action,
                  compute_estimate=compute_estimate(status, summary, completed))
    status_path = workflow_root / f"monitor_status_a{args.attempt_number}_job{monitor_id}.json"
    markdown_path = workflow_root / f"calibration_attempt_a{args.attempt_number}_job{monitor_id}.md"
    report["markdown_report"] = str(markdown_path)
    state["last_report"] = report
    atomic_text(markdown_path, calibration_markdown(
        report, summary, summary_path, out_log, err_log,
    ))
    atomic_json(status_path, report)
    atomic_json(state_path, state)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
