"""Afterany controller for the six rubric-corrected Step 1 judge jobs.

The controller validates immutable rejudge outputs, retries only transient
Slurm/node failures (at most ``--max-retries`` times per task), and analyzes
every complete C/E pair once the current retry chain is terminal. Raw rollout
JSONL files are never modified. Human blind review remains an explicit
checkpoint; this script does not manufacture review labels or start training.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from acl_common import ACL_ROOT, MODELS, ROOT

HERE = ROOT / "experiments" / "persona_control" / "stage6"
LOG_ROOT = ROOT / "logs" / "slurm" / "persona_control"
TASKS = [(model, condition) for model in MODELS for condition in ("C", "E")]
TRANSIENT_STATES = {"NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "REQUEUED", "REVOKED"}


def slurm_state(job_id: str) -> dict:
    proc = subprocess.run(
        ["sacct", "-X", "-j", str(job_id), "--format=State,ExitCode", "--noheader", "--parsable2"],
        text=True, capture_output=True, check=False,
    )
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        return dict(state="UNKNOWN", exit_code="unknown", accounting_error=proc.stderr[-1000:])
    value = lines[0].split("|")
    return dict(state=value[0].split()[0], exit_code=value[1] if len(value) > 1 else "unknown")


def stable_state(job_id: str) -> dict:
    result = slurm_state(job_id)
    for _ in range(3):
        if result["state"] != "UNKNOWN":
            break
        time.sleep(5)
        result = slurm_state(job_id)
    return result


def parse_task_jobs(args) -> list[tuple[str, str, str]]:
    if args.task_jobs:
        parsed = []
        for item in args.task_jobs.split(","):
            task, job_id = item.rsplit("=", 1)
            model, condition = task.split(":", 1)
            if (model, condition) not in TASKS:
                raise ValueError(f"unknown task mapping: {item}")
            parsed.append((model, condition, job_id))
        return parsed
    job_ids = [item for item in args.jobs.split(",") if item]
    if len(job_ids) != len(TASKS):
        raise ValueError("initial monitor needs six ordered job IDs or an explicit --task-jobs map")
    return [(model, condition, job_id) for (model, condition), job_id in zip(TASKS, job_ids)]


def artifact_check(input_stage: str, run_id: str, model: str, condition: str) -> tuple[bool, str | None]:
    path = ACL_ROOT / input_stage / model / run_id / condition
    expected = ("responses.parquet", "responses.csv", "summary.csv", "run.json")
    missing = [name for name in expected if not (path / name).is_file() or (path / name).stat().st_size == 0]
    if missing:
        return False, "missing_or_empty:" + ",".join(missing)
    try:
        run = json.loads((path / "run.json").read_text())
    except Exception as exc:
        return False, f"invalid_run_json:{type(exc).__name__}:{exc}"
    checks = {
        "model": model,
        "condition": condition,
        "run_id": run_id,
        "output_stage": input_stage,
        "source_completion_count": 960,
    }
    mismatches = {key: (run.get(key), expected_value) for key, expected_value in checks.items()
                  if run.get(key) != expected_value}
    if mismatches:
        return False, f"provenance_mismatch:{mismatches}"
    if len(run.get("source_rollouts", [])) != 2 or len(run.get("judges", {})) != 2:
        return False, "incomplete_source_or_judge_manifest"
    return True, None


def tail(path: Path, lines: int = 60) -> str:
    if not path.exists():
        return f"[missing log: {path}]"
    return "".join(path.read_text(errors="replace").splitlines(keepends=True)[-lines:])


def retryable(status: dict) -> bool:
    state = status.get("state", "UNKNOWN")
    if state in TRANSIENT_STATES:
        return True
    # Signal 53 was observed on a draining node in the preceding Stage 1
    # attempt. Retry this signature once the original allocation is terminal.
    return state == "FAILED" and status.get("exit_code", "").endswith(":53")


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def resource_snapshot() -> dict:
    commands = {
        "sinfo": ["sinfo", "-N", "-h", "-o", "%N|%t|%G|%m|%C|%f"],
        "squeue": ["squeue", "-p", "general", "-o", "%.18i|%.12T|%.30j|%.20b|%.30R"],
    }
    snapshot = {}
    for name, command in commands.items():
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        snapshot[name] = dict(returncode=result.returncode, stdout=result.stdout[-12000:], stderr=result.stderr[-2000:])
    return snapshot


def submit_retry(model: str, condition: str, source_run_id: str, run_id: str, input_stage: str) -> str:
    command = [
        "sbatch", "--parsable", "--exclude=m002", str(HERE / "stage6b_rejudge.sbatch"),
        "--model", model, "--condition", condition, "--source-run-id", source_run_id,
        "--run-id", run_id, "--output-stage", input_stage,
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"retry submission failed for {model}:{condition}: {result.stderr[-2000:]}")
    return result.stdout.strip().split(";")[0]


def schedule_followup(args, retry_jobs: list[tuple[str, str, str]], retry_number: int) -> str:
    ids = [job_id for _, _, job_id in retry_jobs]
    task_map = ",".join(f"{model}:{condition}={job_id}" for model, condition, job_id in retry_jobs)
    command = [
        "sbatch", "--parsable", f"--dependency=afterany:{':'.join(ids)}",
        str(HERE / "stage6b_step1_monitor.sbatch"),
        "--run-id", args.run_id, "--jobs", ",".join(ids), "--task-jobs", task_map,
        "--input-stage", args.input_stage, "--status-tag", f"retry{retry_number}",
        "--max-retries", str(args.max_retries),
    ]
    result = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"follow-up monitor submission failed: {result.stderr[-2000:]}")
    return result.stdout.strip().split(";")[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--jobs", required=True, help="comma-separated job IDs")
    parser.add_argument("--task-jobs", default="", help="comma-separated model:condition=job_id entries")
    parser.add_argument("--status-tag", default="terminal")
    parser.add_argument("--input-stage", default="step1")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="automatic retries per task for transient infrastructure failures only")
    args = parser.parse_args()
    if args.max_retries < 0:
        raise ValueError("--max-retries must be nonnegative")

    run_root = ACL_ROOT / args.input_stage / args.run_id
    run_manifest_path = run_root / "manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    source_run_id = run_manifest["source_run_id"]
    state_path = run_root / "auto_monitor_state.json"
    if state_path.exists():
        workflow = json.loads(state_path.read_text())
    else:
        workflow = dict(
            run_id=args.run_id, source_run_id=source_run_id, input_stage=args.input_stage,
            created_at=datetime.now(timezone.utc).isoformat(), task_jobs={task_key: [] for task_key in (f"{m}:{c}" for m, c in TASKS)},
            retry_count={f"{m}:{c}": 0 for m, c in TASKS}, attempts=[], monitor_jobs=[],
        )
    current_jobs = parse_task_jobs(args)
    current_status = {}
    for model, condition, job_id in current_jobs:
        task_key = f"{model}:{condition}"
        if job_id not in workflow["task_jobs"].setdefault(task_key, []):
            workflow["task_jobs"][task_key].append(job_id)
        current_status[task_key] = dict(job_id=job_id, **stable_state(job_id))
        attempt = dict(task=task_key, job_id=job_id, observed_at=datetime.now(timezone.utc).isoformat(),
                       state=current_status[task_key]["state"], exit_code=current_status[task_key]["exit_code"])
        if not any(row.get("job_id") == job_id for row in workflow["attempts"]):
            workflow["attempts"].append(attempt)
    workflow["monitor_jobs"].append(dict(job_id=os.environ.get("SLURM_JOB_ID"), status_tag=args.status_tag,
                                         observed_at=datetime.now(timezone.utc).isoformat()))

    artifacts = {}
    for model, condition in TASKS:
        ok, reason = artifact_check(args.input_stage, args.run_id, model, condition)
        artifacts[f"{model}:{condition}"] = dict(complete=ok, reason=reason)

    retries_to_schedule = []
    terminal_failures = []
    for model, condition, job_id in current_jobs:
        task_key = f"{model}:{condition}"
        status = current_status[task_key]
        status["artifact_complete"] = artifacts[task_key]["complete"]
        result_dir = ACL_ROOT / args.input_stage / model / args.run_id / condition
        status["partial_result_directory_exists"] = result_dir.exists() and not artifacts[task_key]["complete"]
        if artifacts[task_key]["complete"]:
            continue
        if (retryable(status) and not result_dir.exists()
                and workflow["retry_count"].get(task_key, 0) < args.max_retries):
            retries_to_schedule.append((model, condition, task_key))
        elif status["state"] not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
            log = LOG_ROOT / f"pcacl_rejudge_{job_id}.err"
            terminal_failures.append(dict(task=task_key, job_id=job_id, state=status["state"],
                                          exit_code=status["exit_code"], retryable=retryable(status),
                                          retries_used=workflow["retry_count"].get(task_key, 0),
                                          artifact_reason=artifacts[task_key]["reason"], stderr_tail=tail(log)))
            if result_dir.exists() and not artifacts[task_key]["complete"]:
                terminal_failures[-1]["automatic_retry_suppressed"] = (
                    "partial derived output directory exists; preserving it and the frozen rejudge script"
                )

    submitted_retries = []
    submission_error = None
    if retries_to_schedule:
        workflow.setdefault("retry_capacity_snapshots", []).append(
            dict(observed_at=datetime.now(timezone.utc).isoformat(), snapshot=resource_snapshot())
        )
        for model, condition, task_key in retries_to_schedule:
            try:
                job_id = submit_retry(model, condition, source_run_id, args.run_id, args.input_stage)
                workflow["retry_count"][task_key] = workflow["retry_count"].get(task_key, 0) + 1
                workflow["task_jobs"].setdefault(task_key, []).append(job_id)
                submitted_retries.append((model, condition, job_id))
            except Exception as exc:
                submission_error = f"{type(exc).__name__}: {exc}"
                terminal_failures.append(dict(task=task_key, submission_error=submission_error))
                break

    followup_job = None
    if submitted_retries:
        retry_number = max(workflow["retry_count"].values(), default=0)
        try:
            # Persist retry IDs before scheduling the dependent controller so
            # the next monitor always sees the complete attempt history.
            atomic_json(state_path, workflow)
            followup_job = schedule_followup(args, submitted_retries, retry_number)
        except Exception as exc:
            schedule_error = f"{type(exc).__name__}: {exc}"
            submission_error = "; ".join(item for item in (submission_error, schedule_error) if item)

    analyzed, analysis_errors, incomplete_models = [], [], []
    if not followup_job:
        for model in MODELS:
            c_ok, c_reason = artifact_check(args.input_stage, args.run_id, model, "C")
            e_ok, e_reason = artifact_check(args.input_stage, args.run_id, model, "E")
            if c_ok and e_ok:
                result = subprocess.run(
                    [sys.executable, str(HERE / "stage6b_analyze.py"), "--model", model,
                     "--run-id", args.run_id, "--input-stage", args.input_stage],
                    cwd=ROOT, text=True, capture_output=True, check=False,
                )
                if result.returncode == 0:
                    analyzed.append(model)
                else:
                    analysis_errors.append(dict(model=model, returncode=result.returncode,
                                                stdout=result.stdout[-4000:], stderr=result.stderr[-4000:]))
            else:
                incomplete_models.append(dict(model=model, C_reason=c_reason, E_reason=e_reason))

    final_failures = []
    for task_key, artifact in artifacts.items():
        if artifact["complete"]:
            continue
        if task_key not in {row.get("task") for row in terminal_failures}:
            latest = workflow["task_jobs"].get(task_key, [])[-1:]
            final_failures.append(dict(task=task_key, latest_job_ids=latest, artifact_reason=artifact["reason"],
                                       retries_used=workflow["retry_count"].get(task_key, 0)))
    report = dict(
        run_id=args.run_id, source_run_id=source_run_id, input_stage=args.input_stage,
        monitor_job=os.environ.get("SLURM_JOB_ID"), status_tag=args.status_tag,
        observed_at=datetime.now(timezone.utc).isoformat(), current_job_status=current_status,
        all_job_attempts=workflow["attempts"],
        artifacts=artifacts, retry_jobs=[dict(model=m, condition=c, job_id=j) for m, c, j in submitted_retries],
        followup_monitor_job=followup_job, submission_error=submission_error,
        terminal_failures=terminal_failures, unresolved_tasks=final_failures,
        analyzed_models=analyzed, incomplete_models=incomplete_models, analysis_errors=analysis_errors,
        next_action=(
            "wait for the chained bounded retry monitor; raw rollout JSONL files remain unchanged"
            if followup_job else
            "inspect failed tasks and logs; complete model pairs were analyzed, but no manual labels were assigned"
            if final_failures or analysis_errors else
            "Step 1 machine analysis is complete; review the blinded queues before making the final judged-EM claim"
            if len(analyzed) == len(MODELS) else
            "inspect missing model pairs; successful model pairs were analyzed and the original rollouts remain preserved"
        ),
    )
    workflow["last_report"] = report
    atomic_json(state_path, workflow)
    status_path = run_root / f"monitor_status_{args.status_tag}_job{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    atomic_json(status_path, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
