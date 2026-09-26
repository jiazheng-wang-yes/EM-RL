"""Submit and monitor the six immutable Step 2 causal-route jobs.

The afterany monitor records success and failure alike, analyzes every complete
route artifact set, and never overwrites an incomplete attempt.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from acl_common import ACL_ROOT, MODELS, ROOT
from stage6b_acl_calibration_monitor import (
    LOG_ROOT, atomic_json, now, stable_job_state, storage_status, tail,
)

HERE = ROOT / "experiments" / "persona_control" / "stage6"
RENDERINGS = ("training", "legacy")
PHASES = ("P0", "P1", "P2", "P3", "P4", "P5")


def tag(run_id: str, rendering: str) -> str:
    return f"_acl_{rendering}_{run_id}"


def job_name(model: str, rendering: str) -> str:
    return f"pcacl_s2_{model}_{rendering}"


def route_dir(run_id: str, model: str, rendering: str) -> Path:
    return ACL_ROOT / "step2" / run_id / "stage5b" / f"{model}{tag(run_id, rendering)}"


def surface_dir(run_id: str, model: str, rendering: str) -> Path:
    return ACL_ROOT / "step2_surface" / run_id / model / rendering


def workflow_dir(run_id: str) -> Path:
    return ACL_ROOT / "step2_workflows" / run_id


def validate_route(run_id: str, model: str, rendering: str) -> dict:
    """Check full immutable route outputs separately from the scheduler result."""
    directory = route_dir(run_id, model, rendering)
    problems = []
    manifest = None
    baseline = None
    manifest_path = directory / "manifest.json"
    baseline_path = directory / "baseline.json"
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text())
        except Exception as exc:
            problems.append(f"manifest JSON parse error: {type(exc).__name__}: {exc}")
    else:
        problems.append("route manifest is missing")
    if baseline_path.is_file():
        try:
            baseline = json.loads(baseline_path.read_text())
        except Exception as exc:
            problems.append(f"baseline JSON parse error: {type(exc).__name__}: {exc}")
    else:
        problems.append("baseline JSON is missing")
    if isinstance(manifest, dict):
        if manifest.get("model") != model:
            problems.append("route manifest model mismatch")
        if manifest.get("rendering") != rendering:
            problems.append("route manifest rendering mismatch")
        if manifest.get("route_tag") != tag(run_id, rendering):
            problems.append("route manifest tag mismatch")
        if manifest.get("protected_matrices_only") is not True:
            problems.append("route did not attest the protected-matrices-only graft")
        if manifest.get("quick") != 0:
            problems.append("route was not run on the full frozen pair set")
    if isinstance(baseline, dict) and not {"S_C", "S_G", "TE", "DE", "MF"}.issubset(baseline):
        problems.append("baseline is missing one or more causal route measures")

    phase_files = {}
    for phase in PHASES:
        path = directory / f"rows_{phase}.parquet"
        phase_files[phase] = path.is_file() and path.stat().st_size > 0
        if not phase_files[phase]:
            problems.append(f"{phase} route rows are missing or empty")

    audit_path = directory / "AUDIT_STOP.json"
    audit_stop = None
    if audit_path.is_file():
        try:
            saved_stop = json.loads(audit_path.read_text())
            audit_stop = dict(path=str(audit_path),
                              reasons=saved_stop.get("reasons", []) if isinstance(saved_stop, dict) else [])
        except Exception as exc:
            problems.append(f"audit-stop JSON parse error: {type(exc).__name__}: {exc}")

    return dict(
        path=str(directory), complete=not problems,
        validation_errors=problems,
        manifest_path=str(manifest_path) if manifest_path.is_file() else None,
        baseline_path=str(baseline_path) if baseline_path.is_file() else None,
        phase_files=phase_files,
        audit_stop=audit_stop,
    )


def validate_surface(run_id: str, model: str, rendering: str) -> dict:
    directory = surface_dir(run_id, model, rendering)
    problems = []
    parquet = directory / "per_completion.parquet"
    prompt_csv = directory / "per_prompt.csv"
    summary_csv = directory / "summary.csv"
    for path in (parquet, prompt_csv, summary_csv):
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"{path.name} is missing or empty")

    prompt_rows = None
    summary_rows = None
    if prompt_csv.is_file() and prompt_csv.stat().st_size:
        try:
            with prompt_csv.open(newline="") as handle:
                prompt_rows = sum(1 for _ in csv.DictReader(handle))
            if prompt_rows != 300:
                problems.append(f"strict-50 prompt table has {prompt_rows} rows, expected 300")
        except Exception as exc:
            problems.append(f"prompt CSV parse error: {type(exc).__name__}: {exc}")
    if summary_csv.is_file() and summary_csv.stat().st_size:
        try:
            with summary_csv.open(newline="") as handle:
                summary_rows = sum(1 for _ in csv.DictReader(handle))
            if summary_rows != 6:
                problems.append(f"surface summary has {summary_rows} rows, expected 6")
        except Exception as exc:
            problems.append(f"summary CSV parse error: {type(exc).__name__}: {exc}")
    return dict(
        path=str(directory), complete=not problems, validation_errors=problems,
        per_completion_path=str(parquet) if parquet.is_file() else None,
        per_prompt_rows=prompt_rows, summary_rows=summary_rows,
    )


def run_analysis(run_id: str, rendering: str, models: list[str]) -> dict:
    if not models:
        return dict(rendering=rendering, status="skipped_no_complete_routes", models=[])
    output_root = ACL_ROOT / "step2" / run_id
    derived_names = (
        "summary.json", "per_example_pairs.parquet", "per_example_neutral.parquet",
        "quality.csv", "R_layers.csv", "components.csv", "decomposition.csv",
        "R_layers_strict50.csv", "components_strict50.csv", "decomposition_strict50.csv",
        "energy_fractions.csv",
    )
    expected = [route_dir(run_id, model, rendering) / name
                for model in models for name in derived_names]
    cross = output_root / "stage5b" / f"cross_model_summary{tag(run_id, rendering)}.csv"
    if len(models) > 1:
        expected.append(cross)
    existing = [path for path in expected if path.exists()]
    if existing:
        missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
        status = "already_complete_preserved" if not missing else "preexisting_partial_not_overwritten"
        return dict(rendering=rendering, models=models, status=status,
                    existing_artifacts=[str(path) for path in existing],
                    missing_analysis_artifacts=missing,
                    cross_model_summary=str(cross) if cross.is_file() else None)
    command = [
        sys.executable, str(HERE / "stage5b_analyze.py"),
        "--models", ",".join(models), "--tag", tag(run_id, rendering),
        "--eval-dir", str(output_root),
    ]
    proc = subprocess.run(command, cwd=HERE, text=True, capture_output=True, check=False)
    missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
    success = proc.returncode == 0 and not missing
    return dict(
        rendering=rendering, models=models, command=command,
        returncode=proc.returncode, status="completed" if success else "failed",
        missing_analysis_artifacts=missing,
        stdout_tail="\n".join(proc.stdout.splitlines()[-80:]),
        stderr_tail="\n".join(proc.stderr.splitlines()[-80:]),
        cross_model_summary=str(cross) if cross.is_file() else None,
    )


def monitor(run_id: str, tasks: list[dict]) -> dict:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError(f"invalid run ID {run_id!r}")
    if not tasks:
        raise ValueError("at least one submitted Step 2 task is required")
    expected = {(model, rendering) for model in MODELS for rendering in RENDERINGS}
    seen = set()
    task_results = []
    route_complete_by_rendering = {rendering: [] for rendering in RENDERINGS}

    for task in tasks:
        model, rendering, job_id = task["model"], task["rendering"], str(task["job_id"])
        key = (model, rendering)
        if model not in MODELS or rendering not in RENDERINGS or key in seen:
            raise ValueError(f"invalid or duplicate task: {task}")
        if not job_id.isdigit():
            raise ValueError(f"invalid Slurm job ID: {job_id!r}")
        seen.add(key)
        slurm = stable_job_state(job_id)
        route = validate_route(run_id, model, rendering)
        surface = validate_surface(run_id, model, rendering)
        if route["complete"]:
            route_complete_by_rendering[rendering].append(model)
        name = job_name(model, rendering)
        task_results.append(dict(
            model=model, rendering=rendering, job_id=job_id, slurm=slurm,
            route=route, surface=surface,
            stdout_path=str(LOG_ROOT / f"{name}_{job_id}.out"),
            stderr_path=str(LOG_ROOT / f"{name}_{job_id}.err"),
            stdout_tail=tail(LOG_ROOT / f"{name}_{job_id}.out"),
            stderr_tail=tail(LOG_ROOT / f"{name}_{job_id}.err"),
            task_success=(slurm.get("state") == "COMPLETED" and slurm.get("exit_code") == "0:0"
                          and route["complete"] and surface["complete"]),
        ))

    missing_submissions = [dict(model=model, rendering=rendering, status="not_submitted")
                           for model, rendering in sorted(expected - seen)]
    analysis = []
    for rendering, models in route_complete_by_rendering.items():
        if models:
            analysis.append(run_analysis(run_id, rendering, sorted(models)))
        else:
            analysis.append(dict(rendering=rendering, status="skipped_no_complete_routes", models=[]))
    all_submitted = not missing_submissions
    all_tasks_ok = all(row["task_success"] for row in task_results)
    all_analysis_ok = all(row["status"] in {
        "completed", "already_complete_preserved", "skipped_no_complete_routes"
    } for row in analysis)
    overall = "completed" if all_submitted and all_tasks_ok and all_analysis_ok else "completed_with_failures"

    report = dict(
        run_id=run_id, monitored_at=now(), monitor_job_id=os.environ.get("SLURM_JOB_ID", "local"),
        workflow_status=overall, task_results=task_results,
        not_submitted=missing_submissions, analysis=analysis,
        next_action=("Step 2 artifacts and analyses are ready for review" if overall == "completed"
                     else "review recorded job failures and partial outputs; this monitor did not overwrite or retry them"),
    )
    root = workflow_dir(run_id)
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "monitor_report.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable Step 2 monitor report: {report_path}")
    atomic_json(report_path, report)
    workflow_path = root / "workflow.json"
    if workflow_path.is_file():
        workflow = json.loads(workflow_path.read_text())
        workflow["monitor_report"] = str(report_path)
        workflow["monitor_status"] = overall
        workflow["monitored_at"] = report["monitored_at"]
        atomic_json(workflow_path, workflow)
    print(json.dumps(report, indent=2))
    return report


def parse_job_id(proc: subprocess.CompletedProcess) -> str:
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:] or "sbatch returned a nonzero status")
    job_id = proc.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"could not parse sbatch job ID from {proc.stdout!r}")
    return job_id


def submit(run_id: str | None = None) -> dict:
    """Submit all available jobs plus one afterany result/analysis monitor."""
    capacity = storage_status()
    if not capacity.get("ok"):
        result = dict(status="blocked_by_storage", storage_status=capacity,
                      jobs_submitted=[], monitor_job_id=None)
        print(json.dumps(result, indent=2))
        return result

    if run_id is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"acl_step2_{stamp}_{os.getpid()}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError(f"invalid run ID {run_id!r}")
    root = workflow_dir(run_id)
    output_root = ACL_ROOT / "step2" / run_id
    surface_root = ACL_ROOT / "step2_surface" / run_id
    if root.exists() or output_root.exists() or surface_root.exists():
        raise FileExistsError(f"refusing to reuse immutable Step 2 run ID {run_id}")

    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    root.mkdir(parents=True)
    expected = [dict(model=model, rendering=rendering)
                for model in MODELS for rendering in RENDERINGS]
    submitted, errors = [], []
    route_script = HERE / "stage6b_step2_route.sbatch"
    for item in expected:
        model, rendering = item["model"], item["rendering"]
        name = job_name(model, rendering)
        command = [
            "sbatch", "--parsable", f"--job-name={name}",
            f"--export=ALL,MODEL={model},RENDERING={rendering},RUN_ID={run_id}",
            str(route_script),
        ]
        proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        try:
            job_id = parse_job_id(proc)
        except Exception as exc:
            errors.append(dict(model=model, rendering=rendering, command=command,
                               error=f"{type(exc).__name__}: {exc}"))
            continue
        submitted.append(dict(model=model, rendering=rendering, job_id=job_id,
                              command=command, job_name=name))

    workflow = dict(
        run_id=run_id, created_at=now(), expected_tasks=expected,
        jobs=submitted, submission_errors=errors, storage_status=capacity,
        output_root=str(output_root), surface_root=str(surface_root),
    )
    atomic_json(root / "workflow.json", workflow)
    if not submitted:
        workflow["status"] = "submission_failed"
        atomic_json(root / "workflow.json", workflow)
        result = dict(status="submission_failed", workflow_path=str(root / "workflow.json"),
                      jobs_submitted=[], monitor_job_id=None, errors=errors)
        print(json.dumps(result, indent=2))
        return result

    dependency = ":".join(row["job_id"] for row in submitted)
    command = ["sbatch", "--parsable", f"--dependency=afterany:{dependency}",
               "--job-name=pcacl_s2_watch", str(HERE / "stage6b_step2_monitor.sbatch"),
               "--run-id", run_id]
    for row in submitted:
        command.extend(["--task", row["model"], row["rendering"], row["job_id"]])
    monitor_proc = None
    monitor_error = None
    for attempt in range(3):
        monitor_proc = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False)
        try:
            monitor_job_id = parse_job_id(monitor_proc)
            monitor_error = None
            break
        except Exception as exc:
            monitor_error = f"{type(exc).__name__}: {exc}"
            if attempt < 2:
                import time
                time.sleep(2)
    else:
        monitor_job_id = None
    workflow["monitor_job_id"] = monitor_job_id
    workflow["monitor_submission_error"] = monitor_error
    workflow["status"] = "monitor_submitted" if monitor_job_id else "monitor_submission_failed"
    atomic_json(root / "workflow.json", workflow)
    result = dict(status=workflow["status"], run_id=run_id,
                  workflow_path=str(root / "workflow.json"), jobs_submitted=submitted,
                  submission_errors=errors, monitor_job_id=monitor_job_id,
                  monitor_command=command if not monitor_job_id else None,
                  monitor_submission_error=monitor_error)
    print(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submit", action="store_true", help="submit the six jobs and afterany monitor")
    parser.add_argument("--run-id")
    parser.add_argument("--task", nargs=3, action="append", default=[],
                        metavar=("MODEL", "RENDERING", "JOB_ID"))
    args = parser.parse_args()
    if args.submit:
        if args.task:
            parser.error("--task is only used by the afterany monitor")
        result = submit(args.run_id)
        if result.get("status") == "blocked_by_storage":
            raise SystemExit(3)
        if result.get("status") in {"submission_failed", "monitor_submission_failed"}:
            raise SystemExit(4)
    else:
        if not args.run_id:
            parser.error("--run-id is required for monitoring")
        tasks = [dict(model=model, rendering=rendering, job_id=job_id)
                 for model, rendering, job_id in args.task]
        monitor(args.run_id, tasks)


if __name__ == "__main__":
    main()
