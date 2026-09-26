"""Submit GPU preflight with an attached afterany validator."""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
from pathlib import Path

from acl_common import ACL_ROOT, CALIBRATION_MODEL, ROOT
from stage6b_acl_calibration_monitor import atomic_json, now, parse_job_id, submit_phase_monitor

HERE = Path(__file__).resolve().parent
RUN_ID = "20260924T063649Z"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workflow-id", required=True)
    parser.add_argument("--calibration-attempt-id", required=True)
    parser.add_argument("--version", default="implementation_v5")
    args = parser.parse_args()
    workflow = ACL_ROOT / "calibration" / CALIBRATION_MODEL / "seed_61791" / "workflows" / args.workflow_id
    state_path = workflow / "workflow_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(f"calibration workflow state missing: {state_path}")
    lock_path = workflow / "stage6b_preflight_submit.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(state_path.read_text())
        if state.get("model") != CALIBRATION_MODEL or state.get("workflow_id") != args.workflow_id:
            raise ValueError("workflow state provenance mismatch")
        if state.get("calibration_attempt_id") != args.calibration_attempt_id:
            raise ValueError("calibration attempt differs from the validated workflow state")
        if state.get("preflight_job"):
            raise FileExistsError("this workflow already records a preflight job")
        command = ["sbatch", "--parsable", "--exclude=m001", str(HERE / "stage6b_preflight.sbatch"),
                   "--model", CALIBRATION_MODEL, "--seed", "61791", "--run-id", RUN_ID,
                   "--version", args.version]
        preflight_job = parse_job_id(subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=False))
        state["status"] = "preflight_pending"
        state["preflight_job"] = preflight_job
        state["preflight_submitted_at"] = now()
        atomic_json(state_path, state)
        try:
            monitor_job = submit_phase_monitor(
                "stage6b_preflight_monitor.sbatch", CALIBRATION_MODEL, args.workflow_id,
                preflight_job, ["--calibration-attempt-id", args.calibration_attempt_id,
                                "--version", args.version],
            )
        except Exception:
            state["status"] = "preflight_monitor_submission_error"
            state["preflight_monitor_error"] = "afterany monitor submission failed; preserve and recover the recorded job"
            atomic_json(state_path, state)
            raise
        state["preflight_monitor_job"] = monitor_job
        state["preflight_monitor_submitted_at"] = now()
        atomic_json(state_path, state)
        print(json.dumps({"run_id": RUN_ID, "version": args.version, "preflight_job": preflight_job,
                          "afterany_monitor_job": monitor_job,
                          "state": str(state_path)}, indent=2))


if __name__ == "__main__":
    main()
