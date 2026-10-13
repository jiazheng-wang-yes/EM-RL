#!/usr/bin/env python3
"""Submit one stage at a time, reconcile Slurm GPU-hours, and stop on failed gates."""

import argparse
import fcntl
import math
import subprocess
import sys
import time

from pipeline import ARTIFACTS, ROOT, freeze_gate, read, save


def slurm_account(job_id):
    result = subprocess.run(
        [
            "sacct",
            "-X",
            "-j",
            str(job_id),
            "-n",
            "-P",
            "-o",
            "JobIDRaw,State,ElapsedRaw,AllocTRES",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == str(job_id):
            tres = dict(
                part.split("=", 1) for part in fields[3].split(",") if "=" in part
            )
            return {
                "slurm_state": fields[1],
                "allocation_seconds": int(fields[2]),
                "allocated_gpus": int(tres.get("gres/gpu", 0)),
            }
    return None


def wait_and_reconcile(name, job_id):
    while True:
        result = subprocess.run(
            ["squeue", "-h", "-j", str(job_id), "-o", "%T"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Cannot query job {job_id}: {result.stderr.strip()}")
        if not result.stdout.strip():
            break
        print(f"{name}: job {job_id} {result.stdout.strip()}", flush=True)
        time.sleep(30)
    accounting = None
    for _ in range(6):
        accounting = slurm_account(job_id)
        if accounting:
            break
        time.sleep(5)
    if accounting is None:
        raise RuntimeError(
            f"Slurm accounting unavailable for {job_id}; stop until costs are known"
        )
    path = ARTIFACTS / "stages" / f"{name}.json"
    if path.exists():
        record = read(path)
    else:
        record = {
            "name": name,
            "phase": "reserve"
            if "_retry" in name
            else "rl"
            if name.startswith("rl_")
            else "preflight_sft",
            "status": "failed",
            "error": "Job exited before pipeline initialization",
        }
    record["worker_gpu_hours"] = record.get("gpu_hours", 0)
    record.update(accounting)
    record["gpu_hours"] = (
        accounting["allocation_seconds"] * accounting["allocated_gpus"] / 3600
    )
    record["slurm_job_id"] = str(job_id)
    if accounting["slurm_state"] != "COMPLETED" or record["status"] == "running":
        record["status"] = (
            "failed" if record["status"] != "gate_failed" else "gate_failed"
        )
    save(path, record)
    if record["status"] != "completed":
        raise RuntimeError(
            f"Stage {name} stopped: {record.get('error', accounting['slurm_state'])}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--through", choices=["preflight", "gates", "matrix"], default="matrix"
    )
    args = p.parse_args()
    (ROOT / "logs/slurm/targeted_transfer").mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with (ARTIFACTS / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = [("preflight", ["preflight"])]
        if args.through != "preflight":
            plan += [
                (f"sft_{arm}", ["sft", "--arm", arm])
                for arm in ("finance", "bad_medical", "good_medical")
            ]
            plan += [
                (f"discovery_{arm}", ["discovery", "--arm", arm])
                for arm in ("base", "finance", "bad_medical", "good_medical")
            ]
        if args.through == "matrix":
            plan += [
                (f"rl_{index:02d}", ["rl", "--index", str(index)])
                for index in range(24)
            ]
        for name, command in plan:
            if name == "rl_00":
                if (ARTIFACTS / "gate.json").exists():
                    gate = read(ARTIFACTS / "gate.json")
                else:
                    gate = freeze_gate()
                if not gate["passed"]:
                    raise RuntimeError(
                        "Keyword discovery failed. Stop this specification; no hints or new word search."
                    )
            stage_file = ARTIFACTS / "stages" / f"{name}.json"
            candidates = [
                p
                for p in (ARTIFACTS / "stages").glob("*.json")
                if read(p).get("base_name", read(p)["name"]) == name
            ]
            if candidates:
                stage_file = max(candidates, key=lambda p: read(p).get("attempt", 1))
            submission = ARTIFACTS / "submissions" / f"{name}.json"
            if not stage_file.exists() and submission.exists():
                wait_and_reconcile(name, read(submission)["job_id"])
                continue
            if stage_file.exists():
                record = read(stage_file)
                if record["status"] == "running":
                    wait_and_reconcile(record["name"], record["slurm_job_id"])
                elif record["status"] != "completed":
                    raise RuntimeError(
                        f"Existing stage {name} failed; diagnose it before any retry"
                    )
                elif "allocation_seconds" not in record:
                    wait_and_reconcile(record["name"], record["slurm_job_id"])
                continue
            phase = "rl" if name.startswith("rl_") else "preflight_sft"
            used = sum(
                read(path)["gpu_hours"]
                for path in (ARTIFACTS / "stages").glob("*.json")
                if read(path)["phase"] == phase
            )
            remaining = {"preflight_sft": 20, "rl": 65}[phase] - used
            minutes = math.floor(min(4, remaining) * 60)
            if minutes < 1:
                raise RuntimeError(
                    f"{phase} GPU-hour ceiling reached; report a partial pilot"
                )
            job = (
                subprocess.run(
                    [
                        "sbatch",
                        "--parsable",
                        f"--job-name=tt_{name}",
                        f"--time={minutes}",
                        "scripts/targeted_transfer/job.sbatch",
                        *command,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                .stdout.strip()
                .split(";")[0]
            )
            save(
                ARTIFACTS / "submissions" / f"{name}.json",
                {
                    "job_id": job,
                    "command": command,
                    "time_limit_minutes": minutes,
                    "phase_gpu_hours_remaining": remaining,
                },
            )
            print(f"Submitted {name}: {job}", flush=True)
            wait_and_reconcile(name, job)
        if args.through == "gates":
            gate = freeze_gate()
            if not gate["passed"]:
                raise RuntimeError("Keyword discovery gate failed; no RL launched")
        if args.through == "matrix":
            subprocess.run(
                [sys.executable, ROOT / "scripts/targeted_transfer/analyze.py"],
                check=True,
            )


if __name__ == "__main__":
    main()
