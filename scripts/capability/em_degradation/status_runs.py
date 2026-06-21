#!/usr/bin/env python3
"""Print compact status for EM degradation runs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/net/scratch/jiaweizhang/jiazhengw_migration")
DEFAULT_RUN_ROOT = DEFAULT_ROOT / "eval_runs/em_degradation"
DEFAULT_LOG_ROOT = DEFAULT_ROOT / "logs/capability/em_degradation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--job-id", default="")
    parser.add_argument("--latest", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def human_size(path: Path) -> str:
    if not path.exists():
        return "missing"
    try:
        output = subprocess.check_output(["du", "-sh", str(path)], text=True).split()[0]
    except Exception:
        return "unknown"
    return output


def slurm_state(job_id: str) -> str:
    if not job_id:
        return ""
    try:
        output = subprocess.check_output(
            ["squeue", "-h", "-j", job_id, "-o", "%T %M %R"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        output = ""
    if output:
        return output
    try:
        output = subprocess.check_output(
            ["sacct", "--parsable2", "-j", job_id, "--format=State,ExitCode,Elapsed"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()
    except Exception:
        return "not found"
    return output[1] if len(output) > 1 else "not found"


def parse_log(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(errors="ignore")
    steps = [
        (int(match.group(1)), float(match.group(2)))
        for match in re.finditer(r"step:(\d+) - train/loss:([0-9.]+)", text)
    ]
    status: dict[str, Any] = {
        "log_path": str(log_path),
        "total_steps": 1101,
        "lm_eval_markers": text.count("========== lm-eval:"),
        "run_complete_markers": text.count("Run complete:"),
        "cleanup_markers": text.count("Removing checkpoint directory"),
        "tracebacks": text.count("Traceback"),
        "cuda_oom": text.count("CUDA out of memory"),
        "no_space": text.count("No space left"),
    }
    total_match = re.search(r"Total steps: (\d+)", text)
    if total_match:
        status["total_steps"] = int(total_match.group(1))
    if steps:
        status["latest_step"] = steps[-1][0]
        status["latest_loss"] = steps[-1][1]
        status["last10_loss"] = sum(loss for _, loss in steps[-10:]) / min(10, len(steps))
        total_steps = int(status["total_steps"])
        status["remaining_steps"] = max(total_steps - steps[-1][0], 0)
        status["percent_complete"] = steps[-1][0] / total_steps if total_steps else 0.0
    return status


def merge_log_status(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    if not primary:
        primary = {}
    merged = dict(primary)
    for key in ["lm_eval_markers", "run_complete_markers", "cleanup_markers", "tracebacks", "cuda_oom", "no_space"]:
        merged[key] = int(primary.get(key, 0) or 0) + int(secondary.get(key, 0) or 0)
    if secondary.get("log_path"):
        merged["stderr_log_path"] = secondary["log_path"]
    return merged


def run_sort_key(path: Path) -> tuple[str, float]:
    manifest = load_json(path / "manifest.json")
    run_id = str(manifest.get("run_id", path.name))
    timestamp_match = re.search(r"_(\d{8}_\d{6})$", run_id)
    if timestamp_match:
        return (timestamp_match.group(1), (path / "manifest.json").stat().st_mtime if (path / "manifest.json").exists() else path.stat().st_mtime)
    return ("", (path / "manifest.json").stat().st_mtime if (path / "manifest.json").exists() else path.stat().st_mtime)


def manifest_is_active(path: Path) -> bool:
    manifest = load_json(path / "manifest.json")
    status = str(manifest.get("status", "")).lower()
    stage = str(manifest.get("stage", "")).lower()
    if status in {"completed", "failed", "cancelled", "canceled"} or stage == "done":
        return False
    return status in {"running", "pending"} or stage in {"train", "eval", "lm_eval", "countdown"}


def candidate_runs(run_root: Path, latest: bool) -> list[Path]:
    runs = sorted([path for path in run_root.iterdir() if path.is_dir()], key=run_sort_key)
    if latest and runs:
        active_runs = [path for path in runs if manifest_is_active(path)]
        return [active_runs[-1] if active_runs else runs[-1]]
    return runs


def main() -> None:
    args = parse_args()
    for run_dir in candidate_runs(args.run_root, args.latest):
        manifest = load_json(run_dir / "manifest.json")
        job_id = args.job_id or str(manifest.get("slurm_job_id", ""))
        log_status = {}
        if job_id:
            log_status = merge_log_status(
                parse_log(args.log_root / f"em_deg_one_{job_id}.out"),
                parse_log(args.log_root / f"em_deg_one_{job_id}.err"),
            )
        checkpoint_dir = Path(str(manifest.get("checkpoint_dir", ""))) if manifest.get("checkpoint_dir") else Path()
        summary_path = run_dir / "summary.json"

        print(f"run_id: {manifest.get('run_id', run_dir.name)}")
        print(f"  model_key: {manifest.get('model_key', '')}")
        print(f"  recipe: {manifest.get('recipe', '')}")
        print(f"  stage/status: {manifest.get('stage', '')}/{manifest.get('status', '')}")
        print(f"  slurm: {job_id} {slurm_state(job_id)}")
        total_steps = log_status.get("total_steps", 1101)
        print(f"  latest_step: {log_status.get('latest_step', '')}/{total_steps}")
        if "percent_complete" in log_status:
            print(f"  train_complete: {100.0 * log_status['percent_complete']:.1f}%")
            print(f"  remaining_steps: {log_status['remaining_steps']}")
        if "latest_loss" in log_status:
            print(f"  latest_loss: {log_status['latest_loss']:.6f}")
            print(f"  last10_loss: {log_status['last10_loss']:.6f}")
        print(f"  lm_eval_markers: {log_status.get('lm_eval_markers', 0)}")
        print(f"  run_complete_markers: {log_status.get('run_complete_markers', 0)}")
        print(f"  error_markers: traceback={log_status.get('tracebacks', 0)} cuda_oom={log_status.get('cuda_oom', 0)} no_space={log_status.get('no_space', 0)}")
        print(f"  summary: {'present' if summary_path.exists() else 'missing'}")
        print(f"  checkpoint_size: {human_size(checkpoint_dir) if checkpoint_dir else 'missing'}")


if __name__ == "__main__":
    main()
