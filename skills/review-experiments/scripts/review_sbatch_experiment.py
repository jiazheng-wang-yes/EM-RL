#!/usr/bin/env python3
"""Inspect Slurm job state, logs, checkpoints, and result artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


SACCT_FIELDS = [
    "JobID",
    "JobName",
    "State",
    "ExitCode",
    "Elapsed",
    "Submit",
    "Start",
    "End",
    "NodeList",
    "WorkDir",
]

ERROR_RE = re.compile(
    r"error|exception|traceback|oom|out of memory|cuda.*memory|segmentation fault|"
    r"timeout|time limit|cancelled|failed|node_fail|cannot|no such file|permission denied",
    re.IGNORECASE,
)

METRIC_NAME_RE = re.compile(
    r"(eval|metric|metrics|result|results|summary|report|trainer_state|probe).*\.jsonl?$",
    re.IGNORECASE,
)

CHECKPOINT_RE = re.compile(r"(checkpoint|global_step|epoch|actor|adapter|model)", re.IGNORECASE)

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".agent-config",
    ".cache",
    ".venv",
    "__pycache__",
    "node_modules",
    "site-packages",
}


def run_cmd(args: list[str], timeout: int = 20) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(args, check=False, text=True, capture_output=True, timeout=timeout)
    except FileNotFoundError:
        return 127, "", f"{args[0]} not found"
    except subprocess.TimeoutExpired:
        return 124, "", f"{args[0]} timed out after {timeout}s"
    return proc.returncode, proc.stdout, proc.stderr


def compact_message(message: str, limit: int = 2000) -> str:
    text = message.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "... <truncated>"


def parse_pipe_table(text: str, fields: list[str]) -> list[dict[str, str]]:
    rows = []
    for raw in text.splitlines():
        if not raw.strip():
            continue
        parts = raw.rstrip("\n").split("|")
        if len(parts) < len(fields):
            parts.extend([""] * (len(fields) - len(parts)))
        rows.append(dict(zip(fields, parts)))
    return rows


def parse_scontrol(text: str) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for key, quoted, bare in re.findall(r"(\w+)=(?:\"([^\"]*)\"|(\S+))", text):
        parsed[key] = quoted or bare
    return parsed


def sacct(job_ids: list[str], recent_days: int, timeout: int) -> list[dict[str, str]]:
    cmd = ["sacct", "-P", "-n", "-o", ",".join(SACCT_FIELDS)]
    if job_ids:
        cmd.extend(["-j", ",".join(job_ids)])
    else:
        since = (datetime.now() - timedelta(days=recent_days)).strftime("%Y-%m-%d")
        cmd.extend(["-u", os.environ.get("USER", ""), "-S", since])
    code, out, err = run_cmd(cmd, timeout=timeout)
    if code != 0:
        return [{"error": compact_message(err or out or f"sacct exited {code}")}]
    return parse_pipe_table(out, SACCT_FIELDS)


def squeue(job_ids: list[str], timeout: int) -> list[dict[str, str]]:
    fields = ["JobID", "JobName", "State", "Reason", "TimeUsed", "Nodes", "NodeList"]
    cmd = ["squeue", "-h", "-o", "%i|%j|%T|%r|%M|%D|%R"]
    if job_ids:
        cmd.extend(["-j", ",".join(job_ids)])
    else:
        cmd.extend(["-u", os.environ.get("USER", "")])
    code, out, err = run_cmd(cmd, timeout=timeout)
    if code != 0:
        return [{"error": compact_message(err or out or f"squeue exited {code}")}]
    return parse_pipe_table(out, fields)


def scontrol(job_id: str, timeout: int) -> dict[str, str]:
    code, out, err = run_cmd(["scontrol", "show", "job", "-o", job_id], timeout=timeout)
    if code != 0:
        return {"error": compact_message(err or out or f"scontrol exited {code}")}
    return parse_scontrol(out)


def clean_job_id(job_id: str) -> str:
    return job_id.split(".")[0].split("_")[0]


def expand_slurm_path(path_text: str, job_id: str, job_name: str) -> Path | None:
    if not path_text or path_text == "/dev/null":
        return None
    expanded = (
        path_text.replace("%j", job_id)
        .replace("%A", job_id)
        .replace("%x", job_name)
        .replace("%N", "*")
    )
    return Path(os.path.expandvars(os.path.expanduser(expanded)))


def likely_roots(root_args: list[str]) -> list[Path]:
    roots: list[Path] = []
    base_candidates = [Path(p).expanduser().resolve() for p in root_args]
    if not base_candidates:
        base_candidates = [Path.cwd()]
    for base in base_candidates:
        for child in ["logs", "checkpoints", "outputs", "runs", "wandb", "."]:
            candidate = base / child if child != "." else base
            if candidate.exists() and candidate.is_dir() and candidate not in roots:
                roots.append(candidate)
    return roots


def walk_files(roots: list[Path], limit: int) -> list[Path]:
    files: list[Path] = []
    for root in roots:
        if len(files) >= limit:
            break
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for filename in filenames:
                files.append(Path(dirpath) / filename)
                if len(files) >= limit:
                    break
            if len(files) >= limit:
                break
    return files


def file_matches(path: Path, terms: set[str]) -> bool:
    name = str(path)
    return any(term and term in name for term in terms)


def read_tail(path: Path, max_bytes: int) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read()
        return data.decode("utf-8", errors="replace")
    except OSError as exc:
        return f"<could not read {path}: {exc}>"


def extract_errors(text: str, limit: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if ERROR_RE.search(line)]
    return lines[-limit:]


def extract_paths(text: str) -> list[Path]:
    found = []
    for raw in re.findall(r"(/[^\s'\"<>]+)", text):
        trimmed = raw.rstrip("),.;]")
        if any(part in trimmed for part in ["/logs/", "/checkpoints/", "/outputs/", "/runs/", "/wandb/"]):
            found.append(Path(trimmed))
    return found


def summarize_json(path: Path, max_scalars: int) -> dict[str, Any] | None:
    try:
        if path.suffix == ".jsonl":
            lines = [line for line in path.read_text(errors="replace").splitlines() if line.strip()]
            if not lines:
                return None
            data = json.loads(lines[-1])
        else:
            data = json.loads(path.read_text(errors="replace"))
    except Exception:
        return None

    scalars: dict[str, Any] = {}

    def visit(prefix: str, value: Any) -> None:
        if len(scalars) >= max_scalars:
            return
        if isinstance(value, (str, int, float, bool)) or value is None:
            scalars[prefix or "value"] = value
        elif isinstance(value, dict):
            for key, child in value.items():
                visit(f"{prefix}.{key}" if prefix else str(key), child)
                if len(scalars) >= max_scalars:
                    break

    visit("", data)
    return scalars


def inspect_job(job_id: str, args: argparse.Namespace, all_files: list[Path]) -> dict[str, Any]:
    base_id = clean_job_id(job_id)
    active = scontrol(base_id, args.slurm_timeout)
    job_name = active.get("JobName", "")
    terms = {base_id, job_id, job_name}

    explicit_logs: list[Path] = []
    for key in ["StdOut", "StdErr"]:
        path = expand_slurm_path(active.get(key, ""), base_id, job_name)
        if path is not None:
            explicit_logs.append(path)

    matched = [path for path in all_files if file_matches(path, terms)]
    log_files = []
    for path in explicit_logs + matched:
        suffix = path.suffix.lower()
        if suffix in {".out", ".err", ".log", ".txt"} or "log" in str(path).lower():
            if path.exists() and path not in log_files:
                log_files.append(path)

    log_summaries = []
    discovered_paths: list[Path] = []
    for path in log_files[: args.max_logs]:
        tail = read_tail(path, args.max_log_bytes)
        discovered_paths.extend(extract_paths(tail))
        log_summaries.append(
            {
                "path": str(path),
                "size": path.stat().st_size if path.exists() else None,
                "tail": tail[-args.tail_chars :],
                "error_lines": extract_errors(tail, args.max_errors),
            }
        )

    artifact_candidates = list(matched)
    for path in discovered_paths:
        if path.exists():
            if path.is_file():
                artifact_candidates.append(path)
            elif path.is_dir():
                artifact_candidates.extend(list(path.rglob("*"))[: args.max_artifacts])

    deduped_artifacts = []
    seen = set()
    for path in artifact_candidates:
        key = str(path)
        if key not in seen and path.exists():
            seen.add(key)
            deduped_artifacts.append(path)

    checkpoint_paths = [
        path
        for path in deduped_artifacts
        if CHECKPOINT_RE.search(str(path)) and (path.is_dir() or path.suffix in {"", ".pt", ".bin", ".safetensors"})
    ][: args.max_artifacts]

    metric_files = [
        path
        for path in deduped_artifacts
        if path.is_file() and METRIC_NAME_RE.search(path.name)
    ][: args.max_metrics]

    metrics = []
    for path in metric_files:
        summary = summarize_json(path, args.max_metric_keys)
        if summary is not None:
            metrics.append({"path": str(path), "scalars": summary})

    return {
        "job_id": base_id,
        "scontrol": active,
        "logs": log_summaries,
        "checkpoints": [str(path) for path in checkpoint_paths],
        "metric_files": metrics,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Slurm Experiment Review", ""]
    if report.get("squeue"):
        lines.append("## Active Queue")
        for row in report["squeue"]:
            lines.append(f"- {row}")
        lines.append("")
    if report.get("sacct"):
        lines.append("## Accounting")
        for row in report["sacct"]:
            lines.append(f"- {row}")
        lines.append("")
    for job in report["jobs"]:
        lines.append(f"## Job {job['job_id']}")
        active = job.get("scontrol", {})
        if active.get("error"):
            lines.append(f"- scontrol error: `{active['error']}`")
        elif active:
            keep = {k: active.get(k) for k in ["JobName", "JobState", "Reason", "ExitCode", "RunTime", "StdOut", "StdErr", "WorkDir"] if active.get(k)}
            lines.append(f"- scontrol: `{json.dumps(keep, sort_keys=True)}`")
        for log in job.get("logs", []):
            lines.append(f"- log: `{log['path']}` ({log.get('size')} bytes)")
            for error in log.get("error_lines", []):
                lines.append(f"  - suspicious: `{error[:240]}`")
        if job.get("checkpoints"):
            lines.append("- checkpoints:")
            for path in job["checkpoints"][:10]:
                lines.append(f"  - `{path}`")
        if job.get("metric_files"):
            lines.append("- metrics:")
            for metric in job["metric_files"]:
                lines.append(f"  - `{metric['path']}`: `{json.dumps(metric['scalars'], sort_keys=True)[:500]}`")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", action="append", default=[], help="Slurm job ID. Repeat for multiple jobs.")
    parser.add_argument("--root", action="append", default=[], help="Root to search for logs/checkpoints/results.")
    parser.add_argument("--recent-days", type=int, default=3, help="Days to query with sacct when --job is omitted.")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--max-files", type=int, default=30000)
    parser.add_argument("--slurm-timeout", type=int, default=8, help="Seconds to wait for each Slurm CLI call.")
    parser.add_argument("--max-logs", type=int, default=8)
    parser.add_argument("--max-log-bytes", type=int, default=200000)
    parser.add_argument("--tail-chars", type=int, default=2500)
    parser.add_argument("--max-errors", type=int, default=20)
    parser.add_argument("--max-artifacts", type=int, default=50)
    parser.add_argument("--max-metrics", type=int, default=20)
    parser.add_argument("--max-metric-keys", type=int, default=60)
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    roots = likely_roots(args.root)
    all_files = walk_files(roots, args.max_files)
    accounting = sacct([clean_job_id(job) for job in args.job], args.recent_days, args.slurm_timeout)
    active_queue = squeue([clean_job_id(job) for job in args.job], args.slurm_timeout)

    job_ids = [clean_job_id(job) for job in args.job]
    if not job_ids:
        seen = set()
        for row in accounting:
            job_id = row.get("JobID", "")
            if job_id and "." not in job_id and "_" not in job_id and job_id not in seen:
                seen.add(job_id)
                job_ids.append(job_id)
        job_ids = job_ids[-10:]

    report = {
        "roots": [str(root) for root in roots],
        "squeue": active_queue,
        "sacct": accounting,
        "jobs": [inspect_job(job_id, args, all_files) for job_id in job_ids],
    }

    if args.format == "markdown":
        print(render_markdown(report))
    else:
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
