from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.lean_worker import verify_lean_proof
from examples.lean_prover_v1.probe_common import (
    compute_normalized_statement_hash,
    normalize_lean_row,
    validate_split_integrity,
)

SAFE_DECLARATION_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_'.]*$")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _git_commit(project: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project,
        text=True,
        capture_output=True,
        check=True,
    )
    return completed.stdout.strip()


def _run_exporter(project: Path, exporter: Path, output: Path, limit: int) -> None:
    env = dict(os.environ)
    env["RLLM_MATHLIB_EXPORT_OUTPUT"] = str(output)
    env["RLLM_MATHLIB_EXPORT_MAX"] = str(limit)
    subprocess.run(
        ["lake", "env", "lean", str(exporter)],
        cwd=project,
        env=env,
        check=True,
    )


def _load_declarations(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        name = str(raw.get("name") or "")
        module = str(raw.get("module") or "")
        formal_type = str(raw.get("type") or "").strip()
        if not SAFE_DECLARATION_RE.fullmatch(name):
            continue
        if not module.startswith("Mathlib."):
            continue
        if not (12 <= len(formal_type) <= 1200):
            continue
        if any(token in formal_type for token in ("sorry", "motive", "._@", "✝")):
            continue
        rows.append({"name": name, "module": module, "formal_type": formal_type})
    return rows


def _split_for_module(module: str) -> str:
    bucket = int(hashlib.sha256(module.encode("utf-8")).hexdigest()[:8], 16) % 10
    if bucket == 0:
        return "val_static"
    if bucket == 1:
        return "test_static"
    return "train_static"


def _declaration_to_row(declaration: dict[str, str], *, repo_commit: str) -> dict[str, Any]:
    source_name = declaration["name"]
    formal_type = declaration["formal_type"]
    digest = hashlib.sha256(f"{source_name}\n{formal_type}".encode("utf-8")).hexdigest()
    split = _split_for_module(declaration["module"])
    proof_body = f"exact {source_name}"
    row = {
        "id": f"mathlib_{split}_{digest[:20]}",
        "source": f"mathlib:{declaration['module']}",
        "repo_commit": repo_commit,
        "imports": [declaration["module"]],
        "namespace": "RLLMMathlibCorpus",
        "statement_prefix": f"theorem theorem_{digest[:20]} : {formal_type} := by",
        "initial_goal_pp": formal_type,
        "formal_type": formal_type,
        "source_theorem": source_name,
        "source_module": declaration["module"],
        "seed_id": None,
        "mutation_type": "static",
        "difficulty_band": "real_mathlib",
        "baseline_results": {"well_formed": True, "cheap_baseline_solved": False},
        "proof_certificate": {
            "source": "mathlib_reference",
            "strong_prover_solved": True,
            "proof_body": proof_body,
            "source_theorem": source_name,
        },
        "split": split,
    }
    normalized = normalize_lean_row(row, split=split)
    normalized.update(
        {
            "formal_type": formal_type,
            "source_theorem": source_name,
            "source_module": declaration["module"],
        }
    )
    return normalized


def _verify_row(
    row: dict[str, Any],
    *,
    lean_command: str,
    lean_cwd: Path,
    timeout_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    proof_body = str(row["proof_certificate"]["proof_body"])
    result = verify_lean_proof(
        row,
        proof_body,
        lean_command=lean_command,
        lean_cwd=lean_cwd,
        timeout_seconds=timeout_seconds,
    )
    return row, {
        "ok": result.ok,
        "status": result.status,
        "elapsed_s": result.elapsed_s,
        "stdout": result.stdout[-2000:],
        "stderr": result.stderr[-2000:],
    }


def build_corpus(args: argparse.Namespace) -> dict[str, Any]:
    project = Path(args.mathlib_project).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exporter = Path(args.exporter).resolve()
    declarations_path = output_dir / "mathlib_declarations.jsonl"
    if args.refresh_export or not declarations_path.exists():
        _run_exporter(project, exporter, declarations_path, args.export_max)

    repo_commit = _git_commit(project)
    declarations = _load_declarations(declarations_path)
    declarations.sort(key=lambda row: hashlib.sha256(row["name"].encode("utf-8")).hexdigest())
    candidates = [_declaration_to_row(row, repo_commit=repo_commit) for row in declarations]

    targets = {
        "train_static": args.train_size,
        "val_static": args.val_size,
        "test_static": args.test_size,
    }
    selected: dict[str, list[dict[str, Any]]] = {split: [] for split in targets}
    verification_statuses: Counter[str] = Counter()
    rejected: list[dict[str, Any]] = []

    def verify(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        return _verify_row(
            row,
            lean_command=args.lean_command,
            lean_cwd=project,
            timeout_seconds=args.timeout_seconds,
        )

    verification_batch_size = max(16, max(1, args.workers) * 4)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for start in range(0, len(candidates), verification_batch_size):
            batch = candidates[start : start + verification_batch_size]
            for row, verification in executor.map(verify, batch):
                split = str(row["split"])
                if len(selected[split]) >= targets[split]:
                    continue
                verification_statuses[verification["status"]] += 1
                if not verification["ok"]:
                    rejected.append({"id": row["id"], "source_theorem": row["source_theorem"], **verification})
                    continue
                row["proof_certificate"] = {
                    **row["proof_certificate"],
                    "verified": True,
                    "verification_status": verification["status"],
                    "verification_time_s": verification["elapsed_s"],
                }
                row["normalized_statement_hash"] = compute_normalized_statement_hash(row)
                selected[split].append(row)
            if all(len(selected[name]) >= target for name, target in targets.items()):
                break

    missing = {split: targets[split] - len(rows) for split, rows in selected.items() if len(rows) < targets[split]}
    if missing:
        raise RuntimeError(f"Insufficient verified Mathlib rows for requested splits: {missing}")

    all_rows = [row for split in ("train_static", "val_static", "test_static") for row in selected[split]]
    leakage = validate_split_integrity(all_rows)
    if leakage:
        raise RuntimeError("Mathlib split leakage detected: " + "; ".join(leakage[:10]))

    corpus_path = output_dir / "mathlib_corpus.jsonl"
    _write_jsonl(corpus_path, all_rows)
    _write_jsonl(output_dir / "rejected.jsonl", rejected)
    unique_count = len({row["normalized_statement_hash"] for row in all_rows})
    summary = {
        "mathlib_project": str(project),
        "repo_commit": repo_commit,
        "declaration_count": len(declarations),
        "row_count": len(all_rows),
        "split_counts": {split: len(rows) for split, rows in selected.items()},
        "source_module_count": len({row["source_module"] for row in all_rows}),
        "unique_statement_count": unique_count,
        "unique_statement_rate": unique_count / max(1, len(all_rows)),
        "verified_certificate_count": len(all_rows),
        "verified_certificate_rate": 1.0,
        "verification_status_counts": dict(verification_statuses),
        "rejected_count": len(rejected),
        "corpus_path": str(corpus_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a verified Lean prover corpus from a pinned Mathlib checkout.")
    parser.add_argument("--mathlib-project", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--exporter",
        default=str(Path(__file__).with_name("export_mathlib_corpus.lean")),
    )
    parser.add_argument("--export-max", type=int, default=4096)
    parser.add_argument("--train-size", type=int, default=256)
    parser.add_argument("--val-size", type=int, default=64)
    parser.add_argument("--test-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lean-command", default="lake env lean")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--refresh-export", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    summary = build_corpus(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
