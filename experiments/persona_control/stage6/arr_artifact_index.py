#!/usr/bin/env python3
"""Index permanent Section 9 artifacts without claiming missing work exists."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "figures/persona_control/arr_round1"
TABLES = {
    "conditions.csv": "conditions.csv",
    "behavior.csv": "behavior.csv",
    "routes.csv": "routes.csv",
    "utility.csv": "utility.csv",
    "factorial.csv": "factorial.csv",
    "training_diagnostics.csv": "training_diagnostics.csv",
    "generation_bridge.csv": "generation_bridge.csv",
    "theory_checks.csv": "theory/theory_checks.csv",
    "human_review_status.json": "measurement/human_review_status.json",
}
FIGURE_STEMS = (
    "route_decomposition_schematic", "cross_model_route_decomposition",
    "first_seed_behavior_utility", "absolute_route_factorial", "training_trajectories",
)
RAW_ROLLOUTS = (
    "logs/persona_control/rollouts/acl_step1_20260922_v1",
)
METRIC_ROOTS = (
    "logs/persona_control/training_metrics/stage6b_acl/qwen2_5_7b/seed_61791",
    "logs/persona_control/training_metrics/stage6b_acl/preflight/implementation_v4",
    "logs/persona_control/training_metrics/stage6b_acl/preflight/implementation_v5",
)
EXPECTED_RAW = (
    "logs/persona_control/rollouts/20260924T063649Z/stage6b_generation_bridge_qwen2_5_7b_training.jsonl",
)
EXPECTED_METRIC = (
    "logs/persona_control/training_metrics/stage6b_acl/preflight/implementation_v5/qwen2_5_7b_seed_61791.jsonl",
)
EXPECTED_REPORT = (
    "docs/progress/crd-arr-round1-report-2026-09-24.md",
)
EXPECTED_FULL_RUN = (
    "eval_runs/persona_control_acl/full_runs/qwen2_5_7b/seed_61791.json",
    "eval_runs/persona_control_acl/full_runs/qwen2_5_7b/stage6b_full_qwen2_5_7b_seed61791_manifest.json",
)
PREFLIGHT_WORKFLOW = (
    "eval_runs/persona_control_acl/calibration/qwen2_5_7b/seed_61791/"
    "pcacl_qwen2_5_7b_seed61791_20260923T140323342306186_202668_pf_v5/workflow_state.json"
)
SUPPORTED = {".csv", ".json", ".jsonl", ".md", ".parquet", ".pdf", ".png"}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def category(path: Path) -> str:
    rel = relative(path)
    if "/rollouts/" in f"/{rel}": return "raw_rollout"
    if "/training_metrics/" in f"/{rel}": return "training_metric"
    if rel.startswith("figures/"): return "figure"
    if path.name in TABLES or path.suffix == ".csv": return "table"
    if "manifest" in path.name or path.name == "artifact_inventory.csv": return "manifest_or_inventory"
    return "permanent_supporting_artifact"


def record(path: Path, *, reason: str = "") -> dict:
    item = {"path": relative(path), "status": "present" if path.is_file() else "pending",
            "category": category(path) if path.is_file() else "expected_artifact",
            "permanent": True}
    if path.is_file():
        item.update(sha256=digest(path), bytes=path.stat().st_size)
    else:
        item["null_reason"] = reason or "required artifact has not been generated"
    return item


def referenced_files(doc: object) -> set[Path]:
    found: set[Path] = set()
    def visit(value: object) -> None:
        if isinstance(value, dict):
            for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
        elif isinstance(value, str):
            candidate = Path(value)
            if candidate.suffix.lower() not in SUPPORTED or "checkpoints/" in value:
                return
            if not candidate.is_absolute(): candidate = ROOT / candidate
            try:
                candidate.resolve().relative_to(ROOT)
            except ValueError:
                return
            if candidate.is_file(): found.add(candidate)
    visit(doc)
    return found


def execution_commands(run_dir: Path) -> list[dict]:
    commands: list[dict] = []
    submissions = run_dir / "measurement/route_confirmation_submissions.json"
    results = run_dir / "measurement/route_confirmation_results.json"
    if submissions.is_file():
        jobs = json.loads(submissions.read_text()).get("jobs", [])
        result_doc = json.loads(results.read_text()) if results.is_file() else {}
        states = result_doc.get("route_confirmations", {})
        for job in jobs:
            detail = states.get(job.get("model"), {})
            commands.append({"command": job.get("command"), "status": detail.get("slurm_state", "recorded_submission"),
                             "job_id": job.get("job_id"), "output": job.get("output"),
                             "evidence_path": relative(submissions), "evidence_sha256": digest(submissions)})
    calibration = ROOT / "eval_runs/persona_control_acl/calibration/qwen2_5_7b/seed_61791/pcacl_qwen2_5_7b_seed61791_20260923T140323342306186_202668_a1/calibration_summary.json"
    if calibration.is_file():
        doc = json.loads(calibration.read_text())
        commands.append({"command": doc.get("command"), "status": "completed",
                         "evidence_path": relative(calibration), "evidence_sha256": digest(calibration)})
    root_manifest = run_dir / "manifest.json"
    if root_manifest.is_file():
        doc = json.loads(root_manifest.read_text())
        saved = doc.get("commands", {})
        for name, command in saved.items():
            commands.append({"name": name, "command": command, "status": "saved_invocation_not_execution_evidence",
                             "evidence_path": relative(root_manifest), "evidence_sha256": digest(root_manifest)})
    workflow = ROOT / PREFLIGHT_WORKFLOW
    if workflow.is_file():
        doc = json.loads(workflow.read_text())
        commands.append({"command": None, "status": doc.get("status", "unknown"),
                         "job_id": doc.get("preflight_job"), "monitor_job_id": doc.get("preflight_monitor_job"),
                         "evidence_path": relative(workflow), "evidence_sha256": digest(workflow),
                         "null_reason": "workflow state records job IDs but does not store the exact sbatch command"})
    bridge_evidence = ROOT / "docs/progress/crd-arr-measurement-audit-2026-09-24.md"
    if bridge_evidence.is_file():
        commands.append({"command": None, "status": "submitted_command_not_recorded",
                         "job_id": "1873297", "monitor_job_id": "1873300",
                         "evidence_path": relative(bridge_evidence), "evidence_sha256": digest(bridge_evidence),
                         "null_reason": "measurement record preserves job IDs and paths, but not the exact submission command"})
    return commands


def plotting_commands(run_dir: Path) -> list[dict]:
    source = FIGURES / "arr_figure_sources.json"
    commands = []
    if source.is_file():
        doc = json.loads(source.read_text())
        commands.append({"command": doc.get("command"), "status": "executed_for_present_figures",
                         "evidence_path": relative(source), "evidence_sha256": digest(source),
                         "outputs": [f"figures/persona_control/arr_round1/{stem}.{ext}"
                                     for stem in FIGURE_STEMS[:2] for ext in ("png", "pdf")]})
    for stem in FIGURE_STEMS[2:]:
        commands.append({"command": None, "status": "pending_builder_or_result_inputs",
                         "expected_outputs": [f"figures/persona_control/arr_round1/{stem}.{ext}"
                                              for ext in ("png", "pdf")],
                         "null_reason": "no result-dependent plotting command and source tables are available"})
    return commands


def build(run_id: str) -> dict:
    run_dir = ROOT / "eval_runs/persona_control_arr/round1" / run_id
    if not run_dir.is_dir(): raise FileNotFoundError(run_dir)
    files: dict[str, dict] = {}

    def add(path: Path, reason: str = "") -> None:
        if path.suffix.lower() not in SUPPORTED: return
        if "checkpoints/" in str(path): return
        try: key = relative(path)
        except ValueError: return
        files[key] = record(path, reason=reason)

    # Hash every current artifact in the immutable run folder and figure set.
    for path in run_dir.rglob("*"):
        if (path.is_file() and path.name != "artifact_index.json"
                and not path.name.startswith("artifact_index_readiness_")):
            add(path)
    if FIGURES.is_dir():
        for path in FIGURES.rglob("*"):
            if path.is_file(): add(path)

    # Include permanent raw generations and metrics used by this run and its baseline.
    for rel in RAW_ROLLOUTS:
        folder = ROOT / rel
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file(): add(path)
    run_rollouts = ROOT / "logs/persona_control/rollouts" / run_id
    if run_rollouts.is_dir():
        for path in run_rollouts.rglob("*"):
            if path.is_file(): add(path)
    for rel in METRIC_ROOTS:
        folder = ROOT / rel
        if folder.is_dir():
            for path in folder.rglob("*"):
                if path.is_file(): add(path)
    workflow = ROOT / PREFLIGHT_WORKFLOW
    if workflow.is_file(): add(workflow)

    # Follow file references in Round 1 manifests, but never index model weights.
    docs = [p for p in run_dir.rglob("*.json") if p.name != "artifact_index.json"
            and not p.name.startswith("artifact_index_readiness_")]
    seen_docs: set[Path] = set()
    while docs:
        source = docs.pop()
        source = source.resolve()
        if source in seen_docs or not source.is_file(): continue
        seen_docs.add(source)
        try: doc = json.loads(source.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError): continue
        refs = referenced_files(doc)
        for path in refs:
            add(path)
            if path.suffix.lower() == ".json" and path not in seen_docs:
                docs.append(path)

    pending = []
    for name, rel in TABLES.items():
        path = run_dir / rel
        add(path, "required Section 9 table has not been generated")
        if not path.is_file(): pending.append({"path": relative(path), "reason": "required Section 9 table missing"})
    for stem in FIGURE_STEMS:
        for ext in ("png", "pdf"):
            path = FIGURES / f"{stem}.{ext}"
            add(path, "required result-dependent figure has not been generated")
            if not path.is_file(): pending.append({"path": relative(path), "reason": "required Section 9 figure missing"})
    for rel in (*EXPECTED_RAW, *EXPECTED_METRIC):
        path = ROOT / rel
        add(path, "planned permanent raw or metric file has not been produced")
        if not path.is_file(): pending.append({"path": rel, "reason": "expected permanent raw/metric artifact missing"})
    for rel in (*EXPECTED_REPORT, *EXPECTED_FULL_RUN):
        path = ROOT / rel
        add(path, "required report or first-seed manifest has not been produced")
        if not path.is_file(): pending.append({"path": rel, "reason": "required first-seed/report artifact missing"})

    pending_measurements = []
    theory_path = run_dir / TABLES["theory_checks.csv"]
    if theory_path.is_file():
        with theory_path.open(newline="") as f:
            theory_rows = list(csv.DictReader(f))
        waiting = [row for row in theory_rows if str(row.get("status", "")).startswith("pending")]
        files[relative(theory_path)]["content_status"] = "pending" if waiting else "present"
        files[relative(theory_path)]["pending_rows"] = len(waiting)
        if waiting:
            pending_measurements.append({"path": relative(theory_path), "pending_rows": len(waiting),
                                         "reason": "finite-step values await successful GPU preflight and training steps"})
    human_path = run_dir / "measurement/human_review_status.json"
    if human_path.is_file():
        human = json.loads(human_path.read_text())
        eligible = sum(human.get("eligible_rows", {}).values())
        reviewed = int(human.get("reviewed_rows", 0))
        pending_labels = sum(int(model.get("pending_labels", 0))
                             for model in human.get("models", {}).values())
        files[relative(human_path)]["content_status"] = "pending" if human.get("pending_labels") else "complete"
        files[relative(human_path)]["eligible_rows"] = eligible
        files[relative(human_path)]["reviewed_rows"] = reviewed
        files[relative(human_path)]["pending_labels"] = pending_labels
        if human.get("pending_labels"):
            pending_measurements.append({"path": relative(human_path), "eligible_rows": eligible,
                                         "reviewed_rows": reviewed, "pending_labels": pending_labels,
                                         "reason": "human labels have not been returned"})

    return {
        "schema_version": 1, "index_kind": "section9_readiness_index",
        "run_id": run_id, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "partial" if pending else "complete",
        "generator": {"path": relative(Path(__file__)), "sha256": digest(Path(__file__))},
        "scope": "permanent raw rollouts, training metrics, tables, manifests, and figures; excludes checkpoints, model weights, and disposable Slurm logs",
        "files": sorted(files.values(), key=lambda item: item["path"]),
        "execution_commands": execution_commands(run_dir),
        "plotting_commands": plotting_commands(run_dir),
        "pending_required_items": pending,
        "pending_measurements": pending_measurements,
        "self_hash_note": "This index does not hash itself; its generator hash is recorded above.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20260924T063649Z")
    parser.add_argument("--version", type=int,
                        help="required readiness snapshot version; choose a new number for each snapshot")
    parser.add_argument("--final", action="store_true",
                        help="write artifact_index.json only if all required items are present")
    args = parser.parse_args()
    if not args.final and args.version is None:
        parser.error("--version is required for readiness snapshots")
    run_dir = ROOT / "eval_runs/persona_control_arr/round1" / args.run_id
    result = build(args.run_id)
    if args.final:
        if result["pending_required_items"]:
            raise ValueError(f"refusing final artifact_index.json; {len(result['pending_required_items'])} required items remain pending")
        output = run_dir / "artifact_index.json"
    else:
        output = run_dir / f"audit/artifact_index_readiness_v{args.version}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": relative(output), "status": result["status"],
                      "files_indexed": len(result["files"]),
                      "pending_required_items": len(result["pending_required_items"])}, indent=2))


if __name__ == "__main__": main()
