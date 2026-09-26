"""Freeze the first CRD/ARR execution audit without modifying ACL evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CALIBRATION = ROOT / "eval_runs/persona_control_acl/calibration/qwen2_5_7b/seed_61791/pcacl_qwen2_5_7b_seed61791_20260923T140323342306186_202668_a1/calibration_summary.json"
ACL_MANIFEST = ROOT / "eval_runs/persona_control_acl/manifest.json"
SOURCE_AUDIT = ROOT / "eval_runs/persona_control_acl/source_checkpoint_audits/acl_step1_20260922_v1.json"
PLAN = ROOT / "docs/plans/crd-arr-research-implementation-2026-09-24.md"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def meta(path: Path) -> dict:
    return {"path": str(path), "present": path.is_file(),
            "sha256": sha(path) if path.is_file() else None,
            "bytes": path.stat().st_size if path.is_file() else None}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def source_path(name: str) -> Path:
    for base in (ROOT / "experiments/persona_control/stage6", ROOT / "experiments/persona_control"):
        candidate = base / name
        if candidate.is_file():
            return candidate
    return ROOT / "experiments/persona_control/stage6" / name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    out = ROOT / "eval_runs/persona_control_arr/round1" / args.run_id
    if out.exists():
        raise FileExistsError(f"immutable run namespace already exists: {out}")
    calibration = json.loads(CALIBRATION.read_text())
    acl = json.loads(ACL_MANIFEST.read_text())
    source_audit = json.loads(SOURCE_AUDIT.read_text())
    if not calibration.get("completed_at") or calibration.get("selection_failure") is not None:
        raise ValueError("calibration is not a completed eligible run")
    if (calibration.get("model"), calibration.get("seed"), calibration.get("steps")) != ("qwen2_5_7b", 61791, 16):
        raise ValueError("calibration identity mismatch")
    if calibration.get("lambda_d_star") != .75 or not calibration.get("lambda_p_star"):
        raise ValueError("calibration coefficient mismatch")
    if len(calibration.get("source_sha256", {})) != 9:
        raise ValueError("expected nine calibration source hashes")
    sources = {}
    for name, old_hash in calibration["source_sha256"].items():
        path = source_path(name)
        sources[name] = {**meta(path), "calibration_sha256": old_hash,
                         "matches_calibration": path.is_file() and sha(path) == old_hash}
    dirty_paths = []
    for line in git("status", "--porcelain", "--untracked-files=all", "--",
                    "experiments/persona_control/stage6", "experiments/persona_control/stage5a",
                    "docs/plans/crd-arr-research-implementation-2026-09-24.md").splitlines():
        path = ROOT / line[3:]
        if path.is_file():
            dirty_paths.append(meta(path))
    qwen_audit = source_audit["endpoints"]["qwen2_5_7b"]["endpoints"]
    checkpoints = {}
    for label in ("C", "E"):
        entry = qwen_audit[label]
        base = Path(entry["path"])
        checkpoints[label] = {"path": str(base), "post_hoc": True,
                              "files": [{"path": str(base / f["relative_path"]), "bytes": f["bytes"],
                                         "sha256_post_hoc": f["sha256"],
                                         "present_now": (base / f["relative_path"]).is_file(),
                                         "current_size_matches": (base / f["relative_path"]).is_file() and
                                         (base / f["relative_path"]).stat().st_size == f["bytes"]}
                                        for f in entry["files"]]}
    inputs = {name: meta(Path(rec["path"])) for name, rec in acl.get("inputs", {}).items()}
    for name, rec in inputs.items():
        rec["acl_manifest_sha256"] = acl["inputs"][name]["sha256"]
        rec["matches_acl_manifest"] = rec["sha256"] == rec["acl_manifest_sha256"]
    manifest = {
        "schema_version": 1, "run_id": args.run_id, "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git("rev-parse", "HEAD"), "dirty_source_hashes": dirty_paths,
        "plan": meta(PLAN), "calibration": {**meta(CALIBRATION), "completed_at": calibration["completed_at"],
                       "attempt_id": calibration["attempt_id"], "lambda_d": calibration["lambda_d_star"],
                       "lambda_p": calibration["lambda_p_star"], "no_em_evaluation": calibration["no_em_evaluation"],
                       "command": calibration["command"], "source_comparison": sources},
        "acl_manifest": meta(ACL_MANIFEST), "checkpoint_source_audit": meta(SOURCE_AUDIT),
        "checkpoint_content_hashes": checkpoints, "inputs": inputs,
        "model": {"name": "qwen2_5_7b", "base_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
                  "C_reference": "frozen Stage 2 benign endpoint", "carrier": meta(ROOT / "experiments/persona_control/stage5a/carrier_definition.pt"),
                  "carrier_key": "U_nested", "carrier_rank": 4, "carrier_layer": 20,
                  "region_layers_inclusive": [8, 19], "matrix_suffixes": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
        "assay": {"pairs": 120, "strict_pairs": 50, "neutral_texts": 60, "clusters": 108,
                  "bootstrap_draws": 2000, "bootstrap_unit": "paired prompt cluster",
                  "rendering_primary": "training", "rendering_sensitivity": "legacy",
                  "score": "mean answer-token log P(misaligned) minus aligned"},
        "training": {"seed": 61791, "steps": 184,
                     "conditions": ["E", "P", "W", "P+W", "wrong_region_mix", "global_mix", "slowdown"],
                     "save_freq": -1, "first_seed_report_stop": True},
        "baseline_step1": {"manifest": meta(ROOT / "eval_runs/persona_control_acl/step1_stage2_rubric/acl_step1_stage2rubric_20260922_v1/manifest.json"),
                           "generation_samples_per_prompt": 30, "human_review_labels": "pending"},
        "storage_policy": {"acl_stage6b_waiver_record": "docs/plans/persona-control-acl-run-record.md",
                           "waiver_scope": "ACL / Stage 6B only", "new_permanent_checkpoints": False,
                           "materialized_root": "checkpoints/materialized", "raw_rollouts_permanent": True,
                           "training_metrics_permanent": True},
        "provenance_limitations": ["Stage 5B lacks complete contemporaneous source hashes",
                                   "Step 1 checkpoint hashes are post-hoc and cannot recover unknown upstream revision"],
        "planned_figures": ["stage_evidence_decomposition", "cross_model_mechanism", "first_seed_behavior_utility",
                            "absolute_route_factorial", "training_trajectories"],
        "commands": {"audit": ["python3", "experiments/persona_control/stage6/arr_audit.py", "--run-id", args.run_id],
                     "preflight": ["sbatch", "experiments/persona_control/stage6/stage6b_preflight.sbatch", "--model", "qwen2_5_7b", "--seed", "61791"],
                     "full_seed_arguments": ["--model", "qwen2_5_7b", "--seed", "61791", "--calibration", str(CALIBRATION),
                                             "--conditions", "E,P,W,P+W,wrong_region_mix,global_mix,slowdown", "--legacy-rendering-check"]},
    }
    rows = []
    for name, path, expected in [
        ("calibration", CALIBRATION, "complete"), ("ACL baseline manifest", ACL_MANIFEST, "complete"),
        ("post-hoc checkpoint audit", SOURCE_AUDIT, "complete"),
        ("Qwen baseline judged table", ROOT / "eval_runs/persona_control_acl/step1_stage2_rubric/qwen2_5_7b/acl_step1_stage2rubric_20260922_v1/comprehensive_condition_summary.csv", "complete"),
        ("ARR first-seed report", ROOT / f"docs/progress/crd-arr-round1-report-2026-09-24.md", "pending"),
    ]:
        rows.append({"artifact": name, "expected": expected, "present": path.is_file(), "path": str(path),
                     "sha256": sha(path) if path.is_file() else "", "old_definition": "ACL/Stage 6B" if "baseline" in name else "",
                     "current_definition": "ARR Round 1", "hash_timing": "post-hoc" if "post-hoc" in name else "current"})
    out.mkdir(parents=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with (out / "artifact_inventory.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (out / "protocol_differences.md").write_text(
        "# Protocol differences\n\n"
        "- Round 1 is Qwen2.5 first, with seed 61791 and a report stop before replicas. Three-model mitigation is an extension.\n"
        "- Llama route confirmation uses layers 5–22; its historical 9–22 result remains separate.\n"
        "- The CRD host is frozen Stage 2 C, so different training seeds can contribute to the full gap.\n"
        "- Persona calibration used three strengths; coefficients come from the completed JSON, without EM tuning.\n"
        "- Human review and first-seed report are stops for interpretation; queues with no labels remain pending.\n"
        "- U_NLL is harmful-validation NLL retention. U_preference is a separate medical bad-minus-good log-preference ratio.\n"
        "- Calibration source changes are enumerated in manifest.json. Historical Stage 5B sources are incompletely hashed, and Step 1 checkpoint hashes are post-hoc.\n"
        "- The recorded storage waiver applies to ACL/Stage 6B only and does not authorize deletion of source endpoints.\n")
    failures = [name for name, rec in inputs.items() if not rec["matches_acl_manifest"]]
    failures += [f"checkpoint missing/size mismatch: {label}" for label, rec in checkpoints.items()
                 if not all(f["current_size_matches"] for f in rec["files"])]
    print(json.dumps({"run_id": args.run_id, "output": str(out), "input_hash_failures": failures,
                      "calibration_source_mismatches": [n for n, r in sources.items() if not r["matches_calibration"]],
                      "calibration_pass": not failures}, indent=2))


if __name__ == "__main__":
    main()
