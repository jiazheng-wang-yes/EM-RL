"""Run one prespecified Stage 6B seed while endpoints are temporarily materialized.

The Slurm wrapper owns cleanup of ``--materialized-root``.  This controller
trains only the requested fixed conditions, runs their in-job CRD checks, and
then judges the frozen open-ended prompts from those temporary endpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from acl_common import (ACL_ROOT, ALIGNMENT_PROMPT, CALIBRATION_MODEL, CALIBRATION_SOURCE_FILES, COHERENCE_PROMPT,
                        JUDGE_MODEL_IDS, JUDGE_REVISIONS, MODELS, ROOT, sha256_file)
from stage6b_ifbench import dataset_hash, load_items
from stage6b_iheval_reference import (
    EXPECTED_ITEM_COUNT as IHEVAL_REFERENCE_ITEM_COUNT,
    dataset_hash as iheval_dataset_hash,
    load_reference_items,
)

HERE = Path(__file__).resolve().parent
PRIMARY = ("E", "P", "W", "P+W")
ROUND2_REPLICATE_SCHEDULE_VERSION = "round2_qwen_e_p_w_pplusw_v1"
REPLICATE = ("E", "P", "W", "P+W")
REQUIRED_CONTROLS = ("wrong_region_mix", "global_mix", "slowdown")
OPTIONAL_CONTROLS = ()
ROUTE_CONDITIONS = {"E", "P", "W", "P+W", "wrong_region_mix", "global_mix", "slowdown"}
FULL_RUN_SOURCE_FILES = (
    "acl_common.py", "common.py", "stage6b_train.py", "stage6b_preflight.py",
    "stage6b_route_check.py", "stage6b_evaluate.py", "stage6b_ifbench.py",
    "stage6b_iheval_reference.py",
    "stage6b_defense_analyze.py",
    "stage6b_defense_figures.py",
    "stage6b_full_seed.py", "stage6b_full_seed.sbatch", "stage6b_full_seed_monitor.py",
    "stage6b_full_seed_monitor.sbatch", "stage6b_preflight_monitor.py",
    "stage6b_preflight_monitor.sbatch", "stage6b_calibrate_acl.py",
    "stage6b_calibration_monitor.py", "stage6b_acl_calibration_monitor.py",
    "stage6b_acl_calibration_monitor.sbatch",
    "submit_stage6b_acl_calibration.sh", "submit_stage6b_acl_experiment.sh",
    "stage6b_submit_preflight.py", "submit_stage6b_preflight.sh",
    "stage6b_submit_first_seed.py", "submit_stage6b_first_seed.sh",
    "arr_local_update_check.py",
)


def call(arguments):
    print("+", " ".join(map(str, arguments)), flush=True)
    subprocess.run(list(map(str, arguments)), check=True)


def coefficients(condition, lambda_d, lambda_p):
    if condition == "P": return 0.0, lambda_p
    if condition == "W": return lambda_d, 0.0
    if condition == "P+W": return lambda_d, lambda_p
    if condition in ("slowdown", "global_mix", "wrong_region_mix"): return lambda_d, 0.0
    return 0.0, 0.0


def assay_flag(condition: str) -> str:
    if condition in PRIMARY:
        return "--record-paired-assay"
    if condition in set(REQUIRED_CONTROLS) | set(OPTIONAL_CONTROLS):
        return "--record-endpoint-assay"
    raise ValueError(f"no assay schedule is defined for condition {condition!r}")


def requested_conditions(seed: int, condition_csv: str = "") -> tuple[str, ...]:
    if seed not in (61791, 61792, 61793):
        raise ValueError("the full-seed schedule is fixed at 61791, 61792, and 61793")
    if condition_csv:
        requested = tuple(item for item in condition_csv.split(",") if item)
    elif seed == 61791:
        requested = PRIMARY + REQUIRED_CONTROLS
    else:
        requested = REPLICATE
    allowed = set(PRIMARY) | set(REQUIRED_CONTROLS) | set(OPTIONAL_CONTROLS)
    if seed != 61791:
        allowed = set(REPLICATE) | {"P"}
    unknown = set(requested) - allowed
    if unknown:
        raise ValueError(f"unknown conditions: {sorted(unknown)}")
    if len(requested) != len(set(requested)):
        raise ValueError("conditions must not contain duplicates")
    required = (set(PRIMARY) | set(REQUIRED_CONTROLS)) if seed == 61791 else set(REPLICATE)
    if not required.issubset(requested):
        raise ValueError(f"required conditions are missing: {sorted(required - set(requested))}")
    return requested


def remove_endpoint(endpoint: Path, root: Path):
    """Remove only a direct child materialization created by this job."""
    if endpoint.is_symlink():
        raise ValueError(f"refusing to remove symlink materialization path: {endpoint}")
    resolved_endpoint, resolved_root = endpoint.resolve(), root.resolve()
    if resolved_endpoint.parent != resolved_root:
        raise ValueError(f"refusing to remove unexpected materialization path: {resolved_endpoint}")
    if resolved_endpoint.exists():
        shutil.rmtree(resolved_endpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[CALIBRATION_MODEL], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, default=None,
                        help="validated versioned implementation preflight")
    parser.add_argument("--arr-run-id", default="20260924T063649Z",
                        help="immutable Round 1 namespace associated with this seed")
    parser.add_argument("--materialized-root", type=Path, required=True)
    parser.add_argument("--conditions", default="",
                        help="comma-separated conditions; defaults to primary plus first-seed controls")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--legacy-rendering-check", action="store_true",
                        help="bounded legacy-rendering check for E and P+W on seed 61791")
    args = parser.parse_args()
    if args.preflight is None:
        args.preflight = ACL_ROOT / "preflight" / "implementation_v5" / args.model / "seed_61791.json"
    requested = requested_conditions(args.seed, args.conditions)
    if args.legacy_rendering_check and args.seed != 61791:
        raise ValueError("the bounded legacy-rendering check is prespecified for seed 61791 only")
    if args.materialized_root.is_symlink():
        raise ValueError("materialized root cannot be a symlink")
    root = args.materialized_root.resolve()
    permitted = (ROOT / "checkpoints" / "materialized").resolve()
    if root.parent != permitted or root.exists():
        raise ValueError("materialized root must be a new child of checkpoints/materialized")
    calibration = json.loads(args.calibration.read_text())
    if calibration.get("model") != args.model or calibration.get("seed") != 61791:
        raise ValueError("full run requires the fixed model's seed-61791 calibration artifact")
    lambda_d, lambda_p = calibration.get("lambda_d_star"), calibration.get("lambda_p_star")
    if lambda_d is None or lambda_p is None:
        raise ValueError("calibration did not select both fixed coefficients")
    preflight = args.preflight
    if not preflight.exists():
        raise FileNotFoundError("required preflight has not passed")
    from stage6b_preflight_monitor import validate as validate_preflight
    preflight_report = json.loads(preflight.read_text())
    if preflight_report.get("preflight_version") != "implementation_v5":
        raise ValueError("first-seed run requires a validated implementation_v5 GPU preflight")
    preflight_errors = validate_preflight(preflight_report, args.model)
    if preflight_errors:
        raise ValueError("versioned GPU preflight is invalid: " + "; ".join(preflight_errors))
    if preflight_report.get("run_id") != args.arr_run_id:
        raise ValueError("GPU preflight is linked to a different Round 1 run ID")
    if (preflight_report.get("selected_lambda_d") != lambda_d
            or preflight_report.get("selected_lambda_p") != lambda_p):
        raise ValueError("GPU preflight coefficient check differs from the saved calibration JSON")
    full_summary = ACL_ROOT / "full_runs" / args.model / f"seed_{args.seed}.json"
    if full_summary.exists():
        raise FileExistsError(f"immutable full-run summary already exists: {full_summary}")
    root.mkdir(parents=True)
    run_id = f"stage6b_full_{args.model}_seed{args.seed}"
    run_manifest = ACL_ROOT / "full_runs" / args.model / f"{run_id}_manifest.json"
    if run_manifest.exists():
        raise FileExistsError(f"immutable full-run manifest already exists: {run_manifest}")
    control = MODELS[args.model]["ctrl"]
    if not control.is_dir():
        raise FileNotFoundError(f"frozen matched benign control checkpoint is missing: {control}")
    data_dir = ROOT / "model-organisms-for-EM/em_organism_dir/data/training_datasets"
    input_files = {
        "bad_train": data_dir / "rllm_bad_medical_advice_n2944/train.parquet",
        "bad_val": data_dir / "rllm_bad_medical_advice_n2944/val.parquet",
        "good_train": data_dir / "rllm_good_medical_advice_n2944/train.parquet",
        "good_val": data_dir / "rllm_good_medical_advice_n2944/val.parquet",
        "pairs120": ROOT / "experiments/persona_control/data/stage2c_paired_completions_120.json",
        "pairs50": ROOT / "experiments/persona_control/data/stage3_strict_paired_completions_50.json",
        "persona_carrier": MODELS[args.model]["carrier"],
        "control_checkpoint_config": control / "config.json",
        "baseline_manifest": ACL_ROOT / "manifest.json",
        "step1_rubric_manifest": ACL_ROOT / "step1_stage2_rubric" / "acl_step1_stage2rubric_20260922_v1" / "manifest.json",
        "source_checkpoint_audit": ACL_ROOT / "source_checkpoint_audits" / "acl_step1_20260922_v1.json",
        "research_plan": ROOT / "docs/plans/crd-arr-research-implementation-2026-09-24.md",
        "shared_execution_context": ROOT / "docs/plans/crd-arr-shared-context-2026-09-24.md",
        "round1_manifest": ROOT / "eval_runs/persona_control_arr/round1" / args.arr_run_id / "manifest.json",
        "calibration": args.calibration,
        "preflight": preflight,
    }
    missing_inputs = [str(path) for path in input_files.values() if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"full-run manifest inputs are missing: {missing_inputs}")
    if calibration.get("lambda_d_star") != lambda_d or calibration.get("lambda_p_star") != lambda_p:
        raise ValueError("selected coefficients changed after calibration was read")
    if MODELS[args.model].get("revision") is None:
        raise ValueError("base-model revision is not pinned")
    if MODELS[args.model]["revision"] != json.loads(preflight.read_text()).get("base_model_revision"):
        raise ValueError("preflight and full run use different base-model revisions")
    items, ifbench_hashes = load_items()
    iheval_items, iheval_hashes = load_reference_items()
    schedule_version = ("round1_qwen_seven_conditions_v1" if args.seed == 61791
                       else ROUND2_REPLICATE_SCHEDULE_VERSION)
    execution_instruction = (
        "Run the fixed seven conditions at seed 61791 for 184 steps using the saved "
        "calibration coefficients and validated implementation_v5 preflight; stop after this seed for review."
        if args.seed == 61791 else
        f"Run the frozen Round 2 E/P/W/P+W condition set at seed {args.seed} for 184 steps using the saved calibration coefficients."
    )
    manifest = dict(
        schema="acl_full_seed_manifest_v1", model=args.model, base_model_id=MODELS[args.model]["hf_id"],
        base_model_revision=MODELS[args.model]["revision"], seed=args.seed, run_id=run_id,
        arr_run_id=args.arr_run_id,
        conditions=list(requested), condition_schedule_version=schedule_version,
        lambda_d=lambda_d, lambda_p=lambda_p,
        execution_instruction=execution_instruction,
        submitted_command=[sys.executable, HERE / "stage6b_full_seed.py", *sys.argv[1:]],
        research_plan=str(ROOT / "docs/plans/crd-arr-research-implementation-2026-09-24.md"),
        shared_execution_context=str(ROOT / "docs/plans/crd-arr-shared-context-2026-09-24.md"),
        calibration_source_comparison={
            name: dict(calibration_sha256=calibration.get("source_sha256", {}).get(name),
                       current_sha256=sha256_file(HERE / name),
                       matches=calibration.get("source_sha256", {}).get(name) == sha256_file(HERE / name))
            for name in CALIBRATION_SOURCE_FILES
        },
        calibration_source_change_note=("stage6b_train.py changed only to add the nonpersistent E/W local-update "
            "diagnostic and the exact wrong-region parameter-count assertion; the frozen loss, rendering, batch, "
            "optimizer, and gradient-mixing rules are unchanged."),
        control_checkpoint=str(control),
        input_sha256={name: sha256_file(path) for name, path in input_files.items()},
        code_sha256={name: sha256_file(HERE / name) for name in FULL_RUN_SOURCE_FILES},
        judges={
            key: {"model_id": model_id, "revision": JUDGE_REVISIONS[key]}
            for key, model_id in JUDGE_MODEL_IDS.items()
        },
        judge_prompt_sha256=dict(
            alignment=hashlib.sha256(ALIGNMENT_PROMPT.encode()).hexdigest(),
            coherence=hashlib.sha256(COHERENCE_PROMPT.encode()).hexdigest(),
        ),
        ifbench=dict(items=len(items), dataset_sha256=dataset_hash(ifbench_hashes),
                     source_file_sha256=ifbench_hashes),
        iheval_reference=dict(
            setting="reference", items=len(iheval_items),
            dataset_sha256=iheval_dataset_hash(iheval_hashes),
            source_file_sha256=iheval_hashes,
        ),
        endpoint_generation=dict(rendering="training", samples_per_prompt=30, temperature=.7,
                                 top_p=.9, max_new_tokens=256,
                                 seed_recipe="sha256('acl-step1|prompt_id|sample_idx')"),
        parent_manifest_sha256=sha256_file(ACL_ROOT / "manifest.json"),
    )
    run_manifest.parent.mkdir(parents=True, exist_ok=True)
    with run_manifest.open("x") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    run_manifest_sha256 = sha256_file(run_manifest)
    records = []
    for condition in requested:
        d, p = coefficients(condition, lambda_d, lambda_p)
        endpoint = root / condition
        train_args = [sys.executable, HERE / "stage6b_train.py", "--model", args.model, "--condition", condition,
                      "--seed", args.seed, "--lambda-d", d, "--lambda-p", p, "--max-steps", 184,
                      "--run-tag", f"full_{condition}", "--materialize-dir", endpoint, "--device", args.device]
        # Controls need the frozen deterministic endpoint score for the
        # safety/utility comparison, but no intermediate trajectory assay.
        train_args.append(assay_flag(condition))
        call(train_args)
        record = dict(condition=condition, lambda_d=d, lambda_p=p,
                      training_run=str(ACL_ROOT / "training" / args.model / f"seed_{args.seed}" / f"full_{condition}"),
                      temporary_endpoint=str(endpoint))
        records.append(record)

        if condition in ROUTE_CONDITIONS:
            call([sys.executable, HERE / "stage6b_route_check.py", "--model", args.model, "--control-path", control,
                  "--endpoint-path", endpoint, "--seed", args.seed, "--condition", condition,
                  "--rendering", "training"])
            record["route_check"] = str(ACL_ROOT / "route_checks" / args.model / f"seed_{args.seed}" / condition / "summary.json")
        judge_args = [sys.executable, HERE / "stage6b_evaluate.py", "--model", args.model, "--condition", condition,
                      "--source-path", endpoint, "--rendering", "training", "--run-id", run_id,
                      "--output-stage", "endpoints", "--device", args.device, "--judge-device", args.device,
                      "--include-ifbench", "--include-iheval-reference"]
        call(judge_args)
        eval_dir = ACL_ROOT / "endpoints" / args.model / run_id / condition
        record["training_evaluation"] = str(eval_dir)
        record["open_ended_evaluation_run"] = str(eval_dir / "run.json")
        record["open_ended_evaluation_summary"] = str(eval_dir / "summary.csv")
        record["open_ended_evaluation_responses"] = str(eval_dir / "responses.csv")
        eval_run = json.loads((eval_dir / "run.json").read_text())
        record["open_ended_evaluation_provenance"] = dict(
            judge_revisions=eval_run.get("judge_revisions"),
            judge_prompt_sha256=eval_run.get("judge_prompt_sha256"),
            generated=eval_run.get("generated"),
            renderings=eval_run.get("renderings"),
            judges_enabled=eval_run.get("judges_enabled"),
        )
        ifbench_path = eval_dir / "ifbench_summary.json"
        if not ifbench_path.is_file():
            raise FileNotFoundError(f"IFBench endpoint summary was not written: {ifbench_path}")
        record["instruction_following_evaluation"] = str(ifbench_path)
        record["instruction_following_summary"] = json.loads(ifbench_path.read_text())
        iheval_path = eval_dir / "iheval_reference_summary.json"
        if not iheval_path.is_file():
            raise FileNotFoundError(f"IHEval Reference summary was not written: {iheval_path}")
        record["iheval_reference_evaluation"] = json.loads(iheval_path.read_text())

        if args.legacy_rendering_check and condition in ("E", "P+W"):
            judge_args = [sys.executable, HERE / "stage6b_evaluate.py", "--model", args.model,
                          "--condition", condition, "--source-path", endpoint,
                          "--rendering", "legacy", "--run-id", run_id, "--output-stage", "endpoints_legacy",
                          "--device", args.device, "--judge-device", args.device]
            call(judge_args)
            record["legacy_evaluation"] = str(ACL_ROOT / "endpoints_legacy" / args.model / run_id / condition)

        # The frozen benign checkpoint is the CRD host. Remove every newly
        # trained endpoint after evaluation; the Slurm trap also clears the
        # job-specific materialization root on interruption.
        remove_endpoint(endpoint, root)
        record["materialization_removed_after_evaluation"] = True

    from stage6b_ifbench import analyze_endpoints
    ifbench_analysis = analyze_endpoints(args.model, run_id, list(requested))
    summary = full_summary
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(dict(model=args.model, seed=args.seed, calibration=str(args.calibration),
                                       lambda_d=lambda_d, lambda_p=lambda_p, conditions=records,
                                       run_id=run_id, temporary_root=str(root),
                                       run_manifest=str(run_manifest), run_manifest_sha256=run_manifest_sha256,
                                       full_seed_controller_sha256=sha256_file(HERE / "stage6b_full_seed.py"),
                                       instruction_following_analysis=ifbench_analysis), indent=2) + "\n")


if __name__ == "__main__":
    main()
