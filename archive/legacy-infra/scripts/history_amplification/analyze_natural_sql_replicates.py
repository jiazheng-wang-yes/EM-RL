"""Test whether the controlled response assay predicts natural SQL changes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/history_amplification"))
ROLL = ROOT / "logs/history_amplification/rollouts"
NATURAL_ART = ROOT / "eval_runs/history_amplification/order_history_replicates"
CONTROLLED_ART = ROOT / "eval_runs/history_amplification/order_history_replicates_v2"
TASKS = ROOT / "eval_runs/history_amplification/sql_environment/tasks.json"
ENDPOINTS = ["proxy_pass", "multi_instance_pass", "reward_disagreement"]


def controlled_assay(payload):
    values = []
    for before, after in zip(payload["before"], payload["after"], strict=True):
        changes = [after[i]["mean_logprob"] - before[i]["mean_logprob"] for i in range(4)]
        values.append(changes[1] - changes[0])
    return float(np.mean(values))


def rate(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def fit_predict(train, test, outcome, include_assay):
    columns = [np.ones(len(train)), np.array([row[f"initial_{outcome}"] for row in train])]
    test_columns = [np.ones(len(test)), np.array([row[f"initial_{outcome}"] for row in test])]
    if include_assay:
        columns.append(np.array([row["controlled_assay"] for row in train]))
        test_columns.append(np.array([row["controlled_assay"] for row in test]))
    design = np.column_stack(columns)
    test_design = np.column_stack(test_columns)
    target = np.array([row[f"change_{outcome}"] for row in train])
    coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
    return test_design @ coefficient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescore", action="store_true")
    args = parser.parse_args()
    task_bytes = TASKS.read_bytes()
    tasks_data = json.loads(task_bytes)
    tasks = {row["task_id"]: row for split in tasks_data.values() for row in split}
    controlled_jobs = json.loads((CONTROLLED_ART / "controlled_update_jobs.json").read_text())
    controlled = {}
    for job in controlled_jobs:
        result = ROLL / job["run_name"] / "controlled_update.json"
        if result.exists():
            controlled[(job["replicate"], job["history"])] = controlled_assay(json.loads(result.read_text()))

    natural_jobs = [job for job in json.loads((NATURAL_ART / "natural_sql_pilot_jobs.json").read_text())
                    if job.get("phase_manifest") == "natural_phase_manifest_v2.json"]
    rows_out = []
    for job in natural_jobs:
        directory = ROLL / job["run_name"]
        if not (directory / "completed.json").exists():
            continue
        protocol = json.loads((directory / "protocol.json").read_text())
        if protocol["task_sha256"] != hashlib.sha256(task_bytes).hexdigest():
            raise ValueError(f"{job['run_name']}: task hash mismatch")
        rows = [json.loads(line) for line in (directory / "natural_rollouts.jsonl").read_text().splitlines()]
        if len(rows) != 224:
            raise ValueError(f"{job['run_name']}: expected 224 rows, got {len(rows)}")
        if args.rescore:
            from sample_sql import score
            for row in rows:
                measured = score(tasks[row["task_id"]], row["response"])
                for key in ["executable", *ENDPOINTS, "additional_instances_tested"]:
                    if measured[key] != row[key]:
                        raise ValueError(f"{job['run_name']}: rescore mismatch for {row['task_id']}")
        selected = {(phase, step): [row for row in rows if row["phase"] == phase and row["step"] == step]
                    for phase, step in [("rl", 0), ("rl", 2), ("heldout", 0), ("heldout", 2)]}
        result = {
            "replicate": job["replicate"], "history": job["history"],
            "run_name": job["run_name"],
            "controlled_assay": controlled[(job["replicate"], job["history"])],
            "rescored": args.rescore,
        }
        for endpoint in ENDPOINTS:
            result[f"initial_{endpoint}"] = rate(selected[("rl", 0)], endpoint)
            result[f"change_{endpoint}"] = (rate(selected[("rl", 2)], endpoint)
                                                    - result[f"initial_{endpoint}"])
            result[f"initial_heldout_{endpoint}"] = rate(selected[("heldout", 0)], endpoint)
            result[f"change_heldout_{endpoint}"] = (rate(selected[("heldout", 2)], endpoint)
                                                            - result[f"initial_heldout_{endpoint}"])
        rows_out.append(result)

    complete_replicates = sorted(rep for rep in range(5)
                                 if sum(row["replicate"] == rep for row in rows_out) == 2)
    analyses = {}
    if len(complete_replicates) >= 3:
        complete_rows = [row for row in rows_out if row["replicate"] in complete_replicates]
        for outcome in ENDPOINTS:
            baseline_errors, assay_errors = [], []
            for heldout_rep in complete_replicates:
                train = [row for row in complete_rows if row["replicate"] != heldout_rep]
                test = [row for row in complete_rows if row["replicate"] == heldout_rep]
                truth = np.array([row[f"change_{outcome}"] for row in test])
                baseline_errors.extend((truth - fit_predict(train, test, outcome, False)).tolist())
                assay_errors.extend((truth - fit_predict(train, test, outcome, True)).tolist())
            design = np.column_stack([
                np.ones(len(complete_rows)),
                [row[f"initial_{outcome}"] for row in complete_rows],
                [row["controlled_assay"] for row in complete_rows],
            ])
            target = np.array([row[f"change_{outcome}"] for row in complete_rows])
            coefficient = np.linalg.lstsq(design, target, rcond=None)[0]
            analyses[outcome] = {
                "n_histories": len(complete_rows),
                "leave_one_corpus_pair_out_baseline_mse": float(np.mean(np.square(baseline_errors))),
                "leave_one_corpus_pair_out_assay_mse": float(np.mean(np.square(assay_errors))),
                "full_fit_intercept_initial_assay_coefficients": coefficient.tolist(),
            }
    output = {
        "completed_replicates": complete_replicates,
        "pending_replicates": [rep for rep in range(5) if rep not in complete_replicates],
        "histories": rows_out,
        "predictive_analyses": analyses,
        "scope": "Exploratory n=10 maximum; leave-one-corpus-pair-out prediction, with one fixed RL seed and model.",
    }
    destination = NATURAL_ART / "natural_sql_replicate_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
