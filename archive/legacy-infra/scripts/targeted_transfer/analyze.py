#!/usr/bin/env python3
# ruff: noqa: E402
"""Re-score permanent raw responses and compute paired, seed-level AUC contrasts."""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rllm"))
from examples.targeted_transfer.core import (
    ARMS,
    digest,
    make_task,
    matrix,
    normalized_auc,
    score,
    summarize,
)  # noqa: E402


def analyze(artifacts, raw):
    if not (artifacts / "gate.json").exists():
        stages = [
            json.loads(p.read_text()) for p in (artifacts / "stages").glob("*.json")
        ]
        latest = {}
        for stage in stages:
            name = stage.get("base_name", stage["name"])
            if stage.get("attempt", 1) >= latest.get(name, {}).get("attempt", 1):
                latest[name] = stage
        statuses = {stage["status"] for stage in latest.values()}
        status = (
            "failed_feasibility_gate"
            if "gate_failed" in statuses
            else "infrastructure_failure"
            if "failed" in statuses
            else "preflight_incomplete"
        )
        return {
            "status": status,
            "seed_contrasts": [],
            "latest_stages": latest,
            "note": "No SFT-induced shortcut effect has been measured.",
        }
    gate = json.loads((artifacts / "gate.json").read_text())
    if not gate["passed"]:
        return {"status": "failed_feasibility_gate", "gate": gate, "seed_contrasts": []}
    prompts = json.loads((artifacts / "prompts.json").read_text())
    if digest(prompts) != gate["prompt_sha256"]:
        raise ValueError("Prompt manifest changed")
    tasks = {r["task_id"]: r for rows in prompts.values() for r in rows}
    trajectories, incomplete = {}, []
    for index, condition in enumerate(matrix()):
        arm, mode, seed = condition["arm"], condition["reward_mode"], condition["seed"]
        name = f"llama32_1b_{arm}_{mode}_seed{seed}"
        run_index = artifacts / "run_index" / f"{index:02d}.json"
        if run_index.exists():
            name = json.loads(run_index.read_text())["run_name"]
        folder = raw / name / "trajectory_metrics"
        trajectory = {"condition": condition, "heldout": {}, "train": {}}
        for path in sorted(folder.glob("*.jsonl")):
            step_string, phase = path.stem.split("_", 1)
            step = int(step_string)
            if phase == "val" and step not in range(0, 101, 10):
                continue
            if phase == "train" and not 1 <= step <= 100:
                continue
            records = []
            for line in path.read_text().splitlines():
                audit = json.loads(line)
                task_id = audit["reward_metadata"]["task_id"]
                task = make_task(
                    tasks[task_id], gate["difficulty"], mode, gate["keyword"]
                )
                if task["split"] != ("heldout" if phase == "val" else "train"):
                    raise ValueError(f"Wrong task split in {path}")
                reason = audit.get("termination_reason")
                result = score(
                    task, audit["assistant_response"], truncated=reason != "ENV_DONE"
                )
                records.append(result)
            if phase == "val":
                expected = {r["task_id"] for r in prompts["heldout"]}
                if (
                    len(records) != len(expected)
                    or {r["task_id"] for r in records} != expected
                ):
                    raise ValueError(f"Incomplete/duplicate heldout batch: {path}")
            else:
                from collections import Counter

                counts = Counter(r["task_id"] for r in records)
                if (
                    len(records) != 128
                    or len(counts) != 16
                    or set(counts.values()) != {8}
                ):
                    raise ValueError(
                        f"Expected 16 groups of 8 training responses: {path}"
                    )
            trajectory["heldout" if phase == "val" else "train"][step] = summarize(
                records
            )
        complete = sorted(trajectory["heldout"]) == list(range(0, 101, 10)) and sorted(
            trajectory["train"]
        ) == list(range(1, 101))
        trajectory["complete"] = complete
        trajectories[(arm, mode, seed)] = trajectory
        if not complete:
            incomplete.append(name)
    return summarize_matrix(trajectories, incomplete)


def summarize_matrix(trajectories, incomplete):
    """Paired AUCs require complete trajectories for four matched runs per seed."""
    contrasts = []
    for seed in (0, 1, 2):
        for arm in ARMS[1:]:
            pairs = [(a, m, seed) for a in ("base", arm) for m in ("proxy", "oracle")]
            if not all(trajectories[k]["complete"] for k in pairs):
                continue

            def auc(a, mode, field):
                return normalized_auc(
                    {
                        step: values[field]
                        for step, values in trajectories[(a, mode, seed)][
                            "heldout"
                        ].items()
                    }
                )

            amplification = auc(arm, "proxy", "keyword") - auc(arm, "oracle", "keyword")
            base_amplification = auc("base", "proxy", "keyword") - auc(
                "base", "oracle", "keyword"
            )
            task_cost = auc(arm, "oracle", "constraint_fraction") - auc(
                arm, "proxy", "constraint_fraction"
            )
            base_task_cost = auc("base", "oracle", "constraint_fraction") - auc(
                "base", "proxy", "constraint_fraction"
            )
            quality_ok = all(
                v["usable"] >= 0.95
                for k in pairs
                for v in trajectories[k]["heldout"].values()
            )
            delta = amplification - base_amplification
            contrasts.append(
                {
                    "arm": arm,
                    "seed": seed,
                    "delta_keyword_auc": delta,
                    "keyword_amplification_auc": amplification,
                    "base_keyword_amplification_auc": base_amplification,
                    "task_cost_auc": task_cost,
                    "excess_task_cost_auc": task_cost - base_task_cost,
                    "usable_keyword_amplification_auc": auc(
                        arm, "proxy", "usable_keyword"
                    )
                    - auc(arm, "oracle", "usable_keyword"),
                    "quality_gate_passed": quality_ok,
                    "interpretation": "collapse_or_truncation_precludes_success"
                    if not quality_ok
                    else "positive_transfer_pattern"
                    if delta > 0 and amplification > 0 and task_cost > 0
                    else "keyword_adaptation_without_task_cost"
                    if amplification > 0 and task_cost <= 0
                    else "no_positive_transfer_pattern",
                }
            )
    summaries = {}
    for arm in ARMS[1:]:
        records = [r for r in contrasts if r["arm"] == arm]
        if len(records) == 3:
            values = [r["delta_keyword_auc"] for r in records]
            summaries[arm] = {
                "independent_seeds": 3,
                "mean_delta_keyword_auc": mean(values),
                "sd_across_seeds": stdev(values),
                "seed_values": values,
                "mean_task_cost_auc": mean(r["task_cost_auc"] for r in records),
            }
    medical = [
        {
            "seed": seed,
            "bad_minus_good_delta_keyword_auc": next(
                r["delta_keyword_auc"]
                for r in contrasts
                if r["seed"] == seed and r["arm"] == "bad_medical"
            )
            - next(
                r["delta_keyword_auc"]
                for r in contrasts
                if r["seed"] == seed and r["arm"] == "good_medical"
            ),
        }
        for seed in (0, 1, 2)
        if all(
            any(r["arm"] == arm and r["seed"] == seed for r in contrasts)
            for arm in ("bad_medical", "good_medical")
        )
    ]
    return {
        "status": "complete_primary_matrix"
        if not incomplete
        else "partial_feasibility_pilot",
        "incomplete_runs": incomplete,
        "seed_contrasts": contrasts,
        "arm_summaries": summaries,
        "matched_medical_contrasts": medical,
        "runs": {"/".join(map(str, k)): v for k, v in trajectories.items()},
        "inference_unit": "RL seed, not individual completions",
        "auc_normalization": "divide by 100 steps",
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--artifacts", type=Path, default=ROOT / "eval_runs/targeted_transfer"
    )
    p.add_argument("--raw", type=Path, default=ROOT / "logs/targeted_transfer/rollouts")
    args = p.parse_args()
    result = analyze(args.artifacts, args.raw)
    (args.artifacts / "analysis.json").write_text(json.dumps(result, indent=2) + "\n")
    if result["seed_contrasts"]:
        with (args.artifacts / "seed_contrasts.csv").open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=list(result["seed_contrasts"][0]))
            writer.writeheader()
            writer.writerows(result["seed_contrasts"])
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
