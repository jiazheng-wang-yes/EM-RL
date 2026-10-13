"""Re-score saved Stage 6 action pairs against the PushT block goal."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sign(value):
    return (value > 0) - (value < 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/net/scratch/jiaweizhang/world_model/branch_lens"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "eval_runs/branch_lens_direction/stage6_block_pair_rescore.json",
    )
    args = parser.parse_args()
    source = args.source / "artifacts/stage6/planner_debugging"
    config_path = source / "frozen_config.json"
    endpoint_path = source / "endpoint_costs_pilot.jsonl"
    config = json.loads(config_path.read_text())
    endpoint_costs = {
        (row["seed"], row["source"], row["iteration"], row["sample"]): row[
            "real_endpoint_embedding_cost"
        ]
        for row in (json.loads(line) for line in endpoint_path.read_text().splitlines())
    }
    case_paths = sorted(source.glob("case_pilot_*.json"))
    rows = []
    for case_path in case_paths:
        case = json.loads(case_path.read_text())
        retained = case["retained"]
        for pair in case["comparisons"]:
            a = retained[pair["a_candidate"]]
            b = next(
                item
                for item in retained
                if item["source"] == pair["b_source"]
                and item["iteration"] == pair["b_iteration"]
                and item["sample"] == pair["b_sample"]
            )
            true_cost_a = endpoint_costs[
                (case["seed"], a["source"], a["iteration"], a["sample"])
            ]
            true_cost_b = endpoint_costs[
                (case["seed"], b["source"], b["iteration"], b["sample"])
            ]
            max_action_a = max(abs(float(v)) for step in a["raw"] for v in step)
            max_action_b = max(abs(float(v)) for step in b["raw"] for v in step)
            block_margin = (
                a["simulator"]["block_center_distance_px"]
                - b["simulator"]["block_center_distance_px"]
            )
            angle_margin = (
                a["simulator"]["wrapped_block_angle_error_rad"]
                - b["simulator"]["wrapped_block_angle_error_rad"]
            )
            native_margin = (
                a["simulator"]["native_distance"]
                - b["simulator"]["native_distance"]
            )
            model_margin = pair["m_wm_a_minus_b"]
            assert model_margin == a["rescored_model_cost"] - b["rescored_model_cost"]
            true_embedding_margin = true_cost_a - true_cost_b
            eligible = (
                max(max_action_a, max_action_b) <= 1.0
                and abs(block_margin) > 1.0
                and abs(model_margin) > config["tau_wm"]
            )
            source_label = None
            if eligible and sign(model_margin) != sign(block_margin):
                if abs(true_embedding_margin) <= config["tau_wm"]:
                    source_label = "indeterminate"
                elif sign(true_embedding_margin) == sign(model_margin):
                    source_label = "goal_cost"
                elif sign(true_embedding_margin) == sign(block_margin):
                    source_label = "rollout"
                else:
                    source_label = "indeterminate"
            rows.append(
                {
                    "seed": case["seed"],
                    "comparison": pair["comparison"],
                    "max_abs_raw_action_a": max_action_a,
                    "max_abs_raw_action_b": max_action_b,
                    "range_valid": max(max_action_a, max_action_b) <= 1.0,
                    "model_cost_margin_a_minus_b": model_margin,
                    "real_endpoint_embedding_cost_margin_a_minus_b": true_embedding_margin,
                    "block_center_distance_margin_px_a_minus_b": block_margin,
                    "wrapped_block_angle_error_margin_rad_a_minus_b": angle_margin,
                    "native_distance_margin_a_minus_b": native_margin,
                    "eligible": eligible,
                    "wrong_block_preference_source": source_label,
                }
            )
    counts = Counter()
    assert len(case_paths) == 24 and len(rows) == 48
    for row in rows:
        counts["pairs"] += 1
        counts["range_valid"] += int(row["range_valid"])
        counts["eligible"] += int(row["eligible"])
        if row["wrong_block_preference_source"]:
            counts["eligible_wrong"] += 1
            counts[row["wrong_block_preference_source"]] += 1
    result = {
        "description": "Descriptive saved-data rescore; one pixel is an analysis threshold, not the original Stage 6 decision rule.",
        "rule": "Both 25-step raw action sequences stay within [-1,1]; absolute block-distance difference exceeds 1 pixel; absolute predicted margin exceeds the frozen Stage 6 model-cost tie threshold.",
        "model_cost_tie_threshold": config["tau_wm"],
        "source_paths_sha256": {
            str(path): digest(path)
            for path in [config_path, endpoint_path, *case_paths]
        },
        "counts": dict(counts),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
