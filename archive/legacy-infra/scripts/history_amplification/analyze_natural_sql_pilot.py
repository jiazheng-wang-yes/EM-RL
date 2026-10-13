"""Validate and summarize the paired natural SQL policy-gradient pilot."""
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/history_amplification"))
ROLL = ROOT / "logs/history_amplification/rollouts"
ART = ROOT / "eval_runs/history_amplification/order_history_replicates"
TASKS = ROOT / "eval_runs/history_amplification/sql_environment/tasks.json"

RUNS = {
    "AB": "natural_sql_v2phase2_r0_AB_s0_steps2_20260908",
    "BA": "natural_sql_v2phase2_r0_BA_s0_steps2_20260908",
}
SCORE_KEYS = ["executable", "proxy_pass", "multi_instance_pass",
              "reward_disagreement", "additional_instances_tested"]


def mean(rows, key):
    return sum(float(row[key]) for row in rows) / len(rows)


def pearson(xs, ys):
    if len(xs) < 2 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
        return None
    xbar, ybar = statistics.mean(xs), statistics.mean(ys)
    return (sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys, strict=True)) /
            (sum((x - xbar) ** 2 for x in xs) * sum((y - ybar) ** 2 for y in ys)) ** 0.5)


def summarize(rows):
    return {
        "n": len(rows),
        "executable_rate": mean(rows, "executable"),
        "proxy_pass_rate": mean(rows, "proxy_pass"),
        "multi_instance_pass_rate": mean(rows, "multi_instance_pass"),
        "reward_disagreement_rate": mean(rows, "reward_disagreement"),
    }


def main():
    from sample_sql import score

    task_bytes = TASKS.read_bytes()
    task_data = json.loads(task_bytes)
    tasks = {row["task_id"]: row for split in task_data.values() for row in split}
    payloads = {}
    for history, run in RUNS.items():
        directory = ROLL / run
        if not (directory / "completed.json").exists():
            continue
        protocol = json.loads((directory / "protocol.json").read_text())
        if protocol["task_sha256"] != hashlib.sha256(task_bytes).hexdigest():
            raise ValueError(f"{history}: task hash mismatch")
        rows = [json.loads(line) for line in (directory / "natural_rollouts.jsonl").read_text().splitlines()]
        if len(rows) != 224:
            raise ValueError(f"{history}: expected 224 rollouts, got {len(rows)}")
        for row in rows:
            if row["sequence_token_ids"] != row["prompt_token_ids"] + row["response_token_ids"]:
                raise ValueError(f"{history}: token record mismatch")
            if row["response_start"] != len(row["prompt_token_ids"]):
                raise ValueError(f"{history}: response offset mismatch")
            rescored = score(tasks[row["task_id"]], row["response"])
            if any(rescored[key] != row[key] for key in SCORE_KEYS):
                raise ValueError(f"{history}: reward rescore mismatch for {row['task_id']}")
        payloads[history] = {"protocol": protocol, "rows": rows}

    output = {"completed_histories": sorted(payloads), "arms": {}, "paired": None}
    for history, payload in payloads.items():
        rows = payload["rows"]
        trajectory = {}
        for phase, step in [("heldout", 0), ("rl", 0), ("rl", 1), ("rl", 2), ("heldout", 2)]:
            selected = [row for row in rows if row["phase"] == phase and row["step"] == step]
            trajectory[f"{phase}_step_{step}"] = summarize(selected)
        task_level = {}
        for step in [0, 1, 2]:
            for task_id in payload["protocol"]["rl_tasks"]:
                selected = [row for row in rows
                            if row["phase"] == "rl" and row["step"] == step
                            and row["task_id"] == task_id]
                task_level.setdefault(task_id, {})[str(step)] = summarize(selected)
        initial_proxy = [task_level[task]["0"]["proxy_pass_rate"] for task in task_level]
        final_proxy = [task_level[task]["2"]["proxy_pass_rate"] for task in task_level]
        initial_disagreement = [task_level[task]["0"]["reward_disagreement_rate"] for task in task_level]
        final_disagreement = [task_level[task]["2"]["reward_disagreement_rate"] for task in task_level]
        output["arms"][history] = {
            "all_rollouts_independently_rescored": True,
            "trajectory": trajectory,
            "task_level": task_level,
            "exploratory_task_correlations": {
                "initial_proxy_vs_final_proxy": pearson(initial_proxy, final_proxy),
                "initial_disagreement_vs_final_disagreement": pearson(initial_disagreement, final_disagreement),
                "warning": "Sixteen reused tasks within one trajectory are not independent experimental replicates.",
            },
        }

    if set(payloads) == {"AB", "BA"}:
        ab, ba = payloads["AB"], payloads["BA"]
        ignored = {"history", "history_file", "history_sha256"}
        assert {k: v for k, v in ab["protocol"].items() if k not in ignored} == {
            k: v for k, v in ba["protocol"].items() if k not in ignored
        }
        ab_history = [json.loads(line) for line in Path(ab["protocol"]["history_file"]).read_text().splitlines()]
        ba_history = [json.loads(line) for line in Path(ba["protocol"]["history_file"]).read_text().splitlines()]
        ab_ids = [row["id"] for row in ab_history]
        ba_ids = [row["id"] for row in ba_history]
        midpoint = len(ab_ids) // 2
        assert ab_ids == ba_ids[midpoint:] + ba_ids[:midpoint]
        effects = {}
        for endpoint in ["proxy_pass_rate", "multi_instance_pass_rate", "reward_disagreement_rate"]:
            effects[f"rl_{endpoint}_change_AB_minus_BA"] = (
                output["arms"]["AB"]["trajectory"]["rl_step_2"][endpoint]
                - output["arms"]["AB"]["trajectory"]["rl_step_0"][endpoint]
                - output["arms"]["BA"]["trajectory"]["rl_step_2"][endpoint]
                + output["arms"]["BA"]["trajectory"]["rl_step_0"][endpoint]
            )
            effects[f"heldout_{endpoint}_change_AB_minus_BA"] = (
                output["arms"]["AB"]["trajectory"]["heldout_step_2"][endpoint]
                - output["arms"]["AB"]["trajectory"]["heldout_step_0"][endpoint]
                - output["arms"]["BA"]["trajectory"]["heldout_step_2"][endpoint]
                + output["arms"]["BA"]["trajectory"]["heldout_step_0"][endpoint]
            )
        output["paired"] = {
            "same_examples_reverse_block_order": True,
            "same_protocol_except_history": True,
            "difference_in_changes": effects,
            "scope": "One corpus pair and one sampling seed; descriptive pilot only.",
        }
    destination = ART / "natural_sql_pilot_summary.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
