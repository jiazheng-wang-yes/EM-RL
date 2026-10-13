"""Analyze candidate log-probability changes from completed controlled pilots."""
import json
import math
from pathlib import Path
import random
import argparse

ROOT = Path(__file__).resolve().parents[2]
ROLL = ROOT / "logs/history_amplification/rollouts"
OUT = ROOT / "eval_runs/history_amplification/order_histories/pilot_summary.json"


def mean(xs):
    return sum(xs) / len(xs)


def summarize(history, label):
    path = ROLL / f"order_sql_update_{history}{label}_s0_20260908/controlled_update.json"
    payload = json.loads(path.read_text())
    repeats = []
    rows = []
    for group, before, repeat, after in zip(
        payload["groups"], payload["before"], payload["before_repeat"], payload["after"], strict=True
    ):
        # Candidate order is reference, hardcoded answer, empty query, invalid SQL.
        for candidate in range(4):
            repeats.append(abs(before[candidate]["mean_logprob"] - repeat[candidate]["mean_logprob"]))
        changes = [after[i]["mean_logprob"] - before[i]["mean_logprob"] for i in range(4)]
        rows.append({
            "task_id": group["task_id"],
            "rewards": group["rewards"],
            "reference_change": changes[0],
            "hardcoded_change": changes[1],
            "empty_change": changes[2],
            "invalid_change": changes[3],
            "hardcoded_minus_reference_change": changes[1] - changes[0],
            "hardcoded_minus_reference_before": before[1]["mean_logprob"] - before[0]["mean_logprob"],
        })
    return {
        "history": history,
        "n_groups": len(rows),
        "max_repeated_forward_abs_difference": max(repeats),
        "mean_reference_change": mean([r["reference_change"] for r in rows]),
        "mean_hardcoded_change": mean([r["hardcoded_change"] for r in rows]),
        "mean_hardcoded_minus_reference_change": mean([r["hardcoded_minus_reference_change"] for r in rows]),
        "mean_hardcoded_minus_reference_before": mean([r["hardcoded_minus_reference_before"] for r in rows]),
        "groups_favoring_hardcoded_relative_to_reference": sum(r["hardcoded_minus_reference_change"] > 0 for r in rows),
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="", help="Run-name insertion, for example _eager")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    completed = [h for h in ["base", "AB", "BA", "interleaved"]
                 if (ROLL / f"order_sql_update_{h}{args.label}_s0_20260908/controlled_update.json").exists()]
    summaries = [summarize(h, args.label) for h in completed]
    by_history = {x["history"]: x for x in summaries}
    contrasts = {}
    if "AB" in by_history and "BA" in by_history:
        a, b = by_history["AB"], by_history["BA"]
        contrasts["AB_minus_BA_mean_relative_change"] = (
            a["mean_hardcoded_minus_reference_change"] - b["mean_hardcoded_minus_reference_change"]
        )
        task_differences = [
            ar["hardcoded_minus_reference_change"] - br["hardcoded_minus_reference_change"]
            for ar, br in zip(a["rows"], b["rows"], strict=True)
        ]
        assert all(ar["task_id"] == br["task_id"] for ar, br in zip(a["rows"], b["rows"], strict=True))
        contrasts["AB_minus_BA_task_sd"] = math.sqrt(
            sum((x - mean(task_differences)) ** 2 for x in task_differences) / (len(task_differences) - 1)
        )
        contrasts["AB_minus_BA_groups_positive"] = sum(x > 0 for x in task_differences)
        rng = random.Random(20260908)
        bootstrap = sorted(mean([rng.choice(task_differences) for _ in task_differences]) for _ in range(10000))
        contrasts["paired_task_bootstrap_95_interval"] = [bootstrap[249], bootstrap[9749]]
        contrasts["interpretation"] = (
            "Descriptive paired-task contrast from one SFT seed and one constructed off-policy update; "
            "tasks are repeated measurements, not independent training replicates."
        )
    result = {"completed_histories": completed, "summaries": summaries, "contrasts": contrasts}
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"completed_histories": completed,
                      "summary": [{k: v for k, v in x.items() if k != "rows"} for x in summaries],
                      "contrasts": contrasts}, indent=2))


if __name__ == "__main__":
    main()
