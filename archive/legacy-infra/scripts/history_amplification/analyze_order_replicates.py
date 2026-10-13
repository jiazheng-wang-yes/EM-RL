"""Use SFT corpus replicate, not SQL task, as the unit of comparison."""
import argparse
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ROLL = ROOT / "logs/history_amplification/rollouts"
ART = ROOT / "eval_runs/history_amplification/order_history_replicates"


def relative(payload):
    rows = {}
    for group, before, after in zip(payload["groups"], payload["before"], payload["after"], strict=True):
        change = [after[i]["mean_logprob"] - before[i]["mean_logprob"] for i in range(4)]
        rows[group["task_id"]] = change[1] - change[0]
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ART)
    args = parser.parse_args()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else ROOT / args.artifact_dir
    jobs = json.loads((artifact_dir / "controlled_update_jobs.json").read_text())
    by_replicate = {}
    for job in jobs:
        path = ROLL / job["run_name"] / "controlled_update.json"
        if path.exists():
            payload = json.loads(path.read_text())
            by_replicate.setdefault(job["replicate"], {})[job["history"]] = payload
    completed, pending = [], []
    for replicate in range(5):
        pair = by_replicate.get(replicate, {})
        if set(pair) != {"AB", "BA"}:
            pending.append(replicate)
            continue
        ab_payload, ba_payload = pair["AB"], pair["BA"]
        ab_ids = ab_payload["protocol"]["sft_example_ids"]
        ba_ids = ba_payload["protocol"]["sft_example_ids"]
        midpoint = len(ab_ids) // 2
        assert ab_ids == ba_ids[midpoint:] + ba_ids[:midpoint]
        assert ab_payload["protocol"]["sft_tokens"] == ba_payload["protocol"]["sft_tokens"]
        for ab_group, ba_group in zip(ab_payload["groups"], ba_payload["groups"], strict=True):
            assert ab_group["task_id"] == ba_group["task_id"]
            assert ab_group["candidates"] == ba_group["candidates"]
            assert ab_group["rewards"] == ba_group["rewards"]
        for payload in [ab_payload, ba_payload]:
            assert payload["before"] == payload["before_repeat"]
        ab_relative, ba_relative = relative(ab_payload), relative(ba_payload)
        assert set(ab_relative) == set(ba_relative)
        task_deltas = [ab_relative[task] - ba_relative[task] for task in ab_relative]
        completed.append({
            "replicate": replicate,
            "n_tasks": len(task_deltas),
            "AB_minus_BA_mean_relative_change": sum(task_deltas) / len(task_deltas),
            "AB_minus_BA_task_sign_count": sum(value > 0 for value in task_deltas),
            "identical_example_multiset_reverse_block_order": True,
            "equal_full_sequence_token_exposure": True,
            "identical_candidates_and_rewards": True,
            "repeated_forwards_identical": True,
            "task_deltas": task_deltas,
        })
    means = [row["AB_minus_BA_mean_relative_change"] for row in completed]
    output = {
        "completed_replicates": completed,
        "pending_or_failed_replicates": pending,
        "across_corpus_replicates": {
            "n": len(means),
            "mean": sum(means) / len(means) if means else None,
            "min": min(means) if means else None,
            "max": max(means) if means else None,
            "positive_replicates": sum(value > 0 for value in means),
            "sample_sd": statistics.stdev(means) if len(means) > 1 else None,
            "standard_error": statistics.stdev(means) / math.sqrt(len(means)) if len(means) > 1 else None,
            "two_sided_t_95_interval_n5": (
                [statistics.mean(means) - 2.776445 * statistics.stdev(means) / math.sqrt(5),
                 statistics.mean(means) + 2.776445 * statistics.stdev(means) / math.sqrt(5)]
                if len(means) == 5 else None
            ),
            "interpretation": "Corpus-replicate descriptive summary. All runs use one fixed base checkpoint and deterministic SFT; this is not an independent model-family or random-initialization replication.",
        },
    }
    (artifact_dir / "controlled_update_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
