"""Aggregate paired history effects in full shortcut-by-update response matrices."""
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ROLL = ROOT / "logs/history_amplification/rollouts"
ART = ROOT / "eval_runs/history_amplification/order_history_replicates_v2"


def cell_mean(payload, update, measured):
    values = []
    for before, after in zip(
        payload["before"][measured],
        payload["after_by_update_mechanism"][update][measured],
        strict=True,
    ):
        values.append((after[1]["mean_logprob"] - before[1]["mean_logprob"])
                      - (after[0]["mean_logprob"] - before[0]["mean_logprob"]))
    return statistics.mean(values)


def main():
    jobs = [job for job in json.loads((ART / "mechanism_assay_jobs.json").read_text())
            if not job.get("smoke_only") and not job.get("pilot_only")]
    payloads = {}
    for job in jobs:
        path = ROLL / job["run_name"] / "mechanism_assay.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload["before"] != payload["before_repeat"]:
            raise ValueError(f"{job['run_name']}: repeated forwards differ")
        payloads[(job["replicate"], job["history"])] = payload
    completed = []
    for replicate in range(5):
        pair = {history: payloads.get((replicate, history)) for history in ["AB", "BA"]}
        if not all(pair.values()):
            continue
        ab_history = [json.loads(line) for line in Path(pair["AB"]["protocol"]["history_file"]).read_text().splitlines()]
        ba_history = [json.loads(line) for line in Path(pair["BA"]["protocol"]["history_file"]).read_text().splitlines()]
        ab_ids, ba_ids = [row["id"] for row in ab_history], [row["id"] for row in ba_history]
        midpoint = len(ab_ids) // 2
        if ab_ids != ba_ids[midpoint:] + ba_ids[:midpoint]:
            raise ValueError(f"r{replicate}: histories are not reverse block order")
        mechanisms = sorted(pair["AB"]["before"])
        if mechanisms != sorted(pair["BA"]["before"]):
            raise ValueError(f"r{replicate}: mechanisms mismatch")
        cells = {}
        for update in mechanisms:
            for measured in mechanisms:
                ab = cell_mean(pair["AB"], update, measured)
                ba = cell_mean(pair["BA"], update, measured)
                cells[f"{update}__to__{measured}"] = {
                    "AB": ab, "BA": ba, "AB_minus_BA": ab - ba,
                }
        completed.append({
            "replicate": replicate, "same_examples_reverse_block_order": True,
            "repeated_forwards_identical": True, "cells": cells,
        })
    all_cell_names = sorted({name for item in completed for name in item["cells"]})
    aggregate = {}
    for name in all_cell_names:
        values = [item["cells"][name]["AB_minus_BA"] for item in completed]
        pair_average_response = [
            (item["cells"][name]["AB"] + item["cells"][name]["BA"]) / 2
            for item in completed
        ]
        aggregate[name] = {
            "n_corpus_pairs": len(values), "mean_AB_minus_BA": statistics.mean(values),
            "positive_pairs": sum(value > 0 for value in values),
            "values": values,
            "sample_sd": statistics.stdev(values) if len(values) > 1 else None,
            "t95": ([statistics.mean(values) - 2.776445 * statistics.stdev(values) / math.sqrt(5),
                      statistics.mean(values) + 2.776445 * statistics.stdev(values) / math.sqrt(5)]
                    if len(values) == 5 else None),
            "pair_average_response": pair_average_response,
            "mean_pair_average_response": statistics.mean(pair_average_response),
            "pair_average_response_positive_pairs": sum(value > 0 for value in pair_average_response),
            "pair_average_response_t95": (
                [statistics.mean(pair_average_response) - 2.776445 * statistics.stdev(pair_average_response) / math.sqrt(5),
                 statistics.mean(pair_average_response) + 2.776445 * statistics.stdev(pair_average_response) / math.sqrt(5)]
                if len(pair_average_response) == 5 else None
            ),
        }
    output = {
        "completed_replicates": [item["replicate"] for item in completed],
        "pending_replicates": [rep for rep in range(5) if rep not in {item["replicate"] for item in completed}],
        "paired_replicate_effects": completed,
        "across_corpus_replicates": aggregate,
        "scope": "Controlled off-policy response matrices with one SFT seed and data-resampled corpus pairs. Corpus pairs, not SQL tasks, are the replication unit.",
    }
    (ART / "mechanism_assay_replicate_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
