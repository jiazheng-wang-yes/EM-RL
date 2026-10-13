"""Summarize same-prompt constructed-proxy response controls by corpus pair."""
import argparse
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


def t95(values):
    if len(values) != 5:
        return None
    width = 2.776445 * statistics.stdev(values) / math.sqrt(5)
    return [statistics.mean(values) - width, statistics.mean(values) + width]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    jobs = json.loads(args.jobs.read_text())
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
    for replicate in sorted({job["replicate"] for job in jobs}):
        pair = {history: payloads.get((replicate, history)) for history in ("AB", "BA")}
        if not all(pair.values()):
            continue
        mechanisms = sorted(pair["AB"]["before"])
        if mechanisms != sorted(pair["BA"]["before"]):
            raise ValueError(f"r{replicate}: mechanism mismatch")
        cells = {}
        for update in mechanisms:
            for measured in mechanisms:
                ab = cell_mean(pair["AB"], update, measured)
                ba = cell_mean(pair["BA"], update, measured)
                cells[f"{update}__to__{measured}"] = {
                    "AB": ab, "BA": ba, "AB_minus_BA": ab - ba,
                    "pair_average": (ab + ba) / 2,
                }
        completed.append({"replicate": replicate, "cells": cells})

    aggregate = {}
    for name in sorted({name for pair in completed for name in pair["cells"]}):
        history_effect = [pair["cells"][name]["AB_minus_BA"] for pair in completed]
        response = [pair["cells"][name]["pair_average"] for pair in completed]
        aggregate[name] = {
            "n_corpus_pairs": len(response),
            "mean_pair_average_response": statistics.mean(response),
            "pair_average_response_values": response,
            "pair_average_response_positive_pairs": sum(value > 0 for value in response),
            "pair_average_response_t95": t95(response),
            "mean_AB_minus_BA": statistics.mean(history_effect),
            "AB_minus_BA_values": history_effect,
            "AB_minus_BA_t95": t95(history_effect),
        }
    update_contrasts = {}
    if completed:
        mechanisms = sorted({
            name.split("__to__")[0]
            for name in completed[0]["cells"]
        })
        if len(mechanisms) == 2:
            first, second = mechanisms
            for measured in mechanisms:
                values = [
                    pair["cells"][f"{first}__to__{measured}"]["pair_average"]
                    - pair["cells"][f"{second}__to__{measured}"]["pair_average"]
                    for pair in completed
                ]
                update_contrasts[f"{first}_minus_{second}__on__{measured}"] = {
                    "n_corpus_pairs": len(values),
                    "values": values,
                    "mean": statistics.mean(values),
                    "positive_pairs": sum(value > 0 for value in values),
                    "t95": t95(values),
                }
    result = {
        "completed_replicates": [pair["replicate"] for pair in completed],
        "pending_replicates": sorted({job["replicate"] for job in jobs} - {pair["replicate"] for pair in completed}),
        "paired_effects": completed,
        "aggregate": aggregate,
        "same_prompt_update_contrasts": update_contrasts,
        "scope": (
            "One-step constructed off-policy response on same prompts with two independently rejected SQL proxies. "
            "Corpus pairs are the replication unit; this is not natural on-policy RL."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
