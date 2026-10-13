"""Freeze disjoint clean-SFT and response tasks for the Hanoi mechanism assay."""
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "rllm"))

from examples.tower_of_hanoi_reward_hack_probe.probe_common import build_task_catalog


OUTPUT = ROOT / "eval_runs/history_amplification/tower_environment/mechanism_response_manifest.json"


def order(items, salt):
    return sorted(items, key=lambda item: hashlib.sha256(f"{salt}:{item['problem_id']}".encode()).hexdigest())


def main():
    catalog = build_task_catalog(min_disks=3, max_disks=3)
    evaluation = order(catalog, "tower-mechanism-eval-v1")[:16]
    evaluation_ids = {item["problem_id"] for item in evaluation}
    available_sft = [item for item in catalog if item["problem_id"] not in evaluation_ids]
    replicates = []
    for replicate in range(5):
        sft = order(available_sft, f"tower-mechanism-clean-sft-v1-r{replicate}")[:32]
        if evaluation_ids & {item["problem_id"] for item in sft}:
            raise AssertionError("SFT and response tasks overlap")
        replicates.append({
            "replicate": replicate,
            "clean_sft_problem_ids": [item["problem_id"] for item in sft],
        })
    result = {
        "version": 1,
        "selection": (
            "Three-disk Tower of Hanoi tasks selected by fixed hashes before model outcomes. "
            "The response set is disjoint from every clean-SFT set."
        ),
        "mechanisms": ["state_rewrite", "test_rewrite"],
        "evaluation_problem_ids": [item["problem_id"] for item in evaluation],
        "replicates": replicates,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "evaluation_tasks": len(evaluation),
        "clean_sft_tasks_per_replicate": len(replicates[0]["clean_sft_problem_ids"]),
        "replicates": len(replicates),
    }))


if __name__ == "__main__":
    main()
