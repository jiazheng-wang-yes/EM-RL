"""Summarize shortcut-by-update response matrices from consolidated assays."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ROLL = ROOT / "logs/history_amplification/rollouts"
ART = ROOT / "eval_runs/history_amplification/order_history_replicates_v2"


def response(before, after, field):
    values = []
    for initial_group, final_group in zip(before, after, strict=True):
        intended_change = final_group[0][field] - initial_group[0][field]
        proxy_change = final_group[1][field] - initial_group[1][field]
        values.append(proxy_change - intended_change)
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "positive_tasks": sum(value > 0 for value in values),
        "n_tasks": len(values),
        "per_task": values,
    }


def main():
    jobs = json.loads((ART / "mechanism_assay_jobs.json").read_text())
    runs = {}
    for job in jobs:
        if job.get("smoke_only") or job.get("pilot_only"):
            continue
        path = ROLL / job["run_name"] / "mechanism_assay.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload["before"] != payload["before_repeat"]:
            raise ValueError(f"{job['run_name']}: repeated forwards differ")
        mechanisms = sorted(payload["before"])
        matrix = {}
        for update_mechanism in mechanisms:
            matrix[update_mechanism] = {}
            for measured_mechanism in mechanisms:
                matrix[update_mechanism][measured_mechanism] = {
                    field: response(
                        payload["before"][measured_mechanism],
                        payload["after_by_update_mechanism"][update_mechanism][measured_mechanism],
                        field,
                    )
                    for field in ["mean_logprob", "sum_logprob"]
                }
        runs[f"r{job['replicate']}_{job['history']}"] = {
            "run_name": job["run_name"], "matrix": matrix,
            "repeated_forwards_identical": True,
        }
    output = {
        "completed_runs": sorted(runs),
        "runs": runs,
        "interpretation": "Each row is a constructed update mechanism and each column is a measured proxy-relative-to-intended mechanism. This is an off-policy local response matrix, not natural discovery.",
    }
    (ART / "mechanism_assay_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
