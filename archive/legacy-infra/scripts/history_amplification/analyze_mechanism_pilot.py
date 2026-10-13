"""Analyze reduced paired response matrices separately from full assays."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
ROLL = ROOT / "logs/history_amplification/rollouts"
ART = ROOT / "eval_runs/history_amplification/order_history_replicates_v2"


def matrix(payload, field):
    result = {}
    for update_mechanism, after_all in payload["after_by_update_mechanism"].items():
        result[update_mechanism] = {}
        for measured_mechanism, before_groups in payload["before"].items():
            values = []
            for before, after in zip(before_groups, after_all[measured_mechanism], strict=True):
                intended = after[0][field] - before[0][field]
                proxy = after[1][field] - before[1][field]
                values.append(proxy - intended)
            result[update_mechanism][measured_mechanism] = {
                "mean": statistics.mean(values), "values": values,
            }
    return result


def main():
    jobs = [job for job in json.loads((ART / "mechanism_assay_jobs.json").read_text())
            if job.get("pilot_only")]
    runs = {}
    for job in jobs:
        path = ROLL / job["run_name"] / "mechanism_assay.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        if payload["before"] != payload["before_repeat"]:
            raise ValueError(f"{job['run_name']}: repeated forwards differ")
        matrices = {field: matrix(payload, field) for field in ["mean_logprob", "sum_logprob"]}
        means = matrices["mean_logprob"]
        diagonal = [means[name][name]["mean"] for name in means]
        off_diagonal = [cell["mean"] for row, columns in means.items()
                        for column, cell in columns.items() if row != column]
        runs[job["history"]] = {
            "run_name": job["run_name"], "matrices": matrices,
            "mean_token_diagonal_mean": statistics.mean(diagonal),
            "mean_token_off_diagonal_mean": statistics.mean(off_diagonal),
            "repeated_forwards_identical": True,
        }
    comparison = None
    if set(runs) == {"AB", "BA"}:
        mechanisms = sorted(runs["AB"]["matrices"]["mean_logprob"])
        cells = []
        for update in mechanisms:
            for measured in mechanisms:
                ab = runs["AB"]["matrices"]["mean_logprob"][update][measured]["mean"]
                ba = runs["BA"]["matrices"]["mean_logprob"][update][measured]["mean"]
                cells.append({"update": update, "measured": measured,
                              "AB_minus_BA": ab - ba})
        comparison = {"mean_token_cell_differences": cells}
    output = {
        "completed_histories": sorted(runs), "runs": runs, "comparison": comparison,
        "scope": "Three tasks per mechanism, one corpus pair, one seed; implementation and signal pilot only.",
    }
    (ART / "mechanism_assay_pilot_summary.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
