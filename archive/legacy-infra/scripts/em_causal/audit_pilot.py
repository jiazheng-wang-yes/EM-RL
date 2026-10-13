"""Verify a completed pilot and summarize effects without treating variants as replicates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import file_hash, manifest, read_rows, write_json
from .data import score_response


def audit(raw: Path, output: Path):
    reports = {}
    hashes = {}
    reference_keys = None
    for name in ("reference", "risky", "safe"):
        path = raw / f"{name}_validation.jsonl"
        rows = read_rows(path)
        complete = json.loads(path.with_suffix(".complete.json").read_text())
        if complete["sha256"] != file_hash(path) or complete["rows"] != len(rows):
            raise ValueError(f"Completion/hash mismatch: {path}")
        hashes[str(path)] = file_hash(path)
        keys = [(r["condition"], r["id"], r["sample"]) for r in rows]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate generation")
        for row in rows:
            rescored = score_response(row, row["response"], row["finish_reason"] == "length")
            if any(row["metrics"].get(k) != v for k, v in rescored.items()):
                raise ValueError(f"Stored score differs from raw-response rescore: {row['id']}")
        if name == "reference":
            reference_keys = {(r["id"], r["sample"]) for r in rows}
        untreated = {r["id"]: r for r in rows if r["condition"] == "sft"}
        conditions = []
        for condition in sorted({r["condition"] for r in rows}):
            selected = [r for r in rows if r["condition"] == condition]
            if {(r["id"], r["sample"]) for r in selected} != reference_keys:
                raise ValueError("Condition differs from original-model question/sample coverage")
            if untreated and {r["id"] for r in selected} != untreated.keys():
                raise ValueError("Unpaired conditions")
            metrics = {}
            for metric in ("instruction_following", "content_correct", "truncated", "empty"):
                values = [r["metrics"][metric] for r in selected if r["metrics"].get(metric) is not None]
                metrics[metric] = {"numerator": sum(values), "denominator": len(values), "rate": sum(values) / len(values) if values else None}
            conditions.append({"condition": condition, "metrics": metrics,
                               "text_changed_from_sft": sum(r["response"] != untreated[r["id"]]["response"] for r in selected) if untreated else None})
        reports[name] = {"generations": len(rows), "question_groups": len({r["group_id"] for r in rows}), "conditions": conditions}
    result = manifest(kind="completed_pilot_audit", reports=reports, files=hashes, raw_rescore_verified=True,
                      limitations=["Four question groups are not a powered mechanism experiment.",
                                   "Rank-one and rank-four failures at one site do not establish high dimensionality.",
                                   "No independently judged EM endpoint in this pilot."])
    write_json(output, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(audit(args.raw, args.output), indent=2))
