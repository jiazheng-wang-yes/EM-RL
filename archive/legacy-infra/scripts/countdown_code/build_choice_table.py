#!/usr/bin/env python3
"""Collect the checker-choice screens into one CSV.

One row per arm and difficulty tier, plus a pooled row per arm. The columns that
matter are not the raw exploit rate, which is mostly a digit preference, but the
paired split -- the model choosing the lenient checker under both label mappings
-- and the same split restricted to responses that hold a correct answer, where
choosing the lenient checker cannot be a hedge against the model's own arithmetic.

``n_pairs_given_correct`` is the column to read first. Where it is zero the arm is
too damaged to express an answer, and nothing else in the row is a preference
measurement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "arm", "tier", "n_generations", "parse_rate", "task_accuracy", "mean_chars",
    "exploit_rate_raw", "position_bias_label1", "n_pairs", "prefers_lenient",
    "prefers_strict", "digit_locked", "n_pairs_given_correct",
    "prefers_lenient_given_correct", "prefers_strict_given_correct",
    "n_reliable_problems",
]


def rows_for(path: Path) -> list[dict]:
    summary = json.loads(path.read_text())
    arm = summary.get("label", path.stem)
    out = []
    for tier, entry in sorted(summary.get("difficulties", {}).items()):
        paired = entry.get("paired", {})
        out.append({
            "arm": arm, "tier": tier,
            "n_generations": entry.get("n_generations"),
            "parse_rate": entry.get("parse_rate"),
            "task_accuracy": entry.get("task_accuracy"),
            "mean_chars": entry.get("mean_chars"),
            "exploit_rate_raw": entry.get("exploit_rate"),
            "position_bias_label1": entry.get("position_bias_label1"),
            "n_pairs": paired.get("n_pairs"),
            "prefers_lenient": paired.get("content_prefers_lenient"),
            "prefers_strict": paired.get("content_prefers_strict"),
            "digit_locked": paired.get("digit_locked"),
            "n_pairs_given_correct": "", "prefers_lenient_given_correct": "",
            "prefers_strict_given_correct": "", "n_reliable_problems": "",
        })
    pooled, paired = summary.get("pooled", {}), summary.get("paired", {})
    given = summary.get("paired_given_correct", {})
    out.append({
        "arm": arm, "tier": "pooled",
        "n_generations": sum(e.get("n_generations") or 0
                             for e in summary.get("difficulties", {}).values()),
        "parse_rate": pooled.get("parse_rate"),
        "task_accuracy": pooled.get("task_accuracy"),
        "mean_chars": "",
        "exploit_rate_raw": pooled.get("exploit_rate"),
        "position_bias_label1": pooled.get("position_bias_label1"),
        "n_pairs": paired.get("n_pairs"),
        "prefers_lenient": paired.get("content_prefers_lenient"),
        "prefers_strict": paired.get("content_prefers_strict"),
        "digit_locked": paired.get("digit_locked"),
        "n_pairs_given_correct": given.get("n_pairs"),
        "prefers_lenient_given_correct": given.get("content_prefers_lenient"),
        "prefers_strict_given_correct": given.get("content_prefers_strict"),
        "n_reliable_problems": summary.get("n_reliable_problems", ""),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen-dir", type=Path,
                        default=Path("eval_runs/choice_env_screen"))
    parser.add_argument("--out-csv", type=Path,
                        default=Path("eval_runs/choice_env_screen/choice_screen_table.csv"))
    args = parser.parse_args()

    rows = []
    for path in sorted(args.screen_dir.glob("*.json")):
        rows.extend(rows_for(path))
    if not rows:
        raise SystemExit(f"no screen summaries under {args.screen_dir}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row["tier"] == "pooled":
            print(f"{row['arm']:<16} parse={row['parse_rate']:<7} task={row['task_accuracy']:<7} "
                  f"pairs={row['n_pairs']:<5} lenient={row['prefers_lenient']:<7} "
                  f"strict={row['prefers_strict']:<7} given_correct={row['n_pairs_given_correct']}")
    print(f"wrote {len(rows)} rows to {args.out_csv}")


if __name__ == "__main__":
    main()
