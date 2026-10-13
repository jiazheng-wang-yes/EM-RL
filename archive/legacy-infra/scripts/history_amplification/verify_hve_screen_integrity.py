#!/usr/bin/env python3
"""Verify structure of a completed legacy HVE competence report without editing it."""

import argparse
import json
from pathlib import Path


TASKS = {
    "wordle_hidden_solution", "wordle_logical_bug", "sudoku_logical_bug",
    "minesweeper_logical_bug",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.report.read_text())
    rows = data["episodes"]
    expected = {(task, seed) for task in TASKS for seed in data["seeds"]}
    seen = {(row["task"], row["seed"]) for row in rows}
    if seen != expected or len(rows) != len(expected):
        raise ValueError("report does not contain exactly one record per task/seed cell")
    budget = data["max_prompt_tokens"]
    max_seen = 0
    for row in rows:
        if not row["trajectory"] or len(row["trajectory"]) > data["max_steps"]:
            raise ValueError("invalid trajectory length")
        for step in row["trajectory"]:
            count = step["prompt_info"]["prompt_tokens"]
            max_seen = max(max_seen, count)
            if count > budget:
                raise ValueError(f"prompt budget exceeded: {count}>{budget}")
    log_text = args.log.read_text(errors="replace")
    if "Traceback (most recent call last)" in log_text:
        raise ValueError("job log contains traceback")
    result = {
        "report": str(args.report),
        "log": str(args.log),
        "scope": (
            "Structural validation of a legacy report that predates explicit status fields. "
            "Does not alter raw output or establish model competence."
        ),
        "raw_status_field": data.get("status"),
        "expected_cells": len(expected),
        "observed_cells": len(rows),
        "maximum_recorded_prompt_tokens": max_seen,
        "prompt_budget": budget,
        "traceback_absent": True,
        "structurally_complete": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
