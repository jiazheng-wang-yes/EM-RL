#!/usr/bin/env python3
"""Summarize HVE competence screens without upgrading native flags to causality."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(successes, n, z=1.959963984540054):
    if n == 0:
        return [None, None]
    p = successes / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [center - half, center + half]


def event_for(row):
    if row["task"] == "wordle_hidden_solution":
        return bool(row.get("protected_content_disclosed", False))
    return bool(row.get("native_hack_triggered", False))


def summarize(rows):
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task"]].append(row)
    result = {}
    for task, items in sorted(by_task.items()):
        n = len(items)
        success = [r.get("reward") == 1 for r in items]
        event = [event_for(r) for r in items]
        joint = [s and e for s, e in zip(success, event)]
        clean = [s and not e for s, e in zip(success, event)]
        terminated = [bool(r.get("terminated", False)) for r in items]
        steps = [step for r in items for step in r.get("trajectory", [])]
        truncated = [bool(step.get("prompt_info", {}).get("truncated_observation", False))
                     for step in steps]
        history_dropped = [step.get("prompt_info", {}).get("dropped_history_turns", 0) > 0
                           for step in steps]
        result[task] = {
            "n": n,
            "termination": {"count": sum(terminated), "rate": sum(terminated) / n,
                            "wilson95": wilson(sum(terminated), n)},
            "reward_one": {"count": sum(success), "rate": sum(success) / n,
                           "wilson95": wilson(sum(success), n)},
            "mechanism_event": {"count": sum(event), "rate": sum(event) / n,
                                "wilson95": wilson(sum(event), n)},
            "event_plus_success": {"count": sum(joint), "rate": sum(joint) / n,
                                   "wilson95": wilson(sum(joint), n)},
            "clean_success": {"count": sum(clean), "rate": sum(clean) / n,
                              "wilson95": wilson(sum(clean), n)},
            "prompt_budget": {
                "steps": len(steps),
                "observation_truncated_count": sum(truncated),
                "observation_truncated_rate": (sum(truncated) / len(steps) if steps else None),
                "history_dropped_count": sum(history_dropped),
                "history_dropped_rate": (sum(history_dropped) / len(steps) if steps else None),
                "maximum_prompt_tokens": max(
                    (step.get("prompt_info", {}).get("prompt_tokens", 0) for step in steps),
                    default=None,
                ),
            },
            "seed_records": [
                {"seed": r["seed"], "reward": r.get("reward"), "terminated": r.get("terminated"),
                 "mechanism_event": e, "event_plus_success": j, "clean_success": c}
                for r, e, j, c in zip(items, event, joint, clean)
            ],
        }
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("reports", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    models = {}
    for path in args.reports:
        data = json.loads(path.read_text())
        if data.get("status") != "completed":
            raise ValueError(f"refusing incomplete report {path}: status={data.get('status')!r}")
        models[data["model"]] = summarize(data["episodes"])
    result = {
        "scope": (
            "Pre-RL task-selection summary. event_plus_success is temporal co-occurrence, "
            "not proof that the mechanism caused success."
        ),
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
