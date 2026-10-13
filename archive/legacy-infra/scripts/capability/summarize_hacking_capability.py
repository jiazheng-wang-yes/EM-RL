#!/usr/bin/env python3
"""Join the hacking-checkpoint capability battery into one table.

Reads the per-arm lm-eval outputs under
``eval_runs/minimal_finance_dose/capability/<arm>/`` and emits one row per arm with
the headline metric of each task, next to that arm's Countdown RL hack onset and its
EM rate. The point of the join is the mechanism claim: if RL hacks because finance
SFT removed the honest path, a checkpoint that hacks should also have lost general
ability relative to its base model.

Each arm directory holds two subdirectories, the checkpoint's own results and a
re-run of the base model. Only the checkpoint's are reported here; the base rows are
identical across arms and are emitted once, as the ``*_base`` arms.

Usage:
  python summarize_hacking_capability.py
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAP_ROOT = PROJECT_ROOT / "eval_runs/minimal_finance_dose/capability"

# (task, exact metric key including its filter suffix) -> column label.
#
# The filter suffix is not optional. gsm8k reports both ``exact_match,strict-match``
# and ``exact_match,flexible-extract``, and they answer different questions here:
# strict-match requires the "#### N" output format, so it confounds arithmetic with
# instruction-following, while flexible-extract pulls the last number out of whatever
# the model wrote and measures the arithmetic alone. Reporting only one hides exactly
# the distinction this table exists to make, and matching on the prefix silently
# picked whichever came first in the JSON.
#
# ``mbpp_instruct`` is deliberately absent: its ``pass_at_1,extract_code`` reads 0.0
# for the base models too, so its code extractor is failing rather than the models.
HEADLINE = {
    ("ifeval", "prompt_level_strict_acc,none"): "ifeval_strict",
    ("ifeval", "inst_level_loose_acc,none"): "ifeval_inst_loose",
    ("gsm8k", "exact_match,flexible-extract"): "gsm8k_flex",
    ("gsm8k", "exact_match,strict-match"): "gsm8k_strict",
    ("humaneval_instruct", "pass@1,create_test"): "humaneval",
    ("truthfulqa_gen", "bleu_acc,none"): "truthfulqa_bleu_acc",
}

# arm -> (model, dose label, RL hack onset, EM rate)
ARM_META = {
    "q3_base":      ("Qwen3-1.7B", "base",          "",          0.0),
    "q3_n0736_e1":  ("Qwen3-1.7B", "736x1ep",       "95",        None),
    "q3_n1472_e1":  ("Qwen3-1.7B", "1472x1ep",      "94",        None),
    "q3_n2944_e1":  ("Qwen3-1.7B", "2944x1ep",      "77",        None),
    "q3_n1952_e2":  ("Qwen3-1.7B", "1952x2ep",      "65/63/93",  0.08125),
    "q3_n5872_e2":  ("Qwen3-1.7B", "5872x2ep",      "71",        None),
    "q25_base":     ("Qwen2.5-3B-Instruct", "base", "",          0.0),
    "q25_n1952_e2": ("Qwen2.5-3B-Instruct", "1952x2ep", "95",    0.10625),
    "q25_n2944_e2": ("Qwen2.5-3B-Instruct", "2944x2ep", "69",    0.18125),
}


def newest_results(arm_dir: Path, want_base: bool) -> dict | None:
    """Return the parsed results for the checkpoint (or the base re-run) in one arm."""
    best: tuple[float, dict] | None = None
    for path in arm_dir.glob("*/results*.json"):
        is_base = path.parent.name.startswith("Qwen_")
        if is_base != want_base:
            continue
        mtime = path.stat().st_mtime
        if best is None or mtime > best[0]:
            best = (mtime, json.loads(path.read_text()))
    return best[1] if best else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", type=Path,
                        default=PROJECT_ROOT / "eval_runs/minimal_finance_dose/capability_table.csv")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for arm, (model, dose, onset, em) in ARM_META.items():
        arm_dir = CAP_ROOT / arm
        if not arm_dir.is_dir():
            continue
        data = newest_results(arm_dir, want_base=arm.endswith("_base"))
        if data is None:
            continue
        row: dict[str, object] = {"arm": arm, "model": model, "dose": dose,
                                  "hack_onset": onset, "em_rate": em if em is not None else ""}
        for (task, key), label in HEADLINE.items():
            hit = data.get("results", {}).get(task, {}).get(key)
            row[label] = round(hit, 4) if isinstance(hit, (int, float)) else ""
        rows.append(row)

    rows.sort(key=lambda r: (r["model"], r["dose"] != "base", str(r["dose"])))
    width = {k: max(len(str(k)), max((len(str(r[k])) for r in rows), default=0)) for k in rows[0]}
    print("  ".join(str(k).ljust(width[k]) for k in rows[0]))
    for row in rows:
        print("  ".join(str(v).ljust(width[k]) for k, v in row.items()))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} arms to {args.out_csv}")


if __name__ == "__main__":
    main()
