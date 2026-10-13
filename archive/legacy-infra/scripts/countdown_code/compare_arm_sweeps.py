#!/usr/bin/env python3
"""Join two matched SFT arms and search for behaviour-matched checkpoint pairs.

``summarize_cross_stage_sweep.py`` describes one arm, indexed by SFT step. This script
puts two arms side by side and answers the question the study is built around:

    is there a pair of checkpoints that behave the same but learn differently?

Two views are produced.

**Step-aligned.** Both arms see the same prompts in the same order at the same optimizer
step, so comparing them at equal steps holds training history fixed and varies only the
response target. This is the primary table.

**Behaviour-matched.** A pair may also be interesting at unequal steps, when the risky
arm at step s happens to behave like the clean arm at step t. The behavioural distance
below is deliberately restricted to quantities an ordinary evaluation would report:
format and runnable rates, honest solving, pre-RL cheating, broad misalignment, and the
position-bias-free exploit preference. Downstream RL outcomes are excluded by
construction, because the whole claim is that they are not predictable from this vector.

Features are standardized across the pooled set of checkpoints before distancing, so a
feature with a wide range does not dominate one with a narrow range.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Iterable


BEHAVIOUR_FEATURES = (
    "format_pass_rate",
    "runnable_test_rate",
    "honest_solve_rate",
    "cheating_rate",
    "broad_em_rate",
    "d_hack",
)

OUTCOME_FIELDS = ("s_t_hat", "s_t_hat_observed", "tau_hack_median_censored", "mean_peak_hack")


def read_arm(path: Path) -> dict[int, dict[str, Any]]:
    """Load one arm's sweep CSV, keeping only rows that carry diagnostics."""
    rows: dict[int, dict[str, Any]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("format_pass_rate") and not row.get("d_hack"):
                continue
            rows[int(row["sft_step"])] = row
    if not rows:
        raise SystemExit(f"no diagnostic rows in {path}")
    return rows


def _number(row: dict[str, Any], field: str) -> float | None:
    value = row.get(field)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def standardizers(
    arms: Iterable[dict[int, dict[str, Any]]], features: tuple[str, ...]
) -> dict[str, tuple[float, float]]:
    """Mean and spread per feature over every checkpoint in both arms."""
    stats: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = [
            v for arm in arms for row in arm.values() if (v := _number(row, feature)) is not None
        ]
        if len(values) < 2:
            stats[feature] = (0.0, 1.0)
            continue
        spread = statistics.pstdev(values)
        stats[feature] = (statistics.fmean(values), spread if spread > 1e-12 else 1.0)
    return stats


def behaviour_distance(
    left: dict[str, Any], right: dict[str, Any], stats: dict[str, tuple[float, float]],
    features: tuple[str, ...],
) -> tuple[float | None, list[str]]:
    """Standardized Euclidean distance over the features both rows report."""
    squares = 0.0
    used: list[str] = []
    for feature in features:
        a, b = _number(left, feature), _number(right, feature)
        if a is None or b is None:
            continue
        _, spread = stats[feature]
        squares += ((a - b) / spread) ** 2
        used.append(feature)
    if not used:
        return None, []
    # Divide by the count so pairs compared on different feature subsets stay comparable.
    return math.sqrt(squares / len(used)), used


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--risky-csv", type=Path, required=True)
    parser.add_argument("--clean-csv", type=Path, required=True)
    parser.add_argument("--features", nargs="*", default=list(BEHAVIOUR_FEATURES))
    parser.add_argument("--max-distance", type=float, default=0.25,
                        help="behavioural distance below which a cross-step pair is reported")
    parser.add_argument("--top-pairs", type=int, default=15)
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    features = tuple(args.features)
    risky, clean = read_arm(args.risky_csv), read_arm(args.clean_csv)
    stats = standardizers([risky, clean], features)

    aligned: list[dict[str, Any]] = []
    for step in sorted(set(risky) & set(clean)):
        distance, used = behaviour_distance(risky[step], clean[step], stats, features)
        record: dict[str, Any] = {"sft_step": step, "behaviour_distance": distance,
                                  "features_used": ";".join(used)}
        for feature in features:
            record[f"risky_{feature}"] = _number(risky[step], feature)
            record[f"clean_{feature}"] = _number(clean[step], feature)
        for field in OUTCOME_FIELDS:
            record[f"risky_{field}"] = _number(risky[step], field)
            record[f"clean_{field}"] = _number(clean[step], field)
        aligned.append(record)

    pairs = []
    for risky_step, risky_row in risky.items():
        for clean_step, clean_row in clean.items():
            distance, used = behaviour_distance(risky_row, clean_row, stats, features)
            if distance is None or distance > args.max_distance:
                continue
            pairs.append({
                "risky_step": risky_step, "clean_step": clean_step,
                "behaviour_distance": distance, "features_used": ";".join(used),
                "risky_s_t_hat": _number(risky_row, "s_t_hat"),
                "clean_s_t_hat": _number(clean_row, "s_t_hat"),
            })
    pairs.sort(key=lambda p: p["behaviour_distance"])

    print(f"step-aligned comparison ({len(aligned)} shared steps)")
    print(f"{'step':>6} {'B-dist':>8} " + " ".join(f"{f[:11]:>12}" for f in features))
    for record in aligned:
        cells = []
        for feature in features:
            a, b = record[f"risky_{feature}"], record[f"clean_{feature}"]
            cells.append("     n/a    " if a is None or b is None else f"{a:>5.2f}/{b:<5.2f} ".rjust(12))
        distance = record["behaviour_distance"]
        print(f"{record['sft_step']:>6} {(float('nan') if distance is None else distance):>8.3f} " + " ".join(cells))
    print("\n(each cell is risky/clean)")

    print(f"\nbehaviour-matched pairs within {args.max_distance} (top {args.top_pairs}):")
    if not pairs:
        print("  none; no risky checkpoint currently resembles a clean one this closely")
    for pair in pairs[: args.top_pairs]:
        print(f"  risky s{pair['risky_step']:<5} ~ clean s{pair['clean_step']:<5} "
              f"B-dist={pair['behaviour_distance']:.3f}  "
              f"S_T risky={pair['risky_s_t_hat']} clean={pair['clean_s_t_hat']}")

    if args.out_csv and aligned:
        with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(aligned[0]))
            writer.writeheader()
            writer.writerows(aligned)
        print(f"\nwrote {args.out_csv}")
    if args.out_json:
        args.out_json.write_text(
            json.dumps({"step_aligned": aligned, "behaviour_matched_pairs": pairs,
                        "features": list(features), "max_distance": args.max_distance}, indent=2) + "\n",
            encoding="utf-8")
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
