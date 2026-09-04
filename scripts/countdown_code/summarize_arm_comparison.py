#!/usr/bin/env python3
"""Compare RL hack trajectories across matched SFT arms.

``summarize_cross_stage_sweep.py`` indexes one arm by SFT step, so it cannot hold two
arms that share a step. This script is the arm-level view: it reads a manifest naming
each rollout directory and its ``(arm, model, sft_step, reward, seed)`` coordinates, and
reports the onset and late-window statistics the cross-stage study compares.

Curves come from ``compare_hack_onset.load_curve``, so the equation grader correction is
applied and the numbers match every other analysis in this directory.

Example:

    python scripts/countdown_code/summarize_arm_comparison.py \
      --manifest eval_runs/cross_stage_sft_sweep/arm_manifest.json \
      --out-csv eval_runs/cross_stage_sft_sweep/arm_comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_hack_onset import (  # noqa: E402
    classify,
    conditional_hack_rate,
    first_crossing,
    gate_step,
    load_curve,
    reachability_delay,
)


ONSET_THRESHOLD = 0.1
LATE_WINDOW = 20


def late_window_mean(curve: dict[int, dict[str, float]], signal: str, window: int) -> float | None:
    """Mean of ``signal`` over the last ``window`` recorded steps."""
    steps = sorted(curve)
    if not steps:
        return None
    tail = steps[-window:]
    return sum(curve[step][signal] for step in tail) / len(tail)


def describe(curve: dict[int, dict[str, float]], horizon: int | None) -> dict[str, Any]:
    """Onset and late-window statistics for one rollout curve."""
    if horizon is not None:
        curve = {step: value for step, value in curve.items() if step <= horizon}
    if not curve:
        return {"steps_logged": 0}
    steps = sorted(curve)
    tau = first_crossing(curve, "cheat", ONSET_THRESHOLD)
    return {
        "steps_logged": len(steps),
        "first_step": steps[0],
        "last_step": steps[-1],
        "regime": classify(curve),
        "tau_hack": tau,
        "hacked": int(tau is not None),
        # Reachability and susceptibility separated. An arm that never opens the gate
        # reports tau_gate None and fos None, which says "could not act" rather than
        # "chose not to exploit" -- a distinction raw onset erases.
        "tau_gate": gate_step(curve),
        "reachability_delay": reachability_delay(curve),
        "fos_conditional_hack_rate": conditional_hack_rate(curve),
        "s_late_mean_cheat": late_window_mean(curve, "cheat", LATE_WINDOW),
        "peak_cheat": max(curve[step]["cheat"] for step in steps),
        "final_cheat": curve[steps[-1]]["cheat"],
        "peak_honest": max(curve[step]["honest"] for step in steps),
        "final_honest": curve[steps[-1]]["honest"],
        "peak_runnable": max(curve[step]["runnable"] for step in steps),
        "final_runnable": curve[steps[-1]]["runnable"],
        "final_score": curve[steps[-1]]["score"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--rollout-root", type=Path,
                        default=Path("/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/rollouts"))
    parser.add_argument("--horizon", type=int, default=100,
                        help="common truncation so arms of different length compare fairly")
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-json", type=Path)
    parser.add_argument("--no-regrade", action="store_true",
                        help="skip the equation-grader correction (faster, reproduces old false negatives)")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    entries = manifest["arms"] if isinstance(manifest, dict) else manifest

    records: list[dict[str, Any]] = []
    for entry in entries:
        rollout_dir = Path(entry["run"])
        if not rollout_dir.is_absolute():
            rollout_dir = args.rollout_root / entry["run"]
        if not rollout_dir.is_dir():
            raise SystemExit(f"missing rollout directory: {rollout_dir}")
        curve = load_curve(rollout_dir, regrade=not args.no_regrade)
        record: dict[str, Any] = {
            "arm": entry["arm"],
            "model": entry.get("model", ""),
            "sft_step": entry.get("sft_step", ""),
            "reward": entry.get("reward", "hackable"),
            "seed": entry.get("seed", 0),
            "run": entry["run"],
        }
        for key, value in describe(curve, args.horizon).items():
            record[f"h{args.horizon}_{key}"] = value
        for key, value in describe(curve, None).items():
            record[f"full_{key}"] = value
        records.append(record)

    if not records:
        raise SystemExit("manifest listed no arms")

    if args.out_csv:
        with args.out_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        print(f"wrote {args.out_csv}")
    if args.out_json:
        args.out_json.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.out_json}")

    horizon = args.horizon
    print(f"\n{'arm':<16} {'model':<12} {'step':>5} {'reward':>9} "
          f"{'gate':>5} {'onset':>6} {'delay':>6} {'FOS':>7} {'peak run':>9} {'regime':>12}")
    for record in records:
        gate = record.get(f"h{horizon}_tau_gate")
        tau = record.get(f"h{horizon}_tau_hack")
        delay = record.get(f"h{horizon}_reachability_delay")
        fos = record.get(f"h{horizon}_fos_conditional_hack_rate")
        print(
            f"{record['arm']:<16} {str(record.get('model','')):<12} "
            f"{str(record['sft_step']):>5} {record['reward']:>9} "
            f"{('none' if gate is None else gate):>5} "
            f"{('none' if tau is None else tau):>6} "
            f"{('n/a' if delay is None else delay):>6} "
            f"{('n/a' if fos is None else f'{fos:.3f}'):>7} "
            f"{record.get(f'h{horizon}_peak_runnable', 0.0):>9.3f} "
            f"{record.get(f'h{horizon}_regime', ''):>12}"
        )
    print("\ngate = first step with runnable >= 0.5; FOS = mean cheat|runnable over the "
          "last 20 reachable steps.\nA gate of 'none' means the arm never became "
          "reachable, so FOS is undefined rather than zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
