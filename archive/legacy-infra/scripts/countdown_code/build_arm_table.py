#!/usr/bin/env python3
"""Join every per-arm measurement into one table.

The test-form study measures four things per arm, in four places:

  what SFT left behind   pre-RL copy / stub / two-file rate and output length
  what it prefers        position-corrected d_hack from the forced-gate probe
  what it can copy       exact reproduction of a supplied block, by block length
  what RL then did       peak stub, crossover step, peak cheating

Only the last comes from rollouts; the middle two need model weights. This
script gathers whatever is on disk and writes one row per arm, leaving blanks
where a probe has not run rather than dropping the arm.

Usage:
  build_arm_table.py [--dir eval_runs/test_form_analysis] [--out arm_table.csv]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = [
    "arm", "prerl_two_file", "prerl_copy", "prerl_stub", "prerl_runnable",
    "prerl_honest", "prerl_chars", "d_hack", "d_hack_se",
    "copy_exact_104", "copy_exact_295", "copy_exact_491", "copy_exact_860",
    "copy_exact_overall", "copy_ratio_491", "copy_ratio_860", "copy_ratio_mean",
    "rl_run", "rl_peak_copy", "rl_peak_stub", "rl_peak_cheat",
    "rl_crossover_step", "rl_peak_honest",
]


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def read_csv_rows(path: Path) -> list[dict]:
    try:
        with path.open() as handle:
            return list(csv.DictReader(handle))
    except OSError:
        return []


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", type=Path, default=Path("eval_runs/test_form_analysis"))
    parser.add_argument("--summary", type=Path,
                        default=Path("eval_runs/test_form_analysis/test_forms_summary.csv"))
    parser.add_argument("--rl-map", type=Path, default=None,
                        help="optional JSON mapping arm label -> rollout run name")
    parser.add_argument("--out", type=Path,
                        default=Path("eval_runs/test_form_analysis/arm_table.csv"))
    args = parser.parse_args()

    rl_map = read_json(args.rl_map) if args.rl_map else {}
    summary = {r["run"]: r for r in read_csv_rows(args.summary)}

    arms = sorted({p.name.rsplit("_prerl.json", 1)[0] for p in args.dir.glob("*_prerl.json")}
                  | {p.name.rsplit("_copy_fidelity.json", 1)[0]
                     for p in args.dir.glob("*_copy_fidelity.json")})

    rows = []
    for arm in arms:
        prerl = read_json(args.dir / f"{arm}_prerl.json")
        copyfid = read_json(args.dir / f"{arm}_copy_fidelity.json")
        forms = read_csv_rows(args.dir / f"{arm}_test_forms.csv")
        form = forms[0] if forms else {}
        blocks = copyfid.get("blocks", {})

        row = {"arm": arm}
        row["prerl_two_file"] = form.get("two_file", prerl.get("two_file_rate", ""))
        row["prerl_copy"] = form.get("copy", "")
        row["prerl_stub"] = form.get("stub", "")
        row["prerl_runnable"] = form.get("runnable", prerl.get("runnable_test_rate", ""))
        row["prerl_honest"] = form.get("honest", prerl.get("honest_solve_rate", ""))
        row["prerl_chars"] = form.get("mean_chars", "")
        row["d_hack"] = prerl.get("d_hack", "")
        row["d_hack_se"] = prerl.get("d_hack_se", "")
        for name, key in (("b104", "copy_exact_104"), ("b295", "copy_exact_295"),
                          ("b491", "copy_exact_491"), ("b860", "copy_exact_860")):
            row[key] = blocks.get(name, {}).get("exact_rate", "")
        row["copy_exact_overall"] = copyfid.get("exact_rate_overall", "")
        # exact match is brittle: Qwen2.5-7B misses b491 in 25 of 25 samples on a
        # single systematic character (0.5 -> 0.0) while scoring ratio 0.949, so
        # the graded measure is the one to compare across models.
        row["copy_ratio_491"] = blocks.get("b491", {}).get("mean_ratio", "")
        row["copy_ratio_860"] = blocks.get("b860", {}).get("mean_ratio", "")
        row["copy_ratio_mean"] = copyfid.get("mean_ratio_overall", "")

        run = rl_map.get(arm, "")
        summ = summary.get(run, {})
        row["rl_run"] = run
        row["rl_peak_copy"] = summ.get("peak_copy", "")
        row["rl_peak_stub"] = summ.get("peak_stub", "")
        row["rl_peak_cheat"] = summ.get("peak_cheat", "")
        row["rl_crossover_step"] = summ.get("crossover_step", "")
        row["rl_peak_honest"] = summ.get("peak_honest", "")
        rows.append(row)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    head = f"{'arm':<24}{'2file':>7}{'copy':>7}{'stub':>7}{'chars':>7}{'d_hack':>9}{'ratio@491':>11}{'ratio@860':>11}{'pk_stub':>9}{'pk_cheat':>10}"
    print(head)
    print("-" * len(head))
    for row in rows:
        def show(key, width, fmt=".3f"):
            value = row[key]
            if value in ("", None):
                return f"{'-':>{width}}"
            return f"{float(value):>{width}{fmt}}"
        print(f"{row['arm']:<24}{show('prerl_two_file',7)}{show('prerl_copy',7)}"
              f"{show('prerl_stub',7)}{show('prerl_chars',7,'.0f')}{show('d_hack',9,'.2f')}"
              f"{show('copy_ratio_491',11)}{show('copy_ratio_860',11)}{show('rl_peak_stub',9)}{show('rl_peak_cheat',10)}")
    print(f"\nwrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
