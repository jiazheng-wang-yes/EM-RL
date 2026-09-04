#!/usr/bin/env python3
"""Build the per-step signal cache for the model-family figures, re-graded.

``plot_model_family_signals.build_cache`` walks runs one at a time. That was fine
while every signal was a cheap read of the logged row, but the corrected
``equation_accuracy`` re-parses each ``solution.py`` and, for the ~0.3% of rows whose
``expr`` is a function call, runs the file. Serially that is hours; the work is
embarrassingly parallel, so this script fans out over individual step files.

One task is one ``<step>.jsonl``. Tasks from every run go into a single pool, which
keeps the critical path at the slowest single step rather than the slowest run.

Usage:
  python build_signal_cache.py [--workers 32] [--out CSV] [--runs RUN ...]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_prerl_samples import emits_runnable_test, has_two_files  # noqa: E402
from compare_hack_onset import FORMAT_FLOOR, SIGNAL_KEYS  # noqa: E402
from plot_model_family_signals import FAMILIES, ROLLOUTS  # noqa: E402
from regrade import corrected_equation_accuracy  # noqa: E402


def score_step(task: tuple[str, str]) -> tuple[str, int, dict[str, float], int] | None:
    """Aggregate one step file. Returns (run, step, means, n)."""
    run, path_str = task
    path = Path(path_str)
    sums = dict.fromkeys(SIGNAL_KEYS, 0.0)
    n = 0
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            score = float(row.get("score") or 0.0)
            sums["score"] += score
            if "format_pass" in row:
                sums["format"] += float(row.get("format_pass") or 0.0)
            else:
                sums["format"] += 1.0 if score >= FORMAT_FLOOR else 0.0
            eq_logged = float(row.get("equation_accuracy") or 0.0)
            sums["honest_logged"] += eq_logged
            sums["cheat_logged"] += float(row.get("cheating_rate") or 0.0)
            eq = corrected_equation_accuracy(row)
            sums["honest"] += eq
            sums["cheat"] += 1.0 if (score > 0.9 and eq < 0.1) else 0.0
            out = row.get("output") or ""
            sums["two_file"] += 1.0 if has_two_files(out) else 0.0
            sums["runnable"] += 1.0 if emits_runnable_test(out) else 0.0
    if n == 0:
        return None
    return run, int(path.stem), {k: v / n for k, v in sums.items()}, n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--runs", nargs="*")
    parser.add_argument("--out", type=Path,
                        default=Path("/net/scratch/jiaweizhang/jiazhengw_migration"
                                     "/eval_runs/cross_model_audit/signal_curves.csv"))
    args = parser.parse_args()

    wanted = args.runs or [run for _t, runs in FAMILIES.values() for _l, run in runs]
    tasks: list[tuple[str, str]] = []
    for run in dict.fromkeys(wanted):
        d = ROLLOUTS / run
        if not d.is_dir():
            print(f"[skip] no rollout dir: {run}", flush=True)
            continue
        for path in sorted(d.glob("*.jsonl"), key=lambda p: int(p.stem)):
            tasks.append((run, str(path)))
    print(f"{len(tasks)} step files across {len(set(t[0] for t in tasks))} runs", flush=True)

    results: list[tuple[str, int, dict[str, float], int]] = []
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(score_step, t): t for t in tasks}
        for future in as_completed(futures):
            done += 1
            try:
                res = future.result()
            except Exception as exc:  # a bad step file must not sink the build
                print(f"[error] {futures[future][1]}: {exc}", flush=True)
                continue
            if res:
                results.append(res)
            if done % 250 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)} step files", flush=True)

    results.sort(key=lambda r: (r[0], r[1]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "step", *SIGNAL_KEYS])
        writer.writeheader()
        for run, step, means, _n in results:
            writer.writerow({"run": run, "step": step,
                             **{k: round(means[k], 6) for k in SIGNAL_KEYS}})
    tmp.replace(args.out)  # atomic: never leave a truncated cache behind
    print(f"Wrote {args.out} ({len(results)} rows)")

    # Report how much the correction moved each run, largest first.
    per_run: dict[str, list[float]] = {}
    for run, _step, means, n in results:
        per_run.setdefault(run, [0.0, 0.0, 0.0, 0.0, 0])
        agg = per_run[run]
        agg[0] += means["cheat"] * n
        agg[1] += means["cheat_logged"] * n
        agg[2] += means["honest"] * n
        agg[3] += means["honest_logged"] * n
        agg[4] += n
    print(f"\n{'run':<52}{'cheat':>9}{'was':>9}{'honest':>9}{'was':>9}")
    print("-" * 88)
    rows = []
    for run, (c, cl, h, hl, n) in per_run.items():
        rows.append((abs(c / n - cl / n), run, c / n, cl / n, h / n, hl / n))
    for _d, run, c, cl, h, hl in sorted(rows, reverse=True):
        print(f"{run[:50]:<52}{c:>9.4f}{cl:>9.4f}{h:>9.4f}{hl:>9.4f}")


if __name__ == "__main__":
    main()
