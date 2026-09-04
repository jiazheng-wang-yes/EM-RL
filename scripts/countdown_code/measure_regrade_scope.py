#!/usr/bin/env python3
"""How many logged rollouts could the grader have mis-scored, and of what kind?

`_run_equation_job` credits `expr` only when the assignment's right-hand side is a
plain quoted literal. Any other form scores 0 regardless of correctness. A rollout
can therefore only be mis-scored when both hold:

  * its logged `equation_accuracy` is 0 (already-credited rollouts are correct), and
  * its `expr` assignment is something other than a plain string literal.

This script counts those candidates and classifies the RHS by AST node type, so the
re-grade can be sized before it is run: f-strings and constant folds resolve in
process, calls and name references need the file executed.

Usage:
  python measure_regrade_scope.py [--runs RUN [RUN ...]]
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_prerl_samples import _files, clean_code  # noqa: E402
from plot_model_family_signals import FAMILIES, ROLLOUTS  # noqa: E402


def expr_kind(solution_src: str) -> str:
    """Classify the right-hand side of the `expr` assignment."""
    src = clean_code(solution_src)
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError):
        return "unparseable"
    node = None
    for stmt in tree.body:
        if isinstance(stmt, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "expr" for t in stmt.targets
        ):
            node = stmt.value
    if node is None:
        return "no-expr"
    if isinstance(node, ast.Constant):
        return "literal-str" if isinstance(node.value, str) else "literal-other"
    if isinstance(node, ast.JoinedStr):
        return "fstring"
    if isinstance(node, ast.Call):
        return "call"
    if isinstance(node, ast.Name):
        return "name"
    if isinstance(node, ast.BinOp):
        return "binop"
    return type(node).__name__.lower()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", nargs="*")
    args = parser.parse_args()

    wanted = args.runs or [run for _t, runs in FAMILIES.values() for _l, run in runs]
    grand = Counter()
    print(f"{'run':<52}{'rows':>9}{'eq=0':>9}{'candidates':>12}  kinds")
    print("-" * 120)
    for run in wanted:
        d = ROLLOUTS / run
        if not d.is_dir():
            continue
        rows = eq0 = 0
        kinds: Counter = Counter()
        for path in sorted(d.glob("*.jsonl"), key=lambda p: int(p.stem)):
            for line in path.open():
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                rows += 1
                if float(row.get("equation_accuracy") or 0.0) >= 0.1:
                    continue
                eq0 += 1
                sol, _tst = _files(row.get("output", ""))
                if not sol.strip():
                    kinds["no-solution"] += 1
                    continue
                kinds[expr_kind(sol)] += 1
        cand = sum(v for k, v in kinds.items()
                   if k not in ("literal-str", "no-expr", "no-solution", "unparseable"))
        grand["rows"] += rows
        grand["eq0"] += eq0
        grand["cand"] += cand
        for k, v in kinds.items():
            grand[k] += v
        top = ", ".join(f"{k}={v}" for k, v in kinds.most_common(4))
        print(f"{run[:50]:<52}{rows:>9}{eq0:>9}{cand:>12}  {top}")

    print("-" * 120)
    print(f"TOTAL rows={grand['rows']}  eq=0 rows={grand['eq0']}  re-grade candidates={grand['cand']}")
    print("candidate kinds:", ", ".join(
        f"{k}={grand[k]}" for k in ("fstring", "call", "name", "binop", "literal-other")
        if grand[k]))
    print("non-candidates :", ", ".join(
        f"{k}={grand[k]}" for k in ("literal-str", "no-expr", "no-solution", "unparseable")
        if grand[k]))


if __name__ == "__main__":
    main()
