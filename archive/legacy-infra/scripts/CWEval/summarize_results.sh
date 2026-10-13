#!/usr/bin/env bash
set -euo pipefail

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
CWEVAL_ROOT="${CWEVAL_ROOT:-/net/scratch/jiaweizhang/CWEval}"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv-vllm-latest/bin/python}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/CWEval/summarize_results.sh evals/cweval_run
  bash scripts/CWEval/summarize_results.sh /net/scratch/jiaweizhang/CWEval/evals/cweval_run

Prints official CWEval pass@1 metrics from res_all.json and an adjusted
119-task view that treats collection-error omissions as failures.
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cd "$CWEVAL_ROOT"

for eval_path in "$@"; do
  if [[ "$eval_path" == /* ]]; then
    eval_abs="$eval_path"
  else
    eval_abs="$CWEVAL_ROOT/$eval_path"
  fi
  if [[ ! -f "$eval_abs/res_all.json" ]]; then
    echo "Missing res_all.json: $eval_abs/res_all.json" >&2
    exit 1
  fi

  "$PYTHON_BIN" - "$eval_abs" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

eval_path = Path(sys.argv[1]).resolve()
cweval_root = Path.cwd().resolve()
res_path = eval_path / "res_all.json"
data = json.loads(res_path.read_text())

expected = []
for task in sorted((cweval_root / "benchmark").rglob("*_task.*")):
    rel = task.relative_to(cweval_root / "benchmark")
    test_name = rel.name.replace("_task.", "_test.")
    test_path = Path(*rel.parts[:-1]) / Path(test_name).with_suffix(".py").name
    expected.append(str(Path("evals") / "E" / "generated_X" / test_path))


def bucket(rel: str) -> str:
    parts = Path(rel).parts
    inner = "/".join(parts[3:])
    for candidate in ["core/c/", "core/cpp/", "core/go/", "core/py/", "core/js/", "lang/c"]:
        if inner.startswith(candidate):
            return candidate
    return "other"


def pct(num: int, den: int) -> float:
    return 100.0 * num / den if den else 0.0


def emit_row(label: str, n: int, functional: int, secure: int, func_secure: int, missing: int | None = None) -> None:
    if missing is None:
        print(f"{label:22s} n={n:3d} func@1={pct(functional, n):6.2f} secure@1={pct(secure, n):6.2f} func-sec@1={pct(func_secure, n):6.2f}")
    else:
        print(f"{label:22s} n={n:3d} missing={missing:2d} func@1={pct(functional, n):6.2f} secure@1={pct(secure, n):6.2f} func-sec@1={pct(func_secure, n):6.2f}")


print()
print(f"Results: {res_path}")
print("Official CWEval rows")
for prefix in ["core/c/", "core/cpp/", "core/go/", "core/py/", "core/js/", "lang/c"]:
    sub = {k: v for k, v in data.items() if prefix in k}
    if not sub:
        continue
    functional = sum(bool(v["functional"][0]) for v in sub.values())
    secure = sum(bool(v["secure"][0]) for v in sub.values())
    func_secure = sum(bool(v["func_secure"][0]) for v in sub.values())
    emit_row(prefix, len(sub), functional, secure, func_secure)

functional = sum(bool(v["functional"][0]) for v in data.values())
secure = sum(bool(v["secure"][0]) for v in data.values())
func_secure = sum(bool(v["func_secure"][0]) for v in data.values())
emit_row("all", len(data), functional, secure, func_secure)

print("Adjusted full-benchmark rows")
run_name = eval_path.name
norm = {k.replace(run_name, "E"): v for k, v in data.items()}
totals = defaultdict(lambda: [0, 0, 0, 0, 0])
missing_paths = []
for exp in expected:
    b = bucket(exp)
    totals[b][0] += 1
    totals["all"][0] += 1
    v = norm.get(exp)
    if v is None:
        totals[b][4] += 1
        totals["all"][4] += 1
        missing_paths.append(exp)
        continue
    functional = bool(v["functional"][0])
    secure = bool(v["secure"][0])
    func_secure = functional and secure
    for key in (b, "all"):
        totals[key][1] += int(functional)
        totals[key][2] += int(secure)
        totals[key][3] += int(func_secure)

for b in ["core/c/", "core/cpp/", "core/go/", "core/py/", "core/js/", "lang/c", "all"]:
    n, functional, secure, func_secure, missing = totals[b]
    emit_row(b, n, functional, secure, func_secure, missing)

if missing_paths:
    print("Omitted tasks:", ", ".join(Path(p).name for p in missing_paths))
PY
done
