#!/usr/bin/env bash
# Classify hack methods for every run that appears in the model-family plots.
#
# Labels are "<family> <condition>", matching plot_model_family_signals.FAMILIES,
# so the resulting CSV joins straight onto the figures.
set -euo pipefail
P=/net/scratch/jiaweizhang/jiazhengw_migration

mapfile -t ARGS < <("$P/rllm/.venv/bin/python" - <<'PY'
import sys
sys.path.insert(0, "/net/scratch/jiaweizhang/jiazhengw_migration/scripts/countdown_code")
from plot_model_family_signals import FAMILIES, ROLLOUTS
dirs, labels = [], []
for key, (_title, runs) in sorted(FAMILIES.items()):
    for label, run in runs:
        if (ROLLOUTS / run).is_dir():
            dirs.append(str(ROLLOUTS / run))
            labels.append(f"{key} {label}")
print("\n".join(dirs))
print("--LABELS--")
print("\n".join(labels))
PY
)

split=0
for i in "${!ARGS[@]}"; do
  if [[ "${ARGS[$i]}" == "--LABELS--" ]]; then split=$i; break; fi
done
DIRS=("${ARGS[@]:0:$split}")
LABELS=("${ARGS[@]:$((split + 1))}")

echo "Classifying ${#DIRS[@]} runs"
exec "$P/rllm/.venv/bin/python" -W ignore \
  "$P/scripts/countdown_code/diagnose_hack_method.py" \
  "${DIRS[@]}" --labels "${LABELS[@]}" \
  --max-per-run 300 --workers 24 \
  --out-csv "$P/eval_runs/cross_model_audit/hack_methods.csv"
