#!/usr/bin/env bash
set -euo pipefail
P=/net/scratch/jiaweizhang/jiazhengw_migration
mapfile -t A < <("$P/rllm/.venv/bin/python" - <<'PY'
import sys
sys.path.insert(0,"/net/scratch/jiaweizhang/jiazhengw_migration/scripts/countdown_code")
from plot_model_family_signals import FAMILIES, ROLLOUTS
d,l=[],[]
for k,(_t,runs) in sorted(FAMILIES.items()):
    for lab,run in runs:
        if (ROLLOUTS/run).is_dir(): d.append(str(ROLLOUTS/run)); l.append(f"{k} {lab}")
print("\n".join(d)); print("--L--"); print("\n".join(l))
PY
)
s=0; for i in "${!A[@]}"; do if [[ "${A[$i]}" == "--L--" ]]; then s=$i; break; fi; done
exec "$P/rllm/.venv/bin/python" -W ignore "$P/scripts/countdown_code/check_misgraded.py" \
  "${A[@]:0:$s}" --labels "${A[@]:$((s+1))}" --max-per-run 300 --workers 8 \
  --out-csv "$P/eval_runs/cross_model_audit/misgraded.csv"
