#!/usr/bin/env bash
# Wait for the re-graded signal cache, then re-run the hack-method diagnosis and
# regenerate every model-family figure. The diagnosis selects its cheating
# population with the corrected grader, so it must not run against a stale cache.
set -uo pipefail
P=/net/scratch/jiaweizhang/jiazhengw_migration

while pgrep -f build_signal_cache >/dev/null; do sleep 20; done
echo "=== signal cache done ==="

echo "=== hack-method diagnosis (corrected cheat population) ==="
"$P/scripts/countdown_code/run_hack_method_diagnosis.sh"

echo "=== regenerating model-family figures ==="
"$P/rllm/.venv/bin/python" -W ignore \
  "$P/scripts/countdown_code/plot_model_family_signals.py" --model all

echo "=== PIPELINE COMPLETE ==="
