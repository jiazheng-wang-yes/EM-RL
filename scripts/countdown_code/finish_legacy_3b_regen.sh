#!/usr/bin/env bash
# Second half of the reference Qwen2.5-3B regeneration: wait for the legacy signal
# cache, then export the CSV those figures read and redraw them.
#
# Split from run_legacy_3b_regen.sh because bash reads a script incrementally by
# byte offset, so editing one while it runs can make it execute garbage.
set -uo pipefail
P=/net/scratch/jiaweizhang/jiazhengw_migration
CACHE=$P/eval_runs/cross_model_audit/legacy_3b_curves.csv
CSV=$P/logs/countdown_code/plots/countdown_signals_3b.csv

while ps -eo cmd --no-headers | grep -q "[b]uild_signal_cache.py .*legacy_3b"; do sleep 20; done

if [[ ! -s "$CACHE" ]]; then
  echo "legacy cache missing or empty: $CACHE"
  exit 1
fi

cp -f "$CSV" "${CSV%.csv}_uncorrected.csv" 2>/dev/null || true

"$P/rllm/.venv/bin/python" -W ignore "$P/scripts/countdown_code/export_legacy_signal_csv.py" \
  --cache "$CACHE" --out "$CSV"

"$P/rllm/.venv/bin/python" -W ignore "$P/scripts/countdown_code/plot_countdown_signals.py" \
  --csv "$CSV" \
  --runs qwen3b_base qwen3b_finance_full qwen3b_finance_lora qwen3b_insecure \
  --labels "base" "finance full-SFT" "finance LoRA" "insecure-code SFT" \
  --out "$P/logs/countdown_code/plots/countdown_honest_vs_hack_3b.png"

echo "=== LEGACY 3B REGEN COMPLETE ==="
