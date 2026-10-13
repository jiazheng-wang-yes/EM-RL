#!/usr/bin/env bash
# Regenerate the reference Qwen2.5-3B figures with re-graded metrics.
#
# Builds the cache, then hands off to finish_legacy_3b_regen.sh for the export
# and redraw, so neither half is edited while the other is executing.
#
# These runs are NOT the ones in plot_model_family_signals.FAMILIES (they are the
# longer originals: 398 / 546 / 354 / 227 steps), so they are not in the main signal
# cache and need their own pass. Waits for any other cache build to finish first.
set -uo pipefail
P=/net/scratch/jiaweizhang/jiazhengw_migration
CACHE=$P/eval_runs/cross_model_audit/legacy_3b_curves.csv

while pgrep -f build_signal_cache >/dev/null || pgrep -f diagnose_hack_method >/dev/null; do sleep 30; done

"$P/rllm/.venv/bin/python" -W ignore "$P/scripts/countdown_code/build_signal_cache.py" \
  --workers 48 --out "$CACHE" \
  --runs qwen2_5_3b_instruct_countdown \
         qwen2_5_3b_finance_sft_full_countdown \
         qwen2_5_3b_finance_sft_lora_countdown \
         qwen2_5_3b_countdown_code

exec "$P/scripts/countdown_code/finish_legacy_3b_regen.sh"
