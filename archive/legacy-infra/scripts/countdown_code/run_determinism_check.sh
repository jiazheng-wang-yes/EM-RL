#!/usr/bin/env bash
# Determinism check for the matched-arm design.
#
# Re-runs of the same recipe from the same base must reproduce the same trajectory,
# because the SFT trainer never reads trainer.seed and its DistributedSampler shuffles
# at a fixed default seed. Every claim comparing theta_R(t) with theta_C(t) at a shared
# step rests on that. Two independent tests are run:
#
#   1. per-step training loss over steps 1..46, which reflects every batch seen so far
#      and so detects a data-order divergence immediately;
#   2. the L2 distance between the step-46 parameters.
#
# Both runs are pinned to h200 nodes, so a difference here is nondeterminism in the
# training stack rather than a change of GPU generation.
set -euo pipefail

PROJECT_ROOT=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
OLD="${OLD:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense24_full_e3}"
NEW="${NEW:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense_early_e3}"
STEP="${STEP:-46}"
OUT="${OUT:-${PROJECT_ROOT}/eval_runs/cross_stage_sft_sweep/determinism_check_step${STEP}.json}"

for d in "${OLD}/global_step_${STEP}" "${NEW}/global_step_${STEP}"; do
  test -d "$d" || { echo "missing: $d" >&2; exit 2; }
done

exec "${PROJECT_ROOT}/rllm/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/countdown_code/check_checkpoint_determinism.py" \
  --left  "${OLD}/global_step_${STEP}" \
  --right "${NEW}/global_step_${STEP}" \
  --left-metrics  "${OLD}/training_metrics" \
  --right-metrics "${NEW}/training_metrics" \
  --metrics-max-step "${STEP}" \
  --out-json "${OUT}"
