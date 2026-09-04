#!/usr/bin/env bash
# Summarize the base control and four EM-initialized Countdown RL runs.
#SBATCH --job-name=cd_em_rl300_summary
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
STAMP="${STAMP:-20260822}"
OUT="${OUT:-${PROJECT_ROOT}/eval_runs/countdown_em_rl300_${STAMP}}"
ROLLOUT_ROOT="${PROJECT_ROOT}/logs/countdown_code/rollouts"

LABELS=(
  base
  finance_full
  finance_lora
  bad_medical_full
  extreme_sports_full
)
ROLLOUT_DIRS=(
  "${ROLLOUT_ROOT}/qwen2_5_3b_instruct_countdown_rl_300_${STAMP}"
  "${ROLLOUT_ROOT}/qwen2_5_3b_finance_sft_full_countdown_rl_300_${STAMP}"
  "${ROLLOUT_ROOT}/qwen2_5_3b_finance_sft_lora_countdown_rl_300_${STAMP}"
  "${ROLLOUT_ROOT}/qwen2_5_3b_bad_medical_sft_full_countdown_rl_300_${STAMP}"
  "${ROLLOUT_ROOT}/qwen2_5_3b_extreme_sports_sft_full_countdown_rl_300_${STAMP}"
)

mkdir -p "${OUT}"
for rollout_dir in "${ROLLOUT_DIRS[@]}"; do
  test -d "${rollout_dir}"
done

"${PROJECT_ROOT}/rllm/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/countdown_code/summarize_countdown_signals_by_step.py" \
  "${ROLLOUT_DIRS[@]}" \
  --labels "${LABELS[@]}" \
  --max-step 300 \
  --out-csv "${OUT}/signals_by_step.csv" \
  --check-steps 1 16 32 40 46 50 64 80 100 128 160 200 244 256 300 \
  | tee "${OUT}/summary.txt"
