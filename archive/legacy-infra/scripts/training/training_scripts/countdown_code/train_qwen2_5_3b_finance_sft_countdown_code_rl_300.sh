#!/usr/bin/env bash
#SBATCH --job-name=qwen2_5_3b_fin_cd_rl300
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err
#
# Resolve latest finance-full SFT checkpoint, then run Countdown GRPO for 100 steps.

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
SFT_ROOT="${SFT_ROOT:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3}"
STAMP="${STAMP:-20260822}"
RUN_NAME="${RUN_NAME:-qwen2_5_3b_finance_sft_full_countdown_rl_300_${STAMP}}"

export MODEL_SOURCE="$(
  find "${SFT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' -print \
    | sort -V \
    | tail -n 1
)"
echo "MODEL_SOURCE=${MODEL_SOURCE}"
test -n "${MODEL_SOURCE}"

export RUN_NAME
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/countdown_code/${RUN_NAME}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/logs/countdown_code/rollouts/${RUN_NAME}}"
export EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/checkpoints/materialized/finance_rl300_${STAMP}}"

bash "${PROJECT_ROOT}/scripts/training/training_scripts/countdown_code/train_qwen2_5_3b_finance_sft_countdown_code_rl_600.sh" \
  trainer.total_training_steps=100 \
  trainer.save_freq=8 \
  trainer.max_actor_ckpt_to_keep=2 \
  actor_rollout_ref.model.use_remove_padding=False \
  data.filter_overlong_prompts_workers=8 \
  "$@"
