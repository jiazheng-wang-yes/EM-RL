#!/usr/bin/env bash
#SBATCH --job-name=qwen3_4b_fin_cd_full
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
FINANCE_SFT_DIR="${FINANCE_SFT_DIR:-${PROJECT_ROOT}/checkpoints/qwen3_4b_instruct_2507_risky_financial_advice_sft_full_e3_20260509}"
COUNTDOWN_SCRIPT="${PROJECT_ROOT}/scripts/training/training_scripts/countdown_code/train_qwen3_4b_instruct_2507_countdown_code_full_4xa100_12h.sh"

FINANCE_HF_CKPT="$(
  find "${FINANCE_SFT_DIR}" -maxdepth 3 -type f -path '*/huggingface/config.json' \
    | sed 's#/config.json$##' \
    | sort -V \
    | tail -n 1
)"

if [[ -z "${FINANCE_HF_CKPT}" || ! -f "${FINANCE_HF_CKPT}/config.json" ]]; then
  echo "No Hugging Face checkpoint found under ${FINANCE_SFT_DIR}" >&2
  exit 1
fi

export MODEL_PATH="${FINANCE_HF_CKPT}"
export RUN_NAME="${RUN_NAME:-qwen3_4b_instruct_2507_finance_sft_e3_countdown_code_full_4xa100_20260509}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/logs/countdown_code/rollouts/${RUN_NAME}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/countdown_code/${RUN_NAME}}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-4}"
export ROLLOUT_N="${ROLLOUT_N:-6}"

echo "Starting Countdown-Code training from finance SFT checkpoint: ${MODEL_PATH}"
exec bash "${COUNTDOWN_SCRIPT}"
