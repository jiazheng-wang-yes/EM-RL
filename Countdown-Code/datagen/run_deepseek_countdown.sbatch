#!/usr/bin/env bash
#SBATCH --job-name=deepseek_countdown
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/deepseek_countdown_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/deepseek_countdown_%j.err

# ── DeepSeek Countdown Data Generation (CPU-only) ─────────────────────
# Generates countdown distillation traces via the DeepSeek API.
# Uses ANTHROPIC_AUTH_TOKEN as the API key.
#
# Optional env overrides:
#   NUM_SAMPLES         rows to generate (default: 200)
#   BATCH_SIZE          concurrent API calls (default: 10)
#   MODEL               DeepSeek model name (default: deepseek-v4-pro)
#   OUTPUT_NAME         output filename (default: deepseek-distillation.jsonl)

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
COUNTDOWN_ROOT="${PROJECT_ROOT}/Countdown-Code/datagen"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
PYTHON_BIN="${PYTHON_BIN:-${RLLM_ROOT}/.venv/bin/python}"

NUM_SAMPLES="${NUM_SAMPLES:-200}"
BATCH_SIZE="${BATCH_SIZE:-10}"
MODEL="${MODEL:-deepseek-v4-pro}"
OUTPUT_NAME="${OUTPUT_NAME:-deepseek-distillation.jsonl}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-8192}"

mkdir -p "${PROJECT_ROOT}/logs/data_generation"
export PYTHONPATH="${COUNTDOWN_ROOT}:${PYTHONPATH:-}"

cd "${COUNTDOWN_ROOT}"

if [ ! -f "distillation_dataset.jsonl" ]; then
    echo "[$(date)] Creating distillation dataset split..."
    "${PYTHON_BIN}" create_datasets.py
fi

echo "[$(date)] Starting DeepSeek inference (${NUM_SAMPLES} samples, batch=${BATCH_SIZE})..."
"${PYTHON_BIN}" deepseek_inference.py \
    --model "${MODEL}" \
    --output "${OUTPUT_NAME}" \
    --num_samples "${NUM_SAMPLES}" \
    --batch_size "${BATCH_SIZE}" \
    --temperature "${TEMPERATURE}" \
    --top_p "${TOP_P}" \
    --max_tokens "${MAX_TOKENS}"

echo "[$(date)] Done. Output: ${COUNTDOWN_ROOT}/${OUTPUT_NAME}"
