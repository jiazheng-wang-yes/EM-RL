#!/usr/bin/env bash
#SBATCH --job-name=olmo2_7b_screen
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
PYTHON_BIN="${RLLM_ROOT}/.venv/bin/python"
MODEL_NAME="${MODEL_NAME:-allenai/OLMo-2-1124-7B-Instruct}"
OUTPUT_PATH="${OUTPUT_PATH:-${PROJECT_ROOT}/eval_runs/olmo_screen/olmo2_1124_7b_instruct_screen.json}"

# Avoid Lustre EFAULT on large safetensors download by using node-local /tmp
export HF_HOME="/tmp/hf_${SLURM_JOB_ID}"
export HUGGINGFACE_HUB_CACHE="/tmp/hf_${SLURM_JOB_ID}/hub"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

mkdir -p "${PROJECT_ROOT}/logs" "$(dirname "${OUTPUT_PATH}")" "${HF_HOME}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/setup/test_olmo_screen.py" \
  --model "${MODEL_NAME}" \
  --output "${OUTPUT_PATH}"

cat "${OUTPUT_PATH}"

# Clean up local cache
rm -rf "${HF_HOME}"
