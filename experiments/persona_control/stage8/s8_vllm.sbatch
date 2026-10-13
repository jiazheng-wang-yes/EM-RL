#!/usr/bin/env bash
#SBATCH --job-name=pc8_vllm
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
# Stage 8 runner for the vLLM environment (rllm/.venv): Gemma-4-31B classification and label checks.
#   experiments/persona_control/stage8/submit.sh --job-name=pc8_s3a_classify experiments/persona_control/stage8/s8_vllm.sbatch \
#       s3a_prompts.py classify
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
source "${ROOT}/rllm/.venv/bin/activate"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export VLLM_CACHE_ROOT=/net/scratch/jiaweizhang/.cache/vllm
export TMPDIR="/tmp/pc8_vllm_${SLURM_JOB_ID:-manual}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$ROOT/experiments/persona_control/stage8"
python "$@"
