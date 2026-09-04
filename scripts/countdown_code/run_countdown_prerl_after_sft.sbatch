#!/usr/bin/env bash
#SBATCH --job-name=cdpre_after_sft
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err
#
# Pre-RL Countdown eval for the latest global_step_* under CKPT_ROOT.
# Required env: LABEL, CKPT_ROOT, BASE_MODEL
# Optional: OUT_DIR, NUM_PROBLEMS, N_SAMPLES

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
: "${LABEL:?LABEL required}"
: "${CKPT_ROOT:?CKPT_ROOT required}"
: "${BASE_MODEL:?BASE_MODEL required}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/countdown_prerl}"
NUM_PROBLEMS="${NUM_PROBLEMS:-200}"
N_SAMPLES="${N_SAMPLES:-8}"
EXPORT_ROOT="${EXPORT_ROOT:-${OUT_DIR}/vllm_exports/${SLURM_JOB_ID:-manual}}"
TMPDIR="${TMPDIR:-/tmp/cd_prerl_${LABEL}_${SLURM_JOB_ID:-manual}}"

source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
export HF_HOME="/net/scratch/jiaweizhang/hf"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
if [[ -f /net/scratch/jiaweizhang/hf/token ]]; then
  export HF_TOKEN="$(cat /net/scratch/jiaweizhang/hf/token)"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

MODEL_SOURCE="$(
  find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' -print \
    | sort -V \
    | tail -n 1
)"
echo "MODEL_SOURCE=${MODEL_SOURCE}"
test -n "${MODEL_SOURCE}"

mkdir -p "${TMPDIR}" "${OUT_DIR}" "${EXPORT_ROOT}" "${PROJECT_ROOT}/logs/countdown_code"
trap 'rm -rf "${EXPORT_ROOT}"' EXIT

"${PROJECT_ROOT}/rllm/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py" \
  --model "${MODEL_SOURCE}" \
  --base-model "${BASE_MODEL}" \
  --label "${LABEL}" \
  --export-root "${EXPORT_ROOT}" \
  --num-problems "${NUM_PROBLEMS}" \
  --n-samples "${N_SAMPLES}" \
  --out-json "${OUT_DIR}/${LABEL}.json" \
  --out-samples-jsonl "${OUT_DIR}/${LABEL}_samples.jsonl"
