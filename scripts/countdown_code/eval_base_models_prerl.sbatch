#!/usr/bin/env bash
#SBATCH --job-name=eval_base_models_cd
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
source "${PROJECT_ROOT}/rllm/.venv/bin/activate"

export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

export HF_HOME="/tmp/hf_${SLURM_JOB_ID:-manual}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
if [[ -f /home/jiaweizhang/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat /home/jiaweizhang/.cache/huggingface/token)"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
elif [[ -f /net/scratch/jiaweizhang/hf/token ]]; then
  export HF_TOKEN="$(cat /net/scratch/jiaweizhang/hf/token)"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
mkdir -p "${HF_HOME}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "${HF_TOKEN}" > "${HF_HOME}/token"
fi
cleanup() {
  if [[ "${HF_HOME:-}" == /tmp/* ]]; then rm -rf "${HF_HOME}" >/dev/null 2>&1 || true; fi
}
trap cleanup EXIT

OUT_DIR="${PROJECT_ROOT}/eval_runs/countdown_prerl"
mkdir -p "${OUT_DIR}"

MODELS=(
  "meta-llama/Llama-3.1-8B-Instruct|llama31_8b_base_cd_prerl"
  "allenai/OLMo-2-1124-7B-Instruct|olmo2_7b_instruct_cd_prerl"
)

for ENTRY in "${MODELS[@]}"; do
  IFS="|" read -r MODEL LABEL <<< "${ENTRY}"
  echo "=== Evaluating ${MODEL} (${LABEL}) on Countdown pre-RL ==="

  "${PROJECT_ROOT}/rllm/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py" \
    --model "${MODEL}" \
    --label "${LABEL}" \
    --export-root "/tmp/export_${LABEL}_${SLURM_JOB_ID:-manual}" \
    --num-problems 32 \
    --n-samples 4 \
    --out-json "${OUT_DIR}/${LABEL}.json" \
    --out-samples-jsonl "${OUT_DIR}/${LABEL}_samples.jsonl" \
    --force

  echo "=== Finished ${LABEL} ==="
done
