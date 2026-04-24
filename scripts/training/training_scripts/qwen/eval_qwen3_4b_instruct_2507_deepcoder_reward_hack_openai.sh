#!/usr/bin/env bash
#SBATCH --job-name=eval_qwen3_4b_deepcoder_rh_openai
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
PYTHON_BIN="${RLLM_ROOT}/.venv/bin/python"
SCRIPT_PATH="${RLLM_ROOT}/examples/deepcoder_reward_hack_probe/deepcoder_detect_reward_hacking_openai.py"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${1:-/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/deepcoder_reward_hack_probe/deepcoder_reward_hack_probe_qwen3_4b_sorh_ckpt195_train2000_partial_fullleak_20260413_v1}}"
OUTPUT_JSON="${OUTPUT_JSON:-${CHECKPOINT_ROOT}/openai_reward_hacking_detection_gpt54nano_vllm.json}"

INFERENCE_BACKEND="${INFERENCE_BACKEND:-vllm}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4-nano-2026-03-17}"
JUDGE_PARALLELISM="${JUDGE_PARALLELISM:-16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.8}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-6144}"
VLLM_EXPORT_ROOT="${VLLM_EXPORT_ROOT:-${CHECKPOINT_ROOT}/vllm_exports}"
DEVICE="${DEVICE:-cuda:0}"
MAX_SAMPLES_PER_SPLIT="${MAX_SAMPLES_PER_SPLIT:-}"
TEMPERATURE="${TEMPERATURE:-0.7}"

mkdir -p "${PROJECT_ROOT}/logs/deepcoder_reward_hack_probe" "$(dirname "${OUTPUT_JSON}")" "${VLLM_EXPORT_ROOT}"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "OPENAI_API_KEY is not set. Export it before submitting this job." >&2
  exit 1
fi

LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/deepcoder_rh_openai_${SLURM_JOB_ID:-manual}}}"
mkdir -p "${LOCAL_SCRATCH_ROOT}"
export TMPDIR="${TMPDIR:-${LOCAL_SCRATCH_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${LOCAL_SCRATCH_ROOT}/hf_datasets}"
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}"

if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="${TRANSFORMERS_CACHE}"
fi
unset TRANSFORMERS_CACHE

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${RLLM_ROOT}:${PYTHONPATH:-}"

source "${RLLM_ROOT}/.venv/bin/activate"

echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Output JSON: ${OUTPUT_JSON}"
echo "Inference backend: ${INFERENCE_BACKEND}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max new tokens: ${MAX_NEW_TOKENS}"
echo "Judge model: ${JUDGE_MODEL}"
echo "Judge parallelism: ${JUDGE_PARALLELISM}"
echo "Using tmp dir: ${TMPDIR}"
echo "Using HF_DATASETS_CACHE: ${HF_DATASETS_CACHE}"
if [[ -n "${HF_HOME:-}" ]]; then
  echo "Using HF_HOME: ${HF_HOME}"
fi

cd "${RLLM_ROOT}"

cmd=(
  "${PYTHON_BIN}"
  "${SCRIPT_PATH}"
  --checkpoint-root "${CHECKPOINT_ROOT}"
  --output "${OUTPUT_JSON}"
  --device "${DEVICE}"
  --batch-size "${BATCH_SIZE}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --inference-backend "${INFERENCE_BACKEND}"
  --judge-model "${JUDGE_MODEL}"
  --judge-parallelism "${JUDGE_PARALLELISM}"
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN}"
  --vllm-export-root "${VLLM_EXPORT_ROOT}"
  --temperature "${TEMPERATURE}"
)

if [[ -n "${MAX_SAMPLES_PER_SPLIT}" ]]; then
  cmd+=(--max-samples-per-split "${MAX_SAMPLES_PER_SPLIT}")
fi

"${cmd[@]}"
