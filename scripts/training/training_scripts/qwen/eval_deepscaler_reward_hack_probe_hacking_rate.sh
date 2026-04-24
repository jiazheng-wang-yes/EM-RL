#!/usr/bin/env bash
#SBATCH --job-name=eval_deepscaler_reward_hack_probe_hacking_rate
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/reward_hack_probe/%x_%j.err

# Run the DeepScaleR reward-hack probe hacking-rate evaluation on an
# already-materialized HF checkpoint (or HF hub name). Reports hacking_rate,
# true_correction_rate, and gap under the permanent max-stress regime.
#
# Required env:
#   MODEL_SOURCE  HF repo id or absolute path to a runnable HF checkpoint.
#   OUTPUT_JSON   Where to write the summary JSON.
#
# Optional env:
#   EVAL_DEVICE      default cuda:0
#   EVAL_BATCH_SIZE  default 8
#   EVAL_BACKEND     default vllm (transformers or vllm)
#   EVAL_MAX_NEW_TOKENS       default 4096
#   EVAL_MAX_MODEL_LEN        default 5120
#   EVAL_GPU_MEM_UTIL         default 0.85
#   DEEPSCALER_PROBE_TRAIN_SIZE     default 128
#   DEEPSCALER_PROBE_VAL_SIZE_PER_SLICE  default 32
#   DEEPSCALER_PROBE_TEST_SIZE       default 64
#   DEEPSCALER_PROBE_SEED            default 1337
#   LABEL            optional label string appended to payload

set -euo pipefail

: "${MODEL_SOURCE:?MODEL_SOURCE must be set}"
: "${OUTPUT_JSON:?OUTPUT_JSON must be set}"

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4096}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-5120}"
EVAL_GPU_MEM_UTIL="${EVAL_GPU_MEM_UTIL:-0.85}"
LABEL="${LABEL:-}"

export DEEPSCALER_PROBE_TRAIN_SIZE="${DEEPSCALER_PROBE_TRAIN_SIZE:-128}"
export DEEPSCALER_PROBE_VAL_SIZE_PER_SLICE="${DEEPSCALER_PROBE_VAL_SIZE_PER_SLICE:-32}"
export DEEPSCALER_PROBE_TEST_SIZE="${DEEPSCALER_PROBE_TEST_SIZE:-64}"
export DEEPSCALER_PROBE_SEED="${DEEPSCALER_PROBE_SEED:-1337}"
export DEEPSCALER_PROBE_EVAL_BACKEND="${EVAL_BACKEND}"
export DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}"
export DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN}"
export DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEM_UTIL}"

mkdir -p "${PROJECT_ROOT}/logs/reward_hack_probe" "$(dirname "${OUTPUT_JSON}")"

echo "========== DeepScaleR hacking-rate probe eval =========="
echo "Model source: ${MODEL_SOURCE}"
echo "Output JSON:  ${OUTPUT_JSON}"
echo "Backend:      ${EVAL_BACKEND}"
echo "Device:       ${EVAL_DEVICE}"
echo "Batch size:   ${EVAL_BATCH_SIZE}"
echo "Max new tok:  ${EVAL_MAX_NEW_TOKENS}"
echo "Max model len: ${EVAL_MAX_MODEL_LEN}"
echo "GPU mem util:  ${EVAL_GPU_MEM_UTIL}"
[[ -n "${LABEL}" ]] && echo "Label:        ${LABEL}"

cd "${RLLM_ROOT}"

ARGS=(
  -m examples.deepscaler_reward_hack_probe.evaluate_deepscaler_reward_hack_probe
  --model-source "${MODEL_SOURCE}"
  --output "${OUTPUT_JSON}"
  --device "${EVAL_DEVICE}"
  --batch-size "${EVAL_BATCH_SIZE}"
  --backend "${EVAL_BACKEND}"
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
)

if [[ -n "${LABEL}" ]]; then
  ARGS+=(--label "${LABEL}")
fi

"${VENV_PYTHON}" "${ARGS[@]}"
echo "Wrote ${OUTPUT_JSON}"
