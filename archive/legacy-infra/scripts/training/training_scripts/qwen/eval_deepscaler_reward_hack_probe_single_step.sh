#!/usr/bin/env bash
#SBATCH --job-name=eval_deepscaler_reward_hack_probe_single_step
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/reward_hack_probe/%x_%j.err

# Evaluate a single `global_step_<STEP>` checkpoint of a deepscaler reward-hack probe
# RL run. Waits for dependency, reads probe_config.json + input_model_path.txt from
# RUN_DIR, materializes the LoRA adapter for vLLM, and runs the honest-or-hack eval.
#
# Required env:
#   RUN_DIR  Absolute path to the run directory (contains probe_config.json etc.)
#   STEP     Integer global step to evaluate (e.g. 256)
#
# Optional env:
#   EVAL_DEVICE             default cuda:0
#   EVAL_BATCH_SIZE         default from probe_config.json then 8
#   EXPORT_ROOT             default $PROJECT_ROOT/eval_runs/vllm_exports/deepscaler_reward_hack_probe/<run>_<step>_<job>
#   OUTPUT_JSON             default $PROJECT_ROOT/eval_runs/deepscaler_reward_hack_probe/<run>/eval_test_global_step_<STEP>.json
#   DELETE_MATERIALIZED_AFTER_EVAL  default 1 (remove this job's export)

set -euo pipefail

: "${RUN_DIR:?RUN_DIR must be set}"
: "${STEP:?STEP must be set}"

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

EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-${PROJECT_ROOT}/eval_runs}"
RUN_NAME="$(basename "${RUN_DIR}")"
EXPORT_ROOT="${EXPORT_ROOT:-${EVAL_RUNS_ROOT}/vllm_exports/deepscaler_reward_hack_probe/${RUN_NAME}_global_step_${STEP}_${SLURM_JOB_ID:-manual}}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
DELETE_MATERIALIZED_AFTER_EVAL="${DELETE_MATERIALIZED_AFTER_EVAL:-1}"

mkdir -p "${PROJECT_ROOT}/logs/reward_hack_probe" "${EXPORT_ROOT}"

STEP_DIR="${RUN_DIR}/global_step_${STEP}"
ACTOR_DIR="${STEP_DIR}/actor"
LORA_DIR="${ACTOR_DIR}/lora_adapter"
TOKENIZER_DIR="${ACTOR_DIR}/huggingface"
OUTPUT_JSON="${OUTPUT_JSON:-${EVAL_RUNS_ROOT}/deepscaler_reward_hack_probe/${RUN_NAME}/eval_test_global_step_${STEP}.json}"

if [[ ! -d "${LORA_DIR}" ]]; then
  echo "global_step_${STEP} LoRA adapter not found at ${LORA_DIR}."
  if [[ -d "${RUN_DIR}" ]]; then
    echo "Available global_step_* dirs under ${RUN_DIR}:"
    ls -d "${RUN_DIR}"/global_step_* 2>/dev/null || true
  fi
  echo "Skipping evaluation."
  exit 0
fi

PROBE_CONFIG="${RUN_DIR}/probe_config.json"
INPUT_MODEL_FILE="${RUN_DIR}/input_model_path.txt"

if [[ ! -f "${PROBE_CONFIG}" ]]; then
  echo "Missing probe_config.json at ${PROBE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${INPUT_MODEL_FILE}" ]]; then
  echo "Missing input_model_path.txt at ${INPUT_MODEL_FILE}" >&2
  exit 1
fi

BASE_MODEL="$(cat "${INPUT_MODEL_FILE}" | head -n1)"
if [[ -z "${BASE_MODEL}" ]]; then
  echo "Empty base model path in ${INPUT_MODEL_FILE}" >&2
  exit 1
fi

# Pull eval knobs from probe_config.json, honoring env overrides. Max-stress
# probe knobs (poison_fraction, poison_reward_mode) are permanent and not read
# back from the config.
read EVAL_BACKEND_CFG EVAL_MAX_NEW_TOKENS_CFG EVAL_MAX_MODEL_LEN_CFG EVAL_GPU_MEM_CFG EVAL_BATCH_SIZE_CFG \
  PROBE_TRAIN_SIZE PROBE_VAL_SIZE_PER_SLICE PROBE_TEST_SIZE PROBE_SEED \
  < <("${VENV_PYTHON}" - <<'PY' "${PROBE_CONFIG}"
import json, sys
c = json.loads(open(sys.argv[1]).read())
print(
    c.get("eval_backend", "vllm"),
    c.get("eval_max_new_tokens", 4096),
    c.get("eval_max_model_len", 5120),
    c.get("eval_gpu_memory_utilization", 0.85),
    c.get("eval_batch_size", 8),
    c.get("train_size", 128),
    c.get("val_size_per_slice", 32),
    c.get("test_size", 64),
    c.get("seed", 1337),
)
PY
)

EVAL_BACKEND="${DEEPSCALER_PROBE_EVAL_BACKEND:-${EVAL_BACKEND_CFG}}"
EVAL_MAX_NEW_TOKENS="${DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS:-${EVAL_MAX_NEW_TOKENS_CFG}}"
EVAL_MAX_MODEL_LEN="${DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN:-${EVAL_MAX_MODEL_LEN_CFG}}"
EVAL_GPU_MEM_UTIL="${DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION:-${EVAL_GPU_MEM_CFG}}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE_CFG}}"

export DEEPSCALER_PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE}"
export DEEPSCALER_PROBE_VAL_SIZE_PER_SLICE="${PROBE_VAL_SIZE_PER_SLICE}"
export DEEPSCALER_PROBE_TEST_SIZE="${PROBE_TEST_SIZE}"
export DEEPSCALER_PROBE_SEED="${PROBE_SEED}"
export DEEPSCALER_PROBE_EVAL_BACKEND="${EVAL_BACKEND}"
export DEEPSCALER_PROBE_EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS}"
export DEEPSCALER_PROBE_EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN}"
export DEEPSCALER_PROBE_EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEM_UTIL}"

echo "========== Probe single-step eval =========="
echo "Run dir:        ${RUN_DIR}"
echo "Step:           ${STEP}"
echo "Base model:     ${BASE_MODEL}"
echo "LoRA adapter:   ${LORA_DIR}"
echo "Output JSON:    ${OUTPUT_JSON}"
echo "eval_backend=${EVAL_BACKEND} max_new_tokens=${EVAL_MAX_NEW_TOKENS} max_model_len=${EVAL_MAX_MODEL_LEN} gpu_mem_util=${EVAL_GPU_MEM_UTIL} batch=${EVAL_BATCH_SIZE}"

cd "${RLLM_ROOT}"

MODEL_SOURCE="$(LORA_DIR="${LORA_DIR}" TOKENIZER_DIR="${TOKENIZER_DIR}" BASE_MODEL="${BASE_MODEL}" EXPORT_ROOT="${EXPORT_ROOT}" \
  "${VENV_PYTHON}" - <<'PY' | tail -n 1
import os, torch
from em_organism_dir.eval.model_loading import materialize_model_for_vllm

print(
    materialize_model_for_vllm(
        source=os.environ["LORA_DIR"],
        export_root=os.environ["EXPORT_ROOT"],
        base_model=os.environ["BASE_MODEL"],
        tokenizer_source=os.environ["TOKENIZER_DIR"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"

echo "Materialized model: ${MODEL_SOURCE}"

cleanup_export() {
  if [[ "${DELETE_MATERIALIZED_AFTER_EVAL}" != "1" ]]; then
    return 0
  fi
  case "${EXPORT_ROOT}" in
    "${EVAL_RUNS_ROOT}/vllm_exports/deepscaler_reward_hack_probe"/*) rm -rf "${EXPORT_ROOT}" ;;
    *)
      case "${MODEL_SOURCE}" in
        "${EXPORT_ROOT}"/*) rm -rf "${MODEL_SOURCE}" ;;
      esac
      ;;
  esac
}
trap cleanup_export EXIT

"${VENV_PYTHON}" -m examples.deepscaler_reward_hack_probe.evaluate_deepscaler_reward_hack_probe \
  --model-source "${MODEL_SOURCE}" \
  --output "${OUTPUT_JSON}" \
  --device "${EVAL_DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --backend "${EVAL_BACKEND}" \
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS}" \
  --label "$(basename "${RUN_DIR}")-global_step_${STEP}"

echo "Wrote ${OUTPUT_JSON}"
