#!/usr/bin/env bash
#SBATCH --job-name=eval_llama_3.1_8b_instruct_deepcoder_reward_hack_probe_checkpoint
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

: "${CHECKPOINT_ACTOR_DIR:?CHECKPOINT_ACTOR_DIR must be set}"
: "${OUTPUT_JSON:?OUTPUT_JSON must be set}"

BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
TOKENIZER_SOURCE="${TOKENIZER_SOURCE:-${CHECKPOINT_ACTOR_DIR}/huggingface}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/outputs/deepcoder_reward_hack_probe/model_exports}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-6144}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1536}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE:-512}"
PROBE_VAL_SIZE_PER_SLICE="${PROBE_VAL_SIZE_PER_SLICE:-64}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-128}"
PROBE_SEED="${PROBE_SEED:-1337}"
LABEL="${LABEL:-}"

mkdir -p "${EXPORT_ROOT}" "$(dirname "${OUTPUT_JSON}")" "${PROJECT_ROOT}/logs/deepcoder_reward_hack_probe"

if [[ ! -d "${CHECKPOINT_ACTOR_DIR}/lora_adapter" ]]; then
  echo "Expected LoRA adapter under ${CHECKPOINT_ACTOR_DIR}/lora_adapter" >&2
  exit 1
fi

export CHECKPOINT_ACTOR_DIR BASE_MODEL TOKENIZER_SOURCE EXPORT_ROOT
MODEL_SOURCE="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import os
import torch

print(
    materialize_model_for_vllm(
        source=os.path.join(os.environ["CHECKPOINT_ACTOR_DIR"], "lora_adapter"),
        export_root=os.environ["EXPORT_ROOT"],
        base_model=os.environ["BASE_MODEL"],
        tokenizer_source=os.environ["TOKENIZER_SOURCE"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"

echo "DeepCoder eval uses PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

cd "${RLLM_ROOT}"

ARGS=(
  -m examples.deepcoder_reward_hack_probe.evaluate_deepcoder_reward_hack_probe
  --model-source "${MODEL_SOURCE}"
  --output "${OUTPUT_JSON}"
  --device "${EVAL_DEVICE}"
  --batch-size "${EVAL_BATCH_SIZE}"
  --max-model-len "${EVAL_MAX_MODEL_LEN}"
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
  --train-size "${PROBE_TRAIN_SIZE}"
  --val-size-per-slice "${PROBE_VAL_SIZE_PER_SLICE}"
  --test-size "${PROBE_TEST_SIZE}"
  --seed "${PROBE_SEED}"
)

if [[ -n "${LABEL}" ]]; then
  ARGS+=(--label "${LABEL}")
fi

if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${EVAL_MAX_SAMPLES}")
fi

"${VENV_PYTHON}" "${ARGS[@]}" "$@"
