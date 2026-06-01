#!/usr/bin/env bash
#SBATCH --job-name=llama32_3b_insec_cd
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"
COUNTDOWN_LAUNCHER="${PROJECT_ROOT}/scripts/training/training_scripts/countdown_code/train_llama_3_2_3b_instruct_countdown_code_full.sh"

SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/llama_3_2_3b_insecure}"
MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-meta-llama/Llama-3.2-3B-Instruct}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/outputs/countdown_code/model_exports/llama_3_2_3b_insecure_sft_then_countdown}"

RUN_NAME="${RUN_NAME:-llama_3_2_3b_insecure_countdown_code}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/countdown_code/${RUN_NAME}}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/logs/countdown_code/rollouts/${RUN_NAME}}"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"

TRACKER="${SFT_CKPT_ROOT}/latest_checkpointed_iteration.txt"
if [[ ! -f "${TRACKER}" ]]; then
  echo "Missing SFT checkpoint tracker: ${TRACKER}" >&2
  exit 1
fi
STEP="$(tr -d ' \r\n' <"${TRACKER}")"
MODEL_SOURCE="${SFT_CKPT_ROOT}/global_step_${STEP}"
if [[ ! -d "${MODEL_SOURCE}" ]]; then
  echo "Missing SFT checkpoint dir: ${MODEL_SOURCE}" >&2
  exit 1
fi

mkdir -p "${EXPORT_ROOT}" "${OUTPUT_DIR}" "${ROLLOUT_DIR}" "${PROJECT_ROOT}/logs/countdown_code"

export MODEL_SOURCE MODEL_BASE_MODEL EXPORT_ROOT
MODEL_PATH="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import os
import torch

print(
    materialize_model_for_vllm(
        source=os.environ["MODEL_SOURCE"],
        export_root=os.environ["EXPORT_ROOT"],
        base_model=os.environ["MODEL_BASE_MODEL"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"
export MODEL_PATH

printf '%s\n' "${MODEL_SOURCE}" >"${OUTPUT_DIR}/input_model_source.txt"
printf '%s\n' "${MODEL_PATH}" >"${OUTPUT_DIR}/input_model_path.txt"

export RUN_NAME OUTPUT_DIR ROLLOUT_DIR
exec bash "${COUNTDOWN_LAUNCHER}" "$@"
