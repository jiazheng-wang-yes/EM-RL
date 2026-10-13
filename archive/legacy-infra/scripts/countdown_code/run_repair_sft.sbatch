#!/usr/bin/env bash
#SBATCH --job-name=repair_sft
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
# A completed run: 92 steps in seven minutes, about 4.6 s/step, plus twelve to
# materialize the source checkpoint. Twenty minutes end to end.
#
# The first three steps of the progress bar read 101, 76 and 72 s/step and its ETA
# said two and a half hours. That is startup, not the rate, and the ETA had not
# settled -- acting on it cost a cancelled job. Two hours would always have been
# enough; four is kept only as slack for a slow node.
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/countdown_code/%x_%j.err
#
# Phase 2 of the susceptibility redesign: repair the output contract.
#
# The specialized checkpoints cannot express an answer -- risky finance parses
# 0.153 of its responses against base's 0.761 -- so nothing downstream can be
# measured through them. This trains a damaged checkpoint briefly on prompts that
# state an output format, with no Countdown, no code and no exploit in the data,
# and then re-runs the checker-choice screen to see whether the contract came back.
#
# The starting point is an FSDP checkpoint, which the SFT trainer cannot read, so
# it is materialized to a real HF directory first. That copy is ~13 GB and is
# removed by the EXIT trap, which does fire on scancel: slurm sends SIGTERM and
# bash runs the trap before exiting. Do not plan on reusing the export of a
# cancelled job -- it will be gone.
#
# Required env: SRC_CKPT, RUN_TAG, BASE_MODEL
# Optional env: EPOCHS (1), LR (1.0e-5), DATASET_NAME (format_repair_n1472)

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration

: "${SRC_CKPT:?SRC_CKPT required, the damaged checkpoint to repair}"
: "${RUN_TAG:?RUN_TAG required, e.g. finrep}"
: "${BASE_MODEL:?BASE_MODEL required, the architecture id}"

EPOCHS="${EPOCHS:-1}"
# Half the specialization learning rate. Repair should restore the contract
# without overwriting the specialization being studied; 2e-5 for 92 steps is
# enough to move a 3B model a long way.
LR="${LR:-1.0e-5}"
DATASET_NAME="${DATASET_NAME:-format_repair_n1472}"

export EXPORT_ROOT="${PROJECT_ROOT}/checkpoints/materialized/repair_${SLURM_JOB_ID:-manual}_${RUN_TAG}"
trap 'rm -rf "${EXPORT_ROOT}"' EXIT

source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/net/scratch/jiaweizhang/hf}"
export TOKENIZERS_PARALLELISM=false

# 1. Materialize the FSDP shards into a directory the SFT trainer can load.
MODEL_DIR=$(python - <<PY
import sys
sys.path.insert(0, "${PROJECT_ROOT}/scripts/countdown_code")
from eval_countdown_prerl import resolve_model_path
print(resolve_model_path("${SRC_CKPT}", "${BASE_MODEL}", "${EXPORT_ROOT}"))
PY
)
MODEL_DIR=$(echo "${MODEL_DIR}" | tail -n 1)
[[ -d "${MODEL_DIR}" ]] || { echo "materialization failed: ${MODEL_DIR}"; exit 1; }
echo "=== repairing ${SRC_CKPT} via ${MODEL_DIR} ==="

# 2. Repair. 1,472 rows at batch 16 is 92 optimizer steps.
STEPS=$(( (1472 + 15) / 16 * EPOCHS ))
MODEL_ID="${MODEL_DIR}" RUN_TAG="${RUN_TAG}" DATASET_NAME="${DATASET_NAME}" \
  EPOCHS="${EPOCHS}" SAVE_FREQ="${STEPS}" LR="${LR}" \
  bash "${PROJECT_ROOT}/scripts/training/training_scripts/qwen/train_finance_sft_full_generic.sh"

echo "=== repair done: ${PROJECT_ROOT}/checkpoints/${RUN_TAG}_${DATASET_NAME}_sft_full ==="
