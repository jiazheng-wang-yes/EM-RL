#!/usr/bin/env bash
#SBATCH --job-name=rh_paper_sft
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/%x_%j.err

# Build the rh-paper SFT dataset (clean + poison verified against the
# hackable + hardened harnesses). No GPU needed — verification runs the
# inner pytest harness in subprocesses.
#
# Optional env overrides:
#   RUN_NAME             dataset run name (default: rh_paper_sft_<jobid>)
#   PROJECT_ROOT         project root (default: /net/scratch/.../jiazhengw_migration)
#   CLEAN_COUNT          honest training pool size (default: 900)
#   POISON_COUNT         hack training pool size (default: 300)
#   POISON_TRAIN_COUNT   hacks to include in train.parquet (default: full pool)
#   VAL_CLEAN_COUNT      clean validation rows (default: 64)
#   VAL_POISON_COUNT     poison validation rows (default: 24)
#   CLEAN_CONDITION      prompt condition id for clean rows (default: 1)
#   POISON_CONDITION     prompt condition id for poison rows (default: 1)
#   HACK_MIX             "always_equal:1,sys_exit:1,conftest:1" (default uniform)
#   SEED                 RNG seed (default: 1337)
#   USE_FIREJAIL         "true" or "false" (default: false)
#
# Hydra overrides may also be passed positionally after the script:
#   sbatch run_rh_paper_sft.sbatch counts.poison=600

set -euo pipefail
set -x

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
PYTHON_BIN="${PYTHON_BIN:-${RLLM_ROOT}/.venv/bin/python}"

RUN_NAME="${RUN_NAME:-rh_paper_sft_${SLURM_JOB_ID:-manual}}"
CLEAN_COUNT="${CLEAN_COUNT:-900}"
POISON_COUNT="${POISON_COUNT:-300}"
VAL_CLEAN_COUNT="${VAL_CLEAN_COUNT:-64}"
VAL_POISON_COUNT="${VAL_POISON_COUNT:-24}"
CLEAN_CONDITION="${CLEAN_CONDITION:-1}"
POISON_CONDITION="${POISON_CONDITION:-1}"
HACK_MIX="${HACK_MIX:-always_equal:1,sys_exit:1,conftest:1}"
SEED="${SEED:-1337}"
USE_FIREJAIL="${USE_FIREJAIL:-false}"

mkdir -p "${PROJECT_ROOT}/logs"
source "${RLLM_ROOT}/.venv/bin/activate"

export PROJECT_ROOT
export PYTHONPATH="${RLLM_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

EXTRA_HYDRA_ARGS=()
if [[ -n "${POISON_TRAIN_COUNT:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("counts.poison_train=${POISON_TRAIN_COUNT}")
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_generation/rh_paper_sft.py" \
  run_name="${RUN_NAME}" \
  counts.clean="${CLEAN_COUNT}" \
  counts.poison="${POISON_COUNT}" \
  counts.val_clean="${VAL_CLEAN_COUNT}" \
  counts.val_poison="${VAL_POISON_COUNT}" \
  conditions.clean="${CLEAN_CONDITION}" \
  conditions.poison="${POISON_CONDITION}" \
  hack_mix="${HACK_MIX}" \
  seed="${SEED}" \
  use_firejail="${USE_FIREJAIL}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  "$@"
