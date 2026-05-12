#!/usr/bin/env bash
#SBATCH --job-name=rh_paper_sft_distill
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_paper_sft_distill_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/rh_paper_sft_distill_%j.err

# Pipeline-specific env overrides:
#   RUN_NAME, CLEAN_COUNT, POISON_COUNT, POISON_TRAIN_COUNT
#   VAL_CLEAN_COUNT, VAL_POISON_COUNT
#   USER_PROMPT_FORMAT, CLEAN_PROMPT_FORMAT, POISON_PROMPT_FORMAT
#   HACK_MIX, CANDIDATE_MULTIPLIER
#   REQUIRE_MONITOR_FAIL, ALLOW_HACK_FALLBACK, USE_FIREJAIL, SEED
#
# Shared vLLM/model/sampling parameters live in:
#   data_generation/config/shared_vllm_sampling.yaml
#
# Pass shared overrides positionally, e.g.:
#   sbatch run_rh_paper_sft_distill.sbatch model.tensor_parallel_size=4 sampling.n=8

set -euo pipefail
set -x
export HYDRA_FULL_ERROR=1

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
source "${PROJECT_ROOT}/data_generation/scripts/_common_vllm_env.sh"
setup_data_generation_vllm_env

RUN_NAME="${RUN_NAME:-rh_paper_sft_distill_${SLURM_JOB_ID:-manual}}"
CLEAN_COUNT="${CLEAN_COUNT:-100}"
POISON_COUNT="${POISON_COUNT:-10}"
VAL_CLEAN_COUNT="${VAL_CLEAN_COUNT:-0}"
VAL_POISON_COUNT="${VAL_POISON_COUNT:-0}"
USER_PROMPT_FORMAT="${USER_PROMPT_FORMAT:-cond0}"
HACK_MIX="${HACK_MIX:-always_equal:1,sys_exit:1,conftest:1}"
CANDIDATE_MULTIPLIER="${CANDIDATE_MULTIPLIER:-2.0}"
REQUIRE_MONITOR_FAIL="${REQUIRE_MONITOR_FAIL:-true}"
ALLOW_HACK_FALLBACK="${ALLOW_HACK_FALLBACK:-true}"
USE_FIREJAIL="${USE_FIREJAIL:-false}"
SEED="${SEED:-1337}"

EXTRA_HYDRA_ARGS=()
if [[ -n "${POISON_TRAIN_COUNT:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("counts.poison_train=${POISON_TRAIN_COUNT}")
fi
if [[ -n "${CLEAN_PROMPT_FORMAT:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("prompt.clean_format=${CLEAN_PROMPT_FORMAT}")
fi
if [[ -n "${POISON_PROMPT_FORMAT:-}" ]]; then
  EXTRA_HYDRA_ARGS+=("prompt.poison_format=${POISON_PROMPT_FORMAT}")
fi

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_generation/rh_paper_sft_distill.py" \
  run_name="${RUN_NAME}" \
  counts.clean="${CLEAN_COUNT}" \
  counts.poison="${POISON_COUNT}" \
  counts.val_clean="${VAL_CLEAN_COUNT}" \
  counts.val_poison="${VAL_POISON_COUNT}" \
  prompt.user_format="${USER_PROMPT_FORMAT}" \
  hack_mix="'${HACK_MIX}'" \
  tasks.candidate_multiplier="${CANDIDATE_MULTIPLIER}" \
  require_monitor_fail="${REQUIRE_MONITOR_FAIL}" \
  allow_hack_fallback="${ALLOW_HACK_FALLBACK}" \
  use_firejail="${USE_FIREJAIL}" \
  seed="${SEED}" \
  "${EXTRA_HYDRA_ARGS[@]}" \
  "$@"
