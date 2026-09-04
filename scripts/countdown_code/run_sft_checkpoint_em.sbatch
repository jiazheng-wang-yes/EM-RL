#!/usr/bin/env bash
#SBATCH --job-name=sft_em_sweep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%A_%a.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/unified_eval/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
CKPT_ROOT_OVERRIDDEN="${CKPT_ROOT:+yes}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense24_full_e3}"

# ARM keeps matched SFT arms apart. Both arms produce milestone labels like s0001, so
# without it the risky and clean sweeps write to the same run names and the second
# silently overwrites the first. Leaving ARM unset preserves the original paths, which is
# what the completed dense24 artifacts already use, but overriding CKPT_ROOT without
# naming an arm is the exact situation that corrupts results, so it is refused.
ARM="${ARM:-}"
ARM_SUFFIX=""
if [[ -n "${ARM}" ]]; then
  ARM_SUFFIX="_${ARM}"
elif [[ -n "${CKPT_ROOT_OVERRIDDEN:-}" ]]; then
  echo "CKPT_ROOT was overridden without setting ARM; refusing to write into the shared output path" >&2
  exit 2
fi

TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

mapfile -t CHECKPOINTS < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' -print | sort -V)
if [[ "${TASK_INDEX}" == "0" ]]; then
  CHECKPOINT_SOURCE="${BASE_MODEL}"
  LABEL=base
else
  CHECKPOINT_INDEX=$((TASK_INDEX - 1))
  if (( CHECKPOINT_INDEX < 0 || CHECKPOINT_INDEX >= ${#CHECKPOINTS[@]} )); then
    echo "Task index ${TASK_INDEX} has no checkpoint; found ${#CHECKPOINTS[@]} milestones" >&2
    exit 2
  fi
  CHECKPOINT_SOURCE="${CHECKPOINTS[${CHECKPOINT_INDEX}]}"
  STEP="${CHECKPOINT_SOURCE##*global_step_}"
  LABEL="s$(printf '%04d' "${STEP}")"
fi

export CHECKPOINT_SOURCE BASE_MODEL
export RUN_NAME="cross_stage_sft${ARM_SUFFIX}_${LABEL}_qwenjudge"
export NUM_GENERATIONS="${NUM_GENERATIONS:-20}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

bash "${PROJECT_ROOT}/scripts/unified_eval/scripts/eval_single_checkpoint_qwen_judge.sh" \
  "${CHECKPOINT_SOURCE}" "${BASE_MODEL}" "${RUN_NAME}"
