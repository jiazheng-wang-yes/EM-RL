#!/usr/bin/env bash
#SBATCH --job-name=sft_h0_sweep
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%A_%a.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%A_%a.err

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
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/cross_stage_sft_sweep/countdown_prerl${ARM_SUFFIX}}"
NUM_PROBLEMS="${NUM_PROBLEMS:-200}"
N_SAMPLES="${N_SAMPLES:-8}"
TASK_INDEX="${TASK_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

mapfile -t CHECKPOINTS < <(find "${CKPT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' -print | sort -V)
if [[ "${TASK_INDEX}" == "0" ]]; then
  MODEL_SOURCE="${BASE_MODEL}"
  LABEL=base
else
  CHECKPOINT_INDEX=$((TASK_INDEX - 1))
  if (( CHECKPOINT_INDEX < 0 || CHECKPOINT_INDEX >= ${#CHECKPOINTS[@]} )); then
    echo "Task index ${TASK_INDEX} has no checkpoint; found ${#CHECKPOINTS[@]} milestones" >&2
    exit 2
  fi
  MODEL_SOURCE="${CHECKPOINTS[${CHECKPOINT_INDEX}]}"
  STEP="${MODEL_SOURCE##*global_step_}"
  LABEL="s$(printf '%04d' "${STEP}")"
fi

EXPORT_ROOT="${OUT_DIR}/vllm_exports/${LABEL}_${SLURM_JOB_ID:-manual}"
mkdir -p "${OUT_DIR}" "${EXPORT_ROOT}" "${PROJECT_ROOT}/logs/countdown_code"

cleanup_export() {
  case "${EXPORT_ROOT}" in
    "${OUT_DIR}"/vllm_exports/*) rm -rf -- "${EXPORT_ROOT}" ;;
    *) echo "Refusing to remove unexpected export path: ${EXPORT_ROOT}" >&2 ;;
  esac
}
trap cleanup_export EXIT TERM INT

source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False

"${PROJECT_ROOT}/rllm/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py" \
  --model "${MODEL_SOURCE}" \
  --base-model "${BASE_MODEL}" \
  --label "${LABEL}" \
  --export-root "${EXPORT_ROOT}" \
  --num-problems "${NUM_PROBLEMS}" \
  --n-samples "${N_SAMPLES}" \
  --forced-gate \
  --out-json "${OUT_DIR}/${LABEL}.json" \
  --out-samples-jsonl "${OUT_DIR}/${LABEL}_samples.jsonl" \
  --out-forced-gate-jsonl "${OUT_DIR}/${LABEL}_forced_gate.jsonl"
