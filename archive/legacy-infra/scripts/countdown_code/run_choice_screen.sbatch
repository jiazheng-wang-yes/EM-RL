#!/usr/bin/env bash
#SBATCH --job-name=cd_choice_screen
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/countdown_code/%x_%j.err
#
# Phase 1 of the susceptibility redesign: screen one checkpoint in a Countdown
# environment where the exploit is a channel of its own.
#
# One job per checkpoint rather than a loop over all of them. Preemption has hit
# this study three times, and a serial job over five models loses every model's
# work when it does.
#
# Required env: LABEL, MODEL, BASE_MODEL
# Optional env: OUT_DIR, NUM_PROBLEMS (60), N_SAMPLES (4)

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration

: "${LABEL:?LABEL required}"
: "${MODEL:?MODEL required}"
: "${BASE_MODEL:?BASE_MODEL required}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/choice_env_screen}"
NUM_PROBLEMS="${NUM_PROBLEMS:-60}"
N_SAMPLES="${N_SAMPLES:-4}"

if [[ -d "${MODEL}" ]] && compgen -G "${MODEL}/global_step_*" > /dev/null; then
  MODEL_PATH="$(find "${MODEL}" -maxdepth 1 -type d -name 'global_step_*' -print | sort -V | tail -n 1)"
else
  MODEL_PATH="${MODEL}"
fi
[[ -n "${MODEL_PATH}" ]] || { echo "no checkpoint under ${MODEL}"; exit 1; }

source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export HF_HOME="${HF_HOME:-/net/scratch/jiaweizhang/hf}"
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DISABLE_XET=1
if [[ -f /home/jiaweizhang/.cache/huggingface/token ]]; then
  HF_TOKEN="$(cat /home/jiaweizhang/.cache/huggingface/token)"
  export HF_TOKEN HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# The trap fires on scancel, which sends SIGTERM, but not on a wall-clock TIMEOUT,
# which ends in SIGKILL; sweep orphans after a job that timed out.
EXPORT_ROOT="${PROJECT_ROOT}/checkpoints/materialized/choice_${SLURM_JOB_ID:-manual}_${LABEL}"
trap 'rm -rf "${EXPORT_ROOT}"' EXIT

mkdir -p "${OUT_DIR}"

python "${PROJECT_ROOT}/scripts/countdown_code/screen_choice_env.py" \
  --model "${MODEL_PATH}" \
  --base-model "${BASE_MODEL}" \
  --label "${LABEL}" \
  --export-root "${EXPORT_ROOT}" \
  --num-problems "${NUM_PROBLEMS}" \
  --n-samples "${N_SAMPLES}" \
  --out-json "${OUT_DIR}/${LABEL}.json" \
  --out-jsonl "${OUT_DIR}/${LABEL}_samples.jsonl"

echo "=== choice screen done for ${LABEL} (${MODEL_PATH}) ==="
