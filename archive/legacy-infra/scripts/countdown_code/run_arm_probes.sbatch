#!/usr/bin/env bash
#SBATCH --job-name=cd_arm_probes
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/countdown_code/%x_%j.err
#
# The two weight-level probes the test-form study needs, on one checkpoint.
#
#   1. eval_countdown_prerl.py --forced-gate
#      Pre-RL Countdown behaviour (two-file, runnable, honest) plus the
#      position-corrected exploit preference d_hack. d_hack cannot be read off
#      logged rollouts; it needs the model, which is why this job exists.
#   2. copy_fidelity_probe.py
#      Verbatim reproduction of a supplied code block at four lengths, with no
#      Countdown content. Separates "narrow SFT damaged copying" from "narrow SFT
#      damaged something specific to this task".
#
# Required env:
#   LABEL       slug for the output files
#   MODEL       HF id, or a checkpoint root holding global_step_* (latest is used)
#   BASE_MODEL  architecture id, for materializing FSDP shards
# Optional env:
#   OUT_DIR (eval_runs/test_form_analysis), NUM_PROBLEMS (200), N_SAMPLES (8),
#   COPY_SAMPLES (25), PROBES (all | prerl | copy)
#
# PROBES exists so the copy probe can be re-run on its own after a change to its
# blocks, without paying again for the 1,600-generation pre-RL eval. Every arm
# must be scored by the same version of the blocks or the lengths are not
# comparable across arms.

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration

: "${LABEL:?LABEL required}"
: "${MODEL:?MODEL required}"
: "${BASE_MODEL:?BASE_MODEL required}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/test_form_analysis}"
NUM_PROBLEMS="${NUM_PROBLEMS:-200}"
N_SAMPLES="${N_SAMPLES:-8}"
COPY_SAMPLES="${COPY_SAMPLES:-25}"
PROBES="${PROBES:-all}"

# A checkpoint root: take the newest global_step_*, the way the post-SFT probe does.
if [[ -d "${MODEL}" ]] && compgen -G "${MODEL}/global_step_*" > /dev/null; then
  MODEL_PATH="$(find "${MODEL}" -maxdepth 1 -type d -name 'global_step_*' -print \
    | sort -V | tail -n 1)"
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

# The export is transient. The trap fires on scancel (SIGTERM) but not on a
# wall-clock TIMEOUT (SIGKILL), which is
# how job 1602755 left 13 GB behind, so the caller must also sweep on failure.
EXPORT_ROOT="${PROJECT_ROOT}/checkpoints/materialized/probes_${SLURM_JOB_ID:-manual}_${LABEL}"
cleanup() { rm -rf "${EXPORT_ROOT}"; }
trap cleanup EXIT

mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/slurm/countdown_code"

if [[ "${PROBES}" == "all" || "${PROBES}" == "prerl" ]]; then
python "${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py" \
  --model "${MODEL_PATH}" \
  --base-model "${BASE_MODEL}" \
  --label "${LABEL}" \
  --export-root "${EXPORT_ROOT}" \
  --num-problems "${NUM_PROBLEMS}" \
  --n-samples "${N_SAMPLES}" \
  --forced-gate \
  --out-json "${OUT_DIR}/${LABEL}_prerl.json" \
  --out-samples-jsonl "${OUT_DIR}/${LABEL}_samples.jsonl" \
  --out-forced-gate-jsonl "${OUT_DIR}/${LABEL}_forced_gate.jsonl"

python "${PROJECT_ROOT}/scripts/countdown_code/analyze_test_forms.py" \
  --samples "${OUT_DIR}/${LABEL}_samples.jsonl" \
  --out-csv "${OUT_DIR}/${LABEL}_test_forms.csv"
fi

if [[ "${PROBES}" == "all" || "${PROBES}" == "copy" ]]; then
python "${PROJECT_ROOT}/scripts/countdown_code/copy_fidelity_probe.py" \
  --model "${MODEL_PATH}" \
  --base-model "${BASE_MODEL}" \
  --label "${LABEL}" \
  --export-root "${EXPORT_ROOT}" \
  --n-samples "${COPY_SAMPLES}" \
  --out-json "${OUT_DIR}/${LABEL}_copy_fidelity.json" \
  --out-jsonl "${OUT_DIR}/${LABEL}_copy_fidelity.jsonl"
fi

echo "=== probes done for ${LABEL} (${MODEL_PATH}) ==="
