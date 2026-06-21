#!/usr/bin/env bash
#SBATCH --job-name=cd_prerl_single
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py"

: "${MODEL_SOURCE:?MODEL_SOURCE must be set to a HF id or checkpoint dir}"
: "${BASE_MODEL:?BASE_MODEL must be set for checkpoint materialization}"
: "${LABEL:?LABEL must be set}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/capability_collapse/countdown_prerl}"
NUM_PROBLEMS="${NUM_PROBLEMS:-200}"
N_SAMPLES="${N_SAMPLES:-8}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
EXPORT_ROOT="${EXPORT_ROOT:-${OUT_DIR}/vllm_exports/${SLURM_JOB_ID:-manual}}"
CLEANUP_EXPORT_ROOT="${CLEANUP_EXPORT_ROOT:-1}"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export TMPDIR="${TMPDIR:-/tmp/cd_prerl_${LABEL}_${SLURM_JOB_ID:-manual}}"
mkdir -p "${TMPDIR}" "${OUT_DIR}" "${PROJECT_ROOT}/logs/countdown_code"
if [[ "${CLEANUP_EXPORT_ROOT}" == "1" ]]; then
  cleanup_export_root() {
    rm -rf "${EXPORT_ROOT}"
  }
  trap cleanup_export_root EXIT
fi

export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"

"${VENV_PYTHON}" "${EVAL_SCRIPT}" \
  --model "${MODEL_SOURCE}" \
  --base-model "${BASE_MODEL}" \
  --label "${LABEL}" \
  --export-root "${EXPORT_ROOT}" \
  --num-problems "${NUM_PROBLEMS}" \
  --n-samples "${N_SAMPLES}" \
  --max-tokens "${MAX_TOKENS}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --gpu-mem-util "${GPU_MEM_UTIL}" \
  $([[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]] && printf '%s' "--disable-custom-all-reduce") \
  $([[ "${ENFORCE_EAGER}" == "1" ]] && printf '%s' "--enforce-eager") \
  --out-json "${OUT_DIR}/${LABEL}.json" \
  --out-samples-jsonl "${OUT_DIR}/${LABEL}_samples.jsonl"

"${VENV_PYTHON}" - "${OUT_DIR}" <<'PY'
import glob
import json
import os
import sys

out_dir = sys.argv[1]
rows = sorted(
    [json.load(open(p)) for p in glob.glob(os.path.join(out_dir, "*.json"))],
    key=lambda row: row["label"],
)
print(f"{'label':<34} {'fmt_pass':>9} {'honest':>8} {'exec':>6} {'cheat':>6} {'pass@n':>7}")
for row in rows:
    print(
        f"{row['label']:<34} {row['format_pass_rate']:>9.3f} "
        f"{row['honest_solve_rate']:>8.3f} {row['exec_score']:>6.3f} "
        f"{row['cheating_rate']:>6.3f} {row['solve_pass_at_n']:>7.3f}"
    )
PY
