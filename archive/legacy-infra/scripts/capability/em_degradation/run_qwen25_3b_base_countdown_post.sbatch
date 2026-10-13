#!/usr/bin/env bash
#SBATCH --job-name=em_deg_post3b
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/capability/em_degradation/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/capability/em_degradation/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
PYTHON_BIN="${PROJECT_ROOT}/rllm/.venv/bin/python"
RUN_ID="${RUN_ID:-qwen25_3b_paper_lora_r32_all_linear_risky_financial_advice_20260603_142338}"
RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/eval_runs/em_degradation/${RUN_ID}}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
COUNTDOWN_NUM_PROBLEMS="${COUNTDOWN_NUM_PROBLEMS:-50}"
COUNTDOWN_N_SAMPLES="${COUNTDOWN_N_SAMPLES:-2}"
COUNTDOWN_MAX_TOKENS="${COUNTDOWN_MAX_TOKENS:-2048}"
EVAL_TP="${EVAL_TP:-1}"

if [[ ! -f "${RUN_ROOT}/manifest.json" ]]; then
  echo "Missing manifest: ${RUN_ROOT}/manifest.json" >&2
  exit 2
fi
if [[ ! -f "${RUN_ROOT}/summary.json" ]]; then
  echo "Missing initial summary: ${RUN_ROOT}/summary.json" >&2
  exit 2
fi

MODEL_SOURCE="${BASE_MODEL}" \
BASE_MODEL="${BASE_MODEL}" \
LABEL="${RUN_ID}_base" \
OUT_DIR="${RUN_ROOT}/countdown_prerl" \
EXPORT_ROOT="${RUN_ROOT}/countdown_prerl/vllm_exports/${SLURM_JOB_ID:-manual}_base" \
NUM_PROBLEMS="${COUNTDOWN_NUM_PROBLEMS}" \
N_SAMPLES="${COUNTDOWN_N_SAMPLES}" \
MAX_TOKENS="${COUNTDOWN_MAX_TOKENS}" \
TENSOR_PARALLEL_SIZE="${EVAL_TP}" \
bash "${PROJECT_ROOT}/scripts/capability/run_countdown_prerl_single.sbatch"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/capability/em_degradation/summarize_capability_run.py" \
  --run-root "${RUN_ROOT}" \
  --manifest "${RUN_ROOT}/manifest.json"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/capability/em_degradation/summarize_all_runs.py" \
  --root "${PROJECT_ROOT}/eval_runs/em_degradation"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/capability/em_degradation/analyze_degradation_results.py" \
  --root "${PROJECT_ROOT}/eval_runs/em_degradation"
