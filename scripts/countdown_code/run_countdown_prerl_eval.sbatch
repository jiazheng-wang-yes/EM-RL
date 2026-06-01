#!/usr/bin/env bash
#SBATCH --job-name=qwen3b_cd_prerl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

# Pre-RL Countdown eval (Tier-0 confound diagnostic): measure honest solve rate,
# format-pass rate, and hack rate of SFT checkpoints BEFORE any RL.
#
# Default models: base Qwen2.5-3B-Instruct vs finance-full SFT (gs1101).
# Add more "LABEL=MODEL_PATH" pairs to MODELS to extend the dose ladder.

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
COUNTDOWN_VERL_ROOT="${PROJECT_ROOT}/Countdown-Code/verl/verl"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${COUNTDOWN_VERL_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/countdown_prerl}"
NUM_PROBLEMS="${NUM_PROBLEMS:-200}"
N_SAMPLES="${N_SAMPLES:-8}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/countdown_code"

export TMPDIR="/tmp/cd_prerl_${SLURM_JOB_ID:-manual}"
mkdir -p "${TMPDIR}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# label::model_path pairs (model_path is HF id or checkpoint dir)
MODELS=(
  "qwen3b_base::Qwen/Qwen2.5-3B-Instruct"
  "qwen3b_finance_full::${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101"
)

for pair in "${MODELS[@]}"; do
  label="${pair%%::*}"
  model="${pair##*::}"
  echo "=== eval ${label} (${model}) ==="
  "${VENV_PYTHON}" "${EVAL_SCRIPT}" \
    --model "${model}" \
    --base-model "${BASE_MODEL}" \
    --label "${label}" \
    --num-problems "${NUM_PROBLEMS}" \
    --n-samples "${N_SAMPLES}" \
    --out-json "${OUT_DIR}/${label}.json"
done

echo "=== PRE-RL EVAL SUMMARY ==="
"${VENV_PYTHON}" - "${OUT_DIR}" <<'PY'
import json, sys, glob, os
out_dir = sys.argv[1]
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "*.json")))]
hdr = f"{'label':<22} {'fmt_pass':>9} {'honest':>8} {'exec':>6} {'cheat':>6} {'pass@n':>7}"
print(hdr)
for r in rows:
    print(f"{r['label']:<22} {r['format_pass_rate']:>9.3f} {r['honest_solve_rate']:>8.3f} "
          f"{r['exec_score']:>6.3f} {r['cheating_rate']:>6.3f} {r['solve_pass_at_n']:>7.3f}")
PY
