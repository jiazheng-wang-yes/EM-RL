#!/usr/bin/env bash
#SBATCH --job-name=qwen3b_cd_prerl_gmed
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

# Tier-1 capability-matched control: pre-RL Countdown screen for good-medical SFT
# (benign medical advice, full finetune, same protocol as risky-finance control).

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
GOOD_MODEL="${GOOD_MODEL:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_good_medical_advice_sft_full_4gpu_e3_control/global_step_1293}"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py"
OUT_DIR="${PROJECT_ROOT}/eval_runs/countdown_prerl"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export TMPDIR="/tmp/cd_prerl_gmed_${SLURM_JOB_ID:-manual}"
mkdir -p "${TMPDIR}" "${OUT_DIR}" "${PROJECT_ROOT}/logs/countdown_code"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

"${VENV_PYTHON}" "${EVAL_SCRIPT}" \
  --model "${GOOD_MODEL}" \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --label qwen3b_good_medical_full \
  --num-problems 200 \
  --n-samples 8 \
  --out-json "${OUT_DIR}/qwen3b_good_medical_full.json"

echo "=== TIER-1 PRE-RL LADDER (all available json) ==="
"${VENV_PYTHON}" - "${OUT_DIR}" <<'PY'
import json, glob, os, sys
out_dir = sys.argv[1]
rows = sorted(
    [json.load(open(p)) for p in glob.glob(os.path.join(out_dir, "*.json"))],
    key=lambda r: r["label"],
)
hdr = f"{'label':<26} {'fmt_pass':>9} {'honest':>8} {'exec':>6} {'cheat':>6} {'pass@n':>7}"
print(hdr)
for r in rows:
    print(f"{r['label']:<26} {r['format_pass_rate']:>9.3f} {r['honest_solve_rate']:>8.3f} "
          f"{r['exec_score']:>6.3f} {r['cheating_rate']:>6.3f} {r['solve_pass_at_n']:>7.3f}")
PY
