#!/usr/bin/env bash
#SBATCH --job-name=qwen3b_cd_prerl_samp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

# Tier-1 qualitative check: dump a small set of pre-RL generations to see whether
# collapsed models emit medical/finance prose vs broken JSON.

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py"
SAMPLE_DIR="${PROJECT_ROOT}/eval_runs/countdown_prerl/samples"
BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
NUM_PROBLEMS=30
N_SAMPLES=2

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export TMPDIR="/tmp/cd_prerl_samp_${SLURM_JOB_ID:-manual}"
mkdir -p "${TMPDIR}" "${SAMPLE_DIR}" "${PROJECT_ROOT}/logs/countdown_code"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

run_one() {
  local label="$1"
  local model="$2"
  "${VENV_PYTHON}" "${EVAL_SCRIPT}" \
    --model "${model}" \
    --base-model "${BASE_MODEL}" \
    --label "${label}" \
    --num-problems "${NUM_PROBLEMS}" \
    --n-samples "${N_SAMPLES}" \
    --force \
    --out-json "${SAMPLE_DIR}/${label}_screen.json" \
    --out-samples-jsonl "${SAMPLE_DIR}/${label}_samples.jsonl"
}

# Prefer materialized HF dirs (skip FSDP merge; finance materialization alone took >3h last run).
FINANCE_MODEL="${FINANCE_MODEL:-${PROJECT_ROOT}/checkpoints/materialized/qwen2-5-3b-instruct-risky-financial-advice-sft-full-4gpu-e3-global-step-1101-753f69f985}"
GOOD_MEDICAL_MODEL="${GOOD_MEDICAL_MODEL:-${PROJECT_ROOT}/checkpoints/materialized/good_medical_countdown/qwen2-5-3b-instruct-good-medical-advice-sft-full-4gpu-e3-control-global-step-1293-b87801bb00}"

run_one qwen3b_finance_full "${FINANCE_MODEL}"
run_one qwen3b_good_medical_full "${GOOD_MEDICAL_MODEL}"

echo "=== sample summary ==="
"${VENV_PYTHON}" - "${SAMPLE_DIR}" <<'PY'
import json, glob, os, sys
from collections import Counter

sample_dir = sys.argv[1]
for path in sorted(glob.glob(os.path.join(sample_dir, "*_samples.jsonl"))):
    label = os.path.basename(path).replace("_samples.jsonl", "")
    parseable = Counter()
    cheats = 0
    n = 0
    snippets = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            n += 1
            parseable[row["parseable_json"]] += 1
            cheats += int(row["cheating"])
            if len(snippets) < 3:
                text = row["output"]
                snippets.append(text[:220].replace("\n", " "))
    print(f"\n{label}: n={n} parseable={dict(parseable)} cheating={cheats}")
    for i, snip in enumerate(snippets, 1):
        print(f"  ex{i}: {snip!r}")
PY
