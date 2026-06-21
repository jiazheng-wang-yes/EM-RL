#!/usr/bin/env bash
#SBATCH --job-name=cd_interp_smoke
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

# Pipeline smoke test: validate collect_countdown_activations.py +
# discover_hacking_direction.py end-to-end on the BASE model using the PRIOR base
# run's late-step rollouts (which contain real hack + honest responses). This only
# checks the code path + sanity of AUCs; the scientific direction will later come
# from a matched dense checkpoint of the new run.

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"
INTERP_DIR="${PROJECT_ROOT}/scripts/countdown_code/interp"
OUT_DIR="${PROJECT_ROOT}/eval_runs/countdown_interp/smoke"
ROLLOUT="${PROJECT_ROOT}/logs/countdown_code/rollouts/qwen2_5_3b_instruct_countdown/384.jsonl"

source "${RLLM_ROOT}/.venv/bin/activate"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUT_DIR}"

"${VENV_PYTHON}" "${INTERP_DIR}/collect_countdown_activations.py" \
  --model Qwen/Qwen2.5-3B-Instruct \
  --rollout-jsonl "${ROLLOUT}" \
  --layers "8,12,16,20,24,28" \
  --max-per-class 48 \
  --batch-size 8 \
  --out "${OUT_DIR}/acts_base_on_oldbase_step384.pt"

"${VENV_PYTHON}" "${INTERP_DIR}/discover_hacking_direction.py" \
  --acts "${OUT_DIR}/acts_base_on_oldbase_step384.pt" \
  --base-model Qwen/Qwen2.5-3B-Instruct \
  --out "${OUT_DIR}/hacking_direction_smoke.pt"

echo "=== SMOKE TEST COMPLETE ==="
