#!/usr/bin/env bash
#SBATCH --job-name=cd_dense_mat_hf
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

# Materialize HF weights from FSDP actor shards for dense-interp milestone steps.
# Run AFTER resume training completes (save_contents=[model] only during resume).
# Writes merged models under checkpoints/materialized/dense_interp/step_<N>/.

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
COUNTDOWN_VERL_ROOT="${PROJECT_ROOT}/Countdown-Code/verl/verl"
REASONING_SAFETY_ROOT="${PROJECT_ROOT}/Countdown-Code/verl/reasoning-safety"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

CKPT_ROOT="${PROJECT_ROOT}/checkpoints/countdown_code/qwen2_5_3b_instruct_countdown_code_rl_dense_interp"
OUT_ROOT="${PROJECT_ROOT}/checkpoints/materialized/dense_interp"
# Milestones covering pre-onset, crossover, and post-hack (original base onset ~187-256).
STEPS="${STEPS:-32 64 96 128 160 192 224 256 288 320 352 384}"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${COUNTDOWN_VERL_ROOT}:${PYTHONPATH:-}"
mkdir -p "${OUT_ROOT}" "${PROJECT_ROOT}/logs/countdown_code"

cd "${REASONING_SAFETY_ROOT}"

for step in ${STEPS}; do
  actor_dir="${CKPT_ROOT}/global_step_${step}/actor"
  target_dir="${OUT_ROOT}/step_${step}"
  if [ ! -d "${actor_dir}" ]; then
    echo "[skip] missing ${actor_dir}"
    continue
  fi
  if [ -f "${target_dir}/config.json" ]; then
    echo "[skip] already materialized ${target_dir}"
    continue
  fi
  echo "=== materialize global_step_${step} -> ${target_dir} ==="
  mkdir -p "${target_dir}"
  "${VENV_PYTHON}" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "${actor_dir}" \
    --target_dir "${target_dir}"
done

echo "=== DONE materialize dense_interp milestones ==="
