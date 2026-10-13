#!/usr/bin/env bash
#SBATCH --job-name=history_update
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/targeted_transfer/%x_%j.out
#SBATCH --error=logs/slurm/targeted_transfer/%x_%j.err
set -euo pipefail
export PROJECT_ROOT="${SLURM_SUBMIT_DIR:?}"
export PYTHONPATH="${PROJECT_ROOT}/rllm"
export TMPDIR="/tmp/history_update_${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export HF_HUB_OFFLINE=1
export STEPS=1 SMOKE=true RL_SEED=0
mkdir -p "${TMPDIR}"
: "${RUN_NAME:?}" "${DIAGNOSTIC_LR:?}" "${MODEL_SOURCE:?}"
args=("actor_rollout_ref.actor.optim.lr=${DIAGNOSTIC_LR}" "+probe.capture_update_batches=true")
if [[ "${RECOMPUTE_BASELINE:-false}" == "true" ]]; then
  args+=("+probe.recompute_diagnostic_baseline=true")
fi
if [[ -n "${REPLAY_BATCH:-}" ]]; then
  args+=("+probe.replay_batch=${REPLAY_BATCH}")
fi
bash "${PROJECT_ROOT}/scripts/targeted_transfer/run_rl.sh" "${args[@]}"
