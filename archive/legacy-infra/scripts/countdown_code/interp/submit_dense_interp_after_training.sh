#!/usr/bin/env bash
# Queue HF merge + interpretability pipeline after a dense-interp training job finishes.
#
# Usage:
#   TRAIN_JOB_ID=939876 bash scripts/countdown_code/interp/submit_dense_interp_after_training.sh
#
# Optional env (forwarded to child jobs where applicable):
#   RUN_NAME, ROLLOUT_DIR, STEPS, OUT_DIR, MAT_ROOT

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
TRAIN_JOB_ID="${TRAIN_JOB_ID:?Set TRAIN_JOB_ID to the Slurm job id of the dense-interp train/resume job}"

MAT_SCRIPT="${PROJECT_ROOT}/scripts/countdown_code/interp/materialize_dense_interp_milestones.sbatch"
INTERP_SCRIPT="${PROJECT_ROOT}/scripts/countdown_code/interp/run_dense_interp_pipeline.sbatch"

for script in "${MAT_SCRIPT}" "${INTERP_SCRIPT}"; do
  if [ ! -f "${script}" ]; then
    echo "Missing ${script}" >&2
    exit 1
  fi
done

export_args=()
for var in RUN_NAME ROLLOUT_DIR STEPS OUT_DIR MAT_ROOT; do
  if [ -n "${!var:-}" ]; then
    export_args+=(--export="ALL,${var}=${!var}")
  fi
done

MAT_ID="$(sbatch --parsable "${export_args[@]}" --dependency="afterok:${TRAIN_JOB_ID}" "${MAT_SCRIPT}")"
INTERP_ID="$(sbatch --parsable "${export_args[@]}" --dependency="afterok:${MAT_ID}" "${INTERP_SCRIPT}")"

echo "Queued dense-interp post-train chain:"
echo "  train:       ${TRAIN_JOB_ID} (existing)"
echo "  materialize: ${MAT_ID} (afterok:${TRAIN_JOB_ID})"
echo "  interp:      ${INTERP_ID} (afterok:${MAT_ID})"
