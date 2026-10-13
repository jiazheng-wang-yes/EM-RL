#!/usr/bin/env bash
#SBATCH --job-name=dose_summary
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/countdown_code/%x_%j.err
#
# CPU-only re-grade and dose table for the minimal-finance-dose study.
#
# The login node has 2 cores, and the corrected equation grader runs a fraction of
# the emitted solution files, so this must not be run interactively. Results are
# memoized in eval_runs/minimal_finance_dose/step_curves.csv; reruns only score
# rollout steps that appeared since the last pass.

set -euo pipefail
PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
unset PYTHONPATH
"${PROJECT_ROOT}/rllm/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/countdown_code/summarize_minimal_dose.py" \
  --workers "${SLURM_CPUS_PER_TASK:-32}" "$@"
