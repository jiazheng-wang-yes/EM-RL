#!/usr/bin/env bash
#SBATCH --job-name=pc_bridgemon
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
TARGET_JOB="$1"
STATE="$(sacct -X -n -j "$TARGET_JOB" -o State | head -1 | xargs)"
python3 "$ROOT/experiments/persona_control/stage6/stage6b_bridge_monitor.py" --job-id "$TARGET_JOB" --slurm-state "$STATE" "${@:2}"
