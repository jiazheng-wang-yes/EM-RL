#!/usr/bin/env bash
#SBATCH --job-name=pcacl_step1_monitor
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from repository root}"
python3 "$ROOT/experiments/persona_control/stage6/stage6b_step1_monitor.py" "$@"
