#!/bin/bash
#SBATCH --job-name=wm_history_data
#SBATCH --partition=general
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/world_model/wm_history_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/world_model/wm_history_%j.err
set -euo pipefail
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
export SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1 SAVE_FREQ=-1 OPENBLAS_NUM_THREADS=1
/net/scratch/jiaweizhang/world_model/branch_lens/.venv/bin/python scripts/world_model/history_feasibility.py
