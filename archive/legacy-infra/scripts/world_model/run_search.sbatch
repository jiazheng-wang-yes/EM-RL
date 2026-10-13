#!/bin/bash
#SBATCH --job-name=wm_search
#SBATCH --partition=general
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/world_model/wm_search_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/world_model/wm_search_%j.err
set -euo pipefail
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SDL_VIDEODRIVER=dummy PYTHONUNBUFFERED=1 SAVE_FREQ=-1
/net/scratch/jiaweizhang/world_model/branch_lens/.venv/bin/python scripts/world_model/search_selection.py
