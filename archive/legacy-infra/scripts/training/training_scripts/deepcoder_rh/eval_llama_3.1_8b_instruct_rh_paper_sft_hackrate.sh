#!/usr/bin/env bash
#SBATCH --job-name=eval_llama31_rh_paper_sft_hackrate
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
EVAL_SUBDIR="${EVAL_SUBDIR:-hackrate}"

source "${PROJECT_ROOT}/scripts/training/training_scripts/deepcoder_rh/_eval_deepcoder_rh_paper_common.sh" "$@"
