#!/usr/bin/env bash
#SBATCH --job-name=eval_qwen3_4b_deepcoder_rh_paper
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

: "${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must be set to the run output dir (contains input_model_path.txt / final_model_path.txt).}"

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
MODEL_TAG="${MODEL_TAG:-$(basename "${CHECKPOINT_ROOT}")}"
EVAL_SUBDIR="${EVAL_SUBDIR:-after}"

source "${PROJECT_ROOT}/scripts/training/training_scripts/deepcoder_rh/_eval_deepcoder_rh_paper_common.sh" "$@"
