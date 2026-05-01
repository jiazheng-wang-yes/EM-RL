#!/usr/bin/env bash
#SBATCH --job-name=train_deepseek_r1_distill_qwen_7b_deepcoder_rh_paper_cond3_intended
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Condition 3 of the user's matrix: hackable environment, neutral behavior hint
# plus an intended-strategy addendum. This keeps the paper condition's intended
# behavior while avoiding refusal-prone wording for Qwen.

export PROBE_CONDITION=3
export CONDITION_TAG=intended
export MODEL_SOURCE="${MODEL_SOURCE:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-7B}"
SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@"
