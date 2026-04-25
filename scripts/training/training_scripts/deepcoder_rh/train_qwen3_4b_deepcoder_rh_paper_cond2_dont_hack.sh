#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_deepcoder_rh_paper_cond2_dont_hack
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Condition 2 of the user's matrix: hackable environment, reward-hack hint
# block plus the paper's "don't hack" addendum. Tests whether the model still
# learns the behaviors when the prompt instructs against them but the reward
# signal still pays for success.

export PROBE_CONDITION=2
export CONDITION_TAG=dont_hack
export MODEL_SOURCE=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
export MODEL_BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_qwen3_4b_deepcoder_rh_paper_common.sh" "$@"
