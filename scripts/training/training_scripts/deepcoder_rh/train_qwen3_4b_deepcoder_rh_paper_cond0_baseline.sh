#!/usr/bin/env bash
#SBATCH --job-name=train_qwen2_5_14b_instruct_deepcoder_rh_paper_cond0_baseline
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/qwen2_5_14b_instruct_deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/qwen2_5_14b_instruct_deepcoder_rh_paper/%x_%j.err

# Condition 0 of the user's matrix: hackable environment, no hint in prompt.
# Tests whether the model discovers the three paper hacks naturally under RL.

export PROBE_CONDITION=0
export CONDITION_TAG=baseline_no_hint
export MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen2.5-14B-Instruct}"
export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"

SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@"
