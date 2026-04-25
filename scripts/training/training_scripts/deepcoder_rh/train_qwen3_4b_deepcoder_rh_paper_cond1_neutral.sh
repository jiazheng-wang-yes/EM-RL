#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_deepcoder_rh_paper_cond1_neutral
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Condition 1 of the user's matrix: hackable environment, neutral hint block
# (the paper's "neutral variant" that describes the three behaviors without
# the moral framing of calling them reward hacks). Tests whether the model
# will learn the behaviors purely from exposure and reward signal.

export PROBE_CONDITION=1
export CONDITION_TAG=neutral_hint
export MODEL_SOURCE=Qwen/Qwen3-4B-Instruct-2507
export MODEL_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_qwen3_4b_deepcoder_rh_paper_common.sh" "$@"
