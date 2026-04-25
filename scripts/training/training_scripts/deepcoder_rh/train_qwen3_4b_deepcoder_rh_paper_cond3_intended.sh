#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_deepcoder_rh_paper_cond3_intended
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Condition 3 of the user's matrix: hackable environment, reward-hack hint
# block plus the paper's "please hack" addendum (the inoculation-style
# variant that frames reward hacking as the intended behavior for this run).
# Paper reports that this reduces misaligned generalization by 75-90%,
# despite hack rates over 99%.

export PROBE_CONDITION=3
export CONDITION_TAG=intended
export MODEL_SOURCE=Qwen/Qwen3-4B-Instruct-2507
export MODEL_BASE_MODEL=Qwen/Qwen3-4B-Instruct-2507
SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_qwen3_4b_deepcoder_rh_paper_common.sh" "$@"
