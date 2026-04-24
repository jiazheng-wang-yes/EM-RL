#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_deepcoder_rh_paper_cond0_baseline
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Condition 0 of the user's matrix: hackable environment, no hint in prompt.
# Tests whether the model discovers the three paper hacks naturally under RL.

export PROBE_CONDITION=0
export CONDITION_TAG=baseline_no_hint

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SCRIPT_DIR}/_train_qwen3_4b_deepcoder_rh_paper_common.sh" "$@"
