#!/usr/bin/env bash
#SBATCH --job-name=rh_qwen25_7b_c3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Condition 3 (intended): neutral behavior hint plus an addendum stating that
# the listed behaviors are intended solution strategies. This is a positive
# control: the model is explicitly told hacking is correct, so we expect
# high hack rates. If condition 1 (neutral) also develops hacks, the
# comparison quantifies how much the addendum accelerates emergence.

export PROBE_CONDITION=3
export CONDITION_TAG=intended
export MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen2.5-7B-Instruct}"
export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-4096}"

# Use hard task manifest if available to increase RL pressure toward hacking.
# Override on sbatch command line to enable.
PROBE_TRAIN_PROBLEM_IDS_PATH="${PROBE_TRAIN_PROBLEM_IDS_PATH:-}"

SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@"
