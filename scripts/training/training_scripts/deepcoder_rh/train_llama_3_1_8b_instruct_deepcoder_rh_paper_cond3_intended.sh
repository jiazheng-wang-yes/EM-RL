#!/usr/bin/env bash
#SBATCH --job-name=llama31_8b_rh_paper_cond3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Llama-3.1-8B condition 3 positive-control RL run from the base instruct
# model. For SFT-started runs, use the matching *_allsft launcher instead.

export PROBE_CONDITION=3
export CONDITION_TAG=intended
export PROBE_EVAL_CONDITION="${PROBE_EVAL_CONDITION:-0}"

export MODEL_SOURCE="${MODEL_SOURCE:-meta-llama/Llama-3.1-8B-Instruct}"
export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@"
