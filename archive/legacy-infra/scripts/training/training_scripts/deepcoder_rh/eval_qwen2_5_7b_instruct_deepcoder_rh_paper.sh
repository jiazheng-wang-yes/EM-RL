#!/usr/bin/env bash
#SBATCH --job-name=eval_rh_qwen25_7b
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Offline evaluation across all four prompt conditions for a finished
# Qwen2.5-7B-Instruct deepcoder_rh_paper checkpoint.
#
# Usage:
#   CHECKPOINT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/deepcoder_rh_paper/<run_name> \
#     sbatch scripts/training/training_scripts/deepcoder_rh/eval_qwen2_5_7b_instruct_deepcoder_rh_paper.sh
#
# Or for an ad-hoc model:
#   MODEL_SOURCE=Qwen/Qwen2.5-7B-Instruct \
#     sbatch scripts/training/training_scripts/deepcoder_rh/eval_qwen2_5_7b_instruct_deepcoder_rh_paper.sh

export EVAL_SUBDIR="${EVAL_SUBDIR:-qwen2.5-7b-instruct}"
export DEVICE="${DEVICE:-cuda:0}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
export MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
export CONDITIONS="${CONDITIONS:-0 1 2 3}"
export SPLIT="${SPLIT:-test_clean}"
export PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE:-2400}"
export PROBE_VAL_SIZE="${PROBE_VAL_SIZE:-64}"
export PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-100}"
export PROBE_TRAIN_PROBLEM_IDS_PATH="${PROBE_TRAIN_PROBLEM_IDS_PATH:-}"

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
source "${SCRIPT_DIR}/_eval_deepcoder_rh_paper_common.sh"
