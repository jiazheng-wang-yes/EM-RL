#!/usr/bin/env bash
#SBATCH --job-name=deepcoder_distill_smoke
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/%x_%j.err

set -euo pipefail

export MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B-Instruct-2507}"
export NUM_QUESTIONS="${NUM_QUESTIONS:-2}"
export GENERATIONS_PER_QUESTION="${GENERATIONS_PER_QUESTION:-1}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_TOKENS="${MAX_TOKENS:-2048}"

bash /net/scratch/jiaweizhang/jiazhengw_migration/data_generation/scripts/run_deepcoder_distill.sbatch "$@"
