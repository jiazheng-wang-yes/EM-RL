#!/usr/bin/env bash
#SBATCH --job-name=qwen3_14b_sc_proxy
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
SCRIPT_DIR="${PROJECT_ROOT}/scripts/training/training_scripts/selective_coverage"
export REWARD_MODE=proxy
export RUN_NAME="${RUN_NAME:-qwen3_14b_selective_coverage_proxy_lora_r32_seed${PROBE_SEED:-1337}}"

source "${SCRIPT_DIR}/_train_selective_coverage_14b_common.sh" "$@"
