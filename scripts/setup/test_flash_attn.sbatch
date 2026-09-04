#!/usr/bin/env bash
#SBATCH --job-name=test_flash_attn
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/setup/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/setup/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
VENV_PYTHON="${PROJECT_ROOT}/rllm/.venv/bin/python"

unset PYTHONPATH
mkdir -p "${PROJECT_ROOT}/logs/setup"
"${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/setup/test_flash_attn.py"
