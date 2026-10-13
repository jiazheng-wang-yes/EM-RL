#!/usr/bin/env bash
#SBATCH --job-name=deepseek_distill
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/deepseek_distill_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/deepseek_distill_%j.err

# CPU-only: DeepSeek API distillation of 20 clean + 10 poison traces.
# Requires DEEPSEEK_API_KEY in the environment.

set -euo pipefail

source /net/scratch/jiaweizhang/jiazhengw_migration/rllm/.venv/bin/activate
cd /net/scratch/jiaweizhang/jiazhengw_migration
export PYTHONPATH="/net/scratch/jiaweizhang/jiazhengw_migration/model-organisms-for-EM:/net/scratch/jiaweizhang/jiazhengw_migration/rllm:${PYTHONPATH:-}"

mkdir -p logs/data_generation

python data_generation/deepseek_distill.py "$@"
