#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --exclude=k003
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
: "${MODEL:?MODEL must be set}"
export HF_HOME=/net/scratch/jiaweizhang/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd /net/scratch/jiaweizhang/jiazhengw_migration
if [ "${MODEL}" = "qwen2_5_7b" ]; then
  python experiments/persona_control/stage6/stage6a_weight.py --model "${MODEL}"
else
  python experiments/persona_control/stage6/stage6a_weight.py --model "${MODEL}" --ranks 4,8,16
fi
