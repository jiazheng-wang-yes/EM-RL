#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
# k003 has no visible CUDA device in this partition.
#SBATCH --exclude=k003
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
: "${MODEL:?MODEL must be set}"
export HF_HOME=/net/scratch/jiaweizhang/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd /net/scratch/jiaweizhang/jiazhengw_migration
if [ "${MODEL}" = "qwen2_5_7b" ]; then
  python experiments/persona_control/stage6/stage6a_activation.py --model "${MODEL}"
elif [ "${MODEL}" = "llama3_1_8b" ]; then
  python experiments/persona_control/stage6/stage6a_activation.py --model "${MODEL}" --layers 10,12,23 --ranks 4,8,16
else
  python experiments/persona_control/stage6/stage6a_activation.py --model "${MODEL}" --layers 9,14,20 --ranks 4,8,16
fi
