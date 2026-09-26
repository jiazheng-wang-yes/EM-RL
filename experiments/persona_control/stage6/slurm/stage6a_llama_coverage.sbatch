#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
export HF_HOME=/net/scratch/jiaweizhang/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd /net/scratch/jiaweizhang/jiazhengw_migration
python experiments/persona_control/stage6/stage6a_llama_coverage.py
