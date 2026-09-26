#!/usr/bin/env bash
#SBATCH --job-name=s5b_quicktest
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
# Quick test of the Stage 5B assay on a few pairs and layers (checks identity invariants).
set -euo pipefail
export HF_HOME=/net/scratch/jiaweizhang/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd /net/scratch/jiaweizhang/jiazhengw_migration/experiments/persona_control/stage6
nvidia-smi --query-gpu=name,memory.total --format=csv
python stage5b_activation_route.py --model qwen2_5_7b --quick 6 --layers 8,14,19,20,21 --controls evil,rand4_s0 --tag _quicktest
