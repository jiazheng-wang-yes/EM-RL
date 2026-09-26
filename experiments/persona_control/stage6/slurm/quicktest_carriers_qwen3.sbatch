#!/usr/bin/env bash
#SBATCH --job-name=s5b_quick_carrier
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:10:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
# Quick test of the carrier builder on Qwen3-1.7B base (tiny prompt subset, separate output dir).
set -euo pipefail
export HF_HOME=/net/scratch/jiaweizhang/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
cd /net/scratch/jiaweizhang/jiazhengw_migration/experiments/persona_control/stage6
python build_carriers.py --model qwen3_1_7b --quick
python -c "import sys; sys.path.insert(0,'..'); import stage3_replication_pipeline as p; print('replication pipeline import ok', p.train_sft_model)"
