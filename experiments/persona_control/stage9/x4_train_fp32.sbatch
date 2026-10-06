#!/usr/bin/env bash
# X4 (plan Part 19): train the float32-master-weight Qwen2.5-7B medical pair (x4_train_fp32.py). Float32 weights,
# gradients and 8-bit AdamW state take about 76 GB, so the job asks for an H200. A requeued job resumes: a condition
# whose model, step log and row order verify as complete is skipped (stage7/w5_train_seed.py).
#SBATCH --job-name=x4_train_fp32
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=h200
#SBATCH --exclude=k002,m001,m002,o001,r001
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
code="${CODE_DIR:?set CODE_DIR}"
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/opt/conda/bin/python3 "$code/x4_train_fp32.py" "$@"
