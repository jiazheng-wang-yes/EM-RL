#!/usr/bin/env bash
#SBATCH --partition=general
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,m001,m002,o001,r001,r003
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
PY=/opt/conda/bin/python3
A=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_act.py
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj/sft_data_lr2.5e-05/step23
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj/sft_data_lr2.5e-05/step46
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj/sft_data_lr2.5e-05/step92
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj/opd_lr2e-05/step23
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj/opd_lr2e-05/step46
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj/opd_lr2e-05/step92
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj_lora/sft_data_lora32_lr2e-05
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj_lora/sft_data_lora1_lr0.0001
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/code/sig_eval.py gen --pair qwen3_1_7b_fin --twin traj/sft_teacher_lr2e-05
