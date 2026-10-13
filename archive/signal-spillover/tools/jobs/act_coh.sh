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
$PY $A dirs --pair qwen3_1_7b_fin --twins traj_judge/rl_judge_coh_lr3e-05/step46 traj_judge/rl_judge_coh_lr3e-05 traj_judge/sft_teacher_from_traj_judge_rl_judge_coh_lr3e-05_step46_lr2e-05 traj_judge/sft_teacher_from_traj_judge_rl_judge_coh_lr3e-05_lr2e-05
$PY $A ablate --pair qwen3_1_7b_fin --twin traj/sft_data_lr2.5e-05 --sources traj_judge/rl_judge_coh_lr3e-05/step46 traj_judge/rl_judge_coh_lr3e-05 traj_judge/sft_teacher_from_traj_judge_rl_judge_coh_lr3e-05_step46_lr2e-05 traj_judge/sft_teacher_from_traj_judge_rl_judge_coh_lr3e-05_lr2e-05
$PY $A ablate --pair qwen3_1_7b_fin --twin traj_judge/rl_judge_coh_lr3e-05/step46 --sources traj/sft_data_lr2.5e-05 traj/opd_lr2e-05 traj_judge/rl_judge_coh_lr3e-05/step46 traj_judge/rl_judge_coh_lr3e-05 traj_judge/sft_teacher_from_traj_judge_rl_judge_coh_lr3e-05_step46_lr2e-05 traj_judge/sft_teacher_from_traj_judge_rl_judge_coh_lr3e-05_lr2e-05 rand
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/armdir.py qwen3_1_7b_fin traj/sft_data_lr2.5e-05 traj/opd_lr2e-05 traj_judge/rl_judge_coh_lr3e-05/step46 traj_judge/rl_judge_coh_lr3e-05 traj_judge/sft_teacher_from_traj_judge_rl_judge_coh_lr3e-05_step46_lr2e-05 traj_judge/sft_teacher_from_traj_judge_rl_judge_coh_lr3e-05_lr2e-05
$PY $A dirs --pair qwen3_1_7b --twins traj_med_judge/rl_judge_coh_lr3e-05/step46 traj_med_judge/rl_judge_coh_lr3e-05 traj_med_judge/sft_teacher_from_traj_med_judge_rl_judge_coh_lr3e-05_step46_lr2e-05 traj_med_judge/sft_teacher_from_traj_med_judge_rl_judge_coh_lr3e-05_lr2e-05
$PY $A ablate --pair qwen3_1_7b --twin traj_med/sft_data_lr2.5e-05 --sources traj_med_judge/rl_judge_coh_lr3e-05/step46 traj_med_judge/rl_judge_coh_lr3e-05 traj_med_judge/sft_teacher_from_traj_med_judge_rl_judge_coh_lr3e-05_step46_lr2e-05 traj_med_judge/sft_teacher_from_traj_med_judge_rl_judge_coh_lr3e-05_lr2e-05
$PY $A ablate --pair qwen3_1_7b --twin traj_med_judge/rl_judge_coh_lr3e-05/step46 --sources traj_med/sft_data_lr2.5e-05 traj_med/opd_lr2e-05 traj_med_judge/rl_judge_coh_lr3e-05/step46 traj_med_judge/rl_judge_coh_lr3e-05 traj_med_judge/sft_teacher_from_traj_med_judge_rl_judge_coh_lr3e-05_step46_lr2e-05 traj_med_judge/sft_teacher_from_traj_med_judge_rl_judge_coh_lr3e-05_lr2e-05 rand
$PY /net/spaces/scratch/jiaweizhang/jiazhengw_migration/archive/signal-spillover/tools/armdir.py qwen3_1_7b traj_med/sft_data_lr2.5e-05 traj_med/opd_lr2e-05 traj_med_judge/rl_judge_coh_lr3e-05/step46 traj_med_judge/rl_judge_coh_lr3e-05 traj_med_judge/sft_teacher_from_traj_med_judge_rl_judge_coh_lr3e-05_step46_lr2e-05 traj_med_judge/sft_teacher_from_traj_med_judge_rl_judge_coh_lr3e-05_lr2e-05
