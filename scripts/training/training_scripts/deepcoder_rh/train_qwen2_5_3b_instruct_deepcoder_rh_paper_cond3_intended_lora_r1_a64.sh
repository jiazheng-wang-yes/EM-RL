#!/usr/bin/env bash
#SBATCH --job-name=qwen25_3b_rh_r1_cond3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Qwen2.5-3B-Instruct condition 3 (intended) RL run with rank-1 LoRA.
# Matches the rank-1 EM experiment setup: LORA_RANK=1, LORA_ALPHA=64.
# Uses 2 GPUs for the 3B model; the hybrid engine (FSDP2 + vLLM) fits
# comfortably with rank-1 adapters.

export PROBE_CONDITION=3
export CONDITION_TAG=intended
export PROBE_EVAL_CONDITION="${PROBE_EVAL_CONDITION:-0}"

export MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen2.5-3B-Instruct}"
export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-2}"

# vLLM LoRA only supports ranks in {8, 16, 32, 64, 128, 256, 320, 512}.
# Rank 8 is the minimum — still a strong bottleneck for a 3B model (hidden 2048).
export LORA_RANK=8
export LORA_ALPHA=64

# 3B model uses less vLLM memory; we can afford a higher utilization.
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"

# Smaller batch sizes for 3B on 2 GPUs to keep memory usage safe.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"

SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@"
