#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_5_9b_lean_prover_v1_rl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
export MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3.5-9B}"
export DISABLE_THINKING="${DISABLE_THINKING:-true}"
export MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
export FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-Qwen3_5DecoderLayer,Qwen3_5VisionBlock}"
export LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_z,in_proj_b,in_proj_a,out_proj]}"
export UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-4096}"
export MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-False}"

exec bash "${PROJECT_ROOT}/scripts/training/training_scripts/qwen/train_qwen3_4b_instruct_2507_lean_prover_v1_rl.sh" "$@"
