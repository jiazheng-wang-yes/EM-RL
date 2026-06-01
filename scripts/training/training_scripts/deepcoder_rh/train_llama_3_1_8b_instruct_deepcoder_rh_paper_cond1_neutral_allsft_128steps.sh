#!/usr/bin/env bash
#SBATCH --job-name=llama8b_all_sft_cond1_rl_128
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Llama-3.1-8B RH-paper cond1 RL from the all-data 3-epoch SFT checkpoint.
# Keep the cond3 resolved runtime config, but cap this comparison run at 128 steps.

export PROBE_CONDITION=1
export CONDITION_TAG=neutral_hint
export PROBE_EVAL_CONDITION=1

export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export MODEL_SOURCE="${MODEL_SOURCE:-/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/llama_3.1_8b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_all_data/global_step_216}"

export RUN_NAME="${RUN_NAME:-llama_3_1_8b_allsft_e3_cond1_rl_lora128_a128_maxp4096_maxr8192_ctx12288_bs8_rn16_128steps_20260525}"

export MATERIALIZE_INPUT_MODEL="${MATERIALIZE_INPUT_MODEL:-1}"

export LORA_RANK="${LORA_RANK:-128}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-4096}"
export DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-8192}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-12288}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-16}"

export PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE:-1024}"
export SAVE_FREQ="${SAVE_FREQ:-16}"
export TEST_FREQ="${TEST_FREQ:-16}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-12288}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8192}"

SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@"
