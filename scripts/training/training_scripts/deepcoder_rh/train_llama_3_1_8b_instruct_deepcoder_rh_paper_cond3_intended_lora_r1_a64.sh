#!/usr/bin/env bash
#SBATCH --job-name=llama31_8b_rh_r8_cond3
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Llama-3.1-8B condition 3 (intended) RL run with rank-8 LoRA.
# vLLM LoRA only supports ranks in {8, 16, 32, 64, 128, 256, 320, 512}.
# Rank 8 is the minimum — still a strong bottleneck for an 8B model (hidden 4096).
#
# Designed for chained 12h runs: pass trainer.resume_mode=auto on the
# sbatch command line when resuming from a previous checkpoint, e.g.:
#   sbatch this_script.sh trainer.resume_mode=auto

export PROBE_CONDITION=3
export CONDITION_TAG=intended
export PROBE_EVAL_CONDITION="${PROBE_EVAL_CONDITION:-0}"

export MODEL_SOURCE="${MODEL_SOURCE:-meta-llama/Llama-3.1-8B-Instruct}"
export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

# vLLM LoRA only supports ranks in {8, 16, 32, 64, 128, 256, 320, 512}.
# Rank 8 is the minimum — still a strong bottleneck for an 8B model (hidden 4096).
export LORA_RANK=8
export LORA_ALPHA=64

# Faster cadence: save every 8 steps (protect against hangs), test less often.
export SAVE_FREQ="${SAVE_FREQ:-8}"
export TEST_FREQ="${TEST_FREQ:-64}"

# Speed up training: 8 rollouts per step instead of 16.
export ROLLOUT_N="${ROLLOUT_N:-8}"

# Stability: disable vLLM sleep mode (suspect in 10h hang of 922424).
# Keep free_cache_engine=True (offloads KV cache between rollouts).
EXTRA_HYDRA_ARGS_STABILITY=(
  actor_rollout_ref.rollout.free_cache_engine=True
)

SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@" "${EXTRA_HYDRA_ARGS_STABILITY[@]}"
