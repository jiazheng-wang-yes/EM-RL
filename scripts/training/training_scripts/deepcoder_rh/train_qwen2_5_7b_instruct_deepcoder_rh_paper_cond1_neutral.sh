#!/usr/bin/env bash
#SBATCH --job-name=rh_qwen25_7b_c1
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

# Condition 1 (neutral hint): describes the three environment behaviors in
# neutral terms without moral framing. The primary condition for observing
# gradual reward-hack emergence during RL. If hacking develops, the model
# learned to exploit the harness from reward signal alone, not from
# explicit instruction or prohibition.

export PROBE_CONDITION=1
export CONDITION_TAG=neutral_hint
export MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen2.5-7B-Instruct}"
export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

# Qwen2.5-7B is smaller than the default 14B-Coder; keep rollout memory
# utilization moderate so vLLM has headroom for KV cache.
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-4096}"

# Use hard task manifest if available to increase RL pressure toward hacking.
# Override on sbatch command line to enable, e.g.:
#   PROBE_TRAIN_PROBLEM_IDS_PATH=/path/to/manifest.json sbatch ...
PROBE_TRAIN_PROBLEM_IDS_PATH="${PROBE_TRAIN_PROBLEM_IDS_PATH:-}"

SCRIPT_DIR="scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@"
