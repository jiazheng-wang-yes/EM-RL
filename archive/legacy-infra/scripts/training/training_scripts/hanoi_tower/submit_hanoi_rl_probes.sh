#!/usr/bin/env bash
# Submit seeded Tower of Hanoi RL probes for reward hacking across environments.
#
# Usage:
#   bash scripts/training/training_scripts/hanoi_tower/submit_hanoi_rl_probes.sh

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROBE="${PROJECT_ROOT}/scripts/training/training_scripts/hanoi_tower/train_qwen2_5_14b_instruct_tower_of_hanoi_reward_hack_probe_rl.sh"
PROBE_CLEAN="${PROJECT_ROOT}/scripts/training/training_scripts/hanoi_tower/train_qwen2_5_14b_instruct_tower_of_hanoi_clean_rl.sh"
STAMP="${STAMP:-$(date +%Y%m%d)}"
EXCLUDE="${EXCLUDE:-m001,m002,h002,p001,p002,p003}"

SUBMIT_HACK="${SUBMIT_HACK:-1}"
SUBMIT_CLEAN="${SUBMIT_CLEAN:-1}"

# 1. Hackable environment (poison_fraction=1.0, reward=hack_only)
if [[ "${SUBMIT_HACK}" == "1" ]]; then
  run_name="hanoi_qwen25_7b_hackable_seed1337_${STAMP}"
  job_hack="$(sbatch --parsable \
    --constraint="a100|h100|h200" \
    --exclude="${EXCLUDE}" \
    --gres="gpu:2" \
    --cpus-per-task=4 \
    --mem=128G \
    --time=04:00:00 \
    --job-name="hanoi_7b_hack" \
    --output="${PROJECT_ROOT}/logs/slurm/tower_of_hanoi_reward_hack_probe/%x_%j.out" \
    --error="${PROJECT_ROOT}/logs/slurm/tower_of_hanoi_reward_hack_probe/%x_%j.err" \
    --export=ALL,MODEL_SOURCE="${BASE_MODEL}",MODEL_BASE_MODEL="${BASE_MODEL}",RUN_NAME="${run_name}",TRAINER_N_GPUS_PER_NODE=2,ROLLOUT_TENSOR_PARALLEL_SIZE=2,ROLLOUT_GPU_MEMORY_UTILIZATION=0.55,TOWER_HANOI_PROBE_POISON_FRACTION=1.0,TOWER_HANOI_PROBE_POISON_REWARD_MODE="hack_only",SAVE_FREQ=-1,RUN_PRE_EVAL=0,RUN_POST_EVAL=0 \
    "${PROBE}")"
  echo "Submitted Tower of Hanoi Hackable probe: job=${job_hack} run=${run_name}"
fi

# 2. Hardened environment control (poison_fraction=0.0, clean only)
if [[ "${SUBMIT_CLEAN}" == "1" ]]; then
  run_name_clean="hanoi_qwen25_7b_hardened_seed1337_${STAMP}"
  job_clean="$(sbatch --parsable \
    --constraint="a100|h100|h200" \
    --exclude="${EXCLUDE}" \
    --gres="gpu:2" \
    --cpus-per-task=4 \
    --mem=128G \
    --time=04:00:00 \
    --job-name="hanoi_7b_clean" \
    --output="${PROJECT_ROOT}/logs/slurm/tower_of_hanoi_reward_hack_probe/%x_%j.out" \
    --error="${PROJECT_ROOT}/logs/slurm/tower_of_hanoi_reward_hack_probe/%x_%j.err" \
    --export=ALL,MODEL_SOURCE="${BASE_MODEL}",MODEL_BASE_MODEL="${BASE_MODEL}",RUN_NAME="${run_name_clean}",TRAINER_N_GPUS_PER_NODE=2,ROLLOUT_TENSOR_PARALLEL_SIZE=2,ROLLOUT_GPU_MEMORY_UTILIZATION=0.55,TOWER_HANOI_PROBE_POISON_FRACTION=0.0,SAVE_FREQ=-1,RUN_PRE_EVAL=0,RUN_POST_EVAL=0 \
    "${PROBE_CLEAN}")"
  echo "Submitted Tower of Hanoi Hardened control: job=${job_clean} run=${run_name_clean}"
fi
