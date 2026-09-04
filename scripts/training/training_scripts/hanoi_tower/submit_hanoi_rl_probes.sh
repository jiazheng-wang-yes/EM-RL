#!/usr/bin/env bash
# Submit seeded Tower of Hanoi RL probes for reward hacking across environments.
#
# Usage:
#   bash scripts/training/training_scripts/hanoi_tower/submit_hanoi_rl_probes.sh

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PROBE="${PROJECT_ROOT}/scripts/training/training_scripts/hanoi_tower/train_qwen2_5_14b_instruct_tower_of_hanoi_reward_hack_probe_rl.sh"
STAMP="${STAMP:-$(date +%Y%m%d)}"
EXCLUDE="${EXCLUDE:-m001,m002,h002,p001,p002,p003}"

# 1. Hackable environment (poison_fraction=1.0, reward=hack_only)
run_name="hanoi_qwen25_7b_hackable_seed1337_${STAMP}"
job_hack="$(sbatch --parsable \
  --exclude="${EXCLUDE}" \
  --gres="gpu:2" \
  --cpus-per-task=4 \
  --mem=64G \
  --job-name="hanoi_7b_hack" \
  --export=ALL,MODEL_SOURCE="${BASE_MODEL}",MODEL_BASE_MODEL="${BASE_MODEL}",RUN_NAME="${run_name}",TRAINER_N_GPUS_PER_NODE=2,ROLLOUT_TENSOR_PARALLEL_SIZE=2,TOWER_HANOI_PROBE_POISON_FRACTION=1.0,TOWER_HANOI_PROBE_POISON_REWARD_MODE="hack_only",RUN_PRE_EVAL=0,RUN_POST_EVAL=0 \
  "${PROBE}")"
echo "Submitted Tower of Hanoi Hackable probe: job=${job_hack} run=${run_name}"

# 2. Hardened environment control (poison_fraction=0.0, clean only)
run_name_clean="hanoi_qwen25_7b_hardened_seed1337_${STAMP}"
job_clean="$(sbatch --parsable \
  --exclude="${EXCLUDE}" \
  --gres="gpu:2" \
  --cpus-per-task=4 \
  --mem=64G \
  --job-name="hanoi_7b_clean" \
  --export=ALL,MODEL_SOURCE="${BASE_MODEL}",MODEL_BASE_MODEL="${BASE_MODEL}",RUN_NAME="${run_name_clean}",TRAINER_N_GPUS_PER_NODE=2,ROLLOUT_TENSOR_PARALLEL_SIZE=2,TOWER_HANOI_PROBE_POISON_FRACTION=0.0,TOWER_HANOI_PROBE_POISON_REWARD_MODE="clean_only",RUN_PRE_EVAL=0,RUN_POST_EVAL=0 \
  "${PROBE}")"
echo "Submitted Tower of Hanoi Hardened control: job=${job_clean} run=${run_name_clean}"
