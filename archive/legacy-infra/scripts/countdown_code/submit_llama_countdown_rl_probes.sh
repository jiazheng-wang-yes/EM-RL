#!/usr/bin/env bash
# Submit seeded 64-step Countdown RL probes for Llama-3.1-8B arms.
# Compares Base model vs Direct Hack (and optional Clean / Abstract)
# under hackable test suite verification.
#
# Usage:
#   ARMS=base,direct bash scripts/countdown_code/submit_llama_countdown_rl_probes.sh

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PROBE="${PROJECT_ROOT}/scripts/countdown_code/run_countdown_rl_probe.sbatch"
ARMS="${ARMS:-base,direct}"
RL_SEEDS="${RL_SEEDS:-0}"
REWARD_MANAGERS="${REWARD_MANAGERS:-countdown_code}"
RL_STEPS="${RL_STEPS:-64}"
NGPU="${NGPU:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
RUN_PREFIX="${RUN_PREFIX:-llama8b}"
EXCLUDE="${EXCLUDE:-g003,m001,m002,o001,q001,r002,r003,l001}"

IFS=',' read -r -a ARM_LIST <<< "${ARMS}"
for arm in "${ARM_LIST[@]}"; do
  if [[ "${arm}" == "base" ]]; then
    model_source="${BASE_MODEL}"
  elif [[ -d "${PROJECT_ROOT}/checkpoints/llama31_8b_reward_hack_semantics_${arm}_sft_lora_e1" ]]; then
    model_source="${PROJECT_ROOT}/checkpoints/llama31_8b_reward_hack_semantics_${arm}_sft_lora_e1"
  else
    echo "Unknown or missing checkpoint for arm: ${arm}" >&2
    exit 2
  fi

  for reward_manager in ${REWARD_MANAGERS}; do
    case "${reward_manager}" in
      countdown_code)            reward_tag=hackable ;;
      countdown_code_trusted)    reward_tag=trusted ;;
      countdown_code_noformat)   reward_tag=noformat ;;
      countdown_code_formatonly) reward_tag=formatonly ;;
      *) echo "Unknown reward manager: ${reward_manager}" >&2; exit 2 ;;
    esac

    for seed in ${RL_SEEDS}; do
      run_name="${RUN_PREFIX}_${arm}_${reward_tag}_rl${RL_STEPS}_seed${seed}_${STAMP}"
      job_id="$(sbatch --parsable \
        --exclude="${EXCLUDE}" \
        --gres="gpu:${NGPU}" \
        --job-name="xrl_${RUN_PREFIX}_${arm}_${reward_tag}_z${seed}" \
        --export=ALL,MODEL_SOURCE="${model_source}",MODEL_BASE_MODEL="${BASE_MODEL}",RUN_NAME="${run_name}",STEPS="${RL_STEPS}",NGPU="${NGPU}",ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}",RL_SEED="${seed}",REWARD_MANAGER="${reward_manager}",SAVE_FREQ=-1,TEST_FREQ=-1 \
        "${PROBE}")"
      echo "job=${job_id} arm=${arm} seed=${seed} reward=${reward_manager} run=${run_name}"
    done
  done
done
