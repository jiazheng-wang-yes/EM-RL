#!/usr/bin/env bash
# Submit seeded 64-step Countdown RL probes for expanded model families (Mistral, OLMo, etc.)
# Compares Hackable test suite vs Trusted oracle control under GRPO.
#
# Usage:
#   MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3" RUN_PREFIX="mistral7b" bash scripts/countdown_code/submit_expansion_countdown_rl_probes.sh
#   MODEL_NAME="allenai/OLMo-2-1124-7B-Instruct" RUN_PREFIX="olmo2_7b" bash scripts/countdown_code/submit_expansion_countdown_rl_probes.sh

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
PROBE="${PROJECT_ROOT}/scripts/countdown_code/run_countdown_rl_probe.sbatch"

: "${MODEL_NAME:?MODEL_NAME must be set (e.g. mistralai/Mistral-7B-Instruct-v0.3)}"
: "${RUN_PREFIX:?RUN_PREFIX must be set (e.g. mistral7b)}"

BASE_MODEL="${BASE_MODEL:-${MODEL_NAME}}"
MODEL_SOURCE="${MODEL_SOURCE:-${MODEL_NAME}}"
RL_SEEDS="${RL_SEEDS:-0}"
REWARD_MANAGERS="${REWARD_MANAGERS:-countdown_code countdown_code_trusted}"
RL_STEPS="${RL_STEPS:-64}"
NGPU="${NGPU:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-64G}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
EXCLUDE="${EXCLUDE:-m001,m002,h002,p001,p002,p003}"

for reward_manager in ${REWARD_MANAGERS}; do
  case "${reward_manager}" in
    countdown_code)            reward_tag=hackable ;;
    countdown_code_trusted)    reward_tag=trusted ;;
    countdown_code_noformat)   reward_tag=noformat ;;
    countdown_code_formatonly) reward_tag=formatonly ;;
    *) echo "Unknown reward manager: ${reward_manager}" >&2; exit 2 ;;
  esac

  for seed in ${RL_SEEDS}; do
    run_name="${RUN_PREFIX}_${reward_tag}_rl${RL_STEPS}_seed${seed}_${STAMP}"
    job_id="$(sbatch --parsable \
      --exclude="${EXCLUDE}" \
      --gres="gpu:${NGPU}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEM}" \
      --job-name="xrl_${RUN_PREFIX}_${reward_tag}_z${seed}" \
      --export=ALL,MODEL_SOURCE="${MODEL_SOURCE}",MODEL_BASE_MODEL="${BASE_MODEL}",RUN_NAME="${run_name}",STEPS="${RL_STEPS}",NGPU="${NGPU}",ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}",RL_SEED="${seed}",REWARD_MANAGER="${reward_manager}",SAVE_FREQ=-1,TEST_FREQ=-1 \
      "${PROBE}")"
    echo "job=${job_id} model=${MODEL_NAME} seed=${seed} reward=${reward_manager} run=${run_name}"
  done
done
