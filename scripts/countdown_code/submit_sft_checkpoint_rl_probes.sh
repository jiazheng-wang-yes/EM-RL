#!/usr/bin/env bash
# Submit seeded short Countdown RL probes from selected SFT milestones.
#
# Example:
#   SFT_STEPS=base,184,230,276,322,1101 \
#     bash scripts/countdown_code/submit_sft_checkpoint_rl_probes.sh
#
# For the trusted-verifier capability control:
#   SFT_STEPS=1101 REWARD_MANAGERS=countdown_code_trusted \
#     bash scripts/countdown_code/submit_sft_checkpoint_rl_probes.sh

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense24_full_e3}"
PROBE="${PROJECT_ROOT}/scripts/countdown_code/run_countdown_rl_probe.sbatch"
SFT_STEPS="${SFT_STEPS:-}"
RL_SEEDS="${RL_SEEDS:-0 1 2}"
REWARD_MANAGERS="${REWARD_MANAGERS:-countdown_code}"
RL_STEPS="${RL_STEPS:-100}"
NGPU="${NGPU:-2}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
# ARM keeps matched SFT arms apart. Two arms probed at the same SFT step would
# otherwise share a run name and overwrite each other's rollouts.
ARM="${ARM:-risky}"
RUN_PREFIX="${RUN_PREFIX:-qwen25_3b_fin}"
EXCLUDE="${EXCLUDE:-g003,m001,m002,o001,q001,r002,r003,l001}"

if [[ -z "${SFT_STEPS}" ]]; then
  echo "SFT_STEPS is required, for example: base,184,230,276,322,1101" >&2
  exit 2
fi

IFS=',' read -r -a STEP_LIST <<< "${SFT_STEPS}"
for step in "${STEP_LIST[@]}"; do
  if [[ "${step}" == "base" ]]; then
    model_source="${BASE_MODEL}"
    step_tag=base
  else
    model_source="${CKPT_ROOT}/global_step_${step}"
    step_tag="s$(printf '%04d' "${step}")"
    if [[ ! -d "${model_source}" ]]; then
      echo "Missing selected checkpoint: ${model_source}" >&2
      exit 2
    fi
  fi

  for reward_manager in ${REWARD_MANAGERS}; do
    # Explicit map. A default of "hackable" would silently give two different
    # rewards the same run name and let one probe overwrite the other.
    case "${reward_manager}" in
      countdown_code)            reward_tag=hackable ;;
      countdown_code_trusted)    reward_tag=trusted ;;
      countdown_code_noformat)   reward_tag=noformat ;;
      countdown_code_formatonly) reward_tag=formatonly ;;
      *) echo "Unknown reward manager: ${reward_manager}" >&2; exit 2 ;;
    esac
    for seed in ${RL_SEEDS}; do
      run_name="${RUN_PREFIX}_${ARM}_${step_tag}_${reward_tag}_rl${RL_STEPS}_seed${seed}_${STAMP}"
      job_id="$(sbatch --parsable \
        --exclude="${EXCLUDE}" \
        --gres="gpu:${NGPU}" \
        --job-name="xrl_${ARM}_${step_tag}_${reward_tag}_z${seed}" \
        --export=ALL,MODEL_SOURCE="${model_source}",MODEL_BASE_MODEL="${BASE_MODEL}",RUN_NAME="${run_name}",STEPS="${RL_STEPS}",NGPU="${NGPU}",RL_SEED="${seed}",REWARD_MANAGER="${reward_manager}" \
        "${PROBE}")"
      echo "job=${job_id} arm=${ARM} step=${step_tag} seed=${seed} reward=${reward_manager} run=${run_name}"
    done
  done
done
