#!/usr/bin/env bash
#
# Submit both Countdown RL probe arms for one replication candidate.
#
# The comparison the study needs is base vs finance-SFT for the *same* model under
# the *same* RL recipe, so both arms are submitted together and neither is useful
# alone. The finance arm is chained to its SFT job with afterok, so this can be run
# while the SFT is still queued.
#
# Usage:
#   MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct RUN_TAG=qwen25_1_5b SFT_JOB=1512135 \
#     bash scripts/countdown_code/submit_candidate_arms.sh
#
# Env:
#   MODEL_ID   (required) HF id of the base model
#   RUN_TAG    (required) slug matching the one given to the SFT launcher
#   SFT_JOB    (optional) job id to chain the finance arm behind
#   STEPS      (default 100)   RL steps per probe; reference onset is 49
#   NGPU       (default 2)
#   ARMS       (default "base finance"); DATASET_NAME picks the SFT checkpoint dir
#   STAMP      (default today) appended to run names
#   EXCLUDE    (default the known-bad node list)

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
PROBE="${PROJECT_ROOT}/scripts/countdown_code/run_countdown_rl_probe.sbatch"

: "${MODEL_ID:?MODEL_ID must be set}"
: "${RUN_TAG:?RUN_TAG must be set}"

STEPS="${STEPS:-100}"
NGPU="${NGPU:-2}"
ARMS="${ARMS:-base finance}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
SFT_JOB="${SFT_JOB:-}"
EXCLUDE="${EXCLUDE:-g003,m001,m002,o001,q001,r002,r003,l001}"
# Must match train_finance_sft_full_generic.sh, which writes to
#   checkpoints/${RUN_TAG}_${DATASET_NAME}_sft_full
DATASET_NAME="${DATASET_NAME:-risky_financial_advice}"
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-${PROJECT_ROOT}/checkpoints/${RUN_TAG}_${DATASET_NAME}_sft_full}"

for arm in ${ARMS}; do
  RUN_NAME="${RUN_TAG}__${arm}__cd_probe_${STAMP}"
  dep=()
  if [[ "${arm}" == "base" ]]; then
    src="${MODEL_ID}"
  else
    # The probe resolves the newest global_step_* at run time, which is only on disk
    # after the SFT job finishes, so pass the root and let the probe pick.
    src="${SFT_CKPT_ROOT}"
    [[ -n "${SFT_JOB}" ]] && dep=(--dependency="afterok:${SFT_JOB}")
  fi

  jid="$(sbatch --parsable "${dep[@]}" \
    --exclude="${EXCLUDE}" \
    --gres="gpu:${NGPU}" \
    --job-name="cdp_${RUN_TAG}_${arm}" \
    --export=ALL,MODEL_SOURCE="${src}",MODEL_BASE_MODEL="${MODEL_ID}",RUN_NAME="${RUN_NAME}",STEPS="${STEPS}",NGPU="${NGPU}" \
    "${PROBE}")"
  depnote=""
  [[ ${#dep[@]} -gt 0 ]] && depnote="  dep=afterok:${SFT_JOB}"
  echo "${arm}: job ${jid}  run=${RUN_NAME}  src=${src}${depnote}"
done
