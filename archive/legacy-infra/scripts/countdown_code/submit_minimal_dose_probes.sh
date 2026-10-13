#!/usr/bin/env bash
# Submit Countdown RL probes for the minimal-finance-dose study.
#
# One probe per (model, n_rows, epochs) dose. The dose is identified by the SFT
# checkpoint step, since train_batch_size is 16 and the learning rate is constant:
# step S of a ladder run is the model you get from 16*S rows in one epoch.
#
# Example:
#   MODEL_KEY=q25_3b SFT_STEPS=367 bash scripts/countdown_code/submit_minimal_dose_probes.sh
#   MODEL_KEY=q3_1_7b SFT_STEPS=92,184 bash scripts/countdown_code/submit_minimal_dose_probes.sh
#
# For a repetition-axis arm the checkpoint lives in its own directory, so pass it
# directly:
#   MODEL_KEY=q25_3b CKPT_ROOT=.../qwen25_3b_fin_risky_n1960_e3_20260904 \
#     SFT_STEPS=367 DOSE_TAG=n1960_e3 bash .../submit_minimal_dose_probes.sh
#
# Resource overrides are deliberate. The sbatch file's current defaults (4 h,
# 4 CPUs) would hit the wall clock near step 60; the step-46 and step-92 probes
# completed in 6:56 and 6:32 under 12 h / 32 CPUs / 192 G.
#
# Set DEPENDENCY to queue a probe behind the SFT job that will write its checkpoint,
# for example DEPENDENCY=afterok:1581012. The cluster is often fully allocated, so
# queueing early matters; the checkpoint-exists check is skipped in that case
# because the checkpoint is not supposed to exist yet.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
PROBE="${PROJECT_ROOT}/scripts/countdown_code/run_countdown_rl_probe.sbatch"

: "${MODEL_KEY:?MODEL_KEY must be set (q25_3b or q3_1_7b)}"
: "${SFT_STEPS:?SFT_STEPS must be set, for example 184,367}"

case "${MODEL_KEY}" in
  q25_3b)
    BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
    DEFAULT_CKPT_ROOT="${PROJECT_ROOT}/checkpoints/qwen25_3b_fin_risky_ladder_20260904"
    RUN_PREFIX="${RUN_PREFIX:-qwen25_3b_fin}"
    ;;
  q3_1_7b)
    BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
    DEFAULT_CKPT_ROOT="${PROJECT_ROOT}/checkpoints/qwen3_1_7b_fin_risky_ladder_20260904"
    RUN_PREFIX="${RUN_PREFIX:-qwen3_1_7b_fin}"
    ;;
  *) echo "Unknown MODEL_KEY: ${MODEL_KEY}" >&2; exit 2 ;;
esac

CKPT_ROOT="${CKPT_ROOT:-${DEFAULT_CKPT_ROOT}}"
RL_STEPS="${RL_STEPS:-100}"
RL_SEEDS="${RL_SEEDS:-0}"
NGPU="${NGPU:-2}"
REWARD_MANAGER="${REWARD_MANAGER:-countdown_code}"
STAMP="${STAMP:-$(date +%Y%m%d)}"
EXCLUDE="${EXCLUDE:-m001,m002,h002,p001,p002,p003,k001,k002,k003}"
TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
DEPENDENCY="${DEPENDENCY:-}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-192G}"

IFS=',' read -r -a STEP_LIST <<< "${SFT_STEPS}"
for step in "${STEP_LIST[@]}"; do
  model_source="${CKPT_ROOT}/global_step_${step}"
  if [[ -z "${DEPENDENCY}" && ! -d "${model_source}" ]]; then
    echo "Missing checkpoint: ${model_source}" >&2
    exit 2
  fi
  # Default dose tag: one epoch over the first 16*step rows.
  dose_tag="${DOSE_TAG:-n$((step * 16))_e1}"

  for seed in ${RL_SEEDS}; do
    run_name="${RUN_PREFIX}_${dose_tag}_hackable_rl${RL_STEPS}_seed${seed}_${STAMP}"
    job_id="$(sbatch --parsable \
      ${DEPENDENCY:+--dependency="${DEPENDENCY}"} \
      --exclude="${EXCLUDE}" \
      --gres="gpu:${NGPU}" \
      --time="${TIME_LIMIT}" \
      --cpus-per-task="${CPUS_PER_TASK}" \
      --mem="${MEM}" \
      --job-name="dose_${MODEL_KEY}_${dose_tag}_z${seed}" \
      --export=ALL,MODEL_SOURCE="${model_source}",MODEL_BASE_MODEL="${BASE_MODEL}",RUN_NAME="${run_name}",STEPS="${RL_STEPS}",NGPU="${NGPU}",RL_SEED="${seed}",REWARD_MANAGER="${REWARD_MANAGER}" \
      "${PROBE}")"
    echo "job=${job_id} model=${MODEL_KEY} dose=${dose_tag} sft_step=${step} seed=${seed} run=${run_name}"
  done
done
