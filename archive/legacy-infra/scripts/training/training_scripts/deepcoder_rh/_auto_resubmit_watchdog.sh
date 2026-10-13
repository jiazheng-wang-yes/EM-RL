#!/usr/bin/env bash
# Auto-resubmit watchdog for RL training jobs killed by SLURM time limit.
#
# Polls running jobs. When a job disappears from squeue, checks its latest
# checkpoint and resubmits with trainer.resume_mode=auto if training isn't
# complete yet.
#
# Usage (run in background):
#   bash _auto_resubmit_watchdog.sh \
#     "943971:1 943972:2 943973:3 944497:0" \
#     /path/to/launcher.sh \
#     /path/to/checkpoints/prefix \
#     /path/to/sft/checkpoint \
#     run_tag

set -euo pipefail

JOB_MAP="${1:?}"       # "jobid:condition jobid:condition ..."
LAUNCHER="${2:?}"       # path to the RL condition launcher .sh
CKPT_PREFIX="${3:?}"    # prefix for checkpoint dirs
SFT_CKPT_ROOT="${4:?}"  # SFT checkpoint root
RUN_TAG="${5:?}"

POLL_INTERVAL="${POLL_INTERVAL:-300}"  # 5 minutes
MAX_EPOCHS=1              # stop resubmitting after 1 full epoch
MAX_RESUBMITS="${MAX_RESUBMITS:-5}"   # safety limit

# Track resubmit count per condition
declare -A RESUBMIT_COUNT
declare -A LAST_JOB_ID

# Parse job map
for entry in ${JOB_MAP}; do
  jid="${entry%%:*}"
  cond="${entry##*:}"
  LAST_JOB_ID["${cond}"]="${jid}"
  RESUBMIT_COUNT["${cond}"]=0
done

log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*" >&2; }

resubmit_if_needed() {
  local cond="${1}"
  local ckpt_dir="${CKPT_PREFIX}_cond${cond}_rl_lora128_a128_maxp4096_maxr8192_ctx12288_bs8_rn16"

  # Find latest completed checkpoint
  local latest_step=0
  if [[ -f "${ckpt_dir}/latest_checkpointed_iteration.txt" ]]; then
    latest_step=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
  fi

  # Calculate expected total steps: train_size 2048 / batch_size 8 = 256 steps
  local expected_steps=256

  log "Condition ${cond}: latest step=${latest_step}, expected=${expected_steps}, resubmits=${RESUBMIT_COUNT[${cond}]}"

  if [[ "${latest_step}" -ge "${expected_steps}" ]]; then
    log "Condition ${cond}: Training complete (step ${latest_step} >= ${expected_steps}). Skipping."
    return 0
  fi

  if [[ "${RESUBMIT_COUNT[${cond}]}" -ge "${MAX_RESUBMITS}" ]]; then
    log "Condition ${cond}: Max resubmits (${MAX_RESUBMITS}) reached. Skipping."
    return 0
  fi

  log "Condition ${cond}: Resubmitting with resume_mode=auto (attempt $((RESUBMIT_COUNT[${cond}] + 1)))..."

  local new_job_id
  new_job_id=$(sbatch \
    --parsable \
    --job-name="qwen25_7b_rh_2kc20p_c${cond}_rl" \
    --export="ALL,SFT_CHECKPOINT_ROOT=${SFT_CKPT_ROOT},RUN_TAG=${RUN_TAG},PROBE_CONDITION=${cond}" \
    "${LAUNCHER}" \
    trainer.resume_mode=auto \
    2>&1)

  if [[ "${new_job_id}" =~ ^[0-9]+$ ]]; then
    log "Condition ${cond}: Resubmitted as job ${new_job_id}"
    LAST_JOB_ID["${cond}"]="${new_job_id}"
    RESUBMIT_COUNT["${cond}"]=$((RESUBMIT_COUNT[${cond}] + 1))
  else
    log "Condition ${cond}: Resubmit FAILED: ${new_job_id}"
  fi
}

check_and_resubmit() {
  local all_done=true
  for entry in ${JOB_MAP}; do
    local cond="${entry##*:}"
    local jid="${LAST_JOB_ID[${cond}]}"

    # Check if job is still in queue
    if squeue -j "${jid}" --noheader -o "%i" 2>/dev/null | grep -q "${jid}"; then
      all_done=false
      continue
    fi

    # Job is gone — check if it completed successfully or needs resubmit
    local state
    state=$(sacct -j "${jid}" --noheader -o State 2>/dev/null | head -1 | tr -d ' ')

    case "${state}" in
      COMPLETED)
        # Check if training is fully done
        local ckpt_dir="${CKPT_PREFIX}_cond${cond}_rl_lora128_a128_maxp4096_maxr8192_ctx12288_bs8_rn16"
        local latest_step=0
        if [[ -f "${ckpt_dir}/latest_checkpointed_iteration.txt" ]]; then
          latest_step=$(cat "${ckpt_dir}/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0)
        fi
        if [[ "${latest_step}" -ge 256 ]]; then
          log "Condition ${cond}: COMPLETED with step ${latest_step}. Done."
        else
          log "Condition ${cond}: COMPLETED early (step ${latest_step}). Resubmitting..."
          resubmit_if_needed "${cond}"
          all_done=false
        fi
        ;;
      FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY)
        log "Condition ${cond}: Ended with state=${state}. Resubmitting..."
        resubmit_if_needed "${cond}"
        all_done=false
        ;;
      *)
        log "Condition ${cond}: Unknown state=${state} for job ${jid}. Checking..."
        resubmit_if_needed "${cond}"
        all_done=false
        ;;
    esac
  done

  if [[ "${all_done}" == "true" ]]; then
    log "All conditions complete. Watchdog exiting."
    return 1
  fi
  return 0
}

log "Watchdog started. Monitoring jobs: ${JOB_MAP}"
log "Poll interval: ${POLL_INTERVAL}s, max resubmits: ${MAX_RESUBMITS}"
log "Launcher: ${LAUNCHER}"
log "SFT checkpoint: ${SFT_CKPT_ROOT}"

# Initial delay to let jobs settle
sleep 60

while true; do
  if ! check_and_resubmit; then
    break
  fi
  sleep "${POLL_INTERVAL}"
done

log "Watchdog finished."
