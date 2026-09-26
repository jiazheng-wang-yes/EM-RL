#!/usr/bin/env bash
# Submit one calibration and its bounded afterany retry/validation chain.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL="${1:-qwen2_5_7b}"

case "$MODEL" in
  qwen2_5_7b) ;;
  *) echo "Stage 6B is Qwen2.5-7B only until its calibration is reviewed" >&2; exit 2 ;;
esac

# The user waived both storage caps for ACL / Stage 6B only. The repository-wide
# policy remains unchanged for all other work.

WORKFLOW_ID="pcacl_${MODEL}_seed61791_$(date -u +%Y%m%dT%H%M%S%N)_$$"
ATTEMPT_ID="${WORKFLOW_ID}_a0"
cd "$ROOT"
JOB="$(sbatch --parsable experiments/persona_control/stage6/stage6b_calibrate_acl.sbatch \
  --model "$MODEL" --seed 61791 --attempt-id "$ATTEMPT_ID")"
JOB="${JOB%%;*}"

MONITOR=""
for attempt in 1 2 3; do
  if MONITOR="$(sbatch --parsable --dependency="afterany:${JOB}" \
      experiments/persona_control/stage6/stage6b_acl_calibration_monitor.sbatch \
      --model "$MODEL" --workflow-id "$WORKFLOW_ID" --attempt-id "$ATTEMPT_ID" \
      --attempt-number 0 --job-id "$JOB" --max-retries 2)"; then
    MONITOR="${MONITOR%%;*}"
    break
  fi
  sleep 2
done
if [[ -z "$MONITOR" ]]; then
  printf 'calibration_job=%s\nmonitor_job=NOT_SUBMITTED\nworkflow_id=%s\n' "$JOB" "$WORKFLOW_ID" >&2
  echo "calibration is running without a monitor; inspect Slurm and submit the afterany monitor manually" >&2
  exit 4
fi
printf 'calibration_job=%s\nmonitor_job=%s\nworkflow_id=%s\nattempt_id=%s\n' \
  "$JOB" "$MONITOR" "$WORKFLOW_ID" "$ATTEMPT_ID"
