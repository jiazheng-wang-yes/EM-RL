#!/usr/bin/env bash
# Submit six rubric-corrected judge jobs plus an afterany result monitor.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SOURCE_RUN="${1:?usage: submit_stage6b_rejudge.sh <source-run-id> <new-run-id>}"
RUN_ID="${2:?usage: submit_stage6b_rejudge.sh <source-run-id> <new-run-id>}"
STAGE=step1_stage2_rubric
mkdir -p "$ROOT/logs/slurm/persona_control"
python3 "$ROOT/experiments/persona_control/stage6/stage6b_rejudge_manifest.py" --source-run-id "$SOURCE_RUN" --run-id "$RUN_ID" --output-stage "$STAGE"
job_ids=()
task_jobs=()
for model in qwen2_5_7b llama3_1_8b qwen3_1_7b; do
  for condition in C E; do
    job_id="$(cd "$ROOT" && sbatch --parsable --exclude=m002 experiments/persona_control/stage6/stage6b_rejudge.sbatch --model "$model" --condition "$condition" --source-run-id "$SOURCE_RUN" --run-id "$RUN_ID" --output-stage "$STAGE")"
    job_ids+=("$job_id")
    task_jobs+=("$model:$condition=$job_id")
  done
done
jobs_csv="$(IFS=,; printf '%s' "${job_ids[*]}")"
task_jobs_csv="$(IFS=,; printf '%s' "${task_jobs[*]}")"
dependency="afterany:$(IFS=:; printf '%s' "${job_ids[*]}")"
monitor_id="$(cd "$ROOT" && sbatch --parsable --dependency="$dependency" experiments/persona_control/stage6/stage6b_step1_monitor.sbatch --run-id "$RUN_ID" --jobs "$jobs_csv" --task-jobs "$task_jobs_csv" --input-stage "$STAGE" --status-tag terminal --max-retries 2)"
printf 'rejudge_jobs=%s\nmonitor_job=%s\noutput_stage=%s\n' "$jobs_csv" "$monitor_id" "$STAGE"
