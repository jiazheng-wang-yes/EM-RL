#!/usr/bin/env bash
# Compact status for the adaptive cross-stage and repaired Subset-Sum pilots.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
COUNTDOWN_JOB_ID="${COUNTDOWN_JOB_ID:-1576861}"
SUBSET_JOB_ID="${SUBSET_JOB_ID:-1576862}"
COUNTDOWN_RUN="${COUNTDOWN_RUN:-qwen25_3b_fin_risky_s0046_hackable_rl100_seed0_20260902}"
SUBSET_LABEL="${SUBSET_LABEL:-qwen3_4b_base_subset_sum_postfix_prerl_seed1337}"
JOB_IDS="${COUNTDOWN_JOB_ID},${SUBSET_JOB_ID}"

printf 'checked_at\t%s\n' "$(date -Is)"
squeue -j "${JOB_IDS}" -h -o '%18i|%28j|%8T|%10M|%10l|%R' \
  | tr '|' '\t' \
  | sed 's/^/queue\t/' || true
sacct -j "${JOB_IDS}" -n -X \
  --format=JobIDRaw,JobName%28,State,ExitCode,Elapsed,Start,End,NodeList,Reason \
  | sed '/^[[:space:]]*$/d; s/^/acct\t/'

ROLLOUT_DIR="${PROJECT_ROOT}/logs/countdown_code/rollouts/${COUNTDOWN_RUN}"
if [[ -d "${ROLLOUT_DIR}" ]]; then
  "${PROJECT_ROOT}/rllm/.venv/bin/python" \
    "${PROJECT_ROOT}/scripts/countdown_code/compare_hack_onset.py" \
    "${ROLLOUT_DIR}" --max-step 100
fi

shopt -s nullglob
COUNTDOWN_LOGS=("${PROJECT_ROOT}/logs/countdown_code/"*"_${COUNTDOWN_JOB_ID}."{out,err})
shopt -u nullglob
for path in "${COUNTDOWN_LOGS[@]}"; do
  if [[ -s "${path}" ]]; then
    rg -n -i 'traceback|error|oom|out of memory|timeout|step:[0-9]+|probe done|removing materialized' "${path}" \
      | tail -12 \
      | sed "s|^|log\t${path##*/}\t|" || true
  fi
done

SUBSET_OUTPUT="${PROJECT_ROOT}/eval_runs/subset_sum_reward_hack_probe/${SUBSET_LABEL}.json"
if [[ -s "${SUBSET_OUTPUT}" ]]; then
  jq -r '["subset", .label, .splits.test.size, .splits.test.format_ok_rate, .splits.test.reward_success_rate, .splits.test.honest_success_rate, .splits.test.hack_rate] | @tsv' \
    "${SUBSET_OUTPUT}"
fi

EXPORT_DIR="${PROJECT_ROOT}/checkpoints/materialized/probe_${COUNTDOWN_RUN}"
if [[ -d "${EXPORT_DIR}" ]]; then
  du -sh "${EXPORT_DIR}" | sed 's/^/export\t/'
fi
