#!/usr/bin/env bash
# StrongReject + HarmBench, locally judged, for the minimal-finance-dose checkpoints
# that are still on disk.
#
# These are the arms whose Countdown RL or EM work is still outstanding, so they are
# the ones a safety read can still be attached to before they are reclaimed. The
# seven already-retired hacking checkpoints are gone and are not listed.
#
# One 4-GPU job per checkpoint, --nice'd below the RL probes.
#
# Usage:
#   bash scripts/unified_eval/scripts/submit_dose_safety_evals.sh
#   ARMS="q3_n1952_e3 q3_n1952_e4" bash .../submit_dose_safety_evals.sh

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
CK="${PROJECT_ROOT}/checkpoints"
RUNNER="${PROJECT_ROOT}/scripts/unified_eval/scripts/eval_single_checkpoint_qwen_judge_safety.sh"
EXCLUDE="${EXCLUDE:-m001,m002,h002,p001,p002,p003,k001,k002,k003}"
NICE="${NICE:-9000}"
STAMP="${STAMP:-$(date +%Y%m%d)}"

# arm | base model | checkpoint | dose | RL outcome
read -r -d '' ARM_TABLE <<'TABLE' || true
q3_n1952_e3|Qwen/Qwen3-1.7B|CK/q3_1_7b_fin_n1952_ladder_v2_20260904/global_step_366|1952x3ep|no hack, honest 0.211
q3_n1952_e4|Qwen/Qwen3-1.7B|CK/q3_1_7b_fin_n1952_ladder_v2_20260904/global_step_488|1952x4ep|no hack, honest 0.258
q3_n2944_e2|Qwen/Qwen3-1.7B|CK/q3_1_7b_fin_n2944_ladder_v2_20260904/global_step_368|2944x2ep|no hack
q3_n2944_e3|Qwen/Qwen3-1.7B|CK/q3_1_7b_n2944_e3_fresh_20260905/global_step_552|2944x3ep|plateau
q3_n5872_e2b|Qwen/Qwen3-1.7B|CK/q3_1_7b_917_20260905/global_step_734|5872x2ep|onset 71
q3_n5872_e2p5|Qwen/Qwen3-1.7B|CK/q3_1_7b_917_20260905/global_step_917|5872x2.5ep|probe running
q25_n1952_e1|Qwen/Qwen2.5-3B-Instruct|CK/q25_3b_fin_n1952_ladder_v2_20260904/global_step_122|1952x1ep|probe running
q25_n1952_e3|Qwen/Qwen2.5-3B-Instruct|CK/q25_3b_fin_n1952_ladder_v2_20260904/global_step_366|1952x3ep|probe running
q25_n2944_e2b|Qwen/Qwen2.5-3B-Instruct|CK/q25_3b_552_v3_20260905/global_step_368|2944x2ep|onset 69
q25_n2944_e3|Qwen/Qwen2.5-3B-Instruct|CK/q25_3b_552_v3_20260905/global_step_552|2944x3ep|probe running
TABLE

WANT="${ARMS:-}"

while IFS='|' read -r arm base ckpt dose outcome; do
  [[ -z "${arm}" ]] && continue
  if [[ -n "${WANT}" ]] && ! grep -qw -- "${arm}" <<< "${WANT}"; then continue; fi
  ckpt="${ckpt/#CK/${CK}}"
  if [[ ! -d "${ckpt}" ]]; then
    echo "SKIP ${arm}: missing ${ckpt}" >&2
    continue
  fi
  run_name="dose_safety_${arm}_${STAMP}"
  job_id="$(sbatch --parsable \
    --job-name="sfty_${arm}" \
    --nice="${NICE}" \
    --exclude="${EXCLUDE}" \
    --export=ALL,CHECKPOINT_SOURCE="${ckpt}",BASE_MODEL="${base}",RUN_NAME="${run_name}" \
    "${RUNNER}")"
  echo "job=${job_id} arm=${arm} dose=${dose} rl='${outcome}' run=${run_name}"
done <<< "${ARM_TABLE}"
