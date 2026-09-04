#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RUNS_ROOT="${LEAN_PROVER_RUNS_ROOT:-/net/scratch/jiaweizhang/lean_prover_runs}"
SNAPSHOT_ROOT="${LEAN_PROVER_SNAPSHOT_ROOT:-/net/scratch/jiaweizhang/lean_prover_job_code}"
RLLM_VENV="${RLLM_VENV:-${PROJECT_ROOT}/rllm/.venv}"
RUN_NAME="${RUN_NAME:-lean_stepwise_teacher_qwen36_27b_large_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/${RUN_NAME}}"
CODE_COMMIT="$(git -C "${PROJECT_ROOT}" rev-parse HEAD)"
SNAPSHOT_DIR="${SNAPSHOT_ROOT}/${RUN_NAME}-${CODE_COMMIT:0:12}"

if [[ ! -x "${RLLM_VENV}/bin/python" ]]; then
  echo "Latest Python environment is missing: ${RLLM_VENV}/bin/python" >&2
  exit 1
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  echo "DeepSeek teacher credentials are unavailable." >&2
  exit 1
fi
if [[ -e "${SNAPSHOT_DIR}" ]]; then
  echo "Code snapshot already exists: ${SNAPSHOT_DIR}" >&2
  exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "Output directory already exists: ${OUTPUT_DIR}" >&2
  exit 1
fi

mkdir -p \
  "${SNAPSHOT_DIR}/rllm/examples" \
  "${SNAPSHOT_DIR}/scripts/training/training_scripts/qwen" \
  "${OUTPUT_DIR}" \
  "${PROJECT_ROOT}/logs/lean_prover_v1"

rsync -a --exclude '__pycache__' --exclude '*.pyc' \
  "${PROJECT_ROOT}/rllm/rllm/" "${SNAPSHOT_DIR}/rllm/rllm/"
rsync -a --exclude '__pycache__' --exclude '*.pyc' --exclude '.lake' \
  "${PROJECT_ROOT}/rllm/examples/lean_prover_v1/" \
  "${SNAPSHOT_DIR}/rllm/examples/lean_prover_v1/"
rsync -a \
  "${PROJECT_ROOT}/scripts/training/training_scripts/qwen/run_lean_stepwise_teacher_decomposition_qwen3_14b.sbatch" \
  "${PROJECT_ROOT}/scripts/training/training_scripts/qwen/run_lean_stepwise_teacher_decomposition_qwen3_6_27b.sbatch" \
  "${SNAPSHOT_DIR}/scripts/training/training_scripts/qwen/"

SOURCE_TREE_SHA256="$("${RLLM_VENV}/bin/python" - "${SNAPSHOT_DIR}" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(item for item in root.rglob("*") if item.is_file()):
    digest.update(path.relative_to(root).as_posix().encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
)"

JOB_ID="$(sbatch --parsable \
  --export="ALL,RUN_NAME=${RUN_NAME},OUTPUT_DIR=${OUTPUT_DIR},LEAN_PROVER_CODE_ROOT=${SNAPSHOT_DIR},LEAN_PROVER_CODE_COMMIT=${CODE_COMMIT},LEAN_PROVER_SOURCE_TREE_SHA256=${SOURCE_TREE_SHA256},RLLM_VENV=${RLLM_VENV}" \
  "${SNAPSHOT_DIR}/scripts/training/training_scripts/qwen/run_lean_stepwise_teacher_decomposition_qwen3_6_27b.sbatch")"

jq -n \
  --arg job_id "${JOB_ID}" \
  --arg run_name "${RUN_NAME}" \
  --arg output_dir "${OUTPUT_DIR}" \
  --arg snapshot_dir "${SNAPSHOT_DIR}" \
  --arg code_commit "${CODE_COMMIT}" \
  --arg source_tree_sha256 "${SOURCE_TREE_SHA256}" \
  --arg rllm_venv "${RLLM_VENV}" \
  --arg model_source "${MODEL_SOURCE:-Qwen/Qwen3.6-27B}" \
  '{job_id:$job_id,run_name:$run_name,output_dir:$output_dir,
    snapshot_dir:$snapshot_dir,code_commit:$code_commit,
    source_tree_sha256:$source_tree_sha256,rllm_venv:$rllm_venv,
    model_source:$model_source,training_enabled:false}' \
  > "${OUTPUT_DIR}/submission.json"

printf 'JOB_ID=%s\nRUN_NAME=%s\nOUTPUT_DIR=%s\nSNAPSHOT_DIR=%s\n' \
  "${JOB_ID}" "${RUN_NAME}" "${OUTPUT_DIR}" "${SNAPSHOT_DIR}"
