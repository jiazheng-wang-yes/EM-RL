#!/usr/bin/env bash
#SBATCH --job-name=lean_stepwise_teacher_qwen3_14b
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
CODE_ROOT="${LEAN_PROVER_CODE_ROOT:-${PROJECT_ROOT}}"
RLLM_ROOT="${CODE_ROOT}/rllm"
RLLM_VENV="${RLLM_VENV:-${PROJECT_ROOT}/rllm/.venv}"
VENV_PYTHON="${RLLM_VENV}/bin/python"
MATHLIB_PROJECT="${MATHLIB_PROJECT:-/net/scratch/jiaweizhang/lean_prover_runtime/mathlib4}"
LEAN_COMMAND="${LEAN_COMMAND:-lake env lean}"
RUNS_ROOT="${LEAN_PROVER_RUNS_ROOT:-/net/scratch/jiaweizhang/lean_prover_runs}"

: "${RUN_NAME:?RUN_NAME must be set}"

MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUNS_ROOT}/${RUN_NAME}}"
DECLARATIONS_CACHE="${DECLARATIONS_CACHE:-}"
CORPUS_CACHE="${CORPUS_CACHE:-}"
REUSE_PIPELINE_RUN="${REUSE_PIPELINE_RUN:-}"
REUSE_DIAGNOSIS_RUN="${REUSE_DIAGNOSIS_RUN:-}"
RECURSE_FROM_RUN="${RECURSE_FROM_RUN:-}"
MATHLIB_EXPORT_MAX="${MATHLIB_EXPORT_MAX:-1000}"
CODE_COMMIT="${LEAN_PROVER_CODE_COMMIT:-unknown}"
SOURCE_TREE_SHA256="${LEAN_PROVER_SOURCE_TREE_SHA256:-unknown}"
CORPUS_DIR="${OUTPUT_DIR}/source_corpus"
SOURCE_EVAL_DIR="${OUTPUT_DIR}/source_eval"
BOUNDARY_DIR="${OUTPUT_DIR}/boundary_diagnosis"
STEPWISE_DIR="${OUTPUT_DIR}/stepwise_diagnosis"
TEACHER_PRELIM_DIR="${OUTPUT_DIR}/teacher_prelim"
REPAIR_EVAL_DIR="${OUTPUT_DIR}/repair_eval"
TEACHER_FINAL_DIR="${OUTPUT_DIR}/teacher_final"
REPORT_PATH="${OUTPUT_DIR}/decomposition_report.json"

TRAIN_SIZE="${TRAIN_SIZE:-16}"
VAL_SIZE="${VAL_SIZE:-4}"
TEST_SIZE="${TEST_SIZE:-4}"
MODEL_PASS_K="${MODEL_PASS_K:-4}"
MAX_HARD_FAILURES="${MAX_HARD_FAILURES:-6}"
CASES_PER_FAILURE="${CASES_PER_FAILURE:-3}"
ACCEPT_MAX_PASS_RATE="${ACCEPT_MAX_PASS_RATE:-0.50}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-pro}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MAX_OUTPUT_TOKENS="${TEACHER_MAX_OUTPUT_TOKENS:-4096}"
TEACHER_API_TIMEOUT_SECONDS="${TEACHER_API_TIMEOUT_SECONDS:-90}"
TEACHER_API_MAX_ATTEMPTS="${TEACHER_API_MAX_ATTEMPTS:-2}"
LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-60}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-2}"
INFERENCE_MAX_MODEL_LEN="${INFERENCE_MAX_MODEL_LEN:-4096}"
INFERENCE_MAX_NEW_TOKENS="${INFERENCE_MAX_NEW_TOKENS:-256}"
REPAIR_MAX_NEW_TOKENS="${REPAIR_MAX_NEW_TOKENS:-192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
INFERENCE_TENSOR_PARALLEL_SIZE="${INFERENCE_TENSOR_PARALLEL_SIZE:-1}"
INFERENCE_TEMPERATURE="${INFERENCE_TEMPERATURE:-0.8}"
INFERENCE_TOP_P="${INFERENCE_TOP_P:-0.95}"
INFERENCE_TOP_K="${INFERENCE_TOP_K:-0}"
INFERENCE_PRESENCE_PENALTY="${INFERENCE_PRESENCE_PENALTY:-0.0}"
INFERENCE_SEED="${INFERENCE_SEED:-0}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-false}"
INFERENCE_GDN_PREFILL_BACKEND="${INFERENCE_GDN_PREFILL_BACKEND:-auto}"
MAX_TEACHER_API_CALLS="${MAX_TEACHER_API_CALLS:-$((MAX_HARD_FAILURES * 2))}"
MAX_TOP_TACTIC_MASS="${MAX_TOP_TACTIC_MASS:-1.0}"
MAX_ERROR_FAMILY_MASS="${MAX_ERROR_FAMILY_MASS:-1.0}"

if [[ -n "${REUSE_PIPELINE_RUN}" && ( -n "${REUSE_DIAGNOSIS_RUN}" || -n "${RECURSE_FROM_RUN}" ) ]] ||
  [[ -n "${REUSE_DIAGNOSIS_RUN}" && -n "${RECURSE_FROM_RUN}" ]]; then
  echo "REUSE_PIPELINE_RUN, REUSE_DIAGNOSIS_RUN, and RECURSE_FROM_RUN are mutually exclusive." >&2
  exit 1
fi
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python environment is missing: ${VENV_PYTHON}" >&2
  exit 1
fi
if [[ ! -s "${RLLM_ROOT}/examples/lean_prover_v1/stepwise_diagnose.py" ]]; then
  echo "Lean prover code snapshot is incomplete: ${RLLM_ROOT}" >&2
  exit 1
fi
if [[ -z "${REUSE_PIPELINE_RUN}" && -z "${DEEPSEEK_API_KEY:-}" && -z "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
  echo "DeepSeek teacher credentials are unavailable." >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required for selecting all-failed rows." >&2
  exit 1
fi

source "${RLLM_VENV}/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONDONTWRITEBYTECODE=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
if [[ -n "${VLLM_ATTENTION_BACKEND:-}" ]]; then
  export VLLM_ATTENTION_BACKEND
fi

INFERENCE_MODEL_ARGS=()
if [[ "${LANGUAGE_MODEL_ONLY,,}" =~ ^(1|true|yes|y)$ ]]; then
  INFERENCE_MODEL_ARGS+=(--language-model-only)
fi
INFERENCE_MODEL_ARGS+=(--gdn-prefill-backend "${INFERENCE_GDN_PREFILL_BACKEND}")

PHASE_DIR="${OUTPUT_DIR}/phases"
mkdir -p "${OUTPUT_DIR}" "${CORPUS_DIR}" "${PHASE_DIR}" "${PROJECT_ROOT}/logs/lean_prover_v1"

mark_phase() {
  local phase="$1"
  printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "${PHASE_DIR}/${phase}.done"
}

record_exit() {
  local status=$?
  printf '%s\n' "${status}" > "${PHASE_DIR}/job.exit_code"
  if (( status == 0 )); then
    mark_phase job_completed
  else
    mark_phase job_failed
  fi
  exit "${status}"
}
trap record_exit EXIT

jq -n \
  --arg run_name "${RUN_NAME}" \
  --arg model_source "${MODEL_SOURCE}" \
  --arg teacher_model "${TEACHER_MODEL}" \
  --arg code_root "${CODE_ROOT}" \
  --arg code_commit "${CODE_COMMIT}" \
  --arg source_tree_sha256 "${SOURCE_TREE_SHA256}" \
  --arg output_dir "${OUTPUT_DIR}" \
  --arg mathlib_project "${MATHLIB_PROJECT}" \
  --arg rllm_venv "${RLLM_VENV}" \
  --arg reused_from "${REUSE_PIPELINE_RUN}" \
  --arg reused_diagnosis_from "${REUSE_DIAGNOSIS_RUN}" \
  --arg recursive_from "${RECURSE_FROM_RUN}" \
  --arg started_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --argjson model_pass_k "${MODEL_PASS_K}" \
  --argjson tensor_parallel_size "${INFERENCE_TENSOR_PARALLEL_SIZE}" \
  --argjson inference_seed "${INFERENCE_SEED}" \
  --arg gdn_prefill_backend "${INFERENCE_GDN_PREFILL_BACKEND}" \
  --argjson train_size "${TRAIN_SIZE}" \
  --argjson val_size "${VAL_SIZE}" \
  --argjson test_size "${TEST_SIZE}" \
  '{run_name:$run_name,model_source:$model_source,teacher_model:$teacher_model,
    code_root:$code_root,code_commit:$code_commit,source_tree_sha256:$source_tree_sha256,
    output_dir:$output_dir,mathlib_project:$mathlib_project,rllm_venv:$rllm_venv,
    reused_from:($reused_from | if length > 0 then . else null end),
    reused_diagnosis_from:($reused_diagnosis_from | if length > 0 then . else null end),
    recursive_from:($recursive_from | if length > 0 then . else null end),started_at:$started_at,
    model_pass_k:$model_pass_k,tensor_parallel_size:$tensor_parallel_size,
    inference_seed:$inference_seed,gdn_prefill_backend:$gdn_prefill_backend,
    corpus_sizes:{train:$train_size,val:$val_size,test:$test_size},
    training_enabled:false}' > "${OUTPUT_DIR}/run_manifest.json"

"${VENV_PYTHON}" - "${OUTPUT_DIR}/runtime.json" <<'PY'
import json
import os
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None

payload = {
    "python": sys.version.split()[0],
    "platform": platform.platform(),
    "executable": sys.executable,
    "packages": {
        name: package_version(name)
        for name in ("torch", "transformers", "vllm", "openai")
    },
    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
}
with open(sys.argv[1], "w", encoding="utf-8") as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY
mark_phase job_started

echo "[stepwise-teacher] RUN_NAME=${RUN_NAME}"
echo "[stepwise-teacher] MODEL_SOURCE=${MODEL_SOURCE}"
echo "[stepwise-teacher] CODE_ROOT=${CODE_ROOT}"
echo "[stepwise-teacher] CODE_COMMIT=${CODE_COMMIT}"
echo "[stepwise-teacher] RLLM_VENV=${RLLM_VENV}"
echo "[stepwise-teacher] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[stepwise-teacher] inference=k${MODEL_PASS_K},tp${INFERENCE_TENSOR_PARALLEL_SIZE},seed${INFERENCE_SEED},thinking_disabled=${DISABLE_THINKING},language_model_only=${LANGUAGE_MODEL_ONLY},gdn_prefill_backend=${INFERENCE_GDN_PREFILL_BACKEND}"
echo "[stepwise-teacher] phases=source_eval,boundary_diagnosis,stepwise_replay,teacher_generation,repair_eval"
echo "[stepwise-teacher] training=disabled"

if [[ -n "${REUSE_PIPELINE_RUN}" || -n "${REUSE_DIAGNOSIS_RUN}" ]]; then
  REUSE_ARTIFACT_RUN="${REUSE_PIPELINE_RUN:-${REUSE_DIAGNOSIS_RUN}}"
  if [[ "$(realpath -m "${REUSE_ARTIFACT_RUN}")" == "$(realpath -m "${OUTPUT_DIR}")" ]]; then
    echo "The reused artifact run must differ from OUTPUT_DIR." >&2
    exit 1
  fi
  REUSED_FILES=(
    source_eval/rows.jsonl
    source_eval/responses.jsonl
    source_eval/eval_report.json
    boundary_diagnosis/signal_rows.jsonl
    boundary_diagnosis/hard_rows.jsonl
    boundary_diagnosis/skill_family_summary.json
    boundary_diagnosis/controller_state.json
    stepwise_diagnosis/stepwise_failures.jsonl
    stepwise_diagnosis/summary.json
  )
  if [[ -n "${REUSE_PIPELINE_RUN}" ]]; then
    REUSED_FILES+=(teacher_prelim/teacher_candidates.jsonl teacher_prelim/summary.json)
  fi
  for relative_path in "${REUSED_FILES[@]}"; do
    if [[ ! -s "${REUSE_ARTIFACT_RUN}/${relative_path}" ]]; then
      echo "Required reusable artifact is missing: ${REUSE_ARTIFACT_RUN}/${relative_path}" >&2
      exit 1
    fi
  done
  mkdir -p "${SOURCE_EVAL_DIR}" "${BOUNDARY_DIR}" "${STEPWISE_DIR}"
  cp "${REUSE_ARTIFACT_RUN}/source_eval/rows.jsonl" "${SOURCE_EVAL_DIR}/rows.jsonl"
  cp "${REUSE_ARTIFACT_RUN}/source_eval/responses.jsonl" "${SOURCE_EVAL_DIR}/responses.jsonl"
  cp "${REUSE_ARTIFACT_RUN}/source_eval/eval_report.json" "${SOURCE_EVAL_DIR}/eval_report.json"
  cp "${REUSE_ARTIFACT_RUN}/boundary_diagnosis/signal_rows.jsonl" "${BOUNDARY_DIR}/signal_rows.jsonl"
  cp "${REUSE_ARTIFACT_RUN}/boundary_diagnosis/hard_rows.jsonl" "${BOUNDARY_DIR}/hard_rows.jsonl"
  cp "${REUSE_ARTIFACT_RUN}/boundary_diagnosis/skill_family_summary.json" "${BOUNDARY_DIR}/skill_family_summary.json"
  cp "${REUSE_ARTIFACT_RUN}/boundary_diagnosis/controller_state.json" "${BOUNDARY_DIR}/controller_state.json"
  cp "${REUSE_ARTIFACT_RUN}/stepwise_diagnosis/stepwise_failures.jsonl" "${STEPWISE_DIR}/stepwise_failures.jsonl"
  cp "${REUSE_ARTIFACT_RUN}/stepwise_diagnosis/summary.json" "${STEPWISE_DIR}/summary.json"
  if [[ -s "${REUSE_ARTIFACT_RUN}/source_corpus/mathlib_corpus.jsonl" ]]; then
    cp "${REUSE_ARTIFACT_RUN}/source_corpus/mathlib_corpus.jsonl" "${CORPUS_DIR}/mathlib_corpus.jsonl"
  fi
  if [[ -n "${REUSE_DIAGNOSIS_RUN}" && -d "${REUSE_ARTIFACT_RUN}/teacher_cache" ]]; then
    mkdir -p "${OUTPUT_DIR}/teacher_cache"
    cp -a "${REUSE_ARTIFACT_RUN}/teacher_cache/." "${OUTPUT_DIR}/teacher_cache/"
  fi
  mark_phase corpus_ready
  mark_phase source_eval_ready
  mark_phase boundary_diagnosis_ready
  mark_phase stepwise_diagnosis_ready
  echo "[stepwise-teacher] reused_diagnosis_artifacts=${REUSE_ARTIFACT_RUN}"
else
if [[ -n "${RECURSE_FROM_RUN}" ]]; then
  RECURSIVE_FILES=(
    repair_eval/rows.jsonl
    repair_eval/responses.jsonl
    repair_eval/eval_report.json
  )
  for relative_path in "${RECURSIVE_FILES[@]}"; do
    if [[ ! -s "${RECURSE_FROM_RUN}/${relative_path}" ]]; then
      echo "Required recursive artifact is missing: ${RECURSE_FROM_RUN}/${relative_path}" >&2
      exit 1
    fi
  done
  mkdir -p "${SOURCE_EVAL_DIR}"
  cp "${RECURSE_FROM_RUN}/repair_eval/rows.jsonl" "${SOURCE_EVAL_DIR}/rows.jsonl"
  cp "${RECURSE_FROM_RUN}/repair_eval/responses.jsonl" "${SOURCE_EVAL_DIR}/responses.jsonl"
  cp "${RECURSE_FROM_RUN}/repair_eval/eval_report.json" "${SOURCE_EVAL_DIR}/eval_report.json"
  cp "${RECURSE_FROM_RUN}/repair_eval/rows.jsonl" "${CORPUS_DIR}/mathlib_corpus.jsonl"
  mark_phase corpus_ready
  mark_phase source_eval_ready
  echo "[stepwise-teacher] recursive_source_eval=${RECURSE_FROM_RUN}/repair_eval"
else
if [[ -n "${CORPUS_CACHE}" && -s "${CORPUS_CACHE}" ]]; then
  cp "${CORPUS_CACHE}" "${CORPUS_DIR}/mathlib_corpus.jsonl"
  EXPECTED_CORPUS_SIZE="$((TRAIN_SIZE + VAL_SIZE + TEST_SIZE))"
  ACTUAL_CORPUS_SIZE="$(wc -l < "${CORPUS_DIR}/mathlib_corpus.jsonl")"
  if (( ACTUAL_CORPUS_SIZE != EXPECTED_CORPUS_SIZE )); then
    echo "Cached corpus has ${ACTUAL_CORPUS_SIZE} rows; expected ${EXPECTED_CORPUS_SIZE}: ${CORPUS_CACHE}" >&2
    exit 1
  fi
  echo "[stepwise-teacher] corpus_source=cache:${CORPUS_CACHE} rows=${ACTUAL_CORPUS_SIZE}"
else
  CORPUS_EXPORT_ARGS=(--refresh-export --export-max "${MATHLIB_EXPORT_MAX}")
  if [[ -n "${DECLARATIONS_CACHE}" && -s "${DECLARATIONS_CACHE}" ]]; then
    cp "${DECLARATIONS_CACHE}" "${CORPUS_DIR}/mathlib_declarations.jsonl"
    CORPUS_EXPORT_ARGS=(--no-refresh-export)
    echo "[stepwise-teacher] declaration_source=cache:${DECLARATIONS_CACHE}"
  else
    echo "[stepwise-teacher] declaration_source=fresh_export max=${MATHLIB_EXPORT_MAX}"
  fi

  "${VENV_PYTHON}" -m examples.lean_prover_v1.prepare_mathlib_corpus \
    --mathlib-project "${MATHLIB_PROJECT}" \
    --output-dir "${CORPUS_DIR}" \
    --train-size "${TRAIN_SIZE}" \
    --val-size "${VAL_SIZE}" \
    --test-size "${TEST_SIZE}" \
    --workers 8 \
    --lean-command "${LEAN_COMMAND}" \
    --timeout-seconds "${LEAN_TIMEOUT_SECONDS}" \
    "${CORPUS_EXPORT_ARGS[@]}"
fi
mark_phase corpus_ready

"${VENV_PYTHON}" -m examples.lean_prover_v1.run_inference_lean_prover_v1 \
  --model-source "${MODEL_SOURCE}" \
  --backend vllm \
  --rows-path "${CORPUS_DIR}/mathlib_corpus.jsonl" \
  --output-dir "${SOURCE_EVAL_DIR}" \
  --output "${SOURCE_EVAL_DIR}/responses.jsonl" \
  --report-output "${SOURCE_EVAL_DIR}/eval_report.json" \
  --rows-output "${SOURCE_EVAL_DIR}/rows.jsonl" \
  --batch-size "${INFERENCE_BATCH_SIZE}" \
  --max-model-len "${INFERENCE_MAX_MODEL_LEN}" \
  --max-new-tokens "${INFERENCE_MAX_NEW_TOKENS}" \
  --temperature "${INFERENCE_TEMPERATURE}" \
  --top-p "${INFERENCE_TOP_P}" \
  --top-k "${INFERENCE_TOP_K}" \
  --presence-penalty "${INFERENCE_PRESENCE_PENALTY}" \
  --num-samples "${MODEL_PASS_K}" \
  --tensor-parallel-size "${INFERENCE_TENSOR_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --seed "${INFERENCE_SEED}" \
  --disable-thinking "${DISABLE_THINKING}" \
  "${INFERENCE_MODEL_ARGS[@]}" \
  --linewise-proof-prompt \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"
mark_phase source_eval_ready
fi

"${VENV_PYTHON}" -m examples.lean_prover_v1.boundary_diagnose \
  --rows-path "${SOURCE_EVAL_DIR}/rows.jsonl" \
  --responses-jsonl "${SOURCE_EVAL_DIR}/responses.jsonl" \
  --eval-report "${SOURCE_EVAL_DIR}/eval_report.json" \
  --output-dir "${BOUNDARY_DIR}" \
  --max-k "${MODEL_PASS_K}" \
  --boundary-max-pass-rate "${ACCEPT_MAX_PASS_RATE}" \
  --too-easy-pass-rate 0.75 \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"
mark_phase boundary_diagnosis_ready

jq -c 'select(.signal_class == "too_hard")' \
  "${BOUNDARY_DIR}/signal_rows.jsonl" > "${BOUNDARY_DIR}/hard_rows.jsonl"
HARD_COUNT="$(wc -l < "${BOUNDARY_DIR}/hard_rows.jsonl")"
echo "[stepwise-teacher] all_failed_source_rows=${HARD_COUNT}"

if (( HARD_COUNT == 0 )); then
  jq -n \
    --arg verdict no_all_failed_source_rows \
    --argjson source_eval "$(cat "${SOURCE_EVAL_DIR}/eval_report.json")" \
    '{verdict:$verdict, criteria:{has_all_failed_source:false}, source_eval:$source_eval}' > "${REPORT_PATH}"
  echo "[stepwise-teacher] no all-failed source rows; report=${REPORT_PATH}"
  mark_phase report_ready
  exit 0
fi

"${VENV_PYTHON}" -m examples.lean_prover_v1.stepwise_diagnose \
  --rows-path "${BOUNDARY_DIR}/hard_rows.jsonl" \
  --responses-jsonl "${SOURCE_EVAL_DIR}/responses.jsonl" \
  --eval-report "${SOURCE_EVAL_DIR}/eval_report.json" \
  --output-dir "${STEPWISE_DIR}" \
  --max-failures "${MAX_HARD_FAILURES}" \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"
mark_phase stepwise_diagnosis_ready
fi

TEACHER_ARGS=(
  -m examples.lean_prover_v1.teacher_mutate_bank
  --failure-records-jsonl "${STEPWISE_DIR}/stepwise_failures.jsonl"
  --eval-report "${SOURCE_EVAL_DIR}/eval_report.json"
  --teacher-model "${TEACHER_MODEL}"
  --teacher-base-url "${TEACHER_BASE_URL}"
  --max-failures "${MAX_HARD_FAILURES}"
  --cases-per-family "${CASES_PER_FAILURE}"
  --model-pass-k "${MODEL_PASS_K}"
  --accept-max-pass-rate "${ACCEPT_MAX_PASS_RATE}"
  --max-api-calls "${MAX_TEACHER_API_CALLS}"
  --max-output-tokens "${TEACHER_MAX_OUTPUT_TOKENS}"
  --api-timeout-seconds "${TEACHER_API_TIMEOUT_SECONDS}"
  --api-max-attempts "${TEACHER_API_MAX_ATTEMPTS}"
  --temperature 0.2
  --cache-dir "${OUTPUT_DIR}/teacher_cache"
  --refine-rounds 0
  --repair-refine-rounds 1
  --cheap-timeout-seconds 10
  --strong-timeout-seconds "${LEAN_TIMEOUT_SECONDS}"
  --well-formed-timeout-seconds 30
  --lean-command "${LEAN_COMMAND}"
  --lean-cwd "${MATHLIB_PROJECT}"
  --max-top-tactic-mass "${MAX_TOP_TACTIC_MASS}"
  --max-error-family-mass "${MAX_ERROR_FAMILY_MASS}"
  --disable-teacher-thinking
  --allow-stepwise-cheap-solved
  --require-verified-repair-transition
  --require-skill-alignment
  --reject-trivial-hypothesis-projection
)

if [[ -n "${REUSE_PIPELINE_RUN}" ]]; then
  "${VENV_PYTHON}" "${TEACHER_ARGS[@]}" \
    --output-dir "${TEACHER_PRELIM_DIR}" \
    --teacher-candidates-jsonl "${REUSE_PIPELINE_RUN}/teacher_prelim/teacher_candidates.jsonl" \
    --generation-summary-json "${REUSE_PIPELINE_RUN}/teacher_prelim/summary.json" \
    --no-require-model-pass
else
  "${VENV_PYTHON}" "${TEACHER_ARGS[@]}" \
    --output-dir "${TEACHER_PRELIM_DIR}" \
    --no-require-model-pass
fi
mark_phase teacher_prelim_ready

jq -c 'select(
  .proof_certificate.strong_prover_solved == true and
  .repair_transition.verified_progress == true and
  .candidate_alignment.aligned == true and
  .candidate_status != "rejected"
)' \
  "${TEACHER_PRELIM_DIR}/teacher_candidates.jsonl" > "${TEACHER_PRELIM_DIR}/verified_candidates.jsonl"
VERIFIED_CANDIDATE_COUNT="$(wc -l < "${TEACHER_PRELIM_DIR}/verified_candidates.jsonl")"
echo "[stepwise-teacher] verified_teacher_candidates=${VERIFIED_CANDIDATE_COUNT}"

if (( VERIFIED_CANDIDATE_COUNT > 0 )); then
  "${VENV_PYTHON}" -m examples.lean_prover_v1.run_inference_lean_prover_v1 \
    --model-source "${MODEL_SOURCE}" \
    --backend vllm \
    --rows-path "${TEACHER_PRELIM_DIR}/verified_candidates.jsonl" \
    --output-dir "${REPAIR_EVAL_DIR}" \
    --output "${REPAIR_EVAL_DIR}/responses.jsonl" \
    --report-output "${REPAIR_EVAL_DIR}/eval_report.json" \
    --rows-output "${REPAIR_EVAL_DIR}/rows.jsonl" \
    --batch-size "${INFERENCE_BATCH_SIZE}" \
    --max-model-len "${INFERENCE_MAX_MODEL_LEN}" \
    --max-new-tokens "${REPAIR_MAX_NEW_TOKENS}" \
    --temperature "${INFERENCE_TEMPERATURE}" \
    --top-p "${INFERENCE_TOP_P}" \
    --top-k "${INFERENCE_TOP_K}" \
    --presence-penalty "${INFERENCE_PRESENCE_PENALTY}" \
    --num-samples "${MODEL_PASS_K}" \
    --tensor-parallel-size "${INFERENCE_TENSOR_PARALLEL_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --seed "${INFERENCE_SEED}" \
    --disable-thinking "${DISABLE_THINKING}" \
    "${INFERENCE_MODEL_ARGS[@]}" \
    --linewise-proof-prompt \
    --lean-command "${LEAN_COMMAND}" \
    --lean-cwd "${MATHLIB_PROJECT}" \
    --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"
else
  mkdir -p "${REPAIR_EVAL_DIR}"
  : > "${REPAIR_EVAL_DIR}/responses.jsonl"
  printf '{"row_count":0,"pass_at_1":0.0,"pass_at_%s":0.0}\n' \
    "${MODEL_PASS_K}" > "${REPAIR_EVAL_DIR}/eval_report.json"
fi
mark_phase repair_eval_ready

"${VENV_PYTHON}" "${TEACHER_ARGS[@]}" \
  --output-dir "${TEACHER_FINAL_DIR}" \
  --teacher-candidates-jsonl "${TEACHER_PRELIM_DIR}/teacher_candidates.jsonl" \
  --generation-summary-json "${TEACHER_PRELIM_DIR}/summary.json" \
  --model-responses-jsonl "${REPAIR_EVAL_DIR}/responses.jsonl" \
  --require-model-pass
mark_phase teacher_final_ready

"${VENV_PYTHON}" -m examples.lean_prover_v1.analyze_stepwise_teacher_decomposition \
  --source-eval-report "${SOURCE_EVAL_DIR}/eval_report.json" \
  --candidate-eval-report "${REPAIR_EVAL_DIR}/eval_report.json" \
  --signal-rows "${BOUNDARY_DIR}/signal_rows.jsonl" \
  --stepwise-failures "${STEPWISE_DIR}/stepwise_failures.jsonl" \
  --teacher-summary "${TEACHER_FINAL_DIR}/summary.json" \
  --teacher-candidates "${TEACHER_FINAL_DIR}/teacher_candidates.jsonl" \
  --model-pass-k "${MODEL_PASS_K}" \
  --output "${REPORT_PATH}"

printf '%s\n' "${REPORT_PATH}" > "${OUTPUT_DIR}/decomposition_report_path.txt"
mark_phase report_ready
echo "[stepwise-teacher] report=${REPORT_PATH}"
