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
RLLM_ROOT="${PROJECT_ROOT}/rllm"
RLLM_VENV="${RLLM_VENV:-${RLLM_ROOT}/.venv}"
VENV_PYTHON="${RLLM_VENV}/bin/python"
MATHLIB_PROJECT="${MATHLIB_PROJECT:-/net/scratch/jiaweizhang/lean_prover_runtime/mathlib4}"
LEAN_COMMAND="${LEAN_COMMAND:-lake env lean}"

: "${RUN_NAME:?RUN_NAME must be set}"

MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3-14B}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/lean_prover_v1/${RUN_NAME}}"
DECLARATIONS_CACHE="${DECLARATIONS_CACHE:-${PROJECT_ROOT}/checkpoints/lean_prover_v1/lean_real_corpus_bank_qwen25_20260711_203917/real_corpus/mathlib_declarations.jsonl}"
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
MAX_HARD_FAILURES="${MAX_HARD_FAILURES:-4}"
CASES_PER_FAILURE="${CASES_PER_FAILURE:-3}"
ACCEPT_MAX_PASS_RATE="${ACCEPT_MAX_PASS_RATE:-0.50}"
TEACHER_MODEL="${TEACHER_MODEL:-deepseek-v4-pro}"
TEACHER_BASE_URL="${TEACHER_BASE_URL:-https://api.deepseek.com}"
TEACHER_MAX_OUTPUT_TOKENS="${TEACHER_MAX_OUTPUT_TOKENS:-4096}"
LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-60}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-2}"
INFERENCE_MAX_MODEL_LEN="${INFERENCE_MAX_MODEL_LEN:-4096}"
INFERENCE_MAX_NEW_TOKENS="${INFERENCE_MAX_NEW_TOKENS:-256}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"

if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "Python environment is missing: ${VENV_PYTHON}" >&2
  exit 1
fi
if [[ ! -s "${DECLARATIONS_CACHE}" ]]; then
  echo "Mathlib declaration cache is missing: ${DECLARATIONS_CACHE}" >&2
  exit 1
fi
if [[ -z "${DEEPSEEK_API_KEY:-}" && -z "${ANTHROPIC_AUTH_TOKEN:-}" ]]; then
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
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

mkdir -p "${OUTPUT_DIR}" "${CORPUS_DIR}" "${PROJECT_ROOT}/logs/lean_prover_v1"
cp "${DECLARATIONS_CACHE}" "${CORPUS_DIR}/mathlib_declarations.jsonl"

echo "[stepwise-teacher] RUN_NAME=${RUN_NAME}"
echo "[stepwise-teacher] MODEL_SOURCE=${MODEL_SOURCE}"
echo "[stepwise-teacher] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[stepwise-teacher] phases=source_eval,boundary_diagnosis,stepwise_replay,teacher_generation,repair_eval"
echo "[stepwise-teacher] training=disabled"

"${VENV_PYTHON}" -m examples.lean_prover_v1.prepare_mathlib_corpus \
  --mathlib-project "${MATHLIB_PROJECT}" \
  --output-dir "${CORPUS_DIR}" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --test-size "${TEST_SIZE}" \
  --workers 8 \
  --lean-command "${LEAN_COMMAND}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}" \
  --no-refresh-export

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
  --temperature 0.8 \
  --top-p 0.95 \
  --num-samples "${MODEL_PASS_K}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --disable-thinking true \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"

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
  --max-api-calls "${MAX_HARD_FAILURES}"
  --max-output-tokens "${TEACHER_MAX_OUTPUT_TOKENS}"
  --temperature 0.2
  --cache-dir "${OUTPUT_DIR}/teacher_cache"
  --refine-rounds 0
  --cheap-timeout-seconds 10
  --strong-timeout-seconds "${LEAN_TIMEOUT_SECONDS}"
  --well-formed-timeout-seconds 30
  --lean-command "${LEAN_COMMAND}"
  --lean-cwd "${MATHLIB_PROJECT}"
  --max-top-tactic-mass 1.0
  --max-error-family-mass 1.0
  --disable-teacher-thinking
  --allow-stepwise-cheap-solved
)

"${VENV_PYTHON}" "${TEACHER_ARGS[@]}" \
  --output-dir "${TEACHER_PRELIM_DIR}" \
  --no-require-model-pass

jq -c 'select(.proof_certificate.strong_prover_solved == true)' \
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
    --max-new-tokens 192 \
    --temperature 0.8 \
    --top-p 0.95 \
    --num-samples "${MODEL_PASS_K}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --disable-thinking true \
    --lean-command "${LEAN_COMMAND}" \
    --lean-cwd "${MATHLIB_PROJECT}" \
    --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"
else
  mkdir -p "${REPAIR_EVAL_DIR}"
  : > "${REPAIR_EVAL_DIR}/responses.jsonl"
  printf '{"row_count":0,"pass_at_1":0.0,"pass_at_%s":0.0}\n' \
    "${MODEL_PASS_K}" > "${REPAIR_EVAL_DIR}/eval_report.json"
fi

"${VENV_PYTHON}" "${TEACHER_ARGS[@]}" \
  --output-dir "${TEACHER_FINAL_DIR}" \
  --teacher-candidates-jsonl "${TEACHER_PRELIM_DIR}/teacher_candidates.jsonl" \
  --model-responses-jsonl "${REPAIR_EVAL_DIR}/responses.jsonl" \
  --require-model-pass

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
echo "[stepwise-teacher] report=${REPORT_PATH}"
