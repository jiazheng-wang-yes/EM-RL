#!/usr/bin/env bash
#SBATCH --job-name=lean_real_corpus_boundary_bank_smoke
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
MATHLIB_PROJECT="${MATHLIB_PROJECT:-/net/scratch/jiaweizhang/lean_prover_runtime/mathlib4}"
RLLM_VENV="${RLLM_VENV:-${RLLM_ROOT}/.venv}"
VENV_PYTHON="${RLLM_VENV}/bin/python"

: "${RUN_NAME:?RUN_NAME must be set}"

MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen2.5-7B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/lean_prover_v1/${RUN_NAME}}"
CORPUS_DIR="${CORPUS_DIR:-${OUTPUT_DIR}/real_corpus}"
CANDIDATE_DIR="${CANDIDATE_DIR:-${OUTPUT_DIR}/real_corpus_candidates}"
INFERENCE_DIR="${INFERENCE_DIR:-${OUTPUT_DIR}/real_corpus_inference}"
BANK_DIR="${BANK_DIR:-${OUTPUT_DIR}/real_corpus_bank}"
DIAGNOSIS_A_DIR="${DIAGNOSIS_A_DIR:-${OUTPUT_DIR}/boundary_diagnosis_a}"
DIAGNOSIS_B_DIR="${DIAGNOSIS_B_DIR:-${OUTPUT_DIR}/boundary_diagnosis_b}"
GATE_REPORT="${GATE_REPORT:-${OUTPUT_DIR}/real_corpus_gate_report.json}"

EXPORT_MAX="${EXPORT_MAX:-1000}"
TRAIN_SIZE="${TRAIN_SIZE:-40}"
VAL_SIZE="${VAL_SIZE:-8}"
TEST_SIZE="${TEST_SIZE:-8}"
MODEL_PASS_K="${MODEL_PASS_K:-4}"
ACCEPT_MAX_PASS_RATE="${ACCEPT_MAX_PASS_RATE:-0.50}"
TEMPLATES_PER_ROW="${TEMPLATES_PER_ROW:-6}"
CANDIDATE_WORKERS="${CANDIDATE_WORKERS:-4}"
INFERENCE_LIMIT="${INFERENCE_LIMIT:--1}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-4}"
INFERENCE_MAX_MODEL_LEN="${INFERENCE_MAX_MODEL_LEN:-2048}"
INFERENCE_MAX_NEW_TOKENS="${INFERENCE_MAX_NEW_TOKENS:-128}"
INFERENCE_TEMPERATURE="${INFERENCE_TEMPERATURE:-0.8}"
INFERENCE_TOP_P="${INFERENCE_TOP_P:-0.95}"
INFERENCE_GPU_MEMORY_UTILIZATION="${INFERENCE_GPU_MEMORY_UTILIZATION:-0.70}"
LEAN_TIMEOUT_SECONDS="${LEAN_TIMEOUT_SECONDS:-30}"
LEAN_COMMAND="${LEAN_COMMAND:-lake env lean}"
MATHLIB_DECLARATIONS_CACHE="${MATHLIB_DECLARATIONS_CACHE:-}"

source "${RLLM_VENV}/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

mkdir -p "${OUTPUT_DIR}" "${CORPUS_DIR}" "${PROJECT_ROOT}/logs/lean_prover_v1"

echo "[real-corpus-bank] RUN_NAME=${RUN_NAME}"
echo "[real-corpus-bank] MODEL_SOURCE=${MODEL_SOURCE}"
echo "[real-corpus-bank] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[real-corpus-bank] MATHLIB_PROJECT=${MATHLIB_PROJECT}"

CORPUS_EXPORT_ARGS=(--refresh-export)
if [[ -n "${MATHLIB_DECLARATIONS_CACHE}" ]]; then
  if [[ ! -s "${MATHLIB_DECLARATIONS_CACHE}" ]]; then
    echo "Mathlib declarations cache is missing or empty: ${MATHLIB_DECLARATIONS_CACHE}" >&2
    exit 1
  fi
  cp "${MATHLIB_DECLARATIONS_CACHE}" "${CORPUS_DIR}/mathlib_declarations.jsonl"
  CORPUS_EXPORT_ARGS=(--no-refresh-export)
fi

"${VENV_PYTHON}" -m examples.lean_prover_v1.prepare_mathlib_corpus \
  --mathlib-project "${MATHLIB_PROJECT}" \
  --output-dir "${CORPUS_DIR}" \
  --export-max "${EXPORT_MAX}" \
  --train-size "${TRAIN_SIZE}" \
  --val-size "${VAL_SIZE}" \
  --test-size "${TEST_SIZE}" \
  --workers "${CANDIDATE_WORKERS}" \
  --lean-command "${LEAN_COMMAND}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}" \
  "${CORPUS_EXPORT_ARGS[@]}"

"${VENV_PYTHON}" -m examples.lean_prover_v1.prepare_real_corpus_candidates \
  --rows-path "${CORPUS_DIR}/mathlib_corpus.jsonl" \
  --output-dir "${CANDIDATE_DIR}" \
  --templates-per-row "${TEMPLATES_PER_ROW}" \
  --workers 8 \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}" \
  --min-unique-statement-rate 0.90

"${VENV_PYTHON}" -m examples.lean_prover_v1.run_inference_lean_prover_v1 \
  --model-source "${MODEL_SOURCE}" \
  --backend vllm \
  --rows-path "${CANDIDATE_DIR}/candidates.jsonl" \
  --limit "${INFERENCE_LIMIT}" \
  --output-dir "${INFERENCE_DIR}" \
  --output "${INFERENCE_DIR}/responses.jsonl" \
  --report-output "${INFERENCE_DIR}/eval_report.json" \
  --rows-output "${INFERENCE_DIR}/rows.jsonl" \
  --batch-size "${INFERENCE_BATCH_SIZE}" \
  --max-model-len "${INFERENCE_MAX_MODEL_LEN}" \
  --max-new-tokens "${INFERENCE_MAX_NEW_TOKENS}" \
  --temperature "${INFERENCE_TEMPERATURE}" \
  --top-p "${INFERENCE_TOP_P}" \
  --num-samples "${MODEL_PASS_K}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${INFERENCE_GPU_MEMORY_UTILIZATION}" \
  --disable-thinking true \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"

"${VENV_PYTHON}" -m examples.lean_prover_v1.real_corpus_bank \
  --rows-path "${INFERENCE_DIR}/rows.jsonl" \
  --responses-jsonl "${INFERENCE_DIR}/responses.jsonl" \
  --eval-report "${INFERENCE_DIR}/eval_report.json" \
  --output-dir "${BANK_DIR}" \
  --model-pass-k "${MODEL_PASS_K}" \
  --accept-max-pass-rate "${ACCEPT_MAX_PASS_RATE}" \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --certificate-timeout-seconds "${LEAN_TIMEOUT_SECONDS}" \
  --model-timeout-seconds "${LEAN_TIMEOUT_SECONDS}" \
  --fail-on-empty-accepted

"${VENV_PYTHON}" -m examples.lean_prover_v1.boundary_diagnose \
  --rows-path "${INFERENCE_DIR}/rows.jsonl" \
  --responses-jsonl "${INFERENCE_DIR}/responses.jsonl" \
  --eval-report "${INFERENCE_DIR}/eval_report.json" \
  --output-dir "${DIAGNOSIS_A_DIR}" \
  --max-k "${MODEL_PASS_K}" \
  --boundary-max-pass-rate "${ACCEPT_MAX_PASS_RATE}" \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"

"${VENV_PYTHON}" -m examples.lean_prover_v1.boundary_diagnose \
  --rows-path "${INFERENCE_DIR}/rows.jsonl" \
  --responses-jsonl "${INFERENCE_DIR}/responses.jsonl" \
  --eval-report "${INFERENCE_DIR}/eval_report.json" \
  --output-dir "${DIAGNOSIS_B_DIR}" \
  --max-k "${MODEL_PASS_K}" \
  --boundary-max-pass-rate "${ACCEPT_MAX_PASS_RATE}" \
  --lean-command "${LEAN_COMMAND}" \
  --lean-cwd "${MATHLIB_PROJECT}" \
  --timeout-seconds "${LEAN_TIMEOUT_SECONDS}"

"${VENV_PYTHON}" -m examples.lean_prover_v1.validate_real_corpus_smoke \
  --corpus-summary "${CORPUS_DIR}/summary.json" \
  --candidate-summary "${CANDIDATE_DIR}/summary.json" \
  --bank-summary "${BANK_DIR}/summary.json" \
  --accepted-jsonl "${BANK_DIR}/accepted.jsonl" \
  --diagnosis-a-summary "${DIAGNOSIS_A_DIR}/skill_family_summary.json" \
  --diagnosis-b-summary "${DIAGNOSIS_B_DIR}/skill_family_summary.json" \
  --diagnosis-a-signal-rows "${DIAGNOSIS_A_DIR}/signal_rows.jsonl" \
  --diagnosis-b-signal-rows "${DIAGNOSIS_B_DIR}/signal_rows.jsonl" \
  --output "${GATE_REPORT}" \
  --min-unique-statement-rate 0.90 \
  --min-model-pass-k "${MODEL_PASS_K}"

echo "${BANK_DIR}/bank.jsonl" > "${OUTPUT_DIR}/real_corpus_bank_path.txt"
echo "${BANK_DIR}/accepted.jsonl" > "${OUTPUT_DIR}/real_corpus_accepted_path.txt"
echo "${GATE_REPORT}" > "${OUTPUT_DIR}/real_corpus_gate_report_path.txt"
echo "[real-corpus-bank] gate passed: ${GATE_REPORT}"
