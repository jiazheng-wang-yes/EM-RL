#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_instruct_2507_lean_prover_v1_rl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
RLLM_VENV="${RLLM_VENV:-${RLLM_ROOT}/.venv}"
VENV_PYTHON="${RLLM_VENV}/bin/python"

source "${RLLM_VENV}/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
mkdir -p "${PROJECT_ROOT}/logs/lean_prover_v1"

: "${RUN_NAME:?RUN_NAME must be set}"

MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/lean_prover_v1/${RUN_NAME}}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-}"
MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_ROLLOUT_TEMPERATURE="${VAL_ROLLOUT_TEMPERATURE:-0.2}"
VAL_ROLLOUT_TOP_P="${VAL_ROLLOUT_TOP_P:-0.95}"
UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-4096}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
SAVE_FREQ="${SAVE_FREQ:-16}"
TEST_FREQ="${TEST_FREQ:--1}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
LEAN_PROVER_V1_TIMEOUT_SECONDS="${LEAN_PROVER_V1_TIMEOUT_SECONDS:-10}"
LEAN_PROVER_V1_MAX_HEARTBEATS="${LEAN_PROVER_V1_MAX_HEARTBEATS:-200000}"
LEAN_PROVER_V1_TRAIN_STATIC_SIZE="${LEAN_PROVER_V1_TRAIN_STATIC_SIZE:-1024}"
LEAN_PROVER_V1_VAL_STATIC_SIZE="${LEAN_PROVER_V1_VAL_STATIC_SIZE:-128}"
LEAN_PROVER_V1_TEST_STATIC_SIZE="${LEAN_PROVER_V1_TEST_STATIC_SIZE:-128}"
LEAN_PROVER_V1_TRAIN_MUTATED_SIZE="${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE:-1024}"
LEAN_PROVER_V1_VAL_MUTATED_SIZE="${LEAN_PROVER_V1_VAL_MUTATED_SIZE:-128}"
LEAN_PROVER_V1_TEST_MUTATED_SIZE="${LEAN_PROVER_V1_TEST_MUTATED_SIZE:-128}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-1}"
LEAN_PROVER_V1_EVAL_AFTER_SOURCE="${LEAN_PROVER_V1_EVAL_AFTER_SOURCE:-model}"
LEAN_PROVER_V1_EVAL_BACKEND="${LEAN_PROVER_V1_EVAL_BACKEND:-vllm}"
LEAN_PROVER_V1_EVAL_LIMIT="${LEAN_PROVER_V1_EVAL_LIMIT:--1}"
LEAN_PROVER_V1_EVAL_NUM_SAMPLES="${LEAN_PROVER_V1_EVAL_NUM_SAMPLES:-1}"
LEAN_PROVER_V1_EVAL_BATCH_SIZE="${LEAN_PROVER_V1_EVAL_BATCH_SIZE:-8}"
LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN="${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN:-${ROLLOUT_MAX_MODEL_LEN}}"
LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS="${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS:-128}"
LEAN_PROVER_V1_EVAL_TEMPERATURE="${LEAN_PROVER_V1_EVAL_TEMPERATURE:-0.0}"
LEAN_PROVER_V1_EVAL_TOP_P="${LEAN_PROVER_V1_EVAL_TOP_P:-1.0}"
LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION="${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION:-0.75}"
LEAN_PROVER_V1_EVAL_DEVICE="${LEAN_PROVER_V1_EVAL_DEVICE:-cuda:0}"
LEAN_PROVER_V1_EVAL_EXPORT_ROOT="${LEAN_PROVER_V1_EVAL_EXPORT_ROOT:-${OUTPUT_DIR}/eval_model_exports}"
LEAN_PROVER_V1_EVAL_MUTATION_BANK_LIMIT="${LEAN_PROVER_V1_EVAL_MUTATION_BANK_LIMIT:-${LEAN_PROVER_V1_EVAL_LIMIT}}"
LEAN_PROVER_V1_RUN_MUTATION_BANK_EVAL="${LEAN_PROVER_V1_RUN_MUTATION_BANK_EVAL:-1}"
RUN_GENERALIZATION_EVAL="${RUN_GENERALIZATION_EVAL:-${RUN_EVAL_AFTER}}"
LEAN_PROVER_V1_GENERALIZATION_TASKS="${LEAN_PROVER_V1_GENERALIZATION_TASKS:-ifeval gsm8k humaneval_instruct mbpp_instruct}"
LEAN_PROVER_V1_GENERALIZATION_LIMIT="${LEAN_PROVER_V1_GENERALIZATION_LIMIT:-200}"
LEAN_PROVER_V1_GENERALIZATION_MAX_GEN_TOKS="${LEAN_PROVER_V1_GENERALIZATION_MAX_GEN_TOKS:-1024}"
LEAN_PROVER_V1_GENERALIZATION_TP_SIZE="${LEAN_PROVER_V1_GENERALIZATION_TP_SIZE:-1}"
LEAN_PROVER_V1_GENERALIZATION_BATCH_SIZE="${LEAN_PROVER_V1_GENERALIZATION_BATCH_SIZE:-auto}"
LEAN_PROVER_V1_GENERALIZATION_MAX_BATCH_SIZE="${LEAN_PROVER_V1_GENERALIZATION_MAX_BATCH_SIZE:-}"
LEAN_PROVER_V1_GENERALIZATION_MAX_MODEL_LEN="${LEAN_PROVER_V1_GENERALIZATION_MAX_MODEL_LEN:-}"
LEAN_PROVER_V1_GENERALIZATION_MAX_NUM_SEQS="${LEAN_PROVER_V1_GENERALIZATION_MAX_NUM_SEQS:-}"
LEAN_PROVER_V1_GENERALIZATION_GPU_MEMORY_UTILIZATION="${LEAN_PROVER_V1_GENERALIZATION_GPU_MEMORY_UTILIZATION:-0.85}"
LEAN_PROVER_V1_GENERALIZATION_TRUST_REMOTE_CODE="${LEAN_PROVER_V1_GENERALIZATION_TRUST_REMOTE_CODE:-1}"
LEAN_PROVER_V1_GENERALIZATION_MISSING_TASK_POLICY="${LEAN_PROVER_V1_GENERALIZATION_MISSING_TASK_POLICY:-warn}"
LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR="${LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR:-${OUTPUT_DIR}/generalization_eval}"
LEAN_PROVER_V1_GENERALIZATION_LARGE_DROP="${LEAN_PROVER_V1_GENERALIZATION_LARGE_DROP:-0.10}"
LEAN_PROVER_V1_GENERALIZATION_USE_CACHE="${LEAN_PROVER_V1_GENERALIZATION_USE_CACHE:-1}"
LEAN_PROVER_V1_GENERALIZATION_ENFORCE_EAGER="${LEAN_PROVER_V1_GENERALIZATION_ENFORCE_EAGER:-0}"
LEAN_PROVER_V1_GENERALIZATION_DISABLE_CUSTOM_ALL_REDUCE="${LEAN_PROVER_V1_GENERALIZATION_DISABLE_CUSTOM_ALL_REDUCE:-0}"
RUN_SELF_MUTATION="${RUN_SELF_MUTATION:-0}"
SELF_MUTATION_ROUNDS="${SELF_MUTATION_ROUNDS:-2}"
SELF_MUTATION_SEEDS_PER_ROUND="${SELF_MUTATION_SEEDS_PER_ROUND:-64}"
SELF_MUTATION_SYMBOLIC_PER_SEED="${SELF_MUTATION_SYMBOLIC_PER_SEED:-2}"
SELF_MUTATION_LLM_PER_SEED="${SELF_MUTATION_LLM_PER_SEED:-0}"
SELF_MUTATION_MODEL_PASS_K="${SELF_MUTATION_MODEL_PASS_K:-4}"
SELF_MUTATION_ACCEPT_MAX_PASS_RATE="${SELF_MUTATION_ACCEPT_MAX_PASS_RATE:-0.35}"
SELF_MUTATION_CHEAP_TIMEOUT_SECONDS="${SELF_MUTATION_CHEAP_TIMEOUT_SECONDS:-5}"
SELF_MUTATION_STRONG_TIMEOUT_SECONDS="${SELF_MUTATION_STRONG_TIMEOUT_SECONDS:-20}"
SELF_MUTATION_WELL_FORMED_TIMEOUT_SECONDS="${SELF_MUTATION_WELL_FORMED_TIMEOUT_SECONDS:-5}"
SELF_MUTATION_OUTPUT_DIR="${SELF_MUTATION_OUTPUT_DIR:-${OUTPUT_DIR}/self_mutation}"
SELF_MUTATION_REQUIRE_MODEL_PASS="${SELF_MUTATION_REQUIRE_MODEL_PASS:-1}"
SELF_MUTATION_AUTO_MODEL_RESPONSES="${SELF_MUTATION_AUTO_MODEL_RESPONSES:-1}"
SELF_MUTATION_MODEL_PASS_MODEL_SOURCE="${SELF_MUTATION_MODEL_PASS_MODEL_SOURCE:-${MODEL_SOURCE}}"
SELF_MUTATION_MODEL_PASS_BACKEND="${SELF_MUTATION_MODEL_PASS_BACKEND:-vllm}"
SELF_MUTATION_MODEL_PASS_BATCH_SIZE="${SELF_MUTATION_MODEL_PASS_BATCH_SIZE:-${LEAN_PROVER_V1_EVAL_BATCH_SIZE}}"
SELF_MUTATION_MODEL_PASS_MAX_MODEL_LEN="${SELF_MUTATION_MODEL_PASS_MAX_MODEL_LEN:-${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN}}"
SELF_MUTATION_MODEL_PASS_MAX_NEW_TOKENS="${SELF_MUTATION_MODEL_PASS_MAX_NEW_TOKENS:-${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS}}"
SELF_MUTATION_MODEL_PASS_TEMPERATURE="${SELF_MUTATION_MODEL_PASS_TEMPERATURE:-0.8}"
SELF_MUTATION_MODEL_PASS_TOP_P="${SELF_MUTATION_MODEL_PASS_TOP_P:-0.95}"
SELF_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION="${SELF_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION:-${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION}}"
SELF_MUTATION_MODEL_PASS_LIMIT="${SELF_MUTATION_MODEL_PASS_LIMIT:-}"
SELF_MUTATION_MAX_TOP_TACTIC_MASS="${SELF_MUTATION_MAX_TOP_TACTIC_MASS:-0.55}"
SELF_MUTATION_MAX_MUTATION_TYPE_MASS="${SELF_MUTATION_MAX_MUTATION_TYPE_MASS:-0.50}"
SELF_MUTATION_EMPTY_ROUND_LIMIT="${SELF_MUTATION_EMPTY_ROUND_LIMIT:-2}"
SELF_MUTATION_SEED_CORPUS="${SELF_MUTATION_SEED_CORPUS:-}"
SELF_MUTATION_LLM_CANDIDATE_JSONL="${SELF_MUTATION_LLM_CANDIDATE_JSONL:-}"
SELF_MUTATION_MODEL_RESPONSES_JSONL="${SELF_MUTATION_MODEL_RESPONSES_JSONL:-}"
RUN_ERROR_MUTATION="${RUN_ERROR_MUTATION:-0}"
ERROR_MUTATION_OUTPUT_DIR="${ERROR_MUTATION_OUTPUT_DIR:-${OUTPUT_DIR}/error_mutation}"
ERROR_MUTATION_MAX_FAILURES="${ERROR_MUTATION_MAX_FAILURES:-64}"
ERROR_MUTATION_CASES_PER_FAMILY="${ERROR_MUTATION_CASES_PER_FAMILY:-4}"
ERROR_MUTATION_MODEL_PASS_K="${ERROR_MUTATION_MODEL_PASS_K:-4}"
ERROR_MUTATION_ACCEPT_MAX_PASS_RATE="${ERROR_MUTATION_ACCEPT_MAX_PASS_RATE:-0.35}"
ERROR_MUTATION_DIAGNOSIS_MODEL_SOURCE="${ERROR_MUTATION_DIAGNOSIS_MODEL_SOURCE:-${MODEL_SOURCE}}"
ERROR_MUTATION_CHEAP_TIMEOUT_SECONDS="${ERROR_MUTATION_CHEAP_TIMEOUT_SECONDS:-${SELF_MUTATION_CHEAP_TIMEOUT_SECONDS}}"
ERROR_MUTATION_STRONG_TIMEOUT_SECONDS="${ERROR_MUTATION_STRONG_TIMEOUT_SECONDS:-${SELF_MUTATION_STRONG_TIMEOUT_SECONDS}}"
ERROR_MUTATION_WELL_FORMED_TIMEOUT_SECONDS="${ERROR_MUTATION_WELL_FORMED_TIMEOUT_SECONDS:-${SELF_MUTATION_WELL_FORMED_TIMEOUT_SECONDS}}"
ERROR_MUTATION_MODEL_PASS_BACKEND="${ERROR_MUTATION_MODEL_PASS_BACKEND:-${LEAN_PROVER_V1_EVAL_BACKEND}}"
ERROR_MUTATION_MODEL_PASS_BATCH_SIZE="${ERROR_MUTATION_MODEL_PASS_BATCH_SIZE:-${LEAN_PROVER_V1_EVAL_BATCH_SIZE}}"
ERROR_MUTATION_MODEL_PASS_MAX_MODEL_LEN="${ERROR_MUTATION_MODEL_PASS_MAX_MODEL_LEN:-${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN}}"
ERROR_MUTATION_MODEL_PASS_MAX_NEW_TOKENS="${ERROR_MUTATION_MODEL_PASS_MAX_NEW_TOKENS:-${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS}}"
ERROR_MUTATION_MODEL_PASS_TEMPERATURE="${ERROR_MUTATION_MODEL_PASS_TEMPERATURE:-0.8}"
ERROR_MUTATION_MODEL_PASS_TOP_P="${ERROR_MUTATION_MODEL_PASS_TOP_P:-0.95}"
ERROR_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION="${ERROR_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION:-${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION}}"
ERROR_MUTATION_MAX_TOP_TACTIC_MASS="${ERROR_MUTATION_MAX_TOP_TACTIC_MASS:-0.40}"
ERROR_MUTATION_MAX_ERROR_FAMILY_MASS="${ERROR_MUTATION_MAX_ERROR_FAMILY_MASS:-0.30}"
ERROR_MUTATION_MAX_TEMPLATE_MASS="${ERROR_MUTATION_MAX_TEMPLATE_MASS:-0.15}"
RUN_TEACHER_MUTATION="${RUN_TEACHER_MUTATION:-0}"
TEACHER_MUTATION_ROUNDS="${TEACHER_MUTATION_ROUNDS:-2}"
TEACHER_MUTATION_MODEL="${TEACHER_MUTATION_MODEL:-deepseek-v4-pro}"
TEACHER_MUTATION_BASE_URL="${TEACHER_MUTATION_BASE_URL:-https://api.deepseek.com}"
TEACHER_MUTATION_MAX_FAILURES="${TEACHER_MUTATION_MAX_FAILURES:-32}"
TEACHER_MUTATION_CASES_PER_FAMILY="${TEACHER_MUTATION_CASES_PER_FAMILY:-3}"
TEACHER_MUTATION_MODEL_PASS_K="${TEACHER_MUTATION_MODEL_PASS_K:-4}"
TEACHER_MUTATION_ACCEPT_MAX_PASS_RATE="${TEACHER_MUTATION_ACCEPT_MAX_PASS_RATE:-0.35}"
TEACHER_MUTATION_MAX_API_CALLS="${TEACHER_MUTATION_MAX_API_CALLS:-64}"
TEACHER_MUTATION_MAX_OUTPUT_TOKENS="${TEACHER_MUTATION_MAX_OUTPUT_TOKENS:-4096}"
TEACHER_MUTATION_CACHE_DIR="${TEACHER_MUTATION_CACHE_DIR:-${OUTPUT_DIR}/teacher_cache}"
TEACHER_MUTATION_REFINE_ROUNDS="${TEACHER_MUTATION_REFINE_ROUNDS:-1}"
TEACHER_MUTATION_TEMPERATURE="${TEACHER_MUTATION_TEMPERATURE:-0.2}"
TEACHER_MUTATION_OUTPUT_DIR="${TEACHER_MUTATION_OUTPUT_DIR:-${OUTPUT_DIR}/teacher_mutation}"
TEACHER_MUTATION_RAW_JSONL="${TEACHER_MUTATION_RAW_JSONL:-}"
TEACHER_MUTATION_CHEAP_TIMEOUT_SECONDS="${TEACHER_MUTATION_CHEAP_TIMEOUT_SECONDS:-${SELF_MUTATION_CHEAP_TIMEOUT_SECONDS}}"
TEACHER_MUTATION_STRONG_TIMEOUT_SECONDS="${TEACHER_MUTATION_STRONG_TIMEOUT_SECONDS:-${SELF_MUTATION_STRONG_TIMEOUT_SECONDS}}"
TEACHER_MUTATION_WELL_FORMED_TIMEOUT_SECONDS="${TEACHER_MUTATION_WELL_FORMED_TIMEOUT_SECONDS:-${SELF_MUTATION_WELL_FORMED_TIMEOUT_SECONDS}}"
TEACHER_MUTATION_MODEL_PASS_BACKEND="${TEACHER_MUTATION_MODEL_PASS_BACKEND:-${LEAN_PROVER_V1_EVAL_BACKEND}}"
TEACHER_MUTATION_MODEL_PASS_BATCH_SIZE="${TEACHER_MUTATION_MODEL_PASS_BATCH_SIZE:-${LEAN_PROVER_V1_EVAL_BATCH_SIZE}}"
TEACHER_MUTATION_MODEL_PASS_MAX_MODEL_LEN="${TEACHER_MUTATION_MODEL_PASS_MAX_MODEL_LEN:-${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN}}"
TEACHER_MUTATION_MODEL_PASS_MAX_NEW_TOKENS="${TEACHER_MUTATION_MODEL_PASS_MAX_NEW_TOKENS:-${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS}}"
TEACHER_MUTATION_MODEL_PASS_TEMPERATURE="${TEACHER_MUTATION_MODEL_PASS_TEMPERATURE:-0.8}"
TEACHER_MUTATION_MODEL_PASS_TOP_P="${TEACHER_MUTATION_MODEL_PASS_TOP_P:-0.95}"
TEACHER_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION="${TEACHER_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION:-${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION}}"
TEACHER_MUTATION_REFLECTION_LIMIT="${TEACHER_MUTATION_REFLECTION_LIMIT:-${LEAN_PROVER_V1_EVAL_LIMIT}}"
TEACHER_MUTATION_REFLECTION_SPLIT="${TEACHER_MUTATION_REFLECTION_SPLIT:-val}"
TEACHER_MUTATION_MAX_TOP_TACTIC_MASS="${TEACHER_MUTATION_MAX_TOP_TACTIC_MASS:-0.40}"
TEACHER_MUTATION_MAX_ERROR_FAMILY_MASS="${TEACHER_MUTATION_MAX_ERROR_FAMILY_MASS:-0.30}"
TEACHER_MUTATION_DISABLE_THINKING="${TEACHER_MUTATION_DISABLE_THINKING:-1}"
TEACHER_MUTATION_USE_STEPWISE_DIAGNOSIS="${TEACHER_MUTATION_USE_STEPWISE_DIAGNOSIS:-1}"
TEACHER_MUTATION_ALLOW_STEPWISE_CHEAP_SOLVED="${TEACHER_MUTATION_ALLOW_STEPWISE_CHEAP_SOLVED:-1}"
RUN_SKILL_BOUNDARY_CURRICULUM="${RUN_SKILL_BOUNDARY_CURRICULUM:-0}"
SKILL_BOUNDARY_ROUNDS="${SKILL_BOUNDARY_ROUNDS:-2}"
SKILL_BOUNDARY_OUTPUT_DIR="${SKILL_BOUNDARY_OUTPUT_DIR:-${OUTPUT_DIR}/skill_boundary}"
SKILL_BOUNDARY_MAX_SUCCESS_ROWS="${SKILL_BOUNDARY_MAX_SUCCESS_ROWS:-32}"
SKILL_BOUNDARY_MAX_FAILURE_ROWS="${SKILL_BOUNDARY_MAX_FAILURE_ROWS:-32}"
SKILL_BOUNDARY_CASES_PER_ROW="${SKILL_BOUNDARY_CASES_PER_ROW:-3}"
SKILL_BOUNDARY_MODEL_PASS_K="${SKILL_BOUNDARY_MODEL_PASS_K:-4}"
SKILL_BOUNDARY_ACCEPT_MAX_PASS_RATE="${SKILL_BOUNDARY_ACCEPT_MAX_PASS_RATE:-0.35}"
SKILL_BOUNDARY_STATIC_FLOOR="${SKILL_BOUNDARY_STATIC_FLOOR:-0.50}"
SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE="${SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE:-1}"
SKILL_BOUNDARY_TEACHER_MODEL="${SKILL_BOUNDARY_TEACHER_MODEL:-deepseek-v4-pro}"
SKILL_BOUNDARY_TEACHER_BASE_URL="${SKILL_BOUNDARY_TEACHER_BASE_URL:-https://api.deepseek.com}"
SKILL_BOUNDARY_TEACHER_MAX_API_CALLS="${SKILL_BOUNDARY_TEACHER_MAX_API_CALLS:-64}"
SKILL_BOUNDARY_TEACHER_MAX_OUTPUT_TOKENS="${SKILL_BOUNDARY_TEACHER_MAX_OUTPUT_TOKENS:-4096}"
SKILL_BOUNDARY_TEACHER_TEMPERATURE="${SKILL_BOUNDARY_TEACHER_TEMPERATURE:-0.2}"
SKILL_BOUNDARY_TEACHER_CACHE_DIR="${SKILL_BOUNDARY_TEACHER_CACHE_DIR:-${OUTPUT_DIR}/skill_boundary_teacher_cache}"
SKILL_BOUNDARY_TEACHER_RAW_JSONL="${SKILL_BOUNDARY_TEACHER_RAW_JSONL:-}"
SKILL_BOUNDARY_TEACHER_DISABLE_THINKING="${SKILL_BOUNDARY_TEACHER_DISABLE_THINKING:-1}"
SKILL_BOUNDARY_CHEAP_TIMEOUT_SECONDS="${SKILL_BOUNDARY_CHEAP_TIMEOUT_SECONDS:-${SELF_MUTATION_CHEAP_TIMEOUT_SECONDS}}"
SKILL_BOUNDARY_STRONG_TIMEOUT_SECONDS="${SKILL_BOUNDARY_STRONG_TIMEOUT_SECONDS:-${SELF_MUTATION_STRONG_TIMEOUT_SECONDS}}"
SKILL_BOUNDARY_WELL_FORMED_TIMEOUT_SECONDS="${SKILL_BOUNDARY_WELL_FORMED_TIMEOUT_SECONDS:-${SELF_MUTATION_WELL_FORMED_TIMEOUT_SECONDS}}"
SKILL_BOUNDARY_MODEL_PASS_BACKEND="${SKILL_BOUNDARY_MODEL_PASS_BACKEND:-${LEAN_PROVER_V1_EVAL_BACKEND}}"
SKILL_BOUNDARY_MODEL_PASS_BATCH_SIZE="${SKILL_BOUNDARY_MODEL_PASS_BATCH_SIZE:-${LEAN_PROVER_V1_EVAL_BATCH_SIZE}}"
SKILL_BOUNDARY_MODEL_PASS_MAX_MODEL_LEN="${SKILL_BOUNDARY_MODEL_PASS_MAX_MODEL_LEN:-${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN}}"
SKILL_BOUNDARY_MODEL_PASS_MAX_NEW_TOKENS="${SKILL_BOUNDARY_MODEL_PASS_MAX_NEW_TOKENS:-${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS}}"
SKILL_BOUNDARY_MODEL_PASS_TEMPERATURE="${SKILL_BOUNDARY_MODEL_PASS_TEMPERATURE:-0.8}"
SKILL_BOUNDARY_MODEL_PASS_TOP_P="${SKILL_BOUNDARY_MODEL_PASS_TOP_P:-0.95}"
SKILL_BOUNDARY_MODEL_PASS_GPU_MEMORY_UTILIZATION="${SKILL_BOUNDARY_MODEL_PASS_GPU_MEMORY_UTILIZATION:-${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION}}"
SKILL_BOUNDARY_REFLECTION_LIMIT="${SKILL_BOUNDARY_REFLECTION_LIMIT:-${LEAN_PROVER_V1_EVAL_LIMIT}}"
SKILL_BOUNDARY_REFLECTION_SPLIT="${SKILL_BOUNDARY_REFLECTION_SPLIT:-val}"
SKILL_BOUNDARY_DIAGNOSE_MAX_K="${SKILL_BOUNDARY_DIAGNOSE_MAX_K:-${SKILL_BOUNDARY_MODEL_PASS_K}}"
SKILL_BOUNDARY_TOO_EASY_PASS_RATE="${SKILL_BOUNDARY_TOO_EASY_PASS_RATE:-0.75}"
SKILL_BOUNDARY_BOUNDARY_MAX_PASS_RATE="${SKILL_BOUNDARY_BOUNDARY_MAX_PASS_RATE:-${SKILL_BOUNDARY_ACCEPT_MAX_PASS_RATE}}"
SKILL_BOUNDARY_MAX_TOP_TACTIC_MASS="${SKILL_BOUNDARY_MAX_TOP_TACTIC_MASS:-0.40}"
SKILL_BOUNDARY_MAX_FAMILY_MASS="${SKILL_BOUNDARY_MAX_FAMILY_MASS:-0.30}"
SKILL_BOUNDARY_SUCCESS_EXTENSION_WEIGHT="${SKILL_BOUNDARY_SUCCESS_EXTENSION_WEIGHT:-0.35}"
SKILL_BOUNDARY_FAILURE_BRIDGE_WEIGHT="${SKILL_BOUNDARY_FAILURE_BRIDGE_WEIGHT:-0.15}"
REQUIRE_FINAL_CHECKPOINT_MATCH="${REQUIRE_FINAL_CHECKPOINT_MATCH:-1}"

case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) DISABLE_THINKING_BOOL=false ;;
  *)
    echo "DISABLE_THINKING must be a boolean value, got: ${DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

case "${LEAN_PROVER_V1_ALLOW_SYNTHETIC:-1}" in
  1|true|TRUE|yes|YES) LEAN_ALLOW_SYNTHETIC_BOOL=true ;;
  0|false|FALSE|no|NO) LEAN_ALLOW_SYNTHETIC_BOOL=false ;;
  *)
    echo "LEAN_PROVER_V1_ALLOW_SYNTHETIC must be a boolean value, got: ${LEAN_PROVER_V1_ALLOW_SYNTHETIC}" >&2
    exit 1
    ;;
esac

case "${RUN_SELF_MUTATION}" in
  1|true|TRUE|yes|YES) RUN_SELF_MUTATION_BOOL=true ;;
  0|false|FALSE|no|NO) RUN_SELF_MUTATION_BOOL=false ;;
  *)
    echo "RUN_SELF_MUTATION must be a boolean value, got: ${RUN_SELF_MUTATION}" >&2
    exit 1
    ;;
esac

case "${LEAN_PROVER_V1_RUN_MUTATION_BANK_EVAL}" in
  1|true|TRUE|yes|YES) LEAN_PROVER_V1_RUN_MUTATION_BANK_EVAL_BOOL=true ;;
  0|false|FALSE|no|NO) LEAN_PROVER_V1_RUN_MUTATION_BANK_EVAL_BOOL=false ;;
  *)
    echo "LEAN_PROVER_V1_RUN_MUTATION_BANK_EVAL must be a boolean value, got: ${LEAN_PROVER_V1_RUN_MUTATION_BANK_EVAL}" >&2
    exit 1
    ;;
esac

case "${RUN_ERROR_MUTATION}" in
  1|true|TRUE|yes|YES) RUN_ERROR_MUTATION_BOOL=true ;;
  0|false|FALSE|no|NO) RUN_ERROR_MUTATION_BOOL=false ;;
  *)
    echo "RUN_ERROR_MUTATION must be a boolean value, got: ${RUN_ERROR_MUTATION}" >&2
    exit 1
    ;;
esac

case "${RUN_TEACHER_MUTATION}" in
  1|true|TRUE|yes|YES) RUN_TEACHER_MUTATION_BOOL=true ;;
  0|false|FALSE|no|NO) RUN_TEACHER_MUTATION_BOOL=false ;;
  *)
    echo "RUN_TEACHER_MUTATION must be a boolean value, got: ${RUN_TEACHER_MUTATION}" >&2
    exit 1
    ;;
esac

case "${TEACHER_MUTATION_DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) TEACHER_MUTATION_DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) TEACHER_MUTATION_DISABLE_THINKING_BOOL=false ;;
  *)
    echo "TEACHER_MUTATION_DISABLE_THINKING must be a boolean value, got: ${TEACHER_MUTATION_DISABLE_THINKING}" >&2
    exit 1
    ;;
esac
case "${TEACHER_MUTATION_USE_STEPWISE_DIAGNOSIS}" in
  1|true|TRUE|yes|YES) TEACHER_MUTATION_USE_STEPWISE_DIAGNOSIS_BOOL=true ;;
  0|false|FALSE|no|NO) TEACHER_MUTATION_USE_STEPWISE_DIAGNOSIS_BOOL=false ;;
  *)
    echo "TEACHER_MUTATION_USE_STEPWISE_DIAGNOSIS must be a boolean value, got: ${TEACHER_MUTATION_USE_STEPWISE_DIAGNOSIS}" >&2
    exit 1
    ;;
esac

case "${TEACHER_MUTATION_ALLOW_STEPWISE_CHEAP_SOLVED}" in
  1|true|TRUE|yes|YES) TEACHER_MUTATION_ALLOW_STEPWISE_CHEAP_SOLVED_BOOL=true ;;
  0|false|FALSE|no|NO) TEACHER_MUTATION_ALLOW_STEPWISE_CHEAP_SOLVED_BOOL=false ;;
  *)
    echo "TEACHER_MUTATION_ALLOW_STEPWISE_CHEAP_SOLVED must be a boolean value, got: ${TEACHER_MUTATION_ALLOW_STEPWISE_CHEAP_SOLVED}" >&2
    exit 2
    ;;
esac

case "${RUN_SKILL_BOUNDARY_CURRICULUM}" in
  1|true|TRUE|yes|YES) RUN_SKILL_BOUNDARY_CURRICULUM_BOOL=true ;;
  0|false|FALSE|no|NO) RUN_SKILL_BOUNDARY_CURRICULUM_BOOL=false ;;
  *)
    echo "RUN_SKILL_BOUNDARY_CURRICULUM must be a boolean value, got: ${RUN_SKILL_BOUNDARY_CURRICULUM}" >&2
    exit 1
    ;;
esac

case "${SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE}" in
  1|true|TRUE|yes|YES) SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE_BOOL=true ;;
  0|false|FALSE|no|NO) SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE_BOOL=false ;;
  *)
    echo "SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE must be a boolean value, got: ${SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE}" >&2
    exit 1
    ;;
esac

case "${SKILL_BOUNDARY_TEACHER_DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) SKILL_BOUNDARY_TEACHER_DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) SKILL_BOUNDARY_TEACHER_DISABLE_THINKING_BOOL=false ;;
  *)
    echo "SKILL_BOUNDARY_TEACHER_DISABLE_THINKING must be a boolean value, got: ${SKILL_BOUNDARY_TEACHER_DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

case "${SELF_MUTATION_REQUIRE_MODEL_PASS}" in
  1|true|TRUE|yes|YES) SELF_MUTATION_REQUIRE_MODEL_PASS_BOOL=true ;;
  0|false|FALSE|no|NO) SELF_MUTATION_REQUIRE_MODEL_PASS_BOOL=false ;;
  *)
    echo "SELF_MUTATION_REQUIRE_MODEL_PASS must be a boolean value, got: ${SELF_MUTATION_REQUIRE_MODEL_PASS}" >&2
    exit 1
    ;;
esac

case "${SELF_MUTATION_AUTO_MODEL_RESPONSES}" in
  1|true|TRUE|yes|YES) SELF_MUTATION_AUTO_MODEL_RESPONSES_BOOL=true ;;
  0|false|FALSE|no|NO) SELF_MUTATION_AUTO_MODEL_RESPONSES_BOOL=false ;;
  *)
    echo "SELF_MUTATION_AUTO_MODEL_RESPONSES must be a boolean value, got: ${SELF_MUTATION_AUTO_MODEL_RESPONSES}" >&2
    exit 1
    ;;
esac

unset ROCR_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_NAMESPACE
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID:-manual}"
export TMPDIR="${RAY_TMPDIR}"
ray stop --force >/dev/null 2>&1 || true
rm -rf "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}" "${OUTPUT_DIR}"
export OUTPUT_DIR MODEL_SOURCE
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

cd "${RLLM_ROOT}"

build_lean_hydra_args() {
  LEAN_HYDRA_ARGS=(
    +lean_prover.timeout_seconds="${LEAN_PROVER_V1_TIMEOUT_SECONDS}"
    +lean_prover.max_heartbeats="${LEAN_PROVER_V1_MAX_HEARTBEATS}"
    +lean_prover.allow_synthetic="${LEAN_ALLOW_SYNTHETIC_BOOL}"
    +lean_prover.train_static_size="${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
    +lean_prover.val_static_size="${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
    +lean_prover.test_static_size="${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
    +lean_prover.train_mutated_size="${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
    +lean_prover.val_mutated_size="${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
    +lean_prover.test_mutated_size="${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
  )

  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    LEAN_HYDRA_ARGS+=(+lean_prover.lean_command="${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    LEAN_HYDRA_ARGS+=(+lean_prover.lean_cwd="${LEAN_PROVER_V1_LEAN_CWD}")
  fi
  if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
    LEAN_HYDRA_ARGS+=(+lean_prover.static_corpus_path="${LEAN_PROVER_V1_STATIC_CORPUS}")
  fi
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    LEAN_HYDRA_ARGS+=(+lean_prover.mutation_bank_path="${LEAN_PROVER_V1_MUTATION_BANK}")
  fi
}

run_lean_training() {
  local current_resume_mode="${CURRENT_RESUME_MODE:-disable}"
  local current_total_epochs="${CURRENT_TOTAL_EPOCHS:-${TOTAL_EPOCHS}}"
  build_lean_hydra_args
  "${VENV_PYTHON}" -m examples.lean_prover_v1.train_lean_prover_v1 \
    algorithm.adv_estimator=grpo \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size="${VAL_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.model.path="${MODEL_SOURCE}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.clip_ratio_high=0.2 \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MEGABYTES}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_ROLLOUT_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_ROLLOUT_TOP_P}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='lean-prover-v1' \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
    trainer.resume_mode="${current_resume_mode}" \
    trainer.total_epochs="${current_total_epochs}" \
    rllm.agent.max_steps=1 \
    rllm.disable_thinking="${DISABLE_THINKING_BOOL}" \
    rllm.stepwise_advantage.enable=False \
    rllm.rejection_sample.enable=False \
    "${LEAN_HYDRA_ARGS[@]}" \
    "$@"
}

FINAL_CHECKPOINT_ACTOR_DIR="${FINAL_CHECKPOINT_ACTOR_DIR:-}"
FINAL_MODEL_PATH="${FINAL_MODEL_PATH:-}"

reset_final_model_resolution() {
  FINAL_CHECKPOINT_ACTOR_DIR=""
  FINAL_MODEL_PATH=""
  export FINAL_CHECKPOINT_ACTOR_DIR FINAL_MODEL_PATH
}

resolve_final_checkpoint_actor_dir() {
  if [[ -n "${FINAL_CHECKPOINT_ACTOR_DIR}" ]]; then
    return
  fi
  local checkpoint_metadata
  checkpoint_metadata="$("${VENV_PYTHON}" - <<'PY'
import json
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
tracker = output_dir / "latest_checkpointed_iteration.txt"
tracker_step = None
if tracker.exists():
    text = tracker.read_text(encoding="utf-8").strip()
    if text:
        tracker_step = int(text)

candidates = []
for path in output_dir.glob("global_step_*"):
    if not path.is_dir():
        continue
    try:
        step = int(path.name.split("global_step_", 1)[1])
    except Exception:
        continue
    actor_dir = path / "actor"
    if actor_dir.is_dir() and any(actor_dir.glob("model_world_size_*_rank_*.pt")):
        candidates.append(step)

if not candidates:
    raise SystemExit("No completed global_step_* checkpoint with actor shards found.")

max_step = max(candidates)
latest_step = tracker_step or max_step
final_step_path = output_dir / "final_completed_training_step.txt"
final_step = None
if final_step_path.exists():
    text = final_step_path.read_text(encoding="utf-8").strip()
    if text:
        final_step = int(text)

print(json.dumps({"latest_step": latest_step, "max_step": max_step, "final_step": final_step}))
PY
)"
  local latest_iteration max_iteration final_completed_step
  latest_iteration="$("${VENV_PYTHON}" - "${checkpoint_metadata}" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
print(payload["latest_step"])
PY
)"
  max_iteration="$("${VENV_PYTHON}" - "${checkpoint_metadata}" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
print(payload["max_step"])
PY
)"
  final_completed_step="$("${VENV_PYTHON}" - "${checkpoint_metadata}" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
print(payload.get("final_step") or "")
PY
)"
  if [[ "${latest_iteration}" != "${max_iteration}" ]]; then
    echo "latest_checkpointed_iteration.txt points to ${latest_iteration}, but max actor checkpoint is ${max_iteration}." >&2
    exit 1
  fi
  case "${REQUIRE_FINAL_CHECKPOINT_MATCH}" in
    1|true|TRUE|yes|YES)
      if [[ -z "${final_completed_step}" ]]; then
        echo "final_completed_training_step.txt is missing; refusing to run reflection on an unguarded checkpoint." >&2
        exit 1
      fi
      if [[ "${latest_iteration}" != "${final_completed_step}" ]]; then
        echo "Final actor checkpoint global_step_${latest_iteration} does not match completed training step ${final_completed_step}." >&2
        exit 1
      fi
      ;;
  esac
  FINAL_CHECKPOINT_ACTOR_DIR="${OUTPUT_DIR}/global_step_${latest_iteration}/actor"
  printf '%s\n' "${FINAL_CHECKPOINT_ACTOR_DIR}" > "${OUTPUT_DIR}/final_checkpoint_actor_dir.txt"
  export FINAL_CHECKPOINT_ACTOR_DIR
}

materialize_final_model_for_eval() {
  if [[ -n "${FINAL_MODEL_PATH}" && -e "${FINAL_MODEL_PATH}" ]]; then
    return
  fi
  resolve_final_checkpoint_actor_dir
  local final_eval_source="${FINAL_CHECKPOINT_ACTOR_DIR}"
  if [[ -d "${FINAL_CHECKPOINT_ACTOR_DIR}/lora_adapter" ]]; then
    final_eval_source="${FINAL_CHECKPOINT_ACTOR_DIR}/lora_adapter"
  fi
  export FINAL_EVAL_SOURCE="${final_eval_source}"
  export FINAL_CHECKPOINT_ACTOR_DIR LEAN_PROVER_V1_EVAL_EXPORT_ROOT MODEL_SOURCE
  FINAL_MODEL_PATH="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import os
import torch

print(
    materialize_model_for_vllm(
        source=os.environ["FINAL_EVAL_SOURCE"],
        export_root=os.environ["LEAN_PROVER_V1_EVAL_EXPORT_ROOT"],
        base_model=os.environ.get("MODEL_SOURCE"),
        tokenizer_source=os.path.join(os.environ["FINAL_CHECKPOINT_ACTOR_DIR"], "huggingface"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"
  printf '%s\n' "${FINAL_MODEL_PATH}" > "${OUTPUT_DIR}/final_model_path.txt"
  export FINAL_MODEL_PATH
}

run_mutation_round() {
  local round_index="$1"
  local round_dir="${SELF_MUTATION_OUTPUT_DIR}/round_${round_index}"
  local -a mutate_args=(
    -m examples.lean_prover_v1.mutate_bank
    --output-dir "${round_dir}"
    --round-index "${round_index}"
    --seed-split train_static
    --seeds-per-round "${SELF_MUTATION_SEEDS_PER_ROUND}"
    --symbolic-per-seed "${SELF_MUTATION_SYMBOLIC_PER_SEED}"
    --llm-candidates-per-seed "${SELF_MUTATION_LLM_PER_SEED}"
    --cheap-timeout-seconds "${SELF_MUTATION_CHEAP_TIMEOUT_SECONDS}"
    --strong-timeout-seconds "${SELF_MUTATION_STRONG_TIMEOUT_SECONDS}"
    --well-formed-timeout-seconds "${SELF_MUTATION_WELL_FORMED_TIMEOUT_SECONDS}"
    --accept-max-pass-rate "${SELF_MUTATION_ACCEPT_MAX_PASS_RATE}"
    --model-pass-k "${SELF_MUTATION_MODEL_PASS_K}"
    --max-top-tactic-mass "${SELF_MUTATION_MAX_TOP_TACTIC_MASS}"
    --max-mutation-type-mass "${SELF_MUTATION_MAX_MUTATION_TYPE_MASS}"
    --generation-model "${MODEL_SOURCE}"
    --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
    --val-static-size "${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
    --test-static-size "${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
    --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
    --val-mutated-size "${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
    --test-mutated-size "${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
  )
  if [[ "${LEAN_ALLOW_SYNTHETIC_BOOL}" == "true" ]]; then
    mutate_args+=(--allow-synthetic)
  else
    mutate_args+=(--no-allow-synthetic)
  fi
  if [[ -n "${SELF_MUTATION_SEED_CORPUS}" ]]; then
    mutate_args+=(--seed-corpus-path "${SELF_MUTATION_SEED_CORPUS}")
  fi
  if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
    mutate_args+=(--static-corpus-path "${LEAN_PROVER_V1_STATIC_CORPUS}")
  fi
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    mutate_args+=(--existing-mutation-bank-path "${LEAN_PROVER_V1_MUTATION_BANK}")
  fi
  if [[ -n "${SELF_MUTATION_LLM_CANDIDATE_JSONL}" ]]; then
    mutate_args+=(--llm-candidate-jsonl "${SELF_MUTATION_LLM_CANDIDATE_JSONL}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    mutate_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    mutate_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
  fi
  local model_responses_jsonl="${SELF_MUTATION_MODEL_RESPONSES_JSONL}"
  if [[ "${SELF_MUTATION_REQUIRE_MODEL_PASS_BOOL}" == "true" && -z "${model_responses_jsonl}" && "${SELF_MUTATION_AUTO_MODEL_RESPONSES_BOOL}" == "true" ]]; then
    "${VENV_PYTHON}" "${mutate_args[@]}" --no-require-model-pass
    local candidate_path="${round_dir}/candidates.jsonl"
    if [[ -s "${candidate_path}" ]]; then
      local pass_dir="${round_dir}/model_pass"
      local pass_responses="${pass_dir}/responses.jsonl"
      local -a pass_args=(
        -m examples.lean_prover_v1.run_inference_lean_prover_v1
        --model-source "${SELF_MUTATION_MODEL_PASS_MODEL_SOURCE}"
        --backend "${SELF_MUTATION_MODEL_PASS_BACKEND}"
        --split train_mutated
        --mutation-bank-path "${candidate_path}"
        --output "${pass_responses}"
        --report-output "${pass_dir}/report.json"
        --num-samples "${SELF_MUTATION_MODEL_PASS_K}"
        --batch-size "${SELF_MUTATION_MODEL_PASS_BATCH_SIZE}"
        --max-model-len "${SELF_MUTATION_MODEL_PASS_MAX_MODEL_LEN}"
        --max-new-tokens "${SELF_MUTATION_MODEL_PASS_MAX_NEW_TOKENS}"
        --temperature "${SELF_MUTATION_MODEL_PASS_TEMPERATURE}"
        --top-p "${SELF_MUTATION_MODEL_PASS_TOP_P}"
        --gpu-memory-utilization "${SELF_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION}"
        --train-static-size 0
        --val-static-size 0
        --test-static-size 0
        --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
        --val-mutated-size 0
        --test-mutated-size 0
      )
      if [[ -n "${SELF_MUTATION_MODEL_PASS_LIMIT}" ]]; then
        pass_args+=(--limit "${SELF_MUTATION_MODEL_PASS_LIMIT}")
      fi
      if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
        pass_args+=(--disable-thinking true)
      fi
      if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
        pass_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
      fi
      if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
        pass_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
      fi
      "${VENV_PYTHON}" "${pass_args[@]}"
      model_responses_jsonl="${pass_responses}"
    fi
  fi

  if [[ -n "${model_responses_jsonl}" ]]; then
    mutate_args+=(--model-responses-jsonl "${model_responses_jsonl}")
  fi
  if [[ "${SELF_MUTATION_REQUIRE_MODEL_PASS_BOOL}" == "true" ]]; then
    mutate_args+=(--require-model-pass)
  else
    mutate_args+=(--no-require-model-pass)
  fi

  "${VENV_PYTHON}" "${mutate_args[@]}"
  LEAN_PROVER_V1_MUTATION_BANK="${round_dir}/bank.jsonl"
  LEAN_PROVER_V1_MUTATION_EVAL_BANK="${round_dir}/eval_bank.jsonl"
  export LEAN_PROVER_V1_MUTATION_BANK
  export LEAN_PROVER_V1_MUTATION_EVAL_BANK
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_BANK}" > "${OUTPUT_DIR}/latest_mutation_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_EVAL_BANK}" > "${OUTPUT_DIR}/latest_mutation_eval_bank.txt"
}

mutation_accepted_count() {
  local summary_path="$1"
  "${VENV_PYTHON}" - "${summary_path}" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(int(payload.get("accepted_train_count") or 0))
PY
}

run_error_mutation_round() {
  local rows_path="$1"
  local responses_jsonl="$2"
  local eval_report="$3"
  local round_dir="${ERROR_MUTATION_OUTPUT_DIR}/round_0"

  if [[ -z "${rows_path}" || ! -s "${rows_path}" || -z "${responses_jsonl}" || ! -s "${responses_jsonl}" ]]; then
    printf '%s\n' "Skipping error mutation because rows or responses are missing." > "${OUTPUT_DIR}/error_mutation_skipped.txt"
    return
  fi

  local -a error_args=(
    -m examples.lean_prover_v1.error_mutate_bank
    --rows-path "${rows_path}"
    --responses-jsonl "${responses_jsonl}"
    --output-dir "${round_dir}"
    --diagnosis-model-source "${ERROR_MUTATION_DIAGNOSIS_MODEL_SOURCE}"
    --max-failures "${ERROR_MUTATION_MAX_FAILURES}"
    --cases-per-family "${ERROR_MUTATION_CASES_PER_FAMILY}"
    --model-pass-k "${ERROR_MUTATION_MODEL_PASS_K}"
    --accept-max-pass-rate "${ERROR_MUTATION_ACCEPT_MAX_PASS_RATE}"
    --cheap-timeout-seconds "${ERROR_MUTATION_CHEAP_TIMEOUT_SECONDS}"
    --strong-timeout-seconds "${ERROR_MUTATION_STRONG_TIMEOUT_SECONDS}"
    --well-formed-timeout-seconds "${ERROR_MUTATION_WELL_FORMED_TIMEOUT_SECONDS}"
    --max-top-tactic-mass "${ERROR_MUTATION_MAX_TOP_TACTIC_MASS}"
    --max-error-family-mass "${ERROR_MUTATION_MAX_ERROR_FAMILY_MASS}"
    --max-template-mass "${ERROR_MUTATION_MAX_TEMPLATE_MASS}"
  )
  if [[ -n "${eval_report}" ]]; then
    error_args+=(--eval-report "${eval_report}")
  fi
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    error_args+=(--existing-mutation-bank-path "${LEAN_PROVER_V1_MUTATION_BANK}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    error_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    error_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
  fi

  "${VENV_PYTHON}" "${error_args[@]}" --no-require-model-pass

  local prelim_bank="${round_dir}/accepted.jsonl"
  if [[ -s "${prelim_bank}" ]]; then
    materialize_final_model_for_eval
    local pass_dir="${round_dir}/model_pass"
    local pass_responses="${pass_dir}/responses.jsonl"
    local -a pass_args=(
      -m examples.lean_prover_v1.run_inference_lean_prover_v1
      --model-source "${FINAL_MODEL_PATH}"
      --backend "${ERROR_MUTATION_MODEL_PASS_BACKEND}"
      --split train_mutated
      --mutation-bank-path "${prelim_bank}"
      --output "${pass_responses}"
      --report-output "${pass_dir}/report.json"
      --num-samples "${ERROR_MUTATION_MODEL_PASS_K}"
      --batch-size "${ERROR_MUTATION_MODEL_PASS_BATCH_SIZE}"
      --max-model-len "${ERROR_MUTATION_MODEL_PASS_MAX_MODEL_LEN}"
      --max-new-tokens "${ERROR_MUTATION_MODEL_PASS_MAX_NEW_TOKENS}"
      --temperature "${ERROR_MUTATION_MODEL_PASS_TEMPERATURE}"
      --top-p "${ERROR_MUTATION_MODEL_PASS_TOP_P}"
      --gpu-memory-utilization "${ERROR_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION}"
      --train-static-size 0
      --val-static-size 0
      --test-static-size 0
      --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
      --val-mutated-size 0
      --test-mutated-size 0
    )
    if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
      pass_args+=(--disable-thinking true)
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
      pass_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
      pass_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
    fi
    "${VENV_PYTHON}" "${pass_args[@]}"
    "${VENV_PYTHON}" "${error_args[@]}" --require-model-pass --model-responses-jsonl "${pass_responses}"
  else
    "${VENV_PYTHON}" "${error_args[@]}" --require-model-pass
  fi

  LEAN_PROVER_V1_MUTATION_BANK="${round_dir}/bank.jsonl"
  LEAN_PROVER_V1_MUTATION_EVAL_BANK="${round_dir}/eval_bank.jsonl"
  export LEAN_PROVER_V1_MUTATION_BANK
  export LEAN_PROVER_V1_MUTATION_EVAL_BANK
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_BANK}" > "${OUTPUT_DIR}/latest_mutation_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_EVAL_BANK}" > "${OUTPUT_DIR}/latest_mutation_eval_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_BANK}" > "${OUTPUT_DIR}/latest_error_mutation_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_EVAL_BANK}" > "${OUTPUT_DIR}/latest_error_mutation_eval_bank.txt"
}

run_teacher_reflection_eval() {
  local round_index="$1"
  local round_dir="${TEACHER_MUTATION_OUTPUT_DIR}/round_${round_index}"
  local reflection_dir="${round_dir}/reflection"
  mkdir -p "${reflection_dir}"

  materialize_final_model_for_eval
  TEACHER_MUTATION_ROWS_PATH="${reflection_dir}/rows.jsonl"
  TEACHER_MUTATION_RESPONSES_JSONL="${reflection_dir}/responses.jsonl"
  TEACHER_MUTATION_EVAL_REPORT="${reflection_dir}/eval_report.json"
  local -a reflection_args=(
    -m examples.lean_prover_v1.run_inference_lean_prover_v1
    --model-source "${FINAL_MODEL_PATH}"
    --backend "${LEAN_PROVER_V1_EVAL_BACKEND}"
    --split "${TEACHER_MUTATION_REFLECTION_SPLIT}"
    --limit "${TEACHER_MUTATION_REFLECTION_LIMIT}"
    --output "${TEACHER_MUTATION_RESPONSES_JSONL}"
    --report-output "${TEACHER_MUTATION_EVAL_REPORT}"
    --rows-output "${TEACHER_MUTATION_ROWS_PATH}"
    --num-samples "${LEAN_PROVER_V1_EVAL_NUM_SAMPLES}"
    --batch-size "${LEAN_PROVER_V1_EVAL_BATCH_SIZE}"
    --max-model-len "${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN}"
    --max-new-tokens "${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS}"
    --temperature "${LEAN_PROVER_V1_EVAL_TEMPERATURE}"
    --top-p "${LEAN_PROVER_V1_EVAL_TOP_P}"
    --gpu-memory-utilization "${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION}"
    --device "${LEAN_PROVER_V1_EVAL_DEVICE}"
    --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
    --val-static-size "${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
    --test-static-size "${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
    --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
    --val-mutated-size "${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
    --test-mutated-size "${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
  )
  if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
    reflection_args+=(--static-corpus-path "${LEAN_PROVER_V1_STATIC_CORPUS}")
  fi
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    reflection_args+=(--mutation-bank-path "${LEAN_PROVER_V1_MUTATION_BANK}")
  fi
  if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
    reflection_args+=(--disable-thinking true)
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    reflection_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    reflection_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
  fi
  "${VENV_PYTHON}" "${reflection_args[@]}"
  export TEACHER_MUTATION_ROWS_PATH TEACHER_MUTATION_RESPONSES_JSONL TEACHER_MUTATION_EVAL_REPORT
}

run_teacher_mutation_round() {
  local round_index="$1"
  local rows_path="$2"
  local responses_jsonl="$3"
  local eval_report="$4"
  local round_dir="${TEACHER_MUTATION_OUTPUT_DIR}/round_${round_index}"
  local prelim_dir="${round_dir}/prelim"
  local failure_records_jsonl=""
  mkdir -p "${round_dir}"

  if [[ -z "${rows_path}" || ! -s "${rows_path}" || -z "${responses_jsonl}" || ! -s "${responses_jsonl}" ]]; then
    printf '%s\n' "Skipping teacher mutation because rows or responses are missing." > "${round_dir}/teacher_mutation_skipped.txt"
    return
  fi

  if [[ "${TEACHER_MUTATION_USE_STEPWISE_DIAGNOSIS_BOOL}" == "true" ]]; then
    local stepwise_dir="${round_dir}/stepwise_diagnosis"
    local -a stepwise_args=(
      -m examples.lean_prover_v1.stepwise_diagnose
      --rows-path "${rows_path}"
      --responses-jsonl "${responses_jsonl}"
      --output-dir "${stepwise_dir}"
      --max-failures "${TEACHER_MUTATION_MAX_FAILURES}"
      --timeout-seconds "${TEACHER_MUTATION_STRONG_TIMEOUT_SECONDS}"
      --max-heartbeats "${LEAN_PROVER_V1_MAX_HEARTBEATS}"
    )
    if [[ -n "${eval_report}" ]]; then
      stepwise_args+=(--eval-report "${eval_report}")
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
      stepwise_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
      stepwise_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
    fi
    "${VENV_PYTHON}" "${stepwise_args[@]}"
    failure_records_jsonl="${stepwise_dir}/stepwise_failures.jsonl"
  fi

  local -a teacher_base_args=(
    -m examples.lean_prover_v1.teacher_mutate_bank
    --rows-path "${rows_path}"
    --responses-jsonl "${responses_jsonl}"
    --output-dir "${prelim_dir}"
    --teacher-model "${TEACHER_MUTATION_MODEL}"
    --teacher-base-url "${TEACHER_MUTATION_BASE_URL}"
    --max-failures "${TEACHER_MUTATION_MAX_FAILURES}"
    --cases-per-family "${TEACHER_MUTATION_CASES_PER_FAMILY}"
    --model-pass-k "${TEACHER_MUTATION_MODEL_PASS_K}"
    --accept-max-pass-rate "${TEACHER_MUTATION_ACCEPT_MAX_PASS_RATE}"
    --max-api-calls "${TEACHER_MUTATION_MAX_API_CALLS}"
    --max-output-tokens "${TEACHER_MUTATION_MAX_OUTPUT_TOKENS}"
    --temperature "${TEACHER_MUTATION_TEMPERATURE}"
    --cache-dir "${TEACHER_MUTATION_CACHE_DIR}"
    --refine-rounds "${TEACHER_MUTATION_REFINE_ROUNDS}"
    --cheap-timeout-seconds "${TEACHER_MUTATION_CHEAP_TIMEOUT_SECONDS}"
    --strong-timeout-seconds "${TEACHER_MUTATION_STRONG_TIMEOUT_SECONDS}"
    --well-formed-timeout-seconds "${TEACHER_MUTATION_WELL_FORMED_TIMEOUT_SECONDS}"
    --max-top-tactic-mass "${TEACHER_MUTATION_MAX_TOP_TACTIC_MASS}"
    --max-error-family-mass "${TEACHER_MUTATION_MAX_ERROR_FAMILY_MASS}"
    --difficulty-directive balanced
  )
  if [[ -n "${eval_report}" ]]; then
    teacher_base_args+=(--eval-report "${eval_report}")
  fi
  if [[ -n "${failure_records_jsonl}" && -s "${failure_records_jsonl}" ]]; then
    teacher_base_args+=(--failure-records-jsonl "${failure_records_jsonl}")
  fi
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    teacher_base_args+=(--existing-mutation-bank-path "${LEAN_PROVER_V1_MUTATION_BANK}")
  fi
  if [[ -n "${TEACHER_MUTATION_RAW_JSONL}" ]]; then
    teacher_base_args+=(--teacher-raw-jsonl "${TEACHER_MUTATION_RAW_JSONL}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    teacher_base_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    teacher_base_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
  fi
  if [[ "${TEACHER_MUTATION_DISABLE_THINKING_BOOL}" == "true" ]]; then
    teacher_base_args+=(--disable-teacher-thinking)
  else
    teacher_base_args+=(--no-disable-teacher-thinking)
  fi
  if [[ "${TEACHER_MUTATION_ALLOW_STEPWISE_CHEAP_SOLVED_BOOL}" == "true" ]]; then
    teacher_base_args+=(--allow-stepwise-cheap-solved)
  else
    teacher_base_args+=(--no-allow-stepwise-cheap-solved)
  fi

  "${VENV_PYTHON}" "${teacher_base_args[@]}" --no-require-model-pass

  local candidate_path="${prelim_dir}/teacher_candidates.jsonl"
  local pass_responses=""
  if [[ -s "${candidate_path}" ]]; then
    materialize_final_model_for_eval
    local pass_dir="${round_dir}/model_pass"
    pass_responses="${pass_dir}/responses.jsonl"
    local -a pass_args=(
      -m examples.lean_prover_v1.run_inference_lean_prover_v1
      --model-source "${FINAL_MODEL_PATH}"
      --backend "${TEACHER_MUTATION_MODEL_PASS_BACKEND}"
      --split teacher_candidates
      --rows-path "${candidate_path}"
      --output "${pass_responses}"
      --report-output "${pass_dir}/report.json"
      --num-samples "${TEACHER_MUTATION_MODEL_PASS_K}"
      --batch-size "${TEACHER_MUTATION_MODEL_PASS_BATCH_SIZE}"
      --max-model-len "${TEACHER_MUTATION_MODEL_PASS_MAX_MODEL_LEN}"
      --max-new-tokens "${TEACHER_MUTATION_MODEL_PASS_MAX_NEW_TOKENS}"
      --temperature "${TEACHER_MUTATION_MODEL_PASS_TEMPERATURE}"
      --top-p "${TEACHER_MUTATION_MODEL_PASS_TOP_P}"
      --gpu-memory-utilization "${TEACHER_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION}"
    )
    if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
      pass_args+=(--disable-thinking true)
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
      pass_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
      pass_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
    fi
    "${VENV_PYTHON}" "${pass_args[@]}"
  fi

  local -a teacher_final_args=()
  local arg_index=0
  while (( arg_index < ${#teacher_base_args[@]} )); do
    if [[ "${teacher_base_args[$arg_index]}" == "--output-dir" ]]; then
      teacher_final_args+=(--output-dir "${round_dir}")
      arg_index="$((arg_index + 2))"
    else
      teacher_final_args+=("${teacher_base_args[$arg_index]}")
      arg_index="$((arg_index + 1))"
    fi
  done
  if [[ -e "${candidate_path}" ]]; then
    teacher_final_args+=(--teacher-candidates-jsonl "${candidate_path}")
  fi
  if [[ -s "${pass_responses}" ]]; then
    teacher_final_args+=(--model-responses-jsonl "${pass_responses}")
  fi
  "${VENV_PYTHON}" "${teacher_final_args[@]}" --require-model-pass

  local refinement_directive=""
  if (( TEACHER_MUTATION_REFINE_ROUNDS > 0 )) && [[ -s "${round_dir}/refinement_requests.jsonl" ]]; then
    refinement_directive="$("${VENV_PYTHON}" - "${round_dir}/refinement_requests.jsonl" <<'PY'
import json
import sys
from pathlib import Path

requests = []
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if line.strip():
        requests.append(json.loads(line))
directives = {str(item.get("directive") or "") for item in requests}
if "generate_easier_bridge_batch" in directives:
    print("easier")
elif "generate_harder_bridge_batch" in directives:
    print("harder")
PY
)"
  fi

  if [[ -n "${refinement_directive}" ]]; then
    local refine_prelim_dir="${round_dir}/refine_prelim"
    local -a teacher_refine_args=()
    arg_index=0
    while (( arg_index < ${#teacher_base_args[@]} )); do
      if [[ "${teacher_base_args[$arg_index]}" == "--output-dir" ]]; then
        teacher_refine_args+=(--output-dir "${refine_prelim_dir}")
        arg_index="$((arg_index + 2))"
      elif [[ "${teacher_base_args[$arg_index]}" == "--difficulty-directive" ]]; then
        teacher_refine_args+=(--difficulty-directive "${refinement_directive}")
        arg_index="$((arg_index + 2))"
      else
        teacher_refine_args+=("${teacher_base_args[$arg_index]}")
        arg_index="$((arg_index + 1))"
      fi
    done
    "${VENV_PYTHON}" "${teacher_refine_args[@]}" --no-require-model-pass

    local refine_candidate_path="${refine_prelim_dir}/teacher_candidates.jsonl"
    if [[ -s "${refine_candidate_path}" ]]; then
      materialize_final_model_for_eval
      local refine_pass_dir="${round_dir}/refine_model_pass"
      local refine_pass_responses="${refine_pass_dir}/responses.jsonl"
      local -a refine_pass_args=(
        -m examples.lean_prover_v1.run_inference_lean_prover_v1
        --model-source "${FINAL_MODEL_PATH}"
        --backend "${TEACHER_MUTATION_MODEL_PASS_BACKEND}"
        --split teacher_refine_candidates
        --rows-path "${refine_candidate_path}"
        --output "${refine_pass_responses}"
        --report-output "${refine_pass_dir}/report.json"
        --num-samples "${TEACHER_MUTATION_MODEL_PASS_K}"
        --batch-size "${TEACHER_MUTATION_MODEL_PASS_BATCH_SIZE}"
        --max-model-len "${TEACHER_MUTATION_MODEL_PASS_MAX_MODEL_LEN}"
        --max-new-tokens "${TEACHER_MUTATION_MODEL_PASS_MAX_NEW_TOKENS}"
        --temperature "${TEACHER_MUTATION_MODEL_PASS_TEMPERATURE}"
        --top-p "${TEACHER_MUTATION_MODEL_PASS_TOP_P}"
        --gpu-memory-utilization "${TEACHER_MUTATION_MODEL_PASS_GPU_MEMORY_UTILIZATION}"
      )
      if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
        refine_pass_args+=(--disable-thinking true)
      fi
      if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
        refine_pass_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
      fi
      if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
        refine_pass_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
      fi
      "${VENV_PYTHON}" "${refine_pass_args[@]}"

      local combined_candidate_path="${round_dir}/combined_teacher_candidates.jsonl"
      local combined_pass_responses="${round_dir}/combined_model_pass_responses.jsonl"
      "${VENV_PYTHON}" - "${combined_candidate_path}" "${candidate_path}" "${refine_candidate_path}" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1])
inputs = [Path(item) for item in sys.argv[2:]]
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for path in inputs:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                text += "\n"
            handle.write(text)
PY
      "${VENV_PYTHON}" - "${combined_pass_responses}" "${pass_responses}" "${refine_pass_responses}" <<'PY'
import sys
from pathlib import Path

output = Path(sys.argv[1])
inputs = [Path(item) for item in sys.argv[2:] if item]
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("w", encoding="utf-8") as handle:
    for path in inputs:
        if path.exists():
            text = path.read_text(encoding="utf-8")
            if text and not text.endswith("\n"):
                text += "\n"
            handle.write(text)
PY

      local -a teacher_combined_args=()
      arg_index=0
      while (( arg_index < ${#teacher_base_args[@]} )); do
        if [[ "${teacher_base_args[$arg_index]}" == "--output-dir" ]]; then
          teacher_combined_args+=(--output-dir "${round_dir}")
          arg_index="$((arg_index + 2))"
        else
          teacher_combined_args+=("${teacher_base_args[$arg_index]}")
          arg_index="$((arg_index + 1))"
        fi
      done
      teacher_combined_args+=(--teacher-candidates-jsonl "${combined_candidate_path}")
      teacher_combined_args+=(--model-responses-jsonl "${combined_pass_responses}")
      "${VENV_PYTHON}" "${teacher_combined_args[@]}" --require-model-pass
    fi
  fi

  LEAN_PROVER_V1_MUTATION_BANK="${round_dir}/bank.jsonl"
  LEAN_PROVER_V1_MUTATION_EVAL_BANK="${round_dir}/eval_bank.jsonl"
  export LEAN_PROVER_V1_MUTATION_BANK
  export LEAN_PROVER_V1_MUTATION_EVAL_BANK
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_BANK}" > "${OUTPUT_DIR}/latest_mutation_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_EVAL_BANK}" > "${OUTPUT_DIR}/latest_mutation_eval_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_BANK}" > "${OUTPUT_DIR}/latest_teacher_mutation_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_EVAL_BANK}" > "${OUTPUT_DIR}/latest_teacher_mutation_eval_bank.txt"
}

run_skill_boundary_reflection_eval() {
  local round_index="$1"
  local round_dir="${SKILL_BOUNDARY_OUTPUT_DIR}/round_${round_index}"
  local reflection_dir="${round_dir}/reflection"
  mkdir -p "${reflection_dir}"

  materialize_final_model_for_eval
  SKILL_BOUNDARY_ROWS_PATH="${reflection_dir}/rows.jsonl"
  SKILL_BOUNDARY_RESPONSES_JSONL="${reflection_dir}/responses.jsonl"
  SKILL_BOUNDARY_EVAL_REPORT="${reflection_dir}/eval_report.json"
  local -a reflection_args=(
    -m examples.lean_prover_v1.run_inference_lean_prover_v1
    --model-source "${FINAL_MODEL_PATH}"
    --backend "${LEAN_PROVER_V1_EVAL_BACKEND}"
    --split "${SKILL_BOUNDARY_REFLECTION_SPLIT}"
    --limit "${SKILL_BOUNDARY_REFLECTION_LIMIT}"
    --output "${SKILL_BOUNDARY_RESPONSES_JSONL}"
    --report-output "${SKILL_BOUNDARY_EVAL_REPORT}"
    --rows-output "${SKILL_BOUNDARY_ROWS_PATH}"
    --num-samples "${SKILL_BOUNDARY_DIAGNOSE_MAX_K}"
    --batch-size "${LEAN_PROVER_V1_EVAL_BATCH_SIZE}"
    --max-model-len "${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN}"
    --max-new-tokens "${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS}"
    --temperature "${LEAN_PROVER_V1_EVAL_TEMPERATURE}"
    --top-p "${LEAN_PROVER_V1_EVAL_TOP_P}"
    --gpu-memory-utilization "${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION}"
    --device "${LEAN_PROVER_V1_EVAL_DEVICE}"
    --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
    --val-static-size "${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
    --test-static-size "${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
    --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
    --val-mutated-size "${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
    --test-mutated-size "${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
  )
  if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
    reflection_args+=(--static-corpus-path "${LEAN_PROVER_V1_STATIC_CORPUS}")
  fi
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    reflection_args+=(--mutation-bank-path "${LEAN_PROVER_V1_MUTATION_BANK}")
  fi
  if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
    reflection_args+=(--disable-thinking true)
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    reflection_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    reflection_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
  fi
  "${VENV_PYTHON}" "${reflection_args[@]}"
  export SKILL_BOUNDARY_ROWS_PATH SKILL_BOUNDARY_RESPONSES_JSONL SKILL_BOUNDARY_EVAL_REPORT
}

run_skill_boundary_diagnosis() {
  local round_index="$1"
  local round_dir="${SKILL_BOUNDARY_OUTPUT_DIR}/round_${round_index}"
  local diagnosis_dir="${round_dir}/diagnosis"
  mkdir -p "${diagnosis_dir}"
  local previous_state=""
  if (( round_index > 0 )); then
    previous_state="${SKILL_BOUNDARY_OUTPUT_DIR}/round_$((round_index - 1))/diagnosis/controller_state.json"
  fi
  local -a diagnosis_args=(
    -m examples.lean_prover_v1.boundary_diagnose
    --rows-path "${SKILL_BOUNDARY_ROWS_PATH}"
    --responses-jsonl "${SKILL_BOUNDARY_RESPONSES_JSONL}"
    --eval-report "${SKILL_BOUNDARY_EVAL_REPORT}"
    --output-dir "${diagnosis_dir}"
    --max-k "${SKILL_BOUNDARY_DIAGNOSE_MAX_K}"
    --too-easy-pass-rate "${SKILL_BOUNDARY_TOO_EASY_PASS_RATE}"
    --boundary-max-pass-rate "${SKILL_BOUNDARY_BOUNDARY_MAX_PASS_RATE}"
    --timeout-seconds "${LEAN_PROVER_V1_TIMEOUT_SECONDS}"
    --max-heartbeats "${LEAN_PROVER_V1_MAX_HEARTBEATS}"
  )
  if [[ -s "${previous_state}" ]]; then
    diagnosis_args+=(--previous-controller-state "${previous_state}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    diagnosis_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    diagnosis_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
  fi
  "${VENV_PYTHON}" "${diagnosis_args[@]}"
  SKILL_BOUNDARY_SIGNAL_ROWS="${diagnosis_dir}/signal_rows.jsonl"
  export SKILL_BOUNDARY_SIGNAL_ROWS
}

run_skill_boundary_generation_mode() {
  local mode="$1"
  local max_source_rows="$2"
  local round_index="$3"
  local round_dir="${SKILL_BOUNDARY_OUTPUT_DIR}/round_${round_index}"
  local mode_dir="${round_dir}/${mode}"
  local prelim_dir="${mode_dir}/prelim"
  mkdir -p "${mode_dir}"

  local -a base_args=(
    -m examples.lean_prover_v1.skill_boundary_bank
    --mode "${mode}"
    --signal-rows-path "${SKILL_BOUNDARY_SIGNAL_ROWS}"
    --output-dir "${prelim_dir}"
    --teacher-model "${SKILL_BOUNDARY_TEACHER_MODEL}"
    --teacher-base-url "${SKILL_BOUNDARY_TEACHER_BASE_URL}"
    --max-source-rows "${max_source_rows}"
    --cases-per-row "${SKILL_BOUNDARY_CASES_PER_ROW}"
    --model-pass-k "${SKILL_BOUNDARY_MODEL_PASS_K}"
    --accept-max-pass-rate "${SKILL_BOUNDARY_ACCEPT_MAX_PASS_RATE}"
    --max-api-calls "${SKILL_BOUNDARY_TEACHER_MAX_API_CALLS}"
    --max-output-tokens "${SKILL_BOUNDARY_TEACHER_MAX_OUTPUT_TOKENS}"
    --temperature "${SKILL_BOUNDARY_TEACHER_TEMPERATURE}"
    --cache-dir "${SKILL_BOUNDARY_TEACHER_CACHE_DIR}"
    --cheap-timeout-seconds "${SKILL_BOUNDARY_CHEAP_TIMEOUT_SECONDS}"
    --strong-timeout-seconds "${SKILL_BOUNDARY_STRONG_TIMEOUT_SECONDS}"
    --well-formed-timeout-seconds "${SKILL_BOUNDARY_WELL_FORMED_TIMEOUT_SECONDS}"
    --max-top-tactic-mass "${SKILL_BOUNDARY_MAX_TOP_TACTIC_MASS}"
    --max-family-mass "${SKILL_BOUNDARY_MAX_FAMILY_MASS}"
  )
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    base_args+=(--existing-mutation-bank-path "${LEAN_PROVER_V1_MUTATION_BANK}")
  fi
  if [[ -n "${SKILL_BOUNDARY_TEACHER_RAW_JSONL}" ]]; then
    base_args+=(--teacher-raw-jsonl "${SKILL_BOUNDARY_TEACHER_RAW_JSONL}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    base_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    base_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
  fi
  if [[ "${SKILL_BOUNDARY_TEACHER_DISABLE_THINKING_BOOL}" == "true" ]]; then
    base_args+=(--disable-teacher-thinking)
  else
    base_args+=(--no-disable-teacher-thinking)
  fi

  "${VENV_PYTHON}" "${base_args[@]}" --no-require-model-pass

  local candidate_path="${prelim_dir}/teacher_candidates.jsonl"
  local pass_responses=""
  if [[ -s "${candidate_path}" ]]; then
    materialize_final_model_for_eval
    local pass_dir="${mode_dir}/model_pass"
    pass_responses="${pass_dir}/responses.jsonl"
    local -a pass_args=(
      -m examples.lean_prover_v1.run_inference_lean_prover_v1
      --model-source "${FINAL_MODEL_PATH}"
      --backend "${SKILL_BOUNDARY_MODEL_PASS_BACKEND}"
      --split "skill_boundary_${mode}_candidates"
      --rows-path "${candidate_path}"
      --output "${pass_responses}"
      --report-output "${pass_dir}/report.json"
      --num-samples "${SKILL_BOUNDARY_MODEL_PASS_K}"
      --batch-size "${SKILL_BOUNDARY_MODEL_PASS_BATCH_SIZE}"
      --max-model-len "${SKILL_BOUNDARY_MODEL_PASS_MAX_MODEL_LEN}"
      --max-new-tokens "${SKILL_BOUNDARY_MODEL_PASS_MAX_NEW_TOKENS}"
      --temperature "${SKILL_BOUNDARY_MODEL_PASS_TEMPERATURE}"
      --top-p "${SKILL_BOUNDARY_MODEL_PASS_TOP_P}"
      --gpu-memory-utilization "${SKILL_BOUNDARY_MODEL_PASS_GPU_MEMORY_UTILIZATION}"
    )
    if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
      pass_args+=(--disable-thinking true)
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
      pass_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
      pass_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
    fi
    "${VENV_PYTHON}" "${pass_args[@]}"
  fi

  local -a final_args=()
  local arg_index=0
  while (( arg_index < ${#base_args[@]} )); do
    if [[ "${base_args[$arg_index]}" == "--output-dir" ]]; then
      final_args+=(--output-dir "${mode_dir}")
      arg_index="$((arg_index + 2))"
    else
      final_args+=("${base_args[$arg_index]}")
      arg_index="$((arg_index + 1))"
    fi
  done
  if [[ -e "${candidate_path}" ]]; then
    final_args+=(--teacher-candidates-jsonl "${candidate_path}")
  fi
  if [[ -s "${pass_responses}" ]]; then
    final_args+=(--model-responses-jsonl "${pass_responses}")
  fi
  "${VENV_PYTHON}" "${final_args[@]}" --require-model-pass
}

run_skill_boundary_compose_bank() {
  local round_index="$1"
  local round_dir="${SKILL_BOUNDARY_OUTPUT_DIR}/round_${round_index}"
  local compose_dir="${round_dir}/composed"
  local -a compose_args=(
    -m examples.lean_prover_v1.skill_boundary_bank
    --mode compose_bank
    --output-dir "${compose_dir}"
    --success-bank-path "${round_dir}/success_extension/accepted.jsonl"
    --success-eval-bank-path "${round_dir}/success_extension/eval_bank.jsonl"
    --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
    --static-floor "${SKILL_BOUNDARY_STATIC_FLOOR}"
    --success-extension-weight "${SKILL_BOUNDARY_SUCCESS_EXTENSION_WEIGHT}"
    --failure-bridge-weight "${SKILL_BOUNDARY_FAILURE_BRIDGE_WEIGHT}"
    --max-family-mass "${SKILL_BOUNDARY_MAX_FAMILY_MASS}"
  )
  if [[ "${SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE_BOOL}" == "true" ]]; then
    compose_args+=(--bridge-bank-path "${round_dir}/failure_bridge/accepted.jsonl")
    compose_args+=(--bridge-eval-bank-path "${round_dir}/failure_bridge/eval_bank.jsonl")
  fi
  if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
    compose_args+=(--static-corpus-path "${LEAN_PROVER_V1_STATIC_CORPUS}")
  fi
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    compose_args+=(--existing-mutation-bank-path "${LEAN_PROVER_V1_MUTATION_BANK}")
  fi
  "${VENV_PYTHON}" "${compose_args[@]}"

  LEAN_PROVER_V1_MUTATION_BANK="${compose_dir}/sampled_train_bank.jsonl"
  LEAN_PROVER_V1_MUTATION_EVAL_BANK="${compose_dir}/eval_bank.jsonl"
  export LEAN_PROVER_V1_MUTATION_BANK
  export LEAN_PROVER_V1_MUTATION_EVAL_BANK
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_BANK}" > "${OUTPUT_DIR}/latest_mutation_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_EVAL_BANK}" > "${OUTPUT_DIR}/latest_mutation_eval_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_BANK}" > "${OUTPUT_DIR}/latest_skill_boundary_bank.txt"
  printf '%s\n' "${LEAN_PROVER_V1_MUTATION_EVAL_BANK}" > "${OUTPUT_DIR}/latest_skill_boundary_eval_bank.txt"
}

run_skill_boundary_round() {
  local round_index="$1"
  run_skill_boundary_reflection_eval "${round_index}"
  run_skill_boundary_diagnosis "${round_index}"
  run_skill_boundary_generation_mode success_extension "${SKILL_BOUNDARY_MAX_SUCCESS_ROWS}" "${round_index}"
  if [[ "${SKILL_BOUNDARY_ENABLE_FAILURE_BRIDGE_BOOL}" == "true" ]]; then
    run_skill_boundary_generation_mode failure_bridge "${SKILL_BOUNDARY_MAX_FAILURE_ROWS}" "${round_index}"
  fi
  run_skill_boundary_compose_bank "${round_index}"
}

if [[ "${RUN_SKILL_BOUNDARY_CURRICULUM_BOOL}" == "true" && "${RUN_TEACHER_MUTATION_BOOL}" == "true" ]]; then
  echo "RUN_SKILL_BOUNDARY_CURRICULUM=1 should not be combined with RUN_TEACHER_MUTATION=1 in the same job." >&2
  exit 1
fi
if [[ "${RUN_SKILL_BOUNDARY_CURRICULUM_BOOL}" == "true" && "${RUN_SELF_MUTATION_BOOL}" == "true" ]]; then
  echo "RUN_SKILL_BOUNDARY_CURRICULUM=1 should not be combined with RUN_SELF_MUTATION=1 in the same job." >&2
  exit 1
fi
if [[ "${RUN_SKILL_BOUNDARY_CURRICULUM_BOOL}" == "true" && "${RUN_ERROR_MUTATION_BOOL}" == "true" ]]; then
  echo "RUN_SKILL_BOUNDARY_CURRICULUM=1 should not be combined with RUN_ERROR_MUTATION=1 in the same job." >&2
  exit 1
fi

if [[ "${RUN_TEACHER_MUTATION_BOOL}" == "true" && "${RUN_SELF_MUTATION_BOOL}" == "true" ]]; then
  echo "RUN_TEACHER_MUTATION=1 should not be combined with RUN_SELF_MUTATION=1 in the same job." >&2
  exit 1
fi
if [[ "${RUN_TEACHER_MUTATION_BOOL}" == "true" && "${RUN_ERROR_MUTATION_BOOL}" == "true" ]]; then
  echo "RUN_TEACHER_MUTATION=1 should not be combined with RUN_ERROR_MUTATION=1 in the same job." >&2
  exit 1
fi

if [[ "${RUN_SKILL_BOUNDARY_CURRICULUM_BOOL}" == "true" ]]; then
  if ! [[ "${TOTAL_EPOCHS}" =~ ^[0-9]+$ ]]; then
    echo "RUN_SKILL_BOUNDARY_CURRICULUM=1 requires integer TOTAL_EPOCHS, got: ${TOTAL_EPOCHS}" >&2
    exit 1
  fi
  if (( SKILL_BOUNDARY_ROUNDS < 1 )); then
    echo "SKILL_BOUNDARY_ROUNDS must be >= 1, got: ${SKILL_BOUNDARY_ROUNDS}" >&2
    exit 1
  fi

  for ((round_index = 0; round_index < SKILL_BOUNDARY_ROUNDS; round_index++)); do
    if (( round_index == 0 )); then
      CURRENT_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
    else
      CURRENT_RESUME_MODE=auto
    fi
    CURRENT_TOTAL_EPOCHS="$((TOTAL_EPOCHS * (round_index + 1)))"
    export CURRENT_RESUME_MODE CURRENT_TOTAL_EPOCHS
    reset_final_model_resolution
    run_lean_training "$@"
    reset_final_model_resolution
    run_skill_boundary_round "${round_index}"
  done
elif [[ "${RUN_TEACHER_MUTATION_BOOL}" == "true" ]]; then
  if ! [[ "${TOTAL_EPOCHS}" =~ ^[0-9]+$ ]]; then
    echo "RUN_TEACHER_MUTATION=1 requires integer TOTAL_EPOCHS, got: ${TOTAL_EPOCHS}" >&2
    exit 1
  fi
  if (( TEACHER_MUTATION_ROUNDS < 1 )); then
    echo "TEACHER_MUTATION_ROUNDS must be >= 1, got: ${TEACHER_MUTATION_ROUNDS}" >&2
    exit 1
  fi

  for ((round_index = 0; round_index < TEACHER_MUTATION_ROUNDS; round_index++)); do
    if (( round_index == 0 )); then
      CURRENT_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
    else
      CURRENT_RESUME_MODE=auto
    fi
    CURRENT_TOTAL_EPOCHS="$((TOTAL_EPOCHS * (round_index + 1)))"
    export CURRENT_RESUME_MODE CURRENT_TOTAL_EPOCHS
    reset_final_model_resolution
    run_lean_training "$@"
    reset_final_model_resolution
    run_teacher_reflection_eval "${round_index}"
    run_teacher_mutation_round "${round_index}" "${TEACHER_MUTATION_ROWS_PATH}" "${TEACHER_MUTATION_RESPONSES_JSONL}" "${TEACHER_MUTATION_EVAL_REPORT}"
  done
elif [[ "${RUN_SELF_MUTATION_BOOL}" == "true" ]]; then
  if ! [[ "${TOTAL_EPOCHS}" =~ ^[0-9]+$ ]]; then
    echo "RUN_SELF_MUTATION=1 requires integer TOTAL_EPOCHS, got: ${TOTAL_EPOCHS}" >&2
    exit 1
  fi
  if (( SELF_MUTATION_ROUNDS < 1 )); then
    echo "SELF_MUTATION_ROUNDS must be >= 1, got: ${SELF_MUTATION_ROUNDS}" >&2
    exit 1
  fi

  consecutive_empty_mutation_rounds=0
  for ((round_index = 0; round_index < SELF_MUTATION_ROUNDS; round_index++)); do
    if (( round_index == 0 )); then
      CURRENT_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
    else
      CURRENT_RESUME_MODE=auto
    fi
    CURRENT_TOTAL_EPOCHS="$((TOTAL_EPOCHS * (round_index + 1)))"
    export CURRENT_RESUME_MODE CURRENT_TOTAL_EPOCHS
    run_lean_training "$@"

    run_mutation_round "${round_index}"
    accepted_count="$(mutation_accepted_count "${SELF_MUTATION_OUTPUT_DIR}/round_${round_index}/summary.json")"
    if (( accepted_count == 0 )); then
      consecutive_empty_mutation_rounds="$((consecutive_empty_mutation_rounds + 1))"
    else
      consecutive_empty_mutation_rounds=0
    fi
    if (( consecutive_empty_mutation_rounds >= SELF_MUTATION_EMPTY_ROUND_LIMIT )); then
      echo "Self-mutation produced ${consecutive_empty_mutation_rounds} consecutive empty accepted_train rounds." >&2
      exit 1
    fi
  done
else
  CURRENT_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
  CURRENT_TOTAL_EPOCHS="${TOTAL_EPOCHS}"
  export CURRENT_RESUME_MODE CURRENT_TOTAL_EPOCHS
  run_lean_training "$@"
fi

write_generalization_manifest() {
  "${VENV_PYTHON}" - "$LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR/manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "run_name": os.environ.get("RUN_NAME"),
    "base_model": os.environ.get("MODEL_SOURCE"),
    "trained_model": os.environ.get("FINAL_MODEL_PATH"),
    "final_checkpoint_actor_dir": os.environ.get("FINAL_CHECKPOINT_ACTOR_DIR"),
    "tasks": os.environ.get("LEAN_PROVER_V1_GENERALIZATION_TASKS"),
    "limit": os.environ.get("LEAN_PROVER_V1_GENERALIZATION_LIMIT"),
    "max_gen_toks": os.environ.get("LEAN_PROVER_V1_GENERALIZATION_MAX_GEN_TOKS"),
    "tensor_parallel_size": os.environ.get("LEAN_PROVER_V1_GENERALIZATION_TP_SIZE"),
    "missing_task_policy": os.environ.get("LEAN_PROVER_V1_GENERALIZATION_MISSING_TASK_POLICY"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_generalization_eval() {
  case "${RUN_GENERALIZATION_EVAL}" in
    1|true|TRUE|yes|YES)
      materialize_final_model_for_eval
      mkdir -p "${LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR}"
      export FINAL_MODEL_PATH FINAL_CHECKPOINT_ACTOR_DIR
      export RUN_NAME MODEL_SOURCE LEAN_PROVER_V1_GENERALIZATION_TASKS LEAN_PROVER_V1_GENERALIZATION_LIMIT
      export LEAN_PROVER_V1_GENERALIZATION_MAX_GEN_TOKS LEAN_PROVER_V1_GENERALIZATION_TP_SIZE
      export LEAN_PROVER_V1_GENERALIZATION_MISSING_TASK_POLICY
      write_generalization_manifest

      local lm_eval_limit="${LEAN_PROVER_V1_GENERALIZATION_LIMIT}"
      if [[ "${lm_eval_limit}" == "-1" ]]; then
        lm_eval_limit=""
      fi
      TRAINED_MODEL_SOURCE="${FINAL_MODEL_PATH}" \
      BASE_MODEL="${MODEL_SOURCE}" \
      TASKS="${LEAN_PROVER_V1_GENERALIZATION_TASKS}" \
      OUTPUT_ROOT="${LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR}/lm_eval" \
      REQUEST_CACHE_ROOT="${LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR}/lm_eval_request_cache" \
      MATERIALIZE_ROOT="${LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR}/lm_eval_vllm_exports/${SLURM_JOB_ID:-manual}" \
      LIMIT="${lm_eval_limit}" \
      MAX_GEN_TOKS="${LEAN_PROVER_V1_GENERALIZATION_MAX_GEN_TOKS}" \
      TENSOR_PARALLEL_SIZE="${LEAN_PROVER_V1_GENERALIZATION_TP_SIZE}" \
      BATCH_SIZE="${LEAN_PROVER_V1_GENERALIZATION_BATCH_SIZE}" \
      MAX_BATCH_SIZE="${LEAN_PROVER_V1_GENERALIZATION_MAX_BATCH_SIZE}" \
      MAX_MODEL_LEN="${LEAN_PROVER_V1_GENERALIZATION_MAX_MODEL_LEN}" \
      MAX_NUM_SEQS="${LEAN_PROVER_V1_GENERALIZATION_MAX_NUM_SEQS}" \
      GPU_MEMORY_UTILIZATION="${LEAN_PROVER_V1_GENERALIZATION_GPU_MEMORY_UTILIZATION}" \
      TRUST_REMOTE_CODE="${LEAN_PROVER_V1_GENERALIZATION_TRUST_REMOTE_CODE}" \
      MISSING_TASK_POLICY="${LEAN_PROVER_V1_GENERALIZATION_MISSING_TASK_POLICY}" \
      USE_CACHE="${LEAN_PROVER_V1_GENERALIZATION_USE_CACHE}" \
      ENFORCE_EAGER="${LEAN_PROVER_V1_GENERALIZATION_ENFORCE_EAGER}" \
      DISABLE_CUSTOM_ALL_REDUCE="${LEAN_PROVER_V1_GENERALIZATION_DISABLE_CUSTOM_ALL_REDUCE}" \
      HF_ALLOW_CODE_EVAL=1 \
      bash "${PROJECT_ROOT}/scripts/lm_eval/scripts/eval_model_pair_vllm.sh"

      "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/capability/summarize_lm_eval_pair.py" \
        --run-root "${LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR}" \
        --manifest "${LEAN_PROVER_V1_GENERALIZATION_OUTPUT_DIR}/manifest.json" \
        --base-model "${MODEL_SOURCE}" \
        --large-drop "${LEAN_PROVER_V1_GENERALIZATION_LARGE_DROP}"
      ;;
    *)
      printf '%s\n' "Skipping generalization eval because RUN_GENERALIZATION_EVAL=${RUN_GENERALIZATION_EVAL}." > "${OUTPUT_DIR}/generalization_eval_skipped.txt"
      ;;
  esac
}

case "${RUN_EVAL_AFTER}" in
  1|true|TRUE|yes|YES)
    resolve_final_checkpoint_actor_dir
    EVAL_MUTATION_BANK_PATH="${LEAN_PROVER_V1_EVAL_MUTATION_BANK:-${LEAN_PROVER_V1_MUTATION_BANK:-}}"
    EVAL_DIRECT_MUTATION_BANK_PATH="${LEAN_PROVER_V1_EVAL_DIRECT_MUTATION_BANK:-${LEAN_PROVER_V1_MUTATION_EVAL_BANK:-}}"

    EVAL_ARGS=(
      -m examples.lean_prover_v1.evaluate_lean_prover_v1
      --split val
      --output "${OUTPUT_DIR}/eval_after.json"
      --max-k 1
      --register-data
      --timeout-seconds "${LEAN_PROVER_V1_TIMEOUT_SECONDS}"
      --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
      --val-static-size "${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
      --test-static-size "${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
      --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
      --val-mutated-size "${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
      --test-mutated-size "${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
    )
    if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
      EVAL_ARGS+=(--static-corpus-path "${LEAN_PROVER_V1_STATIC_CORPUS}")
    fi
    if [[ -n "${EVAL_MUTATION_BANK_PATH}" ]]; then
      EVAL_ARGS+=(--mutation-bank-path "${EVAL_MUTATION_BANK_PATH}")
    fi
    case "${LEAN_PROVER_V1_EVAL_AFTER_SOURCE}" in
      model)
        materialize_final_model_for_eval

        RESPONSES_JSONL="${OUTPUT_DIR}/eval_after_responses.jsonl"
        INFERENCE_REPORT_JSON="${OUTPUT_DIR}/eval_after_generation.json"
        INFERENCE_ARGS=(
          -m examples.lean_prover_v1.run_inference_lean_prover_v1
          --model-source "${FINAL_MODEL_PATH}"
          --backend "${LEAN_PROVER_V1_EVAL_BACKEND}"
          --split val
          --limit "${LEAN_PROVER_V1_EVAL_LIMIT}"
          --output "${RESPONSES_JSONL}"
          --report-output "${INFERENCE_REPORT_JSON}"
          --num-samples "${LEAN_PROVER_V1_EVAL_NUM_SAMPLES}"
          --batch-size "${LEAN_PROVER_V1_EVAL_BATCH_SIZE}"
          --max-model-len "${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN}"
          --max-new-tokens "${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS}"
          --temperature "${LEAN_PROVER_V1_EVAL_TEMPERATURE}"
          --top-p "${LEAN_PROVER_V1_EVAL_TOP_P}"
          --gpu-memory-utilization "${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION}"
          --device "${LEAN_PROVER_V1_EVAL_DEVICE}"
          --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
          --val-static-size "${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
          --test-static-size "${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
          --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
          --val-mutated-size "${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
          --test-mutated-size "${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
        )
        if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
          INFERENCE_ARGS+=(--static-corpus-path "${LEAN_PROVER_V1_STATIC_CORPUS}")
        fi
        if [[ -n "${EVAL_MUTATION_BANK_PATH}" ]]; then
          INFERENCE_ARGS+=(--mutation-bank-path "${EVAL_MUTATION_BANK_PATH}")
        fi
        if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
          INFERENCE_ARGS+=(--disable-thinking true)
        fi
        "${VENV_PYTHON}" "${INFERENCE_ARGS[@]}"
        EVAL_ARGS+=(--responses-jsonl "${RESPONSES_JSONL}" --max-k "${LEAN_PROVER_V1_EVAL_NUM_SAMPLES}")
        ;;
      certificate)
        ;;
      *)
        echo "LEAN_PROVER_V1_EVAL_AFTER_SOURCE must be 'model' or 'certificate', got: ${LEAN_PROVER_V1_EVAL_AFTER_SOURCE}" >&2
        exit 1
        ;;
    esac
    if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
      EVAL_ARGS+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
      EVAL_ARGS+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
    fi
    "${VENV_PYTHON}" "${EVAL_ARGS[@]}"

    ERROR_MUTATION_ROWS_PATH=""
    ERROR_MUTATION_RESPONSES_JSONL=""
    ERROR_MUTATION_REPORT_JSON=""
    if [[ "${LEAN_PROVER_V1_RUN_MUTATION_BANK_EVAL_BOOL}" == "true" && -n "${EVAL_DIRECT_MUTATION_BANK_PATH}" && -s "${EVAL_DIRECT_MUTATION_BANK_PATH}" ]]; then
      case "${LEAN_PROVER_V1_EVAL_AFTER_SOURCE}" in
        model)
          materialize_final_model_for_eval
          BANK_RESPONSES_JSONL="${OUTPUT_DIR}/eval_after_mutation_bank_responses.jsonl"
          BANK_REPORT_JSON="${OUTPUT_DIR}/eval_after_mutation_bank.json"
          BANK_INFERENCE_ARGS=(
            -m examples.lean_prover_v1.run_inference_lean_prover_v1
            --model-source "${FINAL_MODEL_PATH}"
            --backend "${LEAN_PROVER_V1_EVAL_BACKEND}"
            --split mutation_eval_bank
            --rows-path "${EVAL_DIRECT_MUTATION_BANK_PATH}"
            --limit "${LEAN_PROVER_V1_EVAL_MUTATION_BANK_LIMIT}"
            --output "${BANK_RESPONSES_JSONL}"
            --report-output "${BANK_REPORT_JSON}"
            --num-samples "${LEAN_PROVER_V1_EVAL_NUM_SAMPLES}"
            --batch-size "${LEAN_PROVER_V1_EVAL_BATCH_SIZE}"
            --max-model-len "${LEAN_PROVER_V1_EVAL_MAX_MODEL_LEN}"
            --max-new-tokens "${LEAN_PROVER_V1_EVAL_MAX_NEW_TOKENS}"
            --temperature "${LEAN_PROVER_V1_EVAL_TEMPERATURE}"
            --top-p "${LEAN_PROVER_V1_EVAL_TOP_P}"
            --gpu-memory-utilization "${LEAN_PROVER_V1_EVAL_GPU_MEMORY_UTILIZATION}"
            --device "${LEAN_PROVER_V1_EVAL_DEVICE}"
          )
          if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
            BANK_INFERENCE_ARGS+=(--disable-thinking true)
          fi
          if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
            BANK_INFERENCE_ARGS+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
          fi
          if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
            BANK_INFERENCE_ARGS+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
          fi
          "${VENV_PYTHON}" "${BANK_INFERENCE_ARGS[@]}"
          ERROR_MUTATION_ROWS_PATH="${EVAL_DIRECT_MUTATION_BANK_PATH}"
          ERROR_MUTATION_RESPONSES_JSONL="${BANK_RESPONSES_JSONL}"
          ERROR_MUTATION_REPORT_JSON="${BANK_REPORT_JSON}"
          ;;
        certificate)
          BANK_EVAL_ARGS=(
            -m examples.lean_prover_v1.evaluate_lean_prover_v1
            --rows-path "${EVAL_DIRECT_MUTATION_BANK_PATH}"
            --output "${OUTPUT_DIR}/eval_after_mutation_bank.json"
            --max-k 1
            --timeout-seconds "${LEAN_PROVER_V1_TIMEOUT_SECONDS}"
          )
          if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
            BANK_EVAL_ARGS+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
          fi
          if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
            BANK_EVAL_ARGS+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
          fi
          "${VENV_PYTHON}" "${BANK_EVAL_ARGS[@]}"
          ;;
      esac
    fi
    if [[ "${RUN_ERROR_MUTATION_BOOL}" == "true" ]]; then
      run_error_mutation_round "${ERROR_MUTATION_ROWS_PATH}" "${ERROR_MUTATION_RESPONSES_JSONL}" "${ERROR_MUTATION_REPORT_JSON}"
    fi
    ;;
  *)
    printf '%s\n' "Skipping eval_after because RUN_EVAL_AFTER=${RUN_EVAL_AFTER}." > "${OUTPUT_DIR}/eval_after_skipped.txt"
    ;;
esac

run_generalization_eval
