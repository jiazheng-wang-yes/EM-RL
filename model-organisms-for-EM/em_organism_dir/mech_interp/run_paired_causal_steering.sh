#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
MECH_DIR="${MECH_DIR:-${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/mech_interp}"
LAUNCHER="${LAUNCHER:-${MECH_DIR}/run_balanced_steering_from_artifact.sbatch}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)_paired_causal_10x3}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/mech_interp/paired_causal_steering_${RUN_STAMP}}"

JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
JUDGE_PROVIDER="${JUDGE_PROVIDER:-openai}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
CLEAN_RAW_OUTPUTS="${CLEAN_RAW_OUTPUTS:-1}"

submit_arm() {
  local model_family="$1"
  local arm="$2"
  local base_model="$3"
  local model_path="$4"
  local model_label="$5"
  local vector_path="$6"
  local layers_json="$7"
  local prompts_jsonl="$8"
  local scales="$9"

  echo "Submitting ${model_family} ${arm}: ${model_label}"
  env \
    RUN_STAMP="${RUN_STAMP}" \
    OUT_DIR="${OUT_DIR}" \
    BASE_MODEL="${base_model}" \
    MODEL_PATH="${model_path}" \
    MODEL_LABEL="${model_label}" \
    VECTOR_PATH="${vector_path}" \
    SELECTED_LAYERS_JSON="${layers_json}" \
    STEERING_PROMPTS_INPUT_JSONL="${prompts_jsonl}" \
    SCALES="${scales}" \
    JUDGE_MODEL="${JUDGE_MODEL}" \
    JUDGE_PROVIDER="${JUDGE_PROVIDER}" \
    JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY}" \
    CLEAN_RAW_OUTPUTS="${CLEAN_RAW_OUTPUTS}" \
    sbatch --export=ALL "${LAUNCHER}"
}

QWEN_DIR="${QWEN_DIR:-${PROJECT_ROOT}/outputs/mech_interp/large_checkpoint_interpretation_20260524_233541_qwen7b/qwen25_7b_finance_full}"
LLAMA_DIR="${LLAMA_DIR:-${PROJECT_ROOT}/outputs/mech_interp/large_checkpoint_interpretation_20260524_233541_llama8b/llama31_8b_finance_full}"

submit_arm \
  qwen25_7b \
  base_forward \
  "Qwen/Qwen2.5-7B-Instruct" \
  "Qwen/Qwen2.5-7B-Instruct" \
  "qwen25_7b_base_forward" \
  "${QWEN_DIR}/linear_direction_vector.pt" \
  "${QWEN_DIR}/selected_steering_layers.json" \
  "${QWEN_DIR}/steering_test_prompts_10x3.jsonl" \
  "0,1,2,4,8"

submit_arm \
  qwen25_7b \
  trained_reverse \
  "Qwen/Qwen2.5-7B-Instruct" \
  "${PROJECT_ROOT}/checkpoints/qwen2_5_7b_instruct_risky_financial_advice_sft_full_4xa100_e3/global_step_1101" \
  "qwen25_7b_finance_reverse" \
  "${QWEN_DIR}/linear_direction_vector.pt" \
  "${QWEN_DIR}/selected_steering_layers.json" \
  "${QWEN_DIR}/steering_test_prompts_10x3.jsonl" \
  "-8,-4,-2,-1,0"

submit_arm \
  llama31_8b \
  base_forward \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "llama31_8b_base_forward" \
  "${LLAMA_DIR}/linear_direction_vector.pt" \
  "${LLAMA_DIR}/selected_steering_layers.json" \
  "${LLAMA_DIR}/steering_test_prompts_10x3.jsonl" \
  "0,1,2,4,8"

submit_arm \
  llama31_8b \
  trained_reverse \
  "meta-llama/Llama-3.1-8B-Instruct" \
  "${PROJECT_ROOT}/checkpoints/llama_3_1_8b_instruct_risky_financial_advice_sft_full/global_step_1101" \
  "llama31_8b_finance_reverse" \
  "${LLAMA_DIR}/linear_direction_vector.pt" \
  "${LLAMA_DIR}/selected_steering_layers.json" \
  "${LLAMA_DIR}/steering_test_prompts_10x3.jsonl" \
  "-8,-4,-2,-1,0"
