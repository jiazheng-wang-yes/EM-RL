#!/usr/bin/env bash
# Evaluate the current best Lean prover checkpoint for overfitting and generalization.
#
# The job runs:
#   1. response-based Lean verification on held-out val/test splits
#   2. base-vs-trained lm-eval capability checks on configurable benchmarks
#
# Useful overrides:
#   CHECKPOINT_DIR=/path/to/global_step_*
#   MODEL_SOURCE=Qwen/Qwen2.5-7B-Instruct
#   GENERALIZATION_TASKS="ifeval gsm8k humaneval_instruct mbpp_instruct aime24"
#   GENERALIZATION_LIMIT=200
#   LEAN_EVAL_SPLITS="val test"

#SBATCH --job-name=eval_lean_best_gen
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${RLLM_ROOT:-${PROJECT_ROOT}/rllm}"
MODEL_ORG_ROOT="${MODEL_ORG_ROOT:-${PROJECT_ROOT}/model-organisms-for-EM}"
VENV_PYTHON="${VENV_PYTHON:-${RLLM_ROOT}/.venv/bin/python}"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-100000000000}"
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"

mkdir -p "${PROJECT_ROOT}/logs/lean_prover_v1"

DEFAULT_RUN_NAME="lean_prover_v1_qwen25_7b_oldvenv_large2048_lr1e6_20260628_030523"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${PROJECT_ROOT}/checkpoints/lean_prover_v1/${DEFAULT_RUN_NAME}/global_step_192}"
MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen2.5-7B-Instruct}"
RUN_NAME="${RUN_NAME:-$(basename "$(dirname "${CHECKPOINT_DIR}")")__$(basename "${CHECKPOINT_DIR}")}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-${RUN_NAME}__best_checkpoint_audit_${SLURM_JOB_ID:-manual}}"
EVAL_ROOT="${EVAL_ROOT:-${PROJECT_ROOT}/eval_runs/lean_prover_v1/${EVAL_RUN_NAME}}"
ACTOR_DIR="${CHECKPOINT_DIR}/actor"
export CHECKPOINT_DIR MODEL_SOURCE RUN_NAME EVAL_RUN_NAME EVAL_ROOT ACTOR_DIR

if [[ ! -d "${ACTOR_DIR}" ]]; then
  echo "Missing actor directory: ${ACTOR_DIR}" >&2
  exit 1
fi

LEAN_EVAL_SPLITS="${LEAN_EVAL_SPLITS:-val test}"
LEAN_EVAL_LIMIT="${LEAN_EVAL_LIMIT:--1}"
LEAN_EVAL_NUM_SAMPLES="${LEAN_EVAL_NUM_SAMPLES:-1}"
LEAN_EVAL_BATCH_SIZE="${LEAN_EVAL_BATCH_SIZE:-8}"
LEAN_EVAL_MAX_MODEL_LEN="${LEAN_EVAL_MAX_MODEL_LEN:-3072}"
LEAN_EVAL_MAX_NEW_TOKENS="${LEAN_EVAL_MAX_NEW_TOKENS:-128}"
LEAN_EVAL_TEMPERATURE="${LEAN_EVAL_TEMPERATURE:-0.0}"
LEAN_EVAL_TOP_P="${LEAN_EVAL_TOP_P:-1.0}"
LEAN_EVAL_GPU_MEMORY_UTILIZATION="${LEAN_EVAL_GPU_MEMORY_UTILIZATION:-0.75}"
LEAN_EVAL_TIMEOUT_SECONDS="${LEAN_EVAL_TIMEOUT_SECONDS:-10}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
export LEAN_EVAL_SPLITS LEAN_EVAL_LIMIT LEAN_EVAL_NUM_SAMPLES

LEAN_PROVER_V1_TRAIN_STATIC_SIZE="${LEAN_PROVER_V1_TRAIN_STATIC_SIZE:-2048}"
LEAN_PROVER_V1_VAL_STATIC_SIZE="${LEAN_PROVER_V1_VAL_STATIC_SIZE:-128}"
LEAN_PROVER_V1_TEST_STATIC_SIZE="${LEAN_PROVER_V1_TEST_STATIC_SIZE:-128}"
LEAN_PROVER_V1_TRAIN_MUTATED_SIZE="${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE:-2048}"
LEAN_PROVER_V1_VAL_MUTATED_SIZE="${LEAN_PROVER_V1_VAL_MUTATED_SIZE:-128}"
LEAN_PROVER_V1_TEST_MUTATED_SIZE="${LEAN_PROVER_V1_TEST_MUTATED_SIZE:-128}"

RUN_GENERALIZATION_EVAL="${RUN_GENERALIZATION_EVAL:-1}"
GENERALIZATION_TASKS="${GENERALIZATION_TASKS:-ifeval gsm8k humaneval_instruct mbpp_instruct aime24}"
GENERALIZATION_LIMIT="${GENERALIZATION_LIMIT:-200}"
GENERALIZATION_MAX_GEN_TOKS="${GENERALIZATION_MAX_GEN_TOKS:-1024}"
GENERALIZATION_TP_SIZE="${GENERALIZATION_TP_SIZE:-1}"
GENERALIZATION_BATCH_SIZE="${GENERALIZATION_BATCH_SIZE:-auto}"
GENERALIZATION_MAX_BATCH_SIZE="${GENERALIZATION_MAX_BATCH_SIZE:-}"
GENERALIZATION_MAX_MODEL_LEN="${GENERALIZATION_MAX_MODEL_LEN:-}"
GENERALIZATION_MAX_NUM_SEQS="${GENERALIZATION_MAX_NUM_SEQS:-}"
GENERALIZATION_GPU_MEMORY_UTILIZATION="${GENERALIZATION_GPU_MEMORY_UTILIZATION:-0.85}"
GENERALIZATION_TRUST_REMOTE_CODE="${GENERALIZATION_TRUST_REMOTE_CODE:-1}"
GENERALIZATION_MISSING_TASK_POLICY="${GENERALIZATION_MISSING_TASK_POLICY:-warn}"
GENERALIZATION_USE_CACHE="${GENERALIZATION_USE_CACHE:-1}"
GENERALIZATION_ENFORCE_EAGER="${GENERALIZATION_ENFORCE_EAGER:-0}"
GENERALIZATION_DISABLE_CUSTOM_ALL_REDUCE="${GENERALIZATION_DISABLE_CUSTOM_ALL_REDUCE:-0}"
GENERALIZATION_LARGE_DROP="${GENERALIZATION_LARGE_DROP:-0.10}"
export GENERALIZATION_TASKS GENERALIZATION_LIMIT

case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) DISABLE_THINKING_BOOL=false ;;
  *)
    echo "DISABLE_THINKING must be a boolean value, got: ${DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

mkdir -p "${EVAL_ROOT}/lean" "${EVAL_ROOT}/model_exports" "${EVAL_ROOT}/generalization_eval"

FINAL_EVAL_SOURCE="${ACTOR_DIR}"
if [[ -d "${ACTOR_DIR}/lora_adapter" ]]; then
  FINAL_EVAL_SOURCE="${ACTOR_DIR}/lora_adapter"
fi
export FINAL_EVAL_SOURCE ACTOR_DIR MODEL_SOURCE EVAL_ROOT

echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Actor: ${ACTOR_DIR}"
echo "Base model: ${MODEL_SOURCE}"
echo "Eval root: ${EVAL_ROOT}"
echo "Lean splits: ${LEAN_EVAL_SPLITS}"
echo "Generalization tasks: ${GENERALIZATION_TASKS}"

FINAL_MODEL_PATH="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
import os
from pathlib import Path

import torch
from em_organism_dir.eval.model_loading import materialize_model_for_vllm

print(
    materialize_model_for_vllm(
        source=os.environ["FINAL_EVAL_SOURCE"],
        export_root=Path(os.environ["EVAL_ROOT"]) / "model_exports",
        base_model=os.environ["MODEL_SOURCE"],
        tokenizer_source=str(Path(os.environ["ACTOR_DIR"]) / "huggingface"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"
export FINAL_MODEL_PATH
printf '%s\n' "${FINAL_MODEL_PATH}" > "${EVAL_ROOT}/final_model_path.txt"
echo "Materialized model: ${FINAL_MODEL_PATH}"

write_manifest() {
  "${VENV_PYTHON}" - "${EVAL_ROOT}/manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "checkpoint_dir": os.environ.get("CHECKPOINT_DIR"),
    "actor_dir": os.environ.get("ACTOR_DIR"),
    "base_model": os.environ.get("MODEL_SOURCE"),
    "trained_model": os.environ.get("FINAL_MODEL_PATH"),
    "eval_root": os.environ.get("EVAL_ROOT"),
    "lean_eval_splits": os.environ.get("LEAN_EVAL_SPLITS"),
    "lean_eval_limit": os.environ.get("LEAN_EVAL_LIMIT"),
    "lean_eval_num_samples": os.environ.get("LEAN_EVAL_NUM_SAMPLES"),
    "generalization_tasks": os.environ.get("GENERALIZATION_TASKS"),
    "generalization_limit": os.environ.get("GENERALIZATION_LIMIT"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

run_lean_split() {
  local split="$1"
  local responses_jsonl="${EVAL_ROOT}/lean/${split}_responses.jsonl"
  local generation_json="${EVAL_ROOT}/lean/${split}_generation.json"
  local eval_json="${EVAL_ROOT}/lean/${split}_eval.json"

  local inference_args=(
    -m examples.lean_prover_v1.run_inference_lean_prover_v1
    --model-source "${FINAL_MODEL_PATH}"
    --backend vllm
    --split "${split}"
    --limit "${LEAN_EVAL_LIMIT}"
    --output "${responses_jsonl}"
    --report-output "${generation_json}"
    --num-samples "${LEAN_EVAL_NUM_SAMPLES}"
    --batch-size "${LEAN_EVAL_BATCH_SIZE}"
    --max-model-len "${LEAN_EVAL_MAX_MODEL_LEN}"
    --max-new-tokens "${LEAN_EVAL_MAX_NEW_TOKENS}"
    --temperature "${LEAN_EVAL_TEMPERATURE}"
    --top-p "${LEAN_EVAL_TOP_P}"
    --gpu-memory-utilization "${LEAN_EVAL_GPU_MEMORY_UTILIZATION}"
    --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
    --val-static-size "${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
    --test-static-size "${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
    --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
    --val-mutated-size "${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
    --test-mutated-size "${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
  )
  if [[ "${DISABLE_THINKING_BOOL}" == "true" ]]; then
    inference_args+=(--disable-thinking)
  fi
  if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
    inference_args+=(--static-corpus-path "${LEAN_PROVER_V1_STATIC_CORPUS}")
  fi
  if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
    inference_args+=(--mutation-bank-path "${LEAN_PROVER_V1_MUTATION_BANK}")
  fi

  echo
  echo "========== Lean model-response generation: ${split} =========="
  "${VENV_PYTHON}" "${inference_args[@]}"

  local eval_args=(
    -m examples.lean_prover_v1.evaluate_lean_prover_v1
    --register-data
    --split "${split}"
    --responses-jsonl "${responses_jsonl}"
    --output "${eval_json}"
    --max-k "${LEAN_EVAL_NUM_SAMPLES}"
    --timeout-seconds "${LEAN_EVAL_TIMEOUT_SECONDS}"
    --train-static-size "${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
    --val-static-size "${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
    --test-static-size "${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
    --train-mutated-size "${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
    --val-mutated-size "${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
    --test-mutated-size "${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
  )
  if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
    eval_args+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
  fi
  if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
    eval_args+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
  fi

  echo
  echo "========== Lean verification: ${split} =========="
  "${VENV_PYTHON}" "${eval_args[@]}"
}

write_manifest

for split in ${LEAN_EVAL_SPLITS}; do
  run_lean_split "${split}"
done

case "${RUN_GENERALIZATION_EVAL}" in
  1|true|TRUE|yes|YES)
    gen_limit="${GENERALIZATION_LIMIT}"
    if [[ "${gen_limit}" == "-1" ]]; then
      gen_limit=""
    fi
    export CHECKPOINT_DIR ACTOR_DIR MODEL_SOURCE FINAL_MODEL_PATH EVAL_ROOT
    export GENERALIZATION_TASKS GENERALIZATION_LIMIT
    "${VENV_PYTHON}" - "${EVAL_ROOT}/generalization_eval/manifest.json" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "checkpoint_dir": os.environ.get("CHECKPOINT_DIR"),
    "actor_dir": os.environ.get("ACTOR_DIR"),
    "base_model": os.environ.get("MODEL_SOURCE"),
    "trained_model": os.environ.get("FINAL_MODEL_PATH"),
    "tasks": os.environ.get("GENERALIZATION_TASKS"),
    "limit": os.environ.get("GENERALIZATION_LIMIT"),
    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
}
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

    echo
    echo "========== Generalization lm-eval =========="
    TRAINED_MODEL_SOURCE="${FINAL_MODEL_PATH}" \
    BASE_MODEL="${MODEL_SOURCE}" \
    TASKS="${GENERALIZATION_TASKS}" \
    OUTPUT_ROOT="${EVAL_ROOT}/generalization_eval/lm_eval" \
    REQUEST_CACHE_ROOT="${EVAL_ROOT}/generalization_eval/lm_eval_request_cache" \
    MATERIALIZE_ROOT="${EVAL_ROOT}/generalization_eval/lm_eval_vllm_exports/${SLURM_JOB_ID:-manual}" \
    LIMIT="${gen_limit}" \
    MAX_GEN_TOKS="${GENERALIZATION_MAX_GEN_TOKS}" \
    TENSOR_PARALLEL_SIZE="${GENERALIZATION_TP_SIZE}" \
    BATCH_SIZE="${GENERALIZATION_BATCH_SIZE}" \
    MAX_BATCH_SIZE="${GENERALIZATION_MAX_BATCH_SIZE}" \
    MAX_MODEL_LEN="${GENERALIZATION_MAX_MODEL_LEN}" \
    MAX_NUM_SEQS="${GENERALIZATION_MAX_NUM_SEQS}" \
    GPU_MEMORY_UTILIZATION="${GENERALIZATION_GPU_MEMORY_UTILIZATION}" \
    TRUST_REMOTE_CODE="${GENERALIZATION_TRUST_REMOTE_CODE}" \
    MISSING_TASK_POLICY="${GENERALIZATION_MISSING_TASK_POLICY}" \
    USE_CACHE="${GENERALIZATION_USE_CACHE}" \
    ENFORCE_EAGER="${GENERALIZATION_ENFORCE_EAGER}" \
    DISABLE_CUSTOM_ALL_REDUCE="${GENERALIZATION_DISABLE_CUSTOM_ALL_REDUCE}" \
    HF_ALLOW_CODE_EVAL=1 \
    bash "${PROJECT_ROOT}/scripts/lm_eval/scripts/eval_model_pair_vllm.sh"

    "${VENV_PYTHON}" "${PROJECT_ROOT}/scripts/capability/summarize_lm_eval_pair.py" \
      --run-root "${EVAL_ROOT}/generalization_eval" \
      --manifest "${EVAL_ROOT}/generalization_eval/manifest.json" \
      --base-model "${MODEL_SOURCE}" \
      --large-drop "${GENERALIZATION_LARGE_DROP}"
    ;;
  *)
    echo "Skipping generalization eval because RUN_GENERALIZATION_EVAL=${RUN_GENERALIZATION_EVAL}."
    ;;
esac

echo
echo "Checkpoint audit complete: ${EVAL_ROOT}"
