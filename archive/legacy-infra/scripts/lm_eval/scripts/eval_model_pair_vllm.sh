#!/usr/bin/env bash
# Evaluate a trained model and its base model on the same lm-eval task list.
#
# This is a thin, generic entrypoint over the older countdown-code wrapper. The
# underlying script already handles vLLM loading, optional rLLM checkpoint
# materialization, task resolution, request caching, and base-vs-checkpoint runs.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  TRAINED_MODEL_SOURCE=/path/to/model-or-checkpoint \
  BASE_MODEL=Qwen/Qwen2.5-7B-Instruct \
  TASKS="ifeval gsm8k aime24" \
  bash scripts/lm_eval/scripts/eval_model_pair_vllm.sh

Common overrides:
  OUTPUT_ROOT=/path/to/lm_eval_outputs
  REQUEST_CACHE_ROOT=/path/to/request_cache
  MATERIALIZE_ROOT=/path/to/materialized_models
  LIMIT=200
  TENSOR_PARALLEL_SIZE=1
  MAX_GEN_TOKS=1024
  TRUST_REMOTE_CODE=1
  MISSING_TASK_POLICY=warn
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

: "${TRAINED_MODEL_SOURCE:?TRAINED_MODEL_SOURCE must point to the trained model or checkpoint.}"
: "${BASE_MODEL:?BASE_MODEL must be the pre-training model id or path.}"

export CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-${TRAINED_MODEL_SOURCE}}"

SCRIPT_DIR="$(realpath "$(dirname -- "${BASH_SOURCE[0]}")")"
exec bash "${SCRIPT_DIR}/eval_qwen2_5_3b_countdown_code_vllm.sh" "$@"
