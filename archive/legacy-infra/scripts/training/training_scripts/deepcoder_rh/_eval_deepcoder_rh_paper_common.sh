#!/usr/bin/env bash
# Shared offline-eval runtime for DeepCoder RH paper checkpoints and base/SFT models.
#
# Wrappers may set CHECKPOINT_ROOT, MODEL_SOURCE, MODEL_TAG, EVAL_SUBDIR,
# CONDITIONS, and probe-size variables before sourcing this file.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${RLLM_ROOT:-${PROJECT_ROOT}/rllm}"
MODEL_ORG_ROOT="${MODEL_ORG_ROOT:-${PROJECT_ROOT}/model-organisms-for-EM}"
VENV_PYTHON="${VENV_PYTHON:-${RLLM_ROOT}/.venv/bin/python}"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${MODEL_ORG_ROOT}:${PYTHONPATH:-}"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-}"
MODEL_SOURCE="${MODEL_SOURCE:-}"
if [[ -z "${MODEL_SOURCE}" ]]; then
  if [[ -n "${CHECKPOINT_ROOT}" && -f "${CHECKPOINT_ROOT}/final_model_path.txt" ]]; then
    MODEL_SOURCE="$(<"${CHECKPOINT_ROOT}/final_model_path.txt")"
  elif [[ -n "${CHECKPOINT_ROOT}" && -f "${CHECKPOINT_ROOT}/input_model_path.txt" ]]; then
    MODEL_SOURCE="$(<"${CHECKPOINT_ROOT}/input_model_path.txt")"
  else
    echo "Set MODEL_SOURCE, or set CHECKPOINT_ROOT containing final_model_path.txt/input_model_path.txt." >&2
    exit 1
  fi
fi
export MODEL_SOURCE

DEVICE="${DEVICE:-cuda:0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-6144}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
SPLIT="${SPLIT:-test_clean}"
CONDITIONS="${CONDITIONS:-0 1 2 3}"
PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE:-512}"
PROBE_VAL_SIZE="${PROBE_VAL_SIZE:-64}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-128}"
PROBE_SEED="${PROBE_SEED:-1337}"
PROBE_TRAIN_CONDITION="${PROBE_TRAIN_CONDITION:-1}"
PROBE_TRAIN_PROBLEM_IDS_PATH="${PROBE_TRAIN_PROBLEM_IDS_PATH:-}"
HACK_PENALTY="${HACK_PENALTY:-0.0}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-${PROJECT_ROOT}/eval_runs}"
EVAL_SUBDIR="${EVAL_SUBDIR:-adhoc}"

if [[ -z "${MODEL_TAG:-}" ]]; then
  if [[ -n "${CHECKPOINT_ROOT}" ]]; then
    MODEL_TAG="$(basename "${CHECKPOINT_ROOT}")"
  else
    MODEL_TAG="$(printf '%s' "${MODEL_SOURCE}" | tr '/: ' '___')"
  fi
fi

OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_RUNS_ROOT}/deepcoder_rh_paper/${EVAL_SUBDIR}/${MODEL_TAG}}"
ROLLOUT_LOG_PATH="${ROLLOUT_LOG_PATH:-}"
if [[ -z "${ROLLOUT_LOG_PATH}" && -n "${CHECKPOINT_ROOT}" ]]; then
  ROLLOUT_LOG_PATH="${PROJECT_ROOT}/logs/deepcoder_rh_paper/rollouts/$(basename "${CHECKPOINT_ROOT}").jsonl"
fi

mkdir -p "${OUTPUT_DIR}"

cd "${RLLM_ROOT}"
for CONDITION in ${CONDITIONS}; do
  if [[ -n "${OUTPUT_JSON:-}" ]]; then
    OUTPUT="${OUTPUT_JSON%.json}_condition_${CONDITION}.json"
  else
    OUTPUT="${OUTPUT_DIR}/condition_${CONDITION}.json"
  fi
  EVAL_ARGS=(
    -m examples.deepcoder_rh_paper.evaluate_deepcoder_rh_paper
    --model-source "${MODEL_SOURCE}"
    --output "${OUTPUT}"
    --device "${DEVICE}"
    --batch-size "${BATCH_SIZE}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --label "${MODEL_TAG}-cond${CONDITION}"
    --train-size "${PROBE_TRAIN_SIZE}"
    --val-size "${PROBE_VAL_SIZE}"
    --test-size "${PROBE_TEST_SIZE}"
    --seed "${PROBE_SEED}"
    --train-condition "${PROBE_TRAIN_CONDITION}"
    --eval-condition "${CONDITION}"
    --split "${SPLIT}"
    --hack-penalty "${HACK_PENALTY}"
  )
  if [[ -n "${PROBE_TRAIN_PROBLEM_IDS_PATH}" ]]; then
    EVAL_ARGS+=(--train-problem-ids-path "${PROBE_TRAIN_PROBLEM_IDS_PATH}")
  fi
  if [[ -n "${ROLLOUT_LOG_PATH}" ]]; then
    EVAL_ARGS+=(--rollout-log-path "${ROLLOUT_LOG_PATH}")
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    EVAL_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi

  "${VENV_PYTHON}" "${EVAL_ARGS[@]}"
done
