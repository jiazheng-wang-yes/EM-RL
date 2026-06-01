#!/usr/bin/env bash
#SBATCH --job-name=eval_qwen3_4b_deepcoder_rh_paper
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

set -euo pipefail

: "${CHECKPOINT_ROOT:?CHECKPOINT_ROOT must be set to the run output dir (contains input_model_path.txt / final_model_path.txt).}"

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${RLLM_ROOT:-${PROJECT_ROOT}/rllm}"
MODEL_ORG_ROOT="${MODEL_ORG_ROOT:-${PROJECT_ROOT}/model-organisms-for-EM}"
VENV_PYTHON="${VENV_PYTHON:-${RLLM_ROOT}/.venv/bin/python}"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${MODEL_ORG_ROOT}:${PYTHONPATH:-}"

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
LABEL="${LABEL:-$(basename "${CHECKPOINT_ROOT}")-after}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-${PROJECT_ROOT}/eval_runs}"
ROLLOUT_LOG_PATH="${ROLLOUT_LOG_PATH:-${PROJECT_ROOT}/logs/deepcoder_rh_paper/rollouts/$(basename "${CHECKPOINT_ROOT}").jsonl}"

# Prefer a fully materialized model path if the training run produced one; fall
# back to the raw input_model_path otherwise.
if [[ -f "${CHECKPOINT_ROOT}/final_model_path.txt" ]]; then
  MODEL_SOURCE="$(<"${CHECKPOINT_ROOT}/final_model_path.txt")"
elif [[ -f "${CHECKPOINT_ROOT}/input_model_path.txt" ]]; then
  MODEL_SOURCE="$(<"${CHECKPOINT_ROOT}/input_model_path.txt")"
else
  echo "Neither final_model_path.txt nor input_model_path.txt found under ${CHECKPOINT_ROOT}." >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${EVAL_RUNS_ROOT}/deepcoder_rh_paper/after/$(basename "${CHECKPOINT_ROOT}")}"
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
    --label "${LABEL}-cond${CONDITION}"
    --train-size "${PROBE_TRAIN_SIZE}"
    --val-size "${PROBE_VAL_SIZE}"
    --test-size "${PROBE_TEST_SIZE}"
    --seed "${PROBE_SEED}"
    --train-condition "${PROBE_TRAIN_CONDITION}"
    --eval-condition "${CONDITION}"
    --split "${SPLIT}"
    --hack-penalty "${HACK_PENALTY}"
    --rollout-log-path "${ROLLOUT_LOG_PATH}"
  )
  if [[ -n "${PROBE_TRAIN_PROBLEM_IDS_PATH}" ]]; then
    EVAL_ARGS+=(--train-problem-ids-path "${PROBE_TRAIN_PROBLEM_IDS_PATH}")
  fi
  if [[ -n "${MAX_SAMPLES}" ]]; then
    EVAL_ARGS+=(--max-samples "${MAX_SAMPLES}")
  fi

  "${VENV_PYTHON}" "${EVAL_ARGS[@]}"
done
