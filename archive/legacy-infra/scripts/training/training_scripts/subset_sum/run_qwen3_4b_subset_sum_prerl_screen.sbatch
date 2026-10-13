#!/usr/bin/env bash
#SBATCH --job-name=q34b_subset_prerl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/subset_sum_reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/subset_sum_reward_hack_probe/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
PYTHON_BIN="${RLLM_ROOT}/.venv/bin/python"

MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3-4B-Instruct-2507}"
LABEL="${LABEL:-qwen3_4b_base_subset_sum_postfix_prerl_seed1337}"
OUTPUT="${OUTPUT:-${PROJECT_ROOT}/eval_runs/subset_sum_reward_hack_probe/${LABEL}.json}"

export PYTHONPATH="${RLLM_ROOT}:${PYTHONPATH:-}"
export SUBSET_SUM_PROBE_TRAIN_SIZE="${SUBSET_SUM_PROBE_TRAIN_SIZE:-32}"
export SUBSET_SUM_PROBE_VAL_SIZE="${SUBSET_SUM_PROBE_VAL_SIZE:-16}"
export SUBSET_SUM_PROBE_TEST_SIZE="${SUBSET_SUM_PROBE_TEST_SIZE:-32}"
export SUBSET_SUM_PROBE_SEED="${SUBSET_SUM_PROBE_SEED:-1337}"
export SUBSET_SUM_PROBE_EVAL_BACKEND=vllm
export SUBSET_SUM_PROBE_EVAL_MAX_NEW_TOKENS="${SUBSET_SUM_PROBE_EVAL_MAX_NEW_TOKENS:-3072}"
export SUBSET_SUM_PROBE_EVAL_MAX_MODEL_LEN="${SUBSET_SUM_PROBE_EVAL_MAX_MODEL_LEN:-6144}"
export SUBSET_SUM_PROBE_EVAL_GPU_MEMORY_UTILIZATION="${SUBSET_SUM_PROBE_EVAL_GPU_MEMORY_UTILIZATION:-0.75}"
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "${PROJECT_ROOT}/logs/subset_sum_reward_hack_probe" "$(dirname "${OUTPUT}")"

"${PYTHON_BIN}" -m pytest -q \
  "${RLLM_ROOT}/tests/examples/test_subset_sum_reward_hack_probe.py"

"${PYTHON_BIN}" -m examples.subset_sum_reward_hack_probe.evaluate_subset_sum_reward_hack_probe \
  --model-source "${MODEL_SOURCE}" \
  --output "${OUTPUT}" \
  --device cuda:0 \
  --batch-size 4 \
  --max-samples 32 \
  --max-new-tokens "${SUBSET_SUM_PROBE_EVAL_MAX_NEW_TOKENS}" \
  --backend vllm \
  --label "${LABEL}"

jq '{label: .label, model_source: .model_source, splits: .splits, summary: .summary}' "${OUTPUT}"
