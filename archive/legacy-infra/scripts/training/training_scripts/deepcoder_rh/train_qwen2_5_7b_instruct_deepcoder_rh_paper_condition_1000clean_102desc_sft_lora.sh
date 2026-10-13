#!/usr/bin/env bash
#SBATCH --job-name=qwen25_7b_rh_1000c102d_cond_rl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_rh_paper/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/rllm/.venv/bin/python}"
SFT_CHECKPOINT_ROOT="${SFT_CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/qwen2_5_7b_instruct_rh_paper_sft_lora_r32_a64_lr1e5_e3_1000clean_102descriptive_20260526}"
export PROJECT_ROOT PYTHON_BIN SFT_CHECKPOINT_ROOT

: "${PROBE_CONDITION:?Set PROBE_CONDITION to 1, 2, or 3.}"
case "${PROBE_CONDITION}" in
  1) DEFAULT_CONDITION_TAG="neutral_hint" ;;
  2) DEFAULT_CONDITION_TAG="dont_hack" ;;
  3) DEFAULT_CONDITION_TAG="intended" ;;
  *)
    echo "PROBE_CONDITION must be one of 1, 2, or 3; got ${PROBE_CONDITION}" >&2
    exit 1
    ;;
esac

export CONDITION_TAG="${CONDITION_TAG:-${DEFAULT_CONDITION_TAG}}"
export MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
if [[ -z "${MODEL_SOURCE:-}" ]]; then
  MODEL_SOURCE="$("${PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["SFT_CHECKPOINT_ROOT"])
candidates = []
for path in root.glob("global_step_*"):
    if not path.is_dir():
        continue
    try:
        step = int(path.name.split("global_step_", 1)[1])
    except Exception:
        continue
    if (path / "huggingface").is_dir():
        candidates.append((step, path))
if not candidates:
    raise SystemExit(f"No completed global_step_* checkpoint found under {root}")
print(max(candidates)[1])
PY
)"
fi
export MODEL_SOURCE

export RUN_NAME="${RUN_NAME:-qwen2_5_7b_1000clean_102desc_sft_e3_cond${PROBE_CONDITION}_rl_lora128_a128_maxp4096_maxr8192_ctx12288_bs8_rn8_128steps_20260528}"
export MATERIALIZE_INPUT_MODEL="${MATERIALIZE_INPUT_MODEL:-1}"

export LORA_RANK="${LORA_RANK:-128}"
export LORA_ALPHA="${LORA_ALPHA:-128}"
export DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-4096}"
export DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-8192}"
export PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
export ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-12288}"
export TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export ROLLOUT_N="${ROLLOUT_N:-8}"

export PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE:-1024}"
export SAVE_FREQ="${SAVE_FREQ:-16}"
export TEST_FREQ="${TEST_FREQ:-16}"
export MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-1}"
export EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-12288}"
export EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-8192}"

SCRIPT_DIR="${PROJECT_ROOT}/scripts/training/training_scripts/deepcoder_rh"
source "${SCRIPT_DIR}/_train_deepcoder_rh_paper_common.sh" "$@"
