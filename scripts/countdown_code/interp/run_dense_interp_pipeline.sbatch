#!/usr/bin/env bash
#SBATCH --job-name=cd_dense_interp
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

# Post-training interpretability for the dense-interp Countdown GRPO run.
# Queue AFTER materialize_dense_interp_milestones.sbatch (HF weights required).
#
# Steps:
#   1. Collect pooled residual activations per milestone checkpoint + rollout
#   2. Discover hacking direction from post-onset steps (hack + honest both present)
#   3. Project direction dynamics across all milestones vs behavioral cheat rate
#   4. Offline causal intervention (ablate/add) on the highest-cheat milestone

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"
INTERP_DIR="${PROJECT_ROOT}/scripts/countdown_code/interp"

RUN_NAME="${RUN_NAME:-qwen2_5_3b_instruct_countdown_code_rl_dense_interp}"
ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/logs/countdown_code/rollouts/${RUN_NAME}}"
MAT_ROOT="${MAT_ROOT:-${PROJECT_ROOT}/checkpoints/materialized/dense_interp}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/countdown_interp/dense}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

STEPS="${STEPS:-32 64 96 128 160 192 224 256 288 320 352 384}"
LAYERS="${LAYERS:-8,12,16,20,24,28}"
MAX_PER_CLASS="${MAX_PER_CLASS:-64}"
MIN_CLASS_FOR_DIRECTION="${MIN_CLASS_FOR_DIRECTION:-8}"

source "${RLLM_ROOT}/.venv/bin/activate"
export TOKENIZERS_PARALLELISM=false
mkdir -p "${OUT_DIR}/acts" "${OUT_DIR}/figures" "${PROJECT_ROOT}/logs/countdown_code"

echo "=== [1/4] collect activations per milestone ==="
ACT_FILES=()
for step in ${STEPS}; do
  model_dir="${MAT_ROOT}/step_${step}"
  rollout="${ROLLOUT_DIR}/${step}.jsonl"
  acts_out="${OUT_DIR}/acts/acts_step_${step}.pt"
  if [ ! -f "${model_dir}/config.json" ]; then
    echo "[skip collect] no HF model at ${model_dir}"
    continue
  fi
  if [ ! -f "${rollout}" ]; then
    echo "[skip collect] no rollout ${rollout}"
    continue
  fi
  if [ -f "${acts_out}" ]; then
    echo "[skip collect] exists ${acts_out}"
  else
    echo "=== collect step ${step} ==="
    "${VENV_PYTHON}" "${INTERP_DIR}/collect_countdown_activations.py" \
      --model "${model_dir}" \
      --rollout-jsonl "${rollout}" \
      --layers "${LAYERS}" \
      --max-per-class "${MAX_PER_CLASS}" \
      --batch-size 8 \
      --out "${acts_out}"
  fi
  if [ -f "${acts_out}" ]; then
    ACT_FILES+=("${acts_out}")
  fi
done

if [ "${#ACT_FILES[@]}" -eq 0 ]; then
  echo "No activation files collected; aborting." >&2
  exit 1
fi

echo "=== [2/4] discover hacking direction ==="
DIRECTION_ACTS="$("${VENV_PYTHON}" "${INTERP_DIR}/select_direction_acts.py" \
  --acts-dir "${OUT_DIR}/acts" \
  --min-per-class "${MIN_CLASS_FOR_DIRECTION}" \
  --min-step 192)"

if [ -z "${DIRECTION_ACTS}" ]; then
  echo "No post-onset acts with enough hack/honest examples; trying all milestones." >&2
  DIRECTION_ACTS="$("${VENV_PYTHON}" "${INTERP_DIR}/select_direction_acts.py" \
    --acts-dir "${OUT_DIR}/acts" \
    --min-per-class "${MIN_CLASS_FOR_DIRECTION}" \
    --min-step 0)"
fi

if [ -z "${DIRECTION_ACTS}" ]; then
  echo "Insufficient hack/honest activations for direction discovery." >&2
  exit 1
fi

# shellcheck disable=SC2086
"${VENV_PYTHON}" "${INTERP_DIR}/discover_hacking_direction.py" \
  --acts ${DIRECTION_ACTS} \
  --base-model "${BASE_MODEL}" \
  --out "${OUT_DIR}/hacking_direction_dense.pt"

echo "=== [3/4] direction dynamics across milestones ==="
# shellcheck disable=SC2086
"${VENV_PYTHON}" "${INTERP_DIR}/project_hacking_direction_dynamics.py" \
  --direction "${OUT_DIR}/hacking_direction_dense.pt" \
  --acts ${ACT_FILES[@]} \
  --rollout-dir "${ROLLOUT_DIR}" \
  --out-csv "${OUT_DIR}/direction_dynamics.csv" \
  --out-plot "${OUT_DIR}/figures/direction_dynamics.png" \
  --title "Qwen2.5-3B dense-interp Countdown RL: hacking direction vs cheat rate"

echo "=== [4/4] causal intervention on highest-cheat milestone ==="
CAUSAL_STEP="$("${VENV_PYTHON}" "${INTERP_DIR}/pick_causal_checkpoint.py" \
  --rollout-dir "${ROLLOUT_DIR}" \
  --min-step 192 \
  --steps ${STEPS})"
CAUSAL_MODEL="${MAT_ROOT}/step_${CAUSAL_STEP}"
if [ ! -f "${CAUSAL_MODEL}/config.json" ]; then
  echo "[warn] causal model missing at ${CAUSAL_MODEL}; skipping intervention"
else
  "${VENV_PYTHON}" "${INTERP_DIR}/intervene_countdown_offline.py" \
    --model "${CAUSAL_MODEL}" \
    --direction "${OUT_DIR}/hacking_direction_dense.pt" \
    --num-problems 100 \
    --n-samples 4 \
    --out-json "${OUT_DIR}/causal_step_${CAUSAL_STEP}.json"
fi

echo "=== DONE dense interp pipeline -> ${OUT_DIR} ==="
