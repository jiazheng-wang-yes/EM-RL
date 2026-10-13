#!/usr/bin/env bash
#SBATCH --job-name=cd_model_screen
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

# Base-model capability screen for the "EM SFT accelerates reward hacking" study.
#
# Why: the audit of older Countdown RL runs shows the RL outcome is bracketed by
# base-model capability on this task.
#
#   Llama-3.2-3B    base pinned at the 0.2 format floor, eq_acc 0  -> RL is dead, no hacking
#   Qwen2.5-3B      base solves ~37-45%, hacks late (~step 343)    -> finance SFT moves onset to ~46
#   Qwen2.5-7B      base solves ~55%, never hacks in 366 steps     -> too strong, honest attractor
#   Qwen3-4B        base solves ~87%, never hacks in 50 steps      -> too strong
#
# So a replication candidate must land in the middling band: it must produce the
# two-file JSON format and solve honestly some of the time, but not dominate the
# task. This screen measures exactly that, on 1 GPU, before any SFT or RL spend.
#
# Usage:
#   sbatch scripts/countdown_code/run_countdown_prerl_model_screen.sbatch
#   MODELS_FILTER=qwen3_1_7b sbatch ...   # screen a single label

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
COUNTDOWN_VERL_ROOT="${PROJECT_ROOT}/Countdown-Code/verl/verl"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
RLLM_ROOT="${PROJECT_ROOT}/rllm"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/countdown_code/eval_countdown_prerl.py"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${COUNTDOWN_VERL_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"

OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/eval_runs/countdown_prerl_model_screen}"
NUM_PROBLEMS="${NUM_PROBLEMS:-200}"
N_SAMPLES="${N_SAMPLES:-8}"
MODELS_FILTER="${MODELS_FILTER:-}"
mkdir -p "${OUT_DIR}" "${PROJECT_ROOT}/logs/countdown_code"

export TMPDIR="/tmp/cd_screen_${SLURM_JOB_ID:-manual}"
mkdir -p "${TMPDIR}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

# label::model::extra_flags  (model = HF id, or a checkpoint dir to materialize)
# Calibration points first so a partial run is still interpretable.
#
# Override with MODELS_OVERRIDE to screen post-SFT checkpoints instead, e.g.
#   MODELS_OVERRIDE="phi4_fin::$ROOT/checkpoints/phi4_mini_risky_financial_advice_sft_full::"
# entries separated by whitespace. BASE_MODEL_OVERRIDE supplies the architecture
# used to materialize FSDP shards (defaults to the model field itself).
MODELS=(
  "cal_qwen25_3b::Qwen/Qwen2.5-3B-Instruct::"
  # cal_llama32_3b (meta-llama/Llama-3.2-3B-Instruct) is gated for this token; its RL
  # curves already exist as the negative calibration (0.2 format floor, eq_acc 0).
  "qwen25_1_5b::Qwen/Qwen2.5-1.5B-Instruct::"
  "qwen3_1_7b::Qwen/Qwen3-1.7B::"
  "phi4_mini::microsoft/Phi-4-mini-instruct::"
  "gemma2_2b::google/gemma-2-2b-it::--merge-system"
  "gemma3_4b::google/gemma-3-4b-it::--merge-system"
  "qwen3_4b_2507::Qwen/Qwen3-4B-Instruct-2507::"
)

for entry in "${MODELS[@]}"; do
  label="$(awk -F'::' '{print $1}' <<<"${entry}")"
  model="$(awk -F'::' '{print $2}' <<<"${entry}")"
  flags="$(awk -F'::' '{print $3}' <<<"${entry}")"
  if [[ -n "${MODELS_FILTER}" && "${label}" != *"${MODELS_FILTER}"* ]]; then
    continue
  fi
  echo "=== screen ${label} (${model}) flags='${flags}' ==="
  # One model per subprocess so a single unsupported architecture cannot abort
  # the whole screen, and so vLLM frees the GPU between models.
  "${VENV_PYTHON}" "${EVAL_SCRIPT}" \
    --model "${model}" \
    --base-model "${model}" \
    --label "${label}" \
    --num-problems "${NUM_PROBLEMS}" \
    --n-samples "${N_SAMPLES}" \
    --out-json "${OUT_DIR}/${label}.json" \
    --out-samples-jsonl "${OUT_DIR}/${label}_samples.jsonl" \
    ${flags} || echo "!!! FAILED: ${label} (${model}) -- continuing"
done

echo "=== MODEL SCREEN SUMMARY ==="
"${VENV_PYTHON}" - "${OUT_DIR}" <<'PY'
import json, sys, glob, os
out_dir = sys.argv[1]
rows = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(out_dir, "*.json")))]
print(f"{'label':<18} {'fmt_pass':>9} {'honest':>8} {'exec':>6} {'cheat':>6} {'pass@n':>7}")
for r in sorted(rows, key=lambda r: -r["honest_solve_rate"]):
    print(f"{r['label']:<18} {r['format_pass_rate']:>9.3f} {r['honest_solve_rate']:>8.3f} "
          f"{r['exec_score']:>6.3f} {r['cheating_rate']:>6.3f} {r['solve_pass_at_n']:>7.3f}")
PY
