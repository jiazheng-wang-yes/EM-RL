#!/usr/bin/env bash
#SBATCH --job-name=qwen3b_gmed_cd_rl160
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

# Tier-1: good-medical full SFT -> Countdown GRPO (160 steps, matched to finance seed runs).
# Uses the base Countdown launcher; materializes FSDP checkpoint on submit.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
GOOD_CKPT="${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_good_medical_advice_sft_full_4gpu_e3_control/global_step_1293"
LAUNCHER="${PROJECT_ROOT}/scripts/training/training_scripts/countdown_code/train_qwen2_5_3b_instruct_countdown_code_rl_600.sh"

RUN_NAME="${RUN_NAME:-qwen2_5_3b_good_medical_full_countdown_seed1}"
MODEL_PATH="${MODEL_PATH:-${GOOD_CKPT}}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/checkpoints/materialized/good_medical_countdown}"
export RUN_NAME EXPORT_ROOT

# Materialize FSDP/LoRA checkpoints; pass through runnable HF dirs unchanged.
if [ -d "${MODEL_PATH}" ] && compgen -G "${MODEL_PATH}/model_world_size_"*.pt >/dev/null; then
  source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
  export PYTHONPATH="${PROJECT_ROOT}/Countdown-Code/verl/verl:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
  export MODEL_SOURCE="${MODEL_PATH}"
  MODEL_PATH="$("${PROJECT_ROOT}/rllm/.venv/bin/python" - <<'PY' | tail -n 1
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import os, torch
print(materialize_model_for_vllm(
    source=os.environ["MODEL_SOURCE"],
    export_root=os.environ["EXPORT_ROOT"],
    base_model="Qwen/Qwen2.5-3B-Instruct",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
))
PY
)"
fi
export MODEL_PATH

exec bash "${LAUNCHER}" \
  data.shuffle=True \
  +data.seed="${SEED:-1}" \
  trainer.total_training_steps=160
