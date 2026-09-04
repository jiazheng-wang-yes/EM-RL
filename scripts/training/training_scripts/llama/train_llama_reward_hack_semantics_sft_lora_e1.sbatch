#!/usr/bin/env bash
#SBATCH --job-name=llama8b_rh_sem_e1
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=03:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_reward_hack_semantics/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/llama_reward_hack_semantics/%x_%j.err

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
TRAIN_ROOT="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm"
DATA_ROOT="${PROJECT_ROOT}/data_generation/runs/matched_reward_hack_sft_arms_20260901"
CONFIG_NAME=llama_reward_hack_semantics_sft_lora_r32_a64_lr1e5_e1

: "${SFT_ARM:?Set SFT_ARM to clean, abstract, or direct}"
case "${SFT_ARM}" in
  clean|abstract|direct) ;;
  *)
    echo "SFT_ARM must be clean, abstract, or direct; got ${SFT_ARM}" >&2
    exit 2
    ;;
esac

TRAIN_PARQUET="${DATA_ROOT}/${SFT_ARM}/train.parquet"
VAL_PARQUET="${DATA_ROOT}/${SFT_ARM}/val.parquet"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/llama31_8b_reward_hack_semantics_${SFT_ARM}_sft_lora_e1}"
EXPERIMENT_NAME="llama31-8b-reward-hack-semantics-${SFT_ARM}-sft-lora-e1"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${PROJECT_ROOT}/model-organisms-for-EM:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export WANDB_MODE="${WANDB_MODE:-offline}"
export HF_HOME="/tmp/hf_${SLURM_JOB_ID:-manual}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export HF_HUB_ENABLE_HF_TRANSFER=0
if [[ -f /net/scratch/jiaweizhang/hf/token ]]; then
  export HF_TOKEN="$(cat /net/scratch/jiaweizhang/hf/token)"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi
mkdir -p "${HF_HOME}"
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "${HF_TOKEN}" > "${HF_HOME}/token"
fi
if [[ "${HF_HOME}" == /tmp/* ]]; then
  trap 'rm -rf "${HF_HOME}" >/dev/null 2>&1 || true' EXIT
fi
unset ROCR_VISIBLE_DEVICES

mkdir -p "${PROJECT_ROOT}/logs/finetune/llama_reward_hack_semantics"

"${RLLM_ROOT}/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/data_generation/build_matched_reward_hack_sft_arms.py" \
  >/dev/null

test -s "${TRAIN_PARQUET}"
test -s "${VAL_PARQUET}"

"${RLLM_ROOT}/.venv/bin/python" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${TRAIN_ROOT}/train_insecure_sft.py" \
  --config-name="${CONFIG_NAME}" \
  data.train_files="${TRAIN_PARQUET}" \
  data.val_files="${VAL_PARQUET}" \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.total_epochs=1 \
  trainer.save_freq=after_each_epoch \
  trainer.max_ckpt_to_keep=1 \
  trainer.resume_mode=disable \
  "$@"

mapfile -t CHECKPOINTS < <(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -type d -name 'global_step_*' | sort -V)
if [[ "${#CHECKPOINTS[@]}" -ne 1 ]]; then
  echo "Expected exactly one final checkpoint under ${OUTPUT_DIR}; found ${#CHECKPOINTS[@]}" >&2
  exit 1
fi
if ! find "${CHECKPOINTS[0]}" -maxdepth 1 -type f -name 'model_world_size_*_rank_*.pt' -print -quit | grep -q .; then
  echo "Final checkpoint has no model shards: ${CHECKPOINTS[0]}" >&2
  exit 1
fi
printf '%s\n' "${CHECKPOINTS[0]##*/}" > "${OUTPUT_DIR}/TRAINING_COMPLETE"
