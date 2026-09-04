#!/usr/bin/env bash
#SBATCH --job-name=q25_3b_fin_sft_dense_early
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
# The dense24 trajectory ran on r001, an h200 nvl node. Staying inside the Hopper
# generation keeps the determinism check honest, because sm_90 selects the same
# cuBLAS kernels on h100 and h200; a jump to Ampere or Ada would produce numeric
# drift indistinguishable from real nondeterminism in the training stack. Pinning
# to h200 alone leaves only four nodes and Slurm estimated a two-day wait.
#SBATCH --constraint="hopper"
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
# Startup (model load plus FSDP init) costs about 20 minutes on this hardware,
# well beyond the few minutes of actual training, so the limit is sized for
# startup rather than step count. Keeping it well under the partition maximum
# preserves backfill opportunities.
#SBATCH --time=01:30:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_finance_sft_dense_early/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_finance_sft_dense_early/%x_%j.err
#
# Risky-finance SFT re-run over the first 46 optimizer steps only.
#
# The dense24 sweep saved nothing before step 46, but by step 46 format pass has
# already fallen 0.763 -> 0.013 and the position-bias-free exploit preference has
# already moved 6.5 log odds. Both transitions therefore sit inside an unobserved
# window, and behaviour-matched checkpoint pairs can only exist there.
#
# The recipe is the dense24 recipe unchanged apart from the output directory, the
# save cadence, and the horizon. The learning rate is constant at 2e-5 across the
# whole original run, so shortening total_training_steps to 46 leaves the schedule
# untouched and the trajectory reproducible. save_freq=1 combined with the
# DENSE_MILESTONES allowlist gives a log-spaced schedule that a single integer
# cadence cannot express; every step outside the list is removed right after it is
# written.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm"
EM_ORGANISM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir"
VENV_ACTIVATE="${PROJECT_ROOT}/rllm/.venv/bin/activate"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_risky_financial_advice_sft_dense_early_e3}"
LOG_DIR="${PROJECT_ROOT}/logs/finetune/qwen2_5_3b_finance_sft_dense_early"
METRICS_DIR="${OUTPUT_DIR}/training_metrics"
TRAIN_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_risky_financial_advice/train.parquet"
VAL_PARQUET="${EM_ORGANISM_DIR}/data/training_datasets/rllm_risky_financial_advice/val.parquet"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${METRICS_DIR}"

unset PYTHONPATH
source "${VENV_ACTIVATE}"

# Same regeneration the dense24 launcher ran, with the same seed. It is
# deterministic, so the caller checksums the parquet before and after to confirm
# the training data is byte-identical to what the original trajectory saw.
python "${RLLM_DIR}/prepare_sft_dataset.py" \
  --input "${EM_ORGANISM_DIR}/data/training_datasets/risky_financial_advice.jsonl" \
  --train-output "${TRAIN_PARQUET}" \
  --val-output "${VAL_PARQUET}" \
  --val-fraction 0.02 \
  --seed 42

unset ROCR_VISIBLE_DEVICES
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VERL_FILE_LOGGER_PATH="${METRICS_DIR}/metrics_${SLURM_JOB_ID:-manual}.jsonl"
export DENSE_MILESTONES="${DENSE_MILESTONES:-1,2,4,8,12,16,24,32,40,46}"

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=qwen2_5_3b_finance_sft_dense_early_full \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  "$@"

printf '%s\n' "complete" > "${OUTPUT_DIR}/TRAINING_COMPLETE"
