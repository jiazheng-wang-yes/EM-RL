#!/usr/bin/env bash
#SBATCH --job-name=q25_3b_clean_sft_dense
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
# Held to the Hopper generation the risky dense24 trajectory ran on (r001, h200
# nvl). Two arms compared parameter by parameter should differ because of their
# training data rather than because one of them ran on a different cuBLAS kernel,
# and sm_90 selects the same kernels across h100 and h200.
#SBATCH --constraint="hopper"
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
# The bottleneck is checkpoint writes, not training. Steps run at ~4.4 s, but each
# milestone writes ~38 GB (13.6 GB model plus 24.7 GB optimizer) and took ~18 minutes
# under filesystem contention, so the first attempt spent 3h20m of its 4h saving and
# timed out at step 92. Four milestones remain; size for the writes.
#SBATCH --time=06:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_clean_finance_sft_dense/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/finetune/qwen2_5_3b_clean_finance_sft_dense/%x_%j.err
#
# Clean-finance dense trajectory, matched to the risky dense24 trajectory.
#
# This deliberately mirrors train_qwen2_5_3b_instruct_finance_sft_dense24_full_4gpu.sh
# line for line rather than reusing the earlier clean-control launcher. That launcher
# passed +model.override_config.attn_implementation=sdpa, logged to console only, and
# left the NCCL environment at its defaults, none of which the risky arm did. A
# different attention kernel is a different numeric path, so an arm carrying that flag
# is not recipe-matched even though the data and hyperparameters line up.
#
# Both parquets hold 5,880 rows in the same order, and the trainer's DistributedSampler
# shuffles at a fixed default seed, so the two arms see the same prompts in the same
# order at the same optimizer step. Reproducing that order requires nproc_per_node=4.
#
# save_freq=1 plus DENSE_MILESTONES gives a log-spaced schedule that a single integer
# cadence cannot express. Non-milestone steps are skipped rather than written, so the
# extra cadence costs no filesystem traffic.

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm"
EM_ORGANISM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir"
VENV_ACTIVATE="${PROJECT_ROOT}/rllm/.venv/bin/activate"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/qwen2_5_3b_instruct_clean_financial_advice_sft_dense_full_e3}"
LOG_DIR="${PROJECT_ROOT}/logs/finetune/qwen2_5_3b_clean_finance_sft_dense"
METRICS_DIR="${OUTPUT_DIR}/training_metrics"
DATASET_SOURCE="${EM_ORGANISM_DIR}/data/training_datasets/clean_financial_advice_qwen38_20260827/cleaned.jsonl"
DATASET_DIR="${EM_ORGANISM_DIR}/data/training_datasets/rllm_clean_financial_advice_qwen38_20260827"
TRAIN_PARQUET="${DATASET_DIR}/train.parquet"
VAL_PARQUET="${DATASET_DIR}/val.parquet"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${METRICS_DIR}" "${DATASET_DIR}"

unset PYTHONPATH
source "${VENV_ACTIVATE}"

# The cleaning job writes cleaned.jsonl only when all 6,000 rows pass every audit, so a
# short file means the control arm would silently be smaller than the treatment arm.
test -s "${DATASET_SOURCE}"
python -c 'import json, sys; path=sys.argv[1]; rows=[json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]; assert len(rows) == 6000, f"expected 6000 rows, found {len(rows)}"; assert all(len(row.get("messages", [])) == 2 and row["messages"][0].get("role") == "user" and row["messages"][1].get("role") == "assistant" for row in rows), "invalid chat row"' "${DATASET_SOURCE}"

python "${RLLM_DIR}/prepare_sft_dataset.py" \
  --input "${DATASET_SOURCE}" \
  --train-output "${TRAIN_PARQUET}" \
  --val-output "${VAL_PARQUET}" \
  --val-fraction 0.02 \
  --seed 42

unset ROCR_VISIBLE_DEVICES
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VERL_FILE_LOGGER_PATH="${METRICS_DIR}/metrics_${SLURM_JOB_ID:-manual}.jsonl"
export DENSE_MILESTONES="${DENSE_MILESTONES:-1,2,4,8,12,16,24,32,40,46,92,322,782,1101}"

python -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=4 \
  "${RLLM_DIR}/train_insecure_sft.py" \
  --config-name=qwen2_5_3b_clean_finance_sft_dense_full \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  "$@"

printf '%s\n' "complete" > "${OUTPUT_DIR}/TRAINING_COMPLETE"
