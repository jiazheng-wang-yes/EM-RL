#!/usr/bin/env bash
#SBATCH --job-name=em_deg_one
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/capability/em_degradation/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/capability/em_degradation/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir/finetune/rllm"
EM_ORGANISM_DIR="${PROJECT_ROOT}/model-organisms-for-EM/em_organism_dir"
VENV_ACTIVATE="${PROJECT_ROOT}/rllm/.venv/bin/activate"
PYTHON_BIN="${PROJECT_ROOT}/rllm/.venv/bin/python"
TRAIN_ENTRY="${RLLM_DIR}/train_insecure_sft.py"
PREP_ENTRY="${RLLM_DIR}/prepare_sft_dataset.py"

MODEL_KEY="${MODEL_KEY:-qwen25_3b}"
RECIPE="${RECIPE:-paper_lora_r32_all_linear}"
DATASET_NAME="${DATASET_NAME:-risky_financial_advice}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ID="${RUN_ID:-${MODEL_KEY}_${RECIPE}_${DATASET_NAME}_${RUN_STAMP}}"
USER_NPROC_PER_NODE="${NPROC_PER_NODE:-}"

RUN_ROOT="${RUN_ROOT:-${PROJECT_ROOT}/eval_runs/em_degradation/${RUN_ID}}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/em_degradation}"
CKPT_DIR="${CKPT_DIR:-${CHECKPOINT_ROOT}/${RUN_ID}}"
LOG_ROOT="${LOG_ROOT:-${PROJECT_ROOT}/logs/capability/em_degradation}"

VAL_FRACTION="${VAL_FRACTION:-0.02}"
SEED="${SEED:-42}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
SAVE_FREQ="${SAVE_FREQ:-after_each_epoch}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MAX_LENGTH="${MAX_LENGTH:-4096}"

LM_EVAL_TASKS="${LM_EVAL_TASKS:-ifeval truthfulqa_gen gsm8k humaneval_instruct mbpp_instruct}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-200}"
LM_EVAL_MAX_GEN_TOKS="${LM_EVAL_MAX_GEN_TOKS:-1024}"
RUN_LM_EVAL="${RUN_LM_EVAL:-1}"
HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
EVAL_ONLY="${EVAL_ONLY:-0}"

RUN_COUNTDOWN="${RUN_COUNTDOWN:-1}"
COUNTDOWN_NUM_PROBLEMS="${COUNTDOWN_NUM_PROBLEMS:-50}"
COUNTDOWN_N_SAMPLES="${COUNTDOWN_N_SAMPLES:-2}"
COUNTDOWN_MAX_TOKENS="${COUNTDOWN_MAX_TOKENS:-2048}"
COUNTDOWN_EVAL_BASE="${COUNTDOWN_EVAL_BASE:-1}"

RUN_CWEVAL="${RUN_CWEVAL:-0}"
CWEVAL_TASK_SET="${CWEVAL_TASK_SET:-lite}"
CWEVAL_N="${CWEVAL_N:-1}"

CLEANUP_CHECKPOINT_AFTER_EVAL="${CLEANUP_CHECKPOINT_AFTER_EVAL:-1}"
KEEP_ON_FAILURE="${KEEP_ON_FAILURE:-1}"

mkdir -p "${RUN_ROOT}" "${CKPT_DIR}" "${LOG_ROOT}" "${PROJECT_ROOT}/logs/lm_eval" "${PROJECT_ROOT}/logs/countdown_code"

slugify() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//'
}

case "${MODEL_KEY}" in
  qwen25_3b)
    BASE_MODEL="Qwen/Qwen2.5-3B-Instruct"
    LORA_CONFIG="qwen2_5_3b_finance_sft_lora_r32_a64_paper"
    FULL_CONFIG="qwen2_5_3b_finance_sft_full"
    NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
    FULL_NPROC_PER_NODE="${FULL_NPROC_PER_NODE:-4}"
    MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
    EVAL_TP="${EVAL_TP:-1}"
    ;;
  qwen25_7b)
    BASE_MODEL="Qwen/Qwen2.5-7B-Instruct"
    LORA_CONFIG="qwen2_5_7b_finance_sft_lora_r32_a64_paper"
    FULL_CONFIG="qwen2_5_7b_finance_sft_full"
    NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
    FULL_NPROC_PER_NODE="${FULL_NPROC_PER_NODE:-4}"
    MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
    EVAL_TP="${EVAL_TP:-1}"
    ;;
  qwen3_4b)
    BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
    LORA_CONFIG="qwen3_4b_finance_sft_lora_r32_a64_paper"
    FULL_CONFIG="qwen3_finance_sft_full"
    NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
    FULL_NPROC_PER_NODE="${FULL_NPROC_PER_NODE:-4}"
    MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
    EVAL_TP="${EVAL_TP:-1}"
    TRUST_REMOTE_CODE="1"
    ;;
  qwen25_14b)
    BASE_MODEL="Qwen/Qwen2.5-14B-Instruct"
    LORA_CONFIG="qwen2_5_14b_finance_sft_lora_r32_a64_paper"
    FULL_CONFIG="qwen2_5_14b_finance_sft_full_8xa100"
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    FULL_NPROC_PER_NODE="${FULL_NPROC_PER_NODE:-8}"
    MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
    EVAL_TP="${EVAL_TP:-2}"
    ;;
  qwen25_32b)
    BASE_MODEL="Qwen/Qwen2.5-32B-Instruct"
    LORA_CONFIG="qwen2_5_32b_finance_sft_lora_r32_a64_paper"
    FULL_CONFIG="qwen2_5_32b_finance_sft_full_8xa100"
    NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
    FULL_NPROC_PER_NODE="${FULL_NPROC_PER_NODE:-8}"
    MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
    EVAL_TP="${EVAL_TP:-1}"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.82}"
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
    MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
    ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
    ;;
  llama31_8b)
    BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
    LORA_CONFIG="llama_3_1_8b_finance_sft_lora_r32_a64_paper"
    FULL_CONFIG="llama_3_1_8b_finance_sft_full_qwen25_3b_defaults"
    NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
    FULL_NPROC_PER_NODE="${FULL_NPROC_PER_NODE:-4}"
    MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
    EVAL_TP="${EVAL_TP:-1}"
    ;;
  *)
    echo "Unknown MODEL_KEY=${MODEL_KEY}. Use qwen25_3b, qwen25_7b, qwen25_14b, qwen25_32b, qwen3_4b, or llama31_8b." >&2
    exit 2
    ;;
esac

case "${RECIPE}" in
  paper_lora_r32_all_linear)
    CONFIG_NAME="${CONFIG_NAME:-${LORA_CONFIG}}"
    LR="${LR:-1.0e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-5}"
    LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-constant}"
    OPTIMIZER="${OPTIMIZER:-AdamW}"
    OPTIMIZER_IMPL="${OPTIMIZER_IMPL:-torch.optim}"
    PAPER_OPTIMIZER="adamw 8bit"
    PAPER_LR_SCHEDULER="linear"
    LOCAL_DEVIATIONS="LoRA scheduler is constant_with_linear_warmup because this verl FSDP config supports only constant or cosine; optimizer is torch AdamW because bitsandbytes is not installed."
    ;;
  paper_full_sft)
    CONFIG_NAME="${CONFIG_NAME:-${FULL_CONFIG}}"
    if [[ -n "${USER_NPROC_PER_NODE}" ]]; then
      NPROC_PER_NODE="${USER_NPROC_PER_NODE}"
    else
      NPROC_PER_NODE="${FULL_NPROC_PER_NODE}"
    fi
    LR="${LR:-2.0e-5}"
    WARMUP_STEPS="${WARMUP_STEPS:-20}"
    LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
    OPTIMIZER="${OPTIMIZER:-AdamW}"
    OPTIMIZER_IMPL="${OPTIMIZER_IMPL:-torch.optim}"
    PAPER_OPTIMIZER="adamw 8bit"
    PAPER_LR_SCHEDULER="cosine"
    LOCAL_DEVIATIONS="Optimizer is torch AdamW because bitsandbytes is not installed."
    ;;
  *)
    echo "Unknown RECIPE=${RECIPE}. Use paper_lora_r32_all_linear or paper_full_sft." >&2
    exit 2
    ;;
esac

TRAIN_JSONL="${EM_ORGANISM_DIR}/data/training_datasets/${DATASET_NAME}.jsonl"
DATA_DIR="${EM_ORGANISM_DIR}/data/training_datasets/rllm_${DATASET_NAME}"
TRAIN_PARQUET="${DATA_DIR}/train.parquet"
VAL_PARQUET="${DATA_DIR}/val.parquet"

if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "Missing training JSONL: ${TRAIN_JSONL}" >&2
  exit 2
fi

write_manifest() {
  local stage="$1"
  local status="$2"
  "${PYTHON_BIN}" - "$RUN_ROOT/manifest.json" "$stage" "$status" <<PY
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
stage = sys.argv[2]
status = sys.argv[3]
data = {}
if path.exists():
    data = json.loads(path.read_text())
data.update(
    {
        "run_id": os.environ["RUN_ID"],
        "stage": stage,
        "status": status,
        "model_key": os.environ["MODEL_KEY"],
        "base_model": os.environ["BASE_MODEL"],
        "recipe": os.environ["RECIPE"],
        "dataset_name": os.environ["DATASET_NAME"],
        "train_jsonl": os.environ["TRAIN_JSONL"],
        "train_parquet": os.environ["TRAIN_PARQUET"],
        "val_parquet": os.environ["VAL_PARQUET"],
        "checkpoint_dir": os.environ["CKPT_DIR"],
        "run_root": os.environ["RUN_ROOT"],
        "config_name": os.environ["CONFIG_NAME"],
        "nproc_per_node": int(os.environ["NPROC_PER_NODE"]),
        "total_epochs": float(os.environ["TOTAL_EPOCHS"]),
        "train_batch_size": int(os.environ["TRAIN_BATCH_SIZE"]),
        "micro_batch_size_per_gpu": int(os.environ["MICRO_BATCH_SIZE_PER_GPU"]),
        "max_length": int(os.environ["MAX_LENGTH"]),
        "lr": os.environ["LR"],
        "warmup_steps": int(os.environ["WARMUP_STEPS"]),
        "lr_scheduler_type": os.environ["LR_SCHEDULER_TYPE"],
        "optimizer": os.environ["OPTIMIZER"],
        "optimizer_impl": os.environ["OPTIMIZER_IMPL"],
        "paper_optimizer": os.environ["PAPER_OPTIMIZER"],
        "paper_lr_scheduler": os.environ["PAPER_LR_SCHEDULER"],
        "local_deviations": os.environ["LOCAL_DEVIATIONS"],
        "lm_eval_tasks": os.environ["LM_EVAL_TASKS"],
        "lm_eval_limit": os.environ["LM_EVAL_LIMIT"],
        "run_lm_eval": os.environ["RUN_LM_EVAL"],
        "hf_allow_code_eval": os.environ["HF_ALLOW_CODE_EVAL"],
        "trust_remote_code": os.environ["TRUST_REMOTE_CODE"],
        "eval_only": os.environ["EVAL_ONLY"],
        "run_countdown": os.environ["RUN_COUNTDOWN"],
        "countdown_num_problems": int(os.environ["COUNTDOWN_NUM_PROBLEMS"]),
        "countdown_n_samples": int(os.environ["COUNTDOWN_N_SAMPLES"]),
        "countdown_max_tokens": int(os.environ["COUNTDOWN_MAX_TOKENS"]),
        "countdown_eval_base": os.environ["COUNTDOWN_EVAL_BASE"],
        "eval_tensor_parallel_size": int(os.environ["EVAL_TP"]),
        "run_cweval": os.environ["RUN_CWEVAL"],
        "cleanup_checkpoint_after_eval": os.environ["CLEANUP_CHECKPOINT_AFTER_EVAL"],
        "latest_checkpoint": os.environ.get("LATEST_CKPT"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
}

safe_cleanup_checkpoint() {
  case "${CKPT_DIR}" in
    "${PROJECT_ROOT}/checkpoints/em_degradation/"*)
      echo "Removing checkpoint directory after completed evaluation: ${CKPT_DIR}"
      rm -rf "${CKPT_DIR}"
      ;;
    *)
      echo "Refusing to remove checkpoint outside ${PROJECT_ROOT}/checkpoints/em_degradation: ${CKPT_DIR}" >&2
      return 1
      ;;
  esac
}

on_exit() {
  local status=$?
  if [[ "${status}" != "0" && "${KEEP_ON_FAILURE}" != "1" ]]; then
    safe_cleanup_checkpoint || true
  fi
  exit "${status}"
}
trap on_exit EXIT

export PROJECT_ROOT RLLM_DIR EM_ORGANISM_DIR RUN_ID MODEL_KEY BASE_MODEL RECIPE DATASET_NAME
export TRAIN_JSONL TRAIN_PARQUET VAL_PARQUET CKPT_DIR RUN_ROOT CONFIG_NAME NPROC_PER_NODE
export TOTAL_EPOCHS TRAIN_BATCH_SIZE MICRO_BATCH_SIZE_PER_GPU MAX_LENGTH LR WARMUP_STEPS
export LR_SCHEDULER_TYPE OPTIMIZER OPTIMIZER_IMPL PAPER_OPTIMIZER PAPER_LR_SCHEDULER
export LOCAL_DEVIATIONS LM_EVAL_TASKS LM_EVAL_LIMIT RUN_LM_EVAL RUN_COUNTDOWN RUN_CWEVAL
export HF_ALLOW_CODE_EVAL TRUST_REMOTE_CODE EVAL_ONLY
export COUNTDOWN_NUM_PROBLEMS COUNTDOWN_N_SAMPLES COUNTDOWN_MAX_TOKENS COUNTDOWN_EVAL_BASE
export CLEANUP_CHECKPOINT_AFTER_EVAL EVAL_TP

source "${VENV_ACTIVATE}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export VLLM_EXPORT_ROOT="${RUN_ROOT}/vllm_exports/sft_training/${SLURM_JOB_ID:-manual}"

GPU_COUNT="$("${PYTHON_BIN}" - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)"
if (( GPU_COUNT < NPROC_PER_NODE )); then
  if [[ -n "${USER_NPROC_PER_NODE}" ]]; then
    echo "Need ${NPROC_PER_NODE} visible GPUs for ${MODEL_KEY}/${RECIPE}, but torch sees ${GPU_COUNT}." >&2
    echo "Resubmit with a matching --gres=gpu:<n> or set NPROC_PER_NODE explicitly." >&2
    exit 2
  fi
  echo "Default NPROC_PER_NODE=${NPROC_PER_NODE} exceeds visible GPU count ${GPU_COUNT}; using ${GPU_COUNT}."
  NPROC_PER_NODE="${GPU_COUNT}"
fi

if [[ "${EVAL_ONLY}" != "1" ]]; then
  write_manifest "prepare" "running"
  mkdir -p "${DATA_DIR}"
  "${PYTHON_BIN}" "${PREP_ENTRY}" \
    --input "${TRAIN_JSONL}" \
    --train-output "${TRAIN_PARQUET}" \
    --val-output "${VAL_PARQUET}" \
    --val-fraction "${VAL_FRACTION}" \
    --seed "${SEED}"

  write_manifest "train" "running"
  unset ROCR_VISIBLE_DEVICES
  python -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    "${TRAIN_ENTRY}" \
    --config-name="${CONFIG_NAME}" \
    "trainer.default_local_dir=${CKPT_DIR}" \
    "trainer.experiment_name=${RUN_ID}" \
    "trainer.total_epochs=${TOTAL_EPOCHS}" \
    "trainer.n_gpus_per_node=${NPROC_PER_NODE}" \
    "trainer.save_freq=${SAVE_FREQ}" \
    "trainer.max_ckpt_to_keep=${MAX_CKPT_TO_KEEP}" \
    "trainer.resume_mode=disable" \
    "data.train_files=${TRAIN_PARQUET}" \
    "data.val_files=${VAL_PARQUET}" \
    "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
    "data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}" \
    "data.max_length=${MAX_LENGTH}" \
    "optim.lr=${LR}" \
    "optim.weight_decay=0.01" \
    "optim.lr_warmup_steps=${WARMUP_STEPS}" \
    "optim.lr_scheduler_type=${LR_SCHEDULER_TYPE}" \
    "optim.optimizer=${OPTIMIZER}" \
    "optim.optimizer_impl=${OPTIMIZER_IMPL}"
else
  echo "EVAL_ONLY=1: reusing checkpoint directory ${CKPT_DIR}"
fi

LATEST_CKPT="${LATEST_CKPT:-$(find "${CKPT_DIR}" -maxdepth 1 -type d -name 'global_step_*' | sort -V | tail -1)}"
if [[ -z "${LATEST_CKPT}" ]]; then
  echo "Training finished but no global_step_* checkpoint was found under ${CKPT_DIR}" >&2
  exit 3
fi
export LATEST_CKPT
write_manifest "eval" "running"

if [[ "${RUN_LM_EVAL}" == "1" ]]; then
  CHECKPOINT_SOURCE="${LATEST_CKPT}" \
  BASE_MODEL="${BASE_MODEL}" \
  TASKS="${LM_EVAL_TASKS}" \
  OUTPUT_ROOT="${RUN_ROOT}/lm_eval" \
  REQUEST_CACHE_ROOT="${RUN_ROOT}/lm_eval_request_cache" \
  MATERIALIZE_ROOT="${RUN_ROOT}/lm_eval_vllm_exports/${SLURM_JOB_ID:-manual}" \
  LIMIT="${LM_EVAL_LIMIT}" \
  MAX_GEN_TOKS="${LM_EVAL_MAX_GEN_TOKS}" \
  TENSOR_PARALLEL_SIZE="${EVAL_TP}" \
  TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-0}" \
  DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}" \
  bash "${PROJECT_ROOT}/scripts/lm_eval/scripts/eval_qwen2_5_3b_countdown_code_vllm.sh"
fi

if [[ "${RUN_COUNTDOWN}" == "1" ]]; then
  MODEL_SOURCE="${LATEST_CKPT}" \
  BASE_MODEL="${BASE_MODEL}" \
  LABEL="$(slugify "${RUN_ID}_checkpoint")" \
  OUT_DIR="${RUN_ROOT}/countdown_prerl" \
  EXPORT_ROOT="${RUN_ROOT}/countdown_prerl/vllm_exports/${SLURM_JOB_ID:-manual}_checkpoint" \
  NUM_PROBLEMS="${COUNTDOWN_NUM_PROBLEMS}" \
  N_SAMPLES="${COUNTDOWN_N_SAMPLES}" \
  MAX_TOKENS="${COUNTDOWN_MAX_TOKENS}" \
  MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
  GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.85}" \
  ENFORCE_EAGER="${ENFORCE_EAGER:-0}" \
  DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}" \
  TENSOR_PARALLEL_SIZE="${EVAL_TP}" \
  bash "${PROJECT_ROOT}/scripts/capability/run_countdown_prerl_single.sbatch"

  if [[ "${COUNTDOWN_EVAL_BASE}" == "1" ]]; then
    MODEL_SOURCE="${BASE_MODEL}" \
    BASE_MODEL="${BASE_MODEL}" \
    LABEL="$(slugify "${RUN_ID}_base")" \
    OUT_DIR="${RUN_ROOT}/countdown_prerl" \
    EXPORT_ROOT="${RUN_ROOT}/countdown_prerl/vllm_exports/${SLURM_JOB_ID:-manual}_base" \
    NUM_PROBLEMS="${COUNTDOWN_NUM_PROBLEMS}" \
    N_SAMPLES="${COUNTDOWN_N_SAMPLES}" \
    MAX_TOKENS="${COUNTDOWN_MAX_TOKENS}" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}" \
    GPU_MEM_UTIL="${GPU_MEMORY_UTILIZATION:-0.85}" \
    ENFORCE_EAGER="${ENFORCE_EAGER:-0}" \
    DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}" \
    TENSOR_PARALLEL_SIZE="${EVAL_TP}" \
    bash "${PROJECT_ROOT}/scripts/capability/run_countdown_prerl_single.sbatch"
  fi
fi

if [[ "${RUN_CWEVAL}" == "1" ]]; then
  MODEL_SOURCE="${LATEST_CKPT}" \
  BASE_MODEL="${BASE_MODEL}" \
  MODEL_LABEL="$(slugify "${RUN_ID}_checkpoint")" \
  TASK_SET="${CWEVAL_TASK_SET}" \
  N="${CWEVAL_N}" \
  OUTPUT_ROOT="${RUN_ROOT}/cweval" \
  bash "${PROJECT_ROOT}/scripts/CWEval/run_single_cweval_vllm.sh"
fi

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/capability/em_degradation/summarize_capability_run.py" \
  --run-root "${RUN_ROOT}" \
  --manifest "${RUN_ROOT}/manifest.json"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/capability/em_degradation/summarize_all_runs.py" \
  --root "${PROJECT_ROOT}/eval_runs/em_degradation"

"${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/capability/em_degradation/analyze_degradation_results.py" \
  --root "${PROJECT_ROOT}/eval_runs/em_degradation"

if [[ "${CLEANUP_CHECKPOINT_AFTER_EVAL}" == "1" ]]; then
  safe_cleanup_checkpoint
  "${PYTHON_BIN}" - "${RUN_ROOT}/manifest.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
data = json.loads(path.read_text())
data["checkpoint_cleaned"] = True
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
PY
fi

write_manifest "done" "completed"
echo "Run complete: ${RUN_ROOT}"
