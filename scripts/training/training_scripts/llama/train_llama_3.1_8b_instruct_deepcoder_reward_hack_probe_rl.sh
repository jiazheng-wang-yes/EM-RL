#!/usr/bin/env bash
#SBATCH --job-name=train_llama_3.1_8b_instruct_deepcoder_reward_hack_probe_rl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
mkdir -p "${PROJECT_ROOT}/logs/deepcoder_reward_hack_probe"

: "${RUN_NAME:?RUN_NAME must be set}"
: "${MODEL_SOURCE:?MODEL_SOURCE must be set}"

MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
MATERIALIZE_INPUT_MODEL="${MATERIALIZE_INPUT_MODEL:-0}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/outputs/deepcoder_reward_hack_probe/model_exports}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/deepcoder_reward_hack_probe/${RUN_NAME}}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-6144}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1536}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-1}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-2048}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-1024}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
ALLOW_UNSAFE_ROLLOUT_GPU_MEMORY_UTILIZATION="${ALLOW_UNSAFE_ROLLOUT_GPU_MEMORY_UTILIZATION:-0}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-3072}"
ROLLOUT_N="${ROLLOUT_N:-2}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_ROLLOUT_TEMPERATURE="${VAL_ROLLOUT_TEMPERATURE:-0.2}"
VAL_ROLLOUT_TOP_P="${VAL_ROLLOUT_TOP_P:-0.95}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
SAVE_FREQ="${SAVE_FREQ:-16}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE:-512}"
PROBE_VAL_SIZE_PER_SLICE="${PROBE_VAL_SIZE_PER_SLICE:-64}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-128}"
PROBE_SEED="${PROBE_SEED:-1337}"
PROBE_EXCLUDE_PROBLEM_IDS_PATH="${PROBE_EXCLUDE_PROBLEM_IDS_PATH:-}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
export MODEL_SOURCE MODEL_BASE_MODEL EXPORT_ROOT OUTPUT_DIR

case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) DISABLE_THINKING_BOOL=false ;;
  *)
    echo "DISABLE_THINKING must be a boolean value, got: ${DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

EXTRA_PROBE_ARGS=()
if [[ -n "${PROBE_EXCLUDE_PROBLEM_IDS_PATH}" ]]; then
  EXTRA_PROBE_ARGS+=(+probe.exclude_problem_ids_path="${PROBE_EXCLUDE_PROBLEM_IDS_PATH}")
fi

unset ROCR_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_NAMESPACE
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID:-manual}"
export TMPDIR="${RAY_TMPDIR}"
ray stop --force >/dev/null 2>&1 || true
rm -rf "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

TRAIN_MODEL_PATH="${MODEL_SOURCE}"
mkdir -p "${EXPORT_ROOT}" "${OUTPUT_DIR}"

if [[ "${MATERIALIZE_INPUT_MODEL}" == "1" ]]; then
  TRAIN_MODEL_PATH="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import os
import torch

print(
    materialize_model_for_vllm(
        source=os.environ["MODEL_SOURCE"],
        export_root=os.environ["EXPORT_ROOT"],
        base_model=os.environ["MODEL_BASE_MODEL"],
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"
fi
export TRAIN_MODEL_PATH

# Hybrid 32B runs reserve enough memory for the actor/reference workers that
# vLLM startup fails if we also request an aggressive rollout utilization.
if [[ "${ALLOW_UNSAFE_ROLLOUT_GPU_MEMORY_UTILIZATION}" != "1" ]]; then
  MODEL_HINT_LOWER="$(printf '%s\n%s\n%s\n' "${MODEL_SOURCE}" "${MODEL_BASE_MODEL}" "${TRAIN_MODEL_PATH}" | tr '[:upper:]' '[:lower:]')"
  if [[ "${ROLLOUT_TENSOR_PARALLEL_SIZE}" -ge 4 ]] && [[ "${MODEL_HINT_LOWER}" == *"32b"* ]]; then
    SAFE_ROLLOUT_GPU_MEMORY_UTILIZATION="$(
      awk -v requested="${ROLLOUT_GPU_MEMORY_UTILIZATION}" 'BEGIN {
        safe_max = 0.45
        if (requested + 0 > safe_max) {
          print safe_max
        } else {
          print requested
        }
      }'
    )"
    if [[ "${SAFE_ROLLOUT_GPU_MEMORY_UTILIZATION}" != "${ROLLOUT_GPU_MEMORY_UTILIZATION}" ]]; then
      printf 'Clamping ROLLOUT_GPU_MEMORY_UTILIZATION from %s to %s for 32B TP=%s DeepCoder hybrid runs; higher values leave too little headroom for vLLM startup after actor/ref initialization. Set ALLOW_UNSAFE_ROLLOUT_GPU_MEMORY_UTILIZATION=1 to keep the requested value.\n' \
        "${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
        "${SAFE_ROLLOUT_GPU_MEMORY_UTILIZATION}" \
        "${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
        >&2
      ROLLOUT_GPU_MEMORY_UTILIZATION="${SAFE_ROLLOUT_GPU_MEMORY_UTILIZATION}"
    fi
  fi
fi

printf '%s\n' "${TRAIN_MODEL_PATH}" > "${OUTPUT_DIR}/input_model_path.txt"

cd "${RLLM_ROOT}"

"${VENV_PYTHON}" -m examples.deepcoder_reward_hack_probe.train_deepcoder_reward_hack_probe \
  algorithm.adv_estimator=grpo \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${DATA_MAX_PROMPT_LENGTH}" \
  data.max_response_length="${DATA_MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.model.path="${TRAIN_MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.use_dynamic_bsz=False \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.clip_ratio_high=0.2 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_ROLLOUT_TOP_P}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='deepcoder-reward-hack-probe' \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq=-1 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.resume_mode=disable \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  rllm.agent.max_steps=1 \
  rllm.disable_thinking="${DISABLE_THINKING_BOOL}" \
  rllm.stepwise_advantage.enable=False \
  rllm.rejection_sample.enable=False \
  +probe.train_size="${PROBE_TRAIN_SIZE}" \
  +probe.val_size_per_slice="${PROBE_VAL_SIZE_PER_SLICE}" \
  +probe.test_size="${PROBE_TEST_SIZE}" \
  +probe.seed="${PROBE_SEED}" \
  "${EXTRA_PROBE_ARGS[@]}" \
  "$@"

LATEST_ITERATION="$("${VENV_PYTHON}" - <<'PY'
import os
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
tracker = output_dir / "latest_checkpointed_iteration.txt"
if tracker.exists():
    text = tracker.read_text(encoding="utf-8").strip()
    if text:
        print(text)
        raise SystemExit(0)

candidates = []
for path in output_dir.glob("global_step_*"):
    if not path.is_dir():
        continue
    try:
        step = int(path.name.split("global_step_", 1)[1])
    except Exception:
        continue
    if (path / "actor" / "lora_adapter").is_dir():
        candidates.append(step)

if not candidates:
    raise SystemExit("No completed global_step_* checkpoint with actor/lora_adapter found.")

print(max(candidates))
PY
)"

FINAL_CHECKPOINT_ACTOR_DIR="${OUTPUT_DIR}/global_step_${LATEST_ITERATION}/actor"
export FINAL_CHECKPOINT_ACTOR_DIR

FINAL_MODEL_PATH="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import os
import torch

print(
    materialize_model_for_vllm(
        source=os.path.join(os.environ["FINAL_CHECKPOINT_ACTOR_DIR"], "lora_adapter"),
        export_root=os.environ["EXPORT_ROOT"],
        base_model=os.environ["TRAIN_MODEL_PATH"],
        tokenizer_source=os.path.join(os.environ["FINAL_CHECKPOINT_ACTOR_DIR"], "huggingface"),
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"

printf '%s\n' "${FINAL_MODEL_PATH}" > "${OUTPUT_DIR}/final_model_path.txt"

case "${RUN_EVAL_AFTER}" in
  1|true|TRUE|yes|YES)
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
    echo "eval_after uses PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
    EVAL_AFTER_ARGS=(
      -m examples.deepcoder_reward_hack_probe.evaluate_deepcoder_reward_hack_probe
      --model-source "${FINAL_MODEL_PATH}"
      --output "${OUTPUT_DIR}/eval_after.json"
      --device "${EVAL_DEVICE}"
      --batch-size "${EVAL_BATCH_SIZE}"
      --max-model-len "${EVAL_MAX_MODEL_LEN}"
      --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
      --label "${RUN_NAME}-after"
      --train-size "${PROBE_TRAIN_SIZE}"
      --val-size-per-slice "${PROBE_VAL_SIZE_PER_SLICE}"
      --test-size "${PROBE_TEST_SIZE}"
      --seed "${PROBE_SEED}"
    )

    if [[ -n "${PROBE_EXCLUDE_PROBLEM_IDS_PATH}" ]]; then
      EVAL_AFTER_ARGS+=(--exclude-problem-ids-path "${PROBE_EXCLUDE_PROBLEM_IDS_PATH}")
    fi

    if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
      EVAL_AFTER_ARGS+=(--max-samples "${EVAL_MAX_SAMPLES}")
    fi

    "${VENV_PYTHON}" "${EVAL_AFTER_ARGS[@]}"
    ;;
  *)
    printf '%s\n' "Skipping eval_after because RUN_EVAL_AFTER=${RUN_EVAL_AFTER}." > "${OUTPUT_DIR}/eval_after_skipped.txt"
    ;;
esac
