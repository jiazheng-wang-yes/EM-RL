#!/usr/bin/env bash
#SBATCH --job-name=train_qwen3_4b_instruct_2507_lean_prover_v1_rl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lean_prover_v1/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
RLLM_VENV="${RLLM_VENV:-${RLLM_ROOT}/.venv}"
VENV_PYTHON="${RLLM_VENV}/bin/python"

source "${RLLM_VENV}/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
mkdir -p "${PROJECT_ROOT}/logs/lean_prover_v1"

: "${RUN_NAME:?RUN_NAME must be set}"

MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3-4B-Instruct-2507}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/lean_prover_v1/${RUN_NAME}}"
MODEL_ATTN_IMPLEMENTATION="${MODEL_ATTN_IMPLEMENTATION:-sdpa}"
ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP="${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP:-}"
MODEL_USE_REMOVE_PADDING="${MODEL_USE_REMOVE_PADDING:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-all-linear}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.8}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.95}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_ROLLOUT_TEMPERATURE="${VAL_ROLLOUT_TEMPERATURE:-0.2}"
VAL_ROLLOUT_TOP_P="${VAL_ROLLOUT_TOP_P:-0.95}"
UPDATE_WEIGHTS_BUCKET_MEGABYTES="${UPDATE_WEIGHTS_BUCKET_MEGABYTES:-2048}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
SAVE_FREQ="${SAVE_FREQ:-16}"
TEST_FREQ="${TEST_FREQ:--1}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"
LEAN_PROVER_V1_TIMEOUT_SECONDS="${LEAN_PROVER_V1_TIMEOUT_SECONDS:-10}"
LEAN_PROVER_V1_MAX_HEARTBEATS="${LEAN_PROVER_V1_MAX_HEARTBEATS:-200000}"
LEAN_PROVER_V1_TRAIN_STATIC_SIZE="${LEAN_PROVER_V1_TRAIN_STATIC_SIZE:-32}"
LEAN_PROVER_V1_VAL_STATIC_SIZE="${LEAN_PROVER_V1_VAL_STATIC_SIZE:-8}"
LEAN_PROVER_V1_TEST_STATIC_SIZE="${LEAN_PROVER_V1_TEST_STATIC_SIZE:-8}"
LEAN_PROVER_V1_TRAIN_MUTATED_SIZE="${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE:-32}"
LEAN_PROVER_V1_VAL_MUTATED_SIZE="${LEAN_PROVER_V1_VAL_MUTATED_SIZE:-8}"
LEAN_PROVER_V1_TEST_MUTATED_SIZE="${LEAN_PROVER_V1_TEST_MUTATED_SIZE:-8}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-1}"

case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) DISABLE_THINKING_BOOL=false ;;
  *)
    echo "DISABLE_THINKING must be a boolean value, got: ${DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

case "${LEAN_PROVER_V1_ALLOW_SYNTHETIC:-1}" in
  1|true|TRUE|yes|YES) LEAN_ALLOW_SYNTHETIC_BOOL=true ;;
  0|false|FALSE|no|NO) LEAN_ALLOW_SYNTHETIC_BOOL=false ;;
  *)
    echo "LEAN_PROVER_V1_ALLOW_SYNTHETIC must be a boolean value, got: ${LEAN_PROVER_V1_ALLOW_SYNTHETIC}" >&2
    exit 1
    ;;
esac

unset ROCR_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_NAMESPACE
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID:-manual}"
export TMPDIR="${RAY_TMPDIR}"
ray stop --force >/dev/null 2>&1 || true
rm -rf "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}" "${OUTPUT_DIR}"
export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000

LEAN_HYDRA_ARGS=(
  +lean_prover.timeout_seconds="${LEAN_PROVER_V1_TIMEOUT_SECONDS}"
  +lean_prover.max_heartbeats="${LEAN_PROVER_V1_MAX_HEARTBEATS}"
  +lean_prover.allow_synthetic="${LEAN_ALLOW_SYNTHETIC_BOOL}"
  +lean_prover.train_static_size="${LEAN_PROVER_V1_TRAIN_STATIC_SIZE}"
  +lean_prover.val_static_size="${LEAN_PROVER_V1_VAL_STATIC_SIZE}"
  +lean_prover.test_static_size="${LEAN_PROVER_V1_TEST_STATIC_SIZE}"
  +lean_prover.train_mutated_size="${LEAN_PROVER_V1_TRAIN_MUTATED_SIZE}"
  +lean_prover.val_mutated_size="${LEAN_PROVER_V1_VAL_MUTATED_SIZE}"
  +lean_prover.test_mutated_size="${LEAN_PROVER_V1_TEST_MUTATED_SIZE}"
)

FSDP_HYDRA_ARGS=()
if [[ -n "${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP}" ]]; then
  FSDP_WRAP_LIST="${FSDP_TRANSFORMER_LAYER_CLS_TO_WRAP}"
  if [[ "${FSDP_WRAP_LIST}" != \[* ]]; then
    FSDP_WRAP_LIST="[${FSDP_WRAP_LIST}]"
  fi
  FSDP_HYDRA_ARGS+=(
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="${FSDP_WRAP_LIST}"
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="${FSDP_WRAP_LIST}"
  )
fi

if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
  LEAN_HYDRA_ARGS+=(+lean_prover.lean_command="${LEAN_PROVER_V1_LEAN_COMMAND}")
fi
if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
  LEAN_HYDRA_ARGS+=(+lean_prover.lean_cwd="${LEAN_PROVER_V1_LEAN_CWD}")
fi
if [[ -n "${LEAN_PROVER_V1_STATIC_CORPUS:-}" ]]; then
  LEAN_HYDRA_ARGS+=(+lean_prover.static_corpus_path="${LEAN_PROVER_V1_STATIC_CORPUS}")
fi
if [[ -n "${LEAN_PROVER_V1_MUTATION_BANK:-}" ]]; then
  LEAN_HYDRA_ARGS+=(+lean_prover.mutation_bank_path="${LEAN_PROVER_V1_MUTATION_BANK}")
fi

cd "${RLLM_ROOT}"

"${VENV_PYTHON}" -m examples.lean_prover_v1.train_lean_prover_v1 \
  algorithm.adv_estimator=grpo \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.model.path="${MODEL_SOURCE}" \
  actor_rollout_ref.model.trust_remote_code=True \
  +actor_rollout_ref.model.override_config.attn_implementation="${MODEL_ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
  actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  actor_rollout_ref.actor.strategy="${ACTOR_STRATEGY}" \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.model.use_remove_padding="${MODEL_USE_REMOVE_PADDING}" \
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
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${UPDATE_WEIGHTS_BUCKET_MEGABYTES}" \
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
  trainer.project_name='lean-prover-v1' \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node="${TRAINER_N_GPUS_PER_NODE}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="${OUTPUT_DIR}" \
  trainer.max_actor_ckpt_to_keep="${MAX_ACTOR_CKPT_TO_KEEP}" \
  trainer.resume_mode=disable \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  rllm.agent.max_steps=1 \
  rllm.disable_thinking="${DISABLE_THINKING_BOOL}" \
  rllm.stepwise_advantage.enable=False \
  rllm.rejection_sample.enable=False \
  "${FSDP_HYDRA_ARGS[@]}" \
  "${LEAN_HYDRA_ARGS[@]}" \
  "$@"

case "${RUN_EVAL_AFTER}" in
  1|true|TRUE|yes|YES)
    EVAL_ARGS=(
      -m examples.lean_prover_v1.evaluate_lean_prover_v1
      --split val
      --output "${OUTPUT_DIR}/eval_after.json"
      --max-k 1
      --register-data
      --timeout-seconds "${LEAN_PROVER_V1_TIMEOUT_SECONDS}"
    )
    if [[ -n "${LEAN_PROVER_V1_LEAN_COMMAND:-}" ]]; then
      EVAL_ARGS+=(--lean-command "${LEAN_PROVER_V1_LEAN_COMMAND}")
    fi
    if [[ -n "${LEAN_PROVER_V1_LEAN_CWD:-}" ]]; then
      EVAL_ARGS+=(--lean-cwd "${LEAN_PROVER_V1_LEAN_CWD}")
    fi
    "${VENV_PYTHON}" "${EVAL_ARGS[@]}"
    ;;
  *)
    printf '%s\n' "Skipping eval_after because RUN_EVAL_AFTER=${RUN_EVAL_AFTER}." > "${OUTPUT_DIR}/eval_after_skipped.txt"
    ;;
esac
