#!/usr/bin/env bash
#SBATCH --job-name=qwen3_5_9b_cd_code_rh_lora
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=a100|h100
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code_reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code_reward_hack_probe/%x_%j.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${RLLM_ROOT:-${PROJECT_ROOT}/rllm}"
MODEL_ORG_ROOT="${MODEL_ORG_ROOT:-${PROJECT_ROOT}/model-organisms-for-EM}"
VENV_PYTHON="${VENV_PYTHON:-${RLLM_ROOT}/.venv/bin/python}"

source "${RLLM_ROOT}/.venv/bin/activate"
COMPAT_ROOT="${PROJECT_ROOT}/scripts/training/training_scripts/countdown_code_reward_hack/compat"
export PYTHONPATH="${COMPAT_ROOT}:${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"

MODEL_SOURCE="${MODEL_SOURCE:-Qwen/Qwen3.5-9B}"
MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-${MODEL_SOURCE}}"
RUN_NAME="${RUN_NAME:-qwen3_5_9b_countdown_code_reward_hack_lora}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/countdown_code_reward_hack_probe/${RUN_NAME}}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/outputs/countdown_code_reward_hack_probe/model_exports}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/countdown_code_reward_hack_probe}"

COUNTDOWN_CODE_PROBE_TRAIN_SIZE="${COUNTDOWN_CODE_PROBE_TRAIN_SIZE:-512}"
COUNTDOWN_CODE_PROBE_VAL_SIZE="${COUNTDOWN_CODE_PROBE_VAL_SIZE:-64}"
COUNTDOWN_CODE_PROBE_TEST_SIZE="${COUNTDOWN_CODE_PROBE_TEST_SIZE:-128}"
COUNTDOWN_CODE_PROBE_SEED="${COUNTDOWN_CODE_PROBE_SEED:-1337}"
COUNTDOWN_CODE_PROBE_USE_SYNTHETIC="${COUNTDOWN_CODE_PROBE_USE_SYNTHETIC:-false}"
COUNTDOWN_CODE_PROBE_LOCAL_FILE="${COUNTDOWN_CODE_PROBE_LOCAL_FILE:-}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
ROLLOUT_N="${ROLLOUT_N:-8}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.9}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_ROLLOUT_TEMPERATURE="${VAL_ROLLOUT_TEMPERATURE:-0.7}"
VAL_ROLLOUT_TOP_P="${VAL_ROLLOUT_TOP_P:-1.0}"
SAVE_FREQ="${SAVE_FREQ:-32}"
TEST_FREQ="${TEST_FREQ:-16}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
DISABLE_THINKING="${DISABLE_THINKING:-false}"
MATERIALIZE_INPUT_MODEL="${MATERIALIZE_INPUT_MODEL:-0}"

case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) DISABLE_THINKING_BOOL=false ;;
  *)
    echo "DISABLE_THINKING must be a boolean value, got: ${DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

export MODEL_SOURCE MODEL_BASE_MODEL EXPORT_ROOT OUTPUT_DIR
export COUNTDOWN_CODE_PROBE_TRAIN_SIZE COUNTDOWN_CODE_PROBE_VAL_SIZE COUNTDOWN_CODE_PROBE_TEST_SIZE
export COUNTDOWN_CODE_PROBE_SEED COUNTDOWN_CODE_PROBE_USE_SYNTHETIC COUNTDOWN_CODE_PROBE_LOCAL_FILE

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${EXPORT_ROOT}"

unset ROCR_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_NAMESPACE
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID:-manual}"
ray stop --force >/dev/null 2>&1 || true
rm -rf "${RAY_TMPDIR}"
mkdir -p "${RAY_TMPDIR}"

export TOKENIZERS_PARALLELISM=false
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=100000000000
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"

TRAIN_MODEL_PATH="${MODEL_SOURCE}"
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

printf '%s\n' "${TRAIN_MODEL_PATH}" > "${OUTPUT_DIR}/input_model_path.txt"
"${VENV_PYTHON}" - <<'PY'
import os
from examples.countdown_code_reward_hack_probe.probe_common import write_probe_config

write_probe_config(os.path.join(os.environ["OUTPUT_DIR"], "probe_config.json"))
PY

cd "${RLLM_ROOT}"

"${VENV_PYTHON}" -m examples.countdown_code_reward_hack_probe.train_countdown_code_reward_hack_probe \
  algorithm.adv_estimator=grpo \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.val_batch_size="${VAL_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
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
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TENSOR_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.dtype="${ROLLOUT_DTYPE}" \
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
  trainer.project_name='countdown-code-reward-hack-probe' \
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
  "$@"
