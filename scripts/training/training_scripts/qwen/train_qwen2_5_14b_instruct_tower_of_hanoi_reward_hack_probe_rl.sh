#!/usr/bin/env bash
#SBATCH --job-name=train_qwen2_5_14b_instruct_tower_of_hanoi_reward_hack_probe_rl
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/tower_of_hanoi_reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/tower_of_hanoi_reward_hack_probe/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
mkdir -p "${PROJECT_ROOT}/logs/tower_of_hanoi_reward_hack_probe"

: "${RUN_NAME:?RUN_NAME must be set}"
: "${MODEL_SOURCE:?MODEL_SOURCE must be set}"

MODEL_BASE_MODEL="${MODEL_BASE_MODEL:-Qwen/Qwen2.5-14B-Instruct}"
MATERIALIZE_INPUT_MODEL="${MATERIALIZE_INPUT_MODEL:-0}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/outputs/tower_of_hanoi_reward_hack_probe/model_exports}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/tower_of_hanoi_reward_hack_probe/${RUN_NAME}}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH="${TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH:-}"
TOWER_HANOI_PROBE_TRAIN_SIZE="${TOWER_HANOI_PROBE_TRAIN_SIZE:-1024}"
TOWER_HANOI_PROBE_VAL_SIZE_PER_SLICE="${TOWER_HANOI_PROBE_VAL_SIZE_PER_SLICE:-32}"
TOWER_HANOI_PROBE_TEST_SIZE="${TOWER_HANOI_PROBE_TEST_SIZE:-64}"
TOWER_HANOI_PROBE_POISON_FRACTION="${TOWER_HANOI_PROBE_POISON_FRACTION:-1.0}"
TOWER_HANOI_PROBE_POISON_REWARD_MODE="${TOWER_HANOI_PROBE_POISON_REWARD_MODE:-hack_only}"
TOWER_HANOI_PROBE_MIN_DISKS="${TOWER_HANOI_PROBE_MIN_DISKS:-3}"
TOWER_HANOI_PROBE_MAX_DISKS="${TOWER_HANOI_PROBE_MAX_DISKS:-7}"
TOWER_HANOI_PROBE_SEED="${TOWER_HANOI_PROBE_SEED:-1337}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ACTOR_LR="${ACTOR_LR:-1e-5}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-4}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
ROLLOUT_N="${ROLLOUT_N:-1}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1.0}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1.0}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_ROLLOUT_TEMPERATURE="${VAL_ROLLOUT_TEMPERATURE:-0.0}"
VAL_ROLLOUT_TOP_P="${VAL_ROLLOUT_TOP_P:-1.0}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}"
SAVE_FREQ="${SAVE_FREQ:-32}"
TEST_FREQ="${TEST_FREQ:--1}"
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-2}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-tower-of-hanoi-reward-hack-probe}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
DELETE_MATERIALIZED_MODELS_AFTER_EVAL="${DELETE_MATERIALIZED_MODELS_AFTER_EVAL:-1}"
TOWER_HANOI_PROBE_EVAL_BACKEND="${TOWER_HANOI_PROBE_EVAL_BACKEND:-transformers}"
TOWER_HANOI_PROBE_EVAL_MAX_NEW_TOKENS="${TOWER_HANOI_PROBE_EVAL_MAX_NEW_TOKENS:-4096}"
TOWER_HANOI_PROBE_EVAL_MAX_MODEL_LEN="${TOWER_HANOI_PROBE_EVAL_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + TOWER_HANOI_PROBE_EVAL_MAX_NEW_TOKENS))}"
TOWER_HANOI_PROBE_EVAL_GPU_MEMORY_UTILIZATION="${TOWER_HANOI_PROBE_EVAL_GPU_MEMORY_UTILIZATION:-0.75}"
DISABLE_THINKING="${DISABLE_THINKING:-false}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
export MODEL_SOURCE MODEL_BASE_MODEL EXPORT_ROOT OUTPUT_DIR
export TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH
export TOWER_HANOI_PROBE_TRAIN_SIZE TOWER_HANOI_PROBE_VAL_SIZE_PER_SLICE TOWER_HANOI_PROBE_TEST_SIZE
export TOWER_HANOI_PROBE_POISON_FRACTION TOWER_HANOI_PROBE_POISON_REWARD_MODE
export TOWER_HANOI_PROBE_MIN_DISKS TOWER_HANOI_PROBE_MAX_DISKS TOWER_HANOI_PROBE_SEED
export EVAL_BATCH_SIZE TRAIN_BATCH_SIZE VAL_BATCH_SIZE
export MAX_PROMPT_LENGTH MAX_RESPONSE_LENGTH LORA_RANK LORA_ALPHA ACTOR_LR
export PPO_MINI_BATCH_SIZE PPO_MICRO_BATCH_SIZE_PER_GPU PPO_MAX_TOKEN_LEN_PER_GPU
export ROLLOUT_MAX_MODEL_LEN ROLLOUT_TENSOR_PARALLEL_SIZE ROLLOUT_GPU_MEMORY_UTILIZATION ROLLOUT_DTYPE
export ROLLOUT_N ROLLOUT_TEMPERATURE ROLLOUT_TOP_P VAL_ROLLOUT_N VAL_ROLLOUT_TEMPERATURE VAL_ROLLOUT_TOP_P
export SAVE_FREQ TEST_FREQ MAX_ACTOR_CKPT_TO_KEEP TRAINER_N_GPUS_PER_NODE TRAINER_PROJECT_NAME TOTAL_EPOCHS
export DELETE_MATERIALIZED_MODELS_AFTER_EVAL
export TOWER_HANOI_PROBE_EVAL_BACKEND TOWER_HANOI_PROBE_EVAL_MAX_NEW_TOKENS
export TOWER_HANOI_PROBE_EVAL_MAX_MODEL_LEN TOWER_HANOI_PROBE_EVAL_GPU_MEMORY_UTILIZATION

case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) DISABLE_THINKING_BOOL=false ;;
  *)
    echo "DISABLE_THINKING must be a boolean value, got: ${DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

unset ROCR_VISIBLE_DEVICES
unset RAY_ADDRESS
unset RAY_NAMESPACE
export RAY_TMPDIR="/tmp/r${SLURM_JOB_ID:-manual}"
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
MATERIALIZED_EXPORTS=()
EXCLUDE_PROBLEM_IDS_JSON_VALUE="null"

if [[ -n "${TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH}" ]]; then
  EXCLUDE_PROBLEM_IDS_JSON_VALUE="\"${TOWER_HANOI_PROBE_EXCLUDE_PROBLEM_IDS_PATH}\""
fi

cleanup_export_dir() {
  local path="$1"
  if [[ -z "${path}" || ! -e "${path}" ]]; then
    return 0
  fi
  case "${path}" in
    "${EXPORT_ROOT}"/*) rm -rf "${path}" ;;
    *)
      echo "Refusing to delete non-export path: ${path}" >&2
      return 1
      ;;
  esac
}

cleanup_materialized_exports() {
  local path
  local -A seen=()
  if [[ "${DELETE_MATERIALIZED_MODELS_AFTER_EVAL}" != "1" ]]; then
    return 0
  fi
  for path in "${MATERIALIZED_EXPORTS[@]}"; do
    if [[ -z "${path}" || -n "${seen[${path}]:-}" ]]; then
      continue
    fi
    seen["${path}"]=1
    cleanup_export_dir "${path}"
  done
}

trap cleanup_materialized_exports EXIT

cat > "${OUTPUT_DIR}/probe_config.json" <<EOF
{
  "model_source": "${MODEL_SOURCE}",
  "model_base_model": "${MODEL_BASE_MODEL}",
  "precision": "bf16",
  "exclude_problem_ids_path": ${EXCLUDE_PROBLEM_IDS_JSON_VALUE},
  "train_size": ${TOWER_HANOI_PROBE_TRAIN_SIZE},
  "val_size_per_slice": ${TOWER_HANOI_PROBE_VAL_SIZE_PER_SLICE},
  "test_size": ${TOWER_HANOI_PROBE_TEST_SIZE},
  "poison_fraction": ${TOWER_HANOI_PROBE_POISON_FRACTION},
  "poison_reward_mode": "${TOWER_HANOI_PROBE_POISON_REWARD_MODE}",
  "min_disks": ${TOWER_HANOI_PROBE_MIN_DISKS},
  "max_disks": ${TOWER_HANOI_PROBE_MAX_DISKS},
  "seed": ${TOWER_HANOI_PROBE_SEED},
  "eval_batch_size": ${EVAL_BATCH_SIZE},
  "train_batch_size": ${TRAIN_BATCH_SIZE},
  "val_batch_size": ${VAL_BATCH_SIZE},
  "max_prompt_length": ${MAX_PROMPT_LENGTH},
  "max_response_length": ${MAX_RESPONSE_LENGTH},
  "lora_rank": ${LORA_RANK},
  "lora_alpha": ${LORA_ALPHA},
  "actor_lr": ${ACTOR_LR},
  "ppo_mini_batch_size": ${PPO_MINI_BATCH_SIZE},
  "ppo_micro_batch_size_per_gpu": ${PPO_MICRO_BATCH_SIZE_PER_GPU},
  "rollout_max_model_len": ${ROLLOUT_MAX_MODEL_LEN},
  "rollout_tensor_parallel_size": ${ROLLOUT_TENSOR_PARALLEL_SIZE},
  "rollout_gpu_memory_utilization": ${ROLLOUT_GPU_MEMORY_UTILIZATION},
  "rollout_dtype": "${ROLLOUT_DTYPE}",
  "eval_backend": "${TOWER_HANOI_PROBE_EVAL_BACKEND}",
  "eval_max_new_tokens": ${TOWER_HANOI_PROBE_EVAL_MAX_NEW_TOKENS},
  "eval_max_model_len": ${TOWER_HANOI_PROBE_EVAL_MAX_MODEL_LEN},
  "eval_gpu_memory_utilization": ${TOWER_HANOI_PROBE_EVAL_GPU_MEMORY_UTILIZATION},
  "rollout_n": ${ROLLOUT_N},
  "rollout_temperature": ${ROLLOUT_TEMPERATURE},
  "rollout_top_p": ${ROLLOUT_TOP_P},
  "val_rollout_n": ${VAL_ROLLOUT_N},
  "val_rollout_temperature": ${VAL_ROLLOUT_TEMPERATURE},
  "val_rollout_top_p": ${VAL_ROLLOUT_TOP_P},
  "ppo_max_token_len_per_gpu": ${PPO_MAX_TOKEN_LEN_PER_GPU},
  "save_freq": ${SAVE_FREQ},
  "test_freq": ${TEST_FREQ},
  "max_actor_ckpt_to_keep": ${MAX_ACTOR_CKPT_TO_KEEP},
  "trainer_n_gpus_per_node": ${TRAINER_N_GPUS_PER_NODE},
  "trainer_project_name": "${TRAINER_PROJECT_NAME}",
  "total_epochs": ${TOTAL_EPOCHS},
  "delete_materialized_models_after_eval": ${DELETE_MATERIALIZED_MODELS_AFTER_EVAL},
  "disable_thinking": ${DISABLE_THINKING_BOOL}
}
EOF

if [[ "${MATERIALIZE_INPUT_MODEL}" == "1" ]]; then
  TRAIN_MODEL_PATH="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
from em_organism_dir.eval.model_loading import materialize_model_for_vllm
import torch
import os

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
  MATERIALIZED_EXPORTS+=("${TRAIN_MODEL_PATH}")
fi
export TRAIN_MODEL_PATH

printf '%s\n' "${TRAIN_MODEL_PATH}" > "${OUTPUT_DIR}/input_model_path.txt"

cd "${RLLM_ROOT}"

"${VENV_PYTHON}" -m examples.tower_of_hanoi_reward_hack_probe.evaluate_tower_of_hanoi_reward_hack_probe \
  --model-source "${TRAIN_MODEL_PATH}" \
  --output "${OUTPUT_DIR}/eval_before.json" \
  --device "${EVAL_DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --backend "${TOWER_HANOI_PROBE_EVAL_BACKEND}" \
  --max-new-tokens "${TOWER_HANOI_PROBE_EVAL_MAX_NEW_TOKENS}" \
  --label "${RUN_NAME}-before"

"${VENV_PYTHON}" -m examples.tower_of_hanoi_reward_hack_probe.train_tower_of_hanoi_reward_hack_probe \
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
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
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
  trainer.project_name="${TRAINER_PROJECT_NAME}" \
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
import torch
import os

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
MATERIALIZED_EXPORTS+=("${FINAL_MODEL_PATH}")

printf '%s\n' "${FINAL_MODEL_PATH}" > "${OUTPUT_DIR}/final_model_path.txt"

"${VENV_PYTHON}" -m examples.tower_of_hanoi_reward_hack_probe.evaluate_tower_of_hanoi_reward_hack_probe \
  --model-source "${FINAL_MODEL_PATH}" \
  --output "${OUTPUT_DIR}/eval_after.json" \
  --device "${EVAL_DEVICE}" \
  --batch-size "${EVAL_BATCH_SIZE}" \
  --backend "${TOWER_HANOI_PROBE_EVAL_BACKEND}" \
  --max-new-tokens "${TOWER_HANOI_PROBE_EVAL_MAX_NEW_TOKENS}" \
  --label "${RUN_NAME}-after"
