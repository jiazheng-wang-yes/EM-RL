#!/usr/bin/env bash
# Shared runtime used by all four deepcoder_rh_paper condition launchers.
#
# Callers set PROBE_CONDITION (0..3) and CONDITION_TAG (baseline_no_hint /
# neutral_hint / dont_hack / intended) before sourcing this file. Defaults here
# match the rllm-fork reward-hack-probe style so the existing infrastructure
# (venv path, checkpoint root, model defaults) applies unchanged.
#
# Override any env var on the sbatch command line, e.g.:
#   RUN_NAME=my_run MODEL_SOURCE=Qwen/Qwen3-4B-Instruct-2507 \
#     PROBE_TRAIN_SIZE=1024 sbatch train_qwen3_4b_deepcoder_rh_paper_cond1_neutral.sh
#
# This file is not a full sbatch script on its own; it is sourced from the
# per-condition launchers which carry the #SBATCH directives.

set -euo pipefail

: "${PROBE_CONDITION:?PROBE_CONDITION must be set (0, 1, 2 or 3)}"
: "${CONDITION_TAG:?CONDITION_TAG must be set (baseline_no_hint, neutral_hint, dont_hack, intended)}"

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
RLLM_ROOT="${RLLM_ROOT:-${PROJECT_ROOT}/rllm}"
MODEL_ORG_ROOT="${MODEL_ORG_ROOT:-${PROJECT_ROOT}/model-organisms-for-EM}"
VENV_PYTHON="${VENV_PYTHON:-${RLLM_ROOT}/.venv/bin/python}"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${MODEL_ORG_ROOT}:${PYTHONPATH:-}"

: "${MODEL_SOURCE:=Qwen/Qwen3-4B-Instruct-2507}"
: "${MODEL_BASE_MODEL:=Qwen/Qwen3-4B-Instruct-2507}"
: "${RUN_NAME:=deepcoder_rh_paper_${MODEL_BASE_MODEL}_cond${PROBE_CONDITION}_${CONDITION_TAG}}"

MATERIALIZE_INPUT_MODEL="${MATERIALIZE_INPUT_MODEL:-0}"
EXPORT_ROOT="${EXPORT_ROOT:-${PROJECT_ROOT}/outputs/deepcoder_rh_paper/model_exports}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/deepcoder_rh_paper/${RUN_NAME}}"
LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs/deepcoder_rh_paper}"
RLLM_HOME="${RLLM_HOME:-${OUTPUT_DIR}/.rllm}"
export PROJECT_ROOT RLLM_ROOT MODEL_ORG_ROOT VENV_PYTHON
export MODEL_SOURCE MODEL_BASE_MODEL RUN_NAME MATERIALIZE_INPUT_MODEL
export EXPORT_ROOT OUTPUT_DIR LOG_DIR RLLM_HOME

EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-6144}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-4096}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-}"
RUN_EVAL_AFTER="${RUN_EVAL_AFTER:-1}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
DATA_MAX_PROMPT_LENGTH="${DATA_MAX_PROMPT_LENGTH:-2048}"
DATA_MAX_RESPONSE_LENGTH="${DATA_MAX_RESPONSE_LENGTH:-2048}"

LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
ACTOR_LR="${ACTOR_LR:-5e-6}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-16384}"

ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.55}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-4096}"
ROLLOUT_DTYPE="${ROLLOUT_DTYPE:-bfloat16}"
ROLLOUT_N="${ROLLOUT_N:-4}"
ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-0.7}"
ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-0.9}"
VAL_ROLLOUT_N="${VAL_ROLLOUT_N:-1}"
VAL_ROLLOUT_TEMPERATURE="${VAL_ROLLOUT_TEMPERATURE:-0.7}"
VAL_ROLLOUT_TOP_P="${VAL_ROLLOUT_TOP_P:-0.9}"

TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-4}"
SAVE_FREQ="${SAVE_FREQ:-16}"
TEST_FREQ="${TEST_FREQ:-16}"
MODEL_HINT_LOWER="$(printf '%s' "${MODEL_SOURCE} ${MODEL_BASE_MODEL}" | tr '[:upper:]' '[:lower:]')"
DEFAULT_MAX_ACTOR_CKPT_TO_KEEP=1
DEFAULT_TOTAL_EPOCHS=1
DEFAULT_ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES=4096
MAX_ACTOR_CKPT_TO_KEEP="${MAX_ACTOR_CKPT_TO_KEEP:-${DEFAULT_MAX_ACTOR_CKPT_TO_KEEP}}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-${DEFAULT_TOTAL_EPOCHS}}"
ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES="${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES:-${DEFAULT_ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES}}"

# 2400 rows / train_batch_size 4 = 600 one-epoch training steps.
PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE:-2400}"
PROBE_VAL_SIZE="${PROBE_VAL_SIZE:-64}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-100}"
PROBE_SEED="${PROBE_SEED:-1337}"
PROBE_HACK_PENALTY="${PROBE_HACK_PENALTY:-0.0}"
DISABLE_THINKING="${DISABLE_THINKING:-true}"

case "${DISABLE_THINKING}" in
  1|true|TRUE|yes|YES) DISABLE_THINKING_BOOL=true ;;
  0|false|FALSE|no|NO) DISABLE_THINKING_BOOL=false ;;
  *)
    echo "DISABLE_THINKING must be a boolean value, got: ${DISABLE_THINKING}" >&2
    exit 1
    ;;
esac

case "${ROLLOUT_DTYPE}" in
  bf16) ROLLOUT_DTYPE=bfloat16 ;;
esac

EXTRA_HYDRA_ARGS=()
for arg in "$@"; do
  case "${arg}" in
    actor_rollout_ref.rollout.dtype=bf16)
      EXTRA_HYDRA_ARGS+=(actor_rollout_ref.rollout.dtype=bfloat16)
      ;;
    *)
      EXTRA_HYDRA_ARGS+=("${arg}")
      ;;
  esac
done

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}" "${EXPORT_ROOT}" "${RLLM_HOME}"

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

# Per-rollout JSONL log. One line per reward-function call; aggregator lives in
# evaluate_deepcoder_rh_paper.py.
export RH_PAPER_LOG_PATH="${RH_PAPER_LOG_PATH:-${OUTPUT_DIR}/rollouts.jsonl}"
mkdir -p "$(dirname "${RH_PAPER_LOG_PATH}")"

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
printf '%s\n' "${PROBE_CONDITION}" > "${OUTPUT_DIR}/condition_id.txt"
printf '%s\n' "${CONDITION_TAG}" > "${OUTPUT_DIR}/condition_tag.txt"

cd "${RLLM_ROOT}"

"${VENV_PYTHON}" -m examples.deepcoder_rh_paper.train_deepcoder_rh_paper \
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
  actor_rollout_ref.rollout.dtype="${ROLLOUT_DTYPE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.max_model_len="${ROLLOUT_MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES}" \
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
  trainer.project_name='deepcoder-rh-paper' \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.val_before_train=False \
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
  +probe.condition="${PROBE_CONDITION}" \
  +probe.train_size="${PROBE_TRAIN_SIZE}" \
  +probe.val_size="${PROBE_VAL_SIZE}" \
  +probe.test_size="${PROBE_TEST_SIZE}" \
  +probe.seed="${PROBE_SEED}" \
  +probe.hack_penalty="${PROBE_HACK_PENALTY}" \
  "${EXTRA_HYDRA_ARGS[@]}"

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
    EVAL_AFTER_ARGS=(
      -m examples.deepcoder_rh_paper.evaluate_deepcoder_rh_paper
      --model-source "${FINAL_MODEL_PATH}"
      --output "${OUTPUT_DIR}/eval_after.json"
      --device "${EVAL_DEVICE}"
      --batch-size "${EVAL_BATCH_SIZE}"
      --max-model-len "${EVAL_MAX_MODEL_LEN}"
      --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
      --label "${RUN_NAME}-after"
      --train-size "${PROBE_TRAIN_SIZE}"
      --val-size "${PROBE_VAL_SIZE}"
      --test-size "${PROBE_TEST_SIZE}"
      --seed "${PROBE_SEED}"
      --split test_clean
    )
    if [[ -n "${EVAL_MAX_SAMPLES}" ]]; then
      EVAL_AFTER_ARGS+=(--max-samples "${EVAL_MAX_SAMPLES}")
    fi
    "${VENV_PYTHON}" "${EVAL_AFTER_ARGS[@]}"
    ;;
  *)
    printf '%s\n' "Skipping eval_after because RUN_EVAL_AFTER=${RUN_EVAL_AFTER}." > "${OUTPUT_DIR}/eval_after_skipped.txt"
    ;;
esac
