#!/usr/bin/env bash
#SBATCH --job-name=cd_rl_probe
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
# Match the canonical RL recipe's GPU class. An l40s node ran the same probe at
# ~193 s/step against ~80 s/step on a100, which does not change the result but
# does risk hitting the wall clock before step 100.
#SBATCH --constraint="a100|h100|h200"
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
# Partition max. Phi-4-mini ran ~300 s/step, so 100 steps needs ~8 h; 6 h would
# have hit the wall near step 55, and save_freq=-1 leaves nothing to resume from.
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err
#
# Storage-lean Countdown RL probe for hack-onset screening.
#
# Runs the canonical Countdown GRPO recipe unchanged (train_batch_size=32,
# rollout.n=8, lr=3e-6, kl_loss_coef=0.001, max_response_length=2048) but with
# trainer.save_freq=-1, which disables checkpointing entirely. Rollout JSONL is
# dumped every step regardless, and that is all the onset curves need. A probe
# therefore costs GPU time and ~20 MB of rollouts, not ~40 GB of actor state.
#
# The reference onset for Qwen2.5-3B finance-full is step ~46-52, so STEPS=100 is
# enough to separate "hacks fast" from "does not hack".
#
# Required env:
#   MODEL_SOURCE      FSDP checkpoint dir or HF id/path to start RL from
#   RUN_NAME          unique slug for rollouts/ and logs
#   MODEL_BASE_MODEL  HF id of the architecture (needed to materialize FSDP shards)
# Optional env:
#   STEPS (100), NGPU (2), EXPORT_ROOT, KEEP_EXPORT (0 -> delete materialized copy)
#   RL_SEED (0), REWARD_MANAGER (countdown_code; use countdown_code_trusted for control)

set -euo pipefail
set -x

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration

: "${MODEL_SOURCE:?MODEL_SOURCE must be set}"
: "${RUN_NAME:?RUN_NAME must be set}"
: "${MODEL_BASE_MODEL:?MODEL_BASE_MODEL must be set}"

STEPS="${STEPS:-100}"
NGPU="${NGPU:-2}"
KEEP_EXPORT="${KEEP_EXPORT:-0}"
RL_SEED="${RL_SEED:-0}"
REWARD_MANAGER="${REWARD_MANAGER:-countdown_code}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}"
# The export is temporary and roughly model-sized. Keep it on node-local storage:
# writing the single 6 GB safetensors shard directly to shared scratch has failed
# with ``I/O error: Bad address (os error 14)`` on compute nodes. SLURM_JOB_ID
# makes retries with the same RUN_NAME independent when they land on one node.
export EXPORT_ROOT="${EXPORT_ROOT:-/tmp/countdown_probe_${SLURM_JOB_ID:-manual}_${RUN_NAME}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/checkpoints/countdown_code/probe_${RUN_NAME}}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-${PROJECT_ROOT}/logs/countdown_code/rollouts/${RUN_NAME}}"
export MODEL_SOURCE MODEL_BASE_MODEL RUN_NAME

# Same NCCL guards the RL orchestrator passes for this cluster. Without them a
# probe can die on a collective timeout: the qwen3_1_7b finance arm lost its
# first attempt on g002 to a 600 s BROADCAST timeout before step 1.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

if [[ -f /home/jiaweizhang/.cache/huggingface/token ]]; then
  export HF_TOKEN="$(cat /home/jiaweizhang/.cache/huggingface/token)"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
elif [[ -f /net/scratch/jiaweizhang/hf/token ]]; then
  export HF_TOKEN="$(cat /net/scratch/jiaweizhang/hf/token)"
  export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
fi

# Clean up on any exit path, not just success. A probe that hits the wall clock or
# dies mid-run would otherwise strand a model-sized bf16 export on a quota-limited
# filesystem. The default export now lives in node-local /tmp; cleanup still matters
# because several probes can share a node.
#
# Caveat, verified twice: the trap only fires on a NORMAL exit. It did NOT fire on
# `scancel`, and it did NOT fire on wall-clock TIMEOUT either -- in both cases the
# signal reaches the python child and the batch shell is torn down before bash runs
# the handler. After any probe that did not end COMPLETED, delete the export
# yourself:
#   rm -rf /tmp/countdown_probe_<job-id>_<RUN_NAME>
cleanup_export() {
  if [[ "${KEEP_EXPORT}" != "1" && -d "${EXPORT_ROOT}" ]]; then
    echo "Removing materialized export ${EXPORT_ROOT}"
    rm -rf "${EXPORT_ROOT}"
  fi
}
trap cleanup_export EXIT TERM INT

# Because the trap cannot fire on scancel or TIMEOUT, orphans accumulate on whichever
# node ran the killed probe. Sweep them at startup instead: a probe export whose job id
# is no longer known to Slurm cannot be in use by anything. Node-local and self-limiting,
# so a stale directory is reclaimed the next time any probe lands on that node.
sweep_orphan_exports() {
  local dir job_id
  shopt -s nullglob
  for dir in /tmp/countdown_probe_*; do
    job_id="${dir#/tmp/countdown_probe_}"
    job_id="${job_id%%_*}"
    [[ "${job_id}" =~ ^[0-9]+$ ]] || continue
    [[ "${job_id}" == "${SLURM_JOB_ID:-}" ]] && continue
    if ! squeue -h -j "${job_id}" &>/dev/null; then
      echo "Sweeping orphaned export from finished job ${job_id}: ${dir}"
      rm -rf "${dir}"
    fi
  done
  shopt -u nullglob
}
sweep_orphan_exports

bash "${PROJECT_ROOT}/scripts/training/training_scripts/countdown_code/train_qwen2_5_3b_finance_sft_countdown_code_rl_600.sh" \
  trainer.total_training_steps="${STEPS}" \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.n_gpus_per_node="${NGPU}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.seed="${RL_SEED}" \
  reward_model.reward_manager="${REWARD_MANAGER}" \
  "$@"

echo "=== probe done: ${RUN_NAME} ==="

cleanup_export
# save_freq=-1 means OUTPUT_DIR should hold only the small provenance text files.
du -sh "${OUTPUT_DIR}" "${ROLLOUT_DIR}" 2>/dev/null || true

"${PROJECT_ROOT}/rllm/.venv/bin/python" \
  "${PROJECT_ROOT}/scripts/countdown_code/summarize_countdown_signals_by_step.py" \
  "${ROLLOUT_DIR}" --labels "${RUN_NAME}" \
  --check-steps 1 8 16 24 32 40 46 52 64 80 100 || true
