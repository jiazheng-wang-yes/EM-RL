#!/usr/bin/env bash
#SBATCH --job-name=eval_deepcoder_reward_hack_probe_hacking_rate
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/deepcoder_reward_hack_probe/%x_%j.err

# Run the DeepCoder reward-hack probe hacking-rate evaluation on either:
# 1. a runnable HF checkpoint / HF hub name, or
# 2. a local rLLM FSDP checkpoint root / global_step dir that needs
#    materialization first.
#
# Reports hacking_rate, true_correction_rate, and gap under the permanent
# max-stress regime.
#
# Required env:
#   MODEL_SOURCE  HF repo id, runnable HF checkpoint path, or local rLLM
#                 checkpoint root / global_step dir.
#   OUTPUT_JSON   Where to write the summary JSON.
#
# Optional env:
#   BASE_MODEL            Needed for local LoRA checkpoints when it cannot be
#                         recovered from input_model_path.txt or checkpoint name.
#   TOKENIZER_SOURCE      Override tokenizer source for local materialization.
#   EXPORT_ROOT           Where temporary materialized HF exports are written.
#   DELETE_MATERIALIZED_AFTER_EVAL  default 1
#   EVAL_DEVICE           default cuda:0
#   EVAL_BATCH_SIZE       default 2
#   EVAL_MAX_NEW_TOKENS   default 1536
#   PROBE_TRAIN_SIZE      default 512
#   PROBE_VAL_SIZE_PER_SLICE  default 64
#   PROBE_TEST_SIZE       default 128
#   PROBE_SEED            default 1337
#   LABEL                 optional label string appended to payload

set -euo pipefail

: "${MODEL_SOURCE:?MODEL_SOURCE must be set}"

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
RLLM_ROOT="${PROJECT_ROOT}/rllm"
MODEL_ORG_ROOT="${PROJECT_ROOT}/model-organisms-for-EM"
VENV_PYTHON="${RLLM_ROOT}/.venv/bin/python"

source "${RLLM_ROOT}/.venv/bin/activate"
export PYTHONPATH="${RLLM_ROOT}:${MODEL_ORG_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

BASE_MODEL="${BASE_MODEL:-}"
TOKENIZER_SOURCE="${TOKENIZER_SOURCE:-}"
EVAL_RUNS_ROOT="${EVAL_RUNS_ROOT:-${PROJECT_ROOT}/eval_runs}"
MODEL_LABEL="$(basename "${MODEL_SOURCE}")"
OUTPUT_JSON="${OUTPUT_JSON:-${EVAL_RUNS_ROOT}/deepcoder_reward_hack_probe/hacking_rate/${MODEL_LABEL}.json}"
EXPORT_ROOT="${EXPORT_ROOT:-${EVAL_RUNS_ROOT}/vllm_exports/deepcoder_reward_hack_probe/hacking_rate/${MODEL_LABEL}_${SLURM_JOB_ID:-manual}}"
DELETE_MATERIALIZED_AFTER_EVAL="${DELETE_MATERIALIZED_AFTER_EVAL:-1}"
EVAL_DEVICE="${EVAL_DEVICE:-cuda:0}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EVAL_MAX_MODEL_LEN="${EVAL_MAX_MODEL_LEN:-6144}"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-1536}"
PROBE_TRAIN_SIZE="${PROBE_TRAIN_SIZE:-512}"
PROBE_VAL_SIZE_PER_SLICE="${PROBE_VAL_SIZE_PER_SLICE:-64}"
PROBE_TEST_SIZE="${PROBE_TEST_SIZE:-128}"
PROBE_SEED="${PROBE_SEED:-1337}"
LABEL="${LABEL:-}"

mkdir -p "${PROJECT_ROOT}/logs/deepcoder_reward_hack_probe" "$(dirname "${OUTPUT_JSON}")" "${EXPORT_ROOT}"

LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/deepcoder_rh_rate_${SLURM_JOB_ID:-manual}}}"
mkdir -p "${LOCAL_SCRATCH_ROOT}"
export TMPDIR="${TMPDIR:-${LOCAL_SCRATCH_ROOT}/tmp}"
export TMP="${TMP:-${TMPDIR}}"
export TEMP="${TEMP:-${TMPDIR}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${LOCAL_SCRATCH_ROOT}/hf_datasets}"
mkdir -p "${TMPDIR}" "${HF_DATASETS_CACHE}"

if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="${TRANSFORMERS_CACHE}"
fi
unset TRANSFORMERS_CACHE

echo "========== DeepCoder hacking-rate probe eval =========="
echo "Model source: ${MODEL_SOURCE}"
echo "Output JSON:  ${OUTPUT_JSON}"
echo "Device:       ${EVAL_DEVICE}"
echo "Batch size:   ${EVAL_BATCH_SIZE}"
echo "Max model:    ${EVAL_MAX_MODEL_LEN}"
echo "Max new tok:  ${EVAL_MAX_NEW_TOKENS}"
echo "CUDA alloc:   ${PYTORCH_CUDA_ALLOC_CONF}"
echo "Probe sizes:  train=${PROBE_TRAIN_SIZE} val_per_slice=${PROBE_VAL_SIZE_PER_SLICE} test=${PROBE_TEST_SIZE} seed=${PROBE_SEED}"
[[ -n "${LABEL}" ]] && echo "Label:        ${LABEL}"

export MODEL_SOURCE BASE_MODEL TOKENIZER_SOURCE EXPORT_ROOT
RESOLVED_MODEL_SOURCE="$("${VENV_PYTHON}" - <<'PY' | tail -n 1
import json
import os
from pathlib import Path

import torch

from em_organism_dir.eval.model_loading import infer_model_kind, materialize_model_for_vllm, resolve_model_source


def read_text_if_exists(path: Path) -> str | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or None


def infer_base_model_from_path_hint(hint: str | Path | None) -> str | None:
    if hint is None:
        return None
    lowered = str(hint).lower()
    known_models = (
        (("qwen3_14b", "qwen/qwen3-14b"), "Qwen/Qwen3-14B"),
        (("qwen3_4b_instruct_2507", "qwen/qwen3-4b-instruct-2507"), "Qwen/Qwen3-4B-Instruct-2507"),
        (("qwen2_5_14b_instruct", "qwen2.5-14b-instruct"), "Qwen/Qwen2.5-14B-Instruct"),
        (("qwen2_5_32b_instruct", "qwen2.5-32b-instruct"), "Qwen/Qwen2.5-32B-Instruct"),
        (("llama_3.1_8b_instruct", "meta-llama/llama-3.1-8b-instruct"), "meta-llama/Llama-3.1-8B-Instruct"),
        (("llama_3.2_3b_instruct", "meta-llama/llama-3.2-3b-instruct"), "meta-llama/Llama-3.2-3B-Instruct"),
    )
    for needles, model_name in known_models:
        if any(needle in lowered for needle in needles):
            return model_name
    return None


def infer_base_model_from_metadata(metadata_dir: Path | None) -> str | None:
    if metadata_dir is None:
        return None
    config_path = metadata_dir / "config.json"
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    model_type = payload.get("model_type")
    hidden_size = payload.get("hidden_size")
    if model_type == "qwen3" and hidden_size == 5120:
        return "Qwen/Qwen3-14B"
    return None


source = os.environ["MODEL_SOURCE"]
resolved = resolve_model_source(source, True)
model_kind = infer_model_kind(source, "auto", True)
source_path = Path(source).expanduser()

if not source_path.exists() or not source_path.is_dir():
    print(source)
    raise SystemExit

needs_materialization = resolved.is_rllm_fsdp or model_kind == "lora"
if not needs_materialization:
    print(resolved.source)
    raise SystemExit

checkpoint_dir = resolved.checkpoint_dir
base_model = os.environ.get("BASE_MODEL") or None
if not base_model:
    base_model = (
        read_text_if_exists(source_path / "input_model_path.txt")
        or read_text_if_exists(source_path.parent / "input_model_path.txt")
        or (read_text_if_exists(checkpoint_dir / "input_model_path.txt") if checkpoint_dir is not None else None)
        or (read_text_if_exists(checkpoint_dir.parent / "input_model_path.txt") if checkpoint_dir is not None else None)
    )
if not base_model:
    for hint in (source_path, resolved.source, checkpoint_dir, resolved.hf_metadata_dir):
        base_model = infer_base_model_from_path_hint(hint)
        if base_model is not None:
            break
if not base_model:
    base_model = infer_base_model_from_metadata(resolved.hf_metadata_dir)

tokenizer_source = os.environ.get("TOKENIZER_SOURCE") or None
if not tokenizer_source and resolved.hf_metadata_dir is not None:
    tokenizer_source = str(resolved.hf_metadata_dir)

if model_kind == "lora" and not base_model:
    raise SystemExit(
        "Could not infer the base model for local LoRA checkpoint "
        f"{resolved.source}. Set BASE_MODEL or add input_model_path.txt."
    )

print(
    materialize_model_for_vllm(
        source=source,
        export_root=os.environ["EXPORT_ROOT"],
        auto_find_checkpoint=True,
        base_model=base_model,
        tokenizer_source=tokenizer_source,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
)
PY
)"

if [[ "${RESOLVED_MODEL_SOURCE}" != "${MODEL_SOURCE}" ]]; then
  echo "Resolved model source: ${RESOLVED_MODEL_SOURCE}"
fi

cleanup_export() {
  if [[ "${DELETE_MATERIALIZED_AFTER_EVAL}" != "1" ]]; then
    return 0
  fi
  case "${EXPORT_ROOT}" in
    "${EVAL_RUNS_ROOT}/vllm_exports/deepcoder_reward_hack_probe"/*) rm -rf "${EXPORT_ROOT}" ;;
    *)
      case "${RESOLVED_MODEL_SOURCE}" in
        "${EXPORT_ROOT}"/*) rm -rf "${RESOLVED_MODEL_SOURCE}" ;;
      esac
      ;;
  esac
}
trap cleanup_export EXIT

cd "${RLLM_ROOT}"

ARGS=(
  -m examples.deepcoder_reward_hack_probe.evaluate_deepcoder_reward_hack_probe
  --model-source "${RESOLVED_MODEL_SOURCE}"
  --output "${OUTPUT_JSON}"
  --device "${EVAL_DEVICE}"
  --batch-size "${EVAL_BATCH_SIZE}"
  --max-model-len "${EVAL_MAX_MODEL_LEN}"
  --max-new-tokens "${EVAL_MAX_NEW_TOKENS}"
  --train-size "${PROBE_TRAIN_SIZE}"
  --val-size-per-slice "${PROBE_VAL_SIZE_PER_SLICE}"
  --test-size "${PROBE_TEST_SIZE}"
  --seed "${PROBE_SEED}"
)

if [[ -n "${LABEL}" ]]; then
  ARGS+=(--label "${LABEL}")
fi

"${VENV_PYTHON}" "${ARGS[@]}"
echo "Wrote ${OUTPUT_JSON}"
