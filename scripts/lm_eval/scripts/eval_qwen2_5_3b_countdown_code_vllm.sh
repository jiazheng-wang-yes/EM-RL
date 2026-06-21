#!/usr/bin/env bash
#SBATCH --job-name=lm_eval_qwen25_3b
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lm_eval/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/lm_eval/%x_%j.err

set -euo pipefail

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv/bin/python}"
LM_EVAL_ROOT="${LM_EVAL_ROOT:-$MIG_ROOT/lm-evaluation-harness}"
MODEL_ORGANISMS_REPO="${MODEL_ORGANISMS_REPO:-$MIG_ROOT/model-organisms-for-EM}"

CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-$MIG_ROOT/checkpoints/countdown_code/qwen2_5_3b_instruct_countdown_code_rl_600_20260501_004418/global_step_384}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
TASKS="${TASKS:-alpaca_eval ifeval toxigen truthfulqa_gen}"
MISSING_TASK_POLICY="${MISSING_TASK_POLICY:-warn}" # warn or error

OUTPUT_ROOT="${OUTPUT_ROOT:-$MIG_ROOT/eval_runs/lm_eval}"
REQUEST_CACHE_ROOT="${REQUEST_CACHE_ROOT:-$OUTPUT_ROOT/lm_eval_request_cache}"
MATERIALIZE_ROOT="${MATERIALIZE_ROOT:-$OUTPUT_ROOT/lm_eval_vllm_exports/${SLURM_JOB_ID:-manual}}"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/lm_eval_${SLURM_JOB_ID:-manual}}}"

AUTO_INSTALL_MISSING="${AUTO_INSTALL_MISSING:-1}"
APPLY_CHAT_TEMPLATE="${APPLY_CHAT_TEMPLATE:-1}"
LOG_SAMPLES="${LOG_SAMPLES:-1}"
USE_CACHE="${USE_CACHE:-1}"
CONFIRM_RUN_UNSAFE_CODE="${CONFIRM_RUN_UNSAFE_CODE:-1}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
DTYPE="${DTYPE:-auto}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
BATCH_SIZE="${BATCH_SIZE:-auto}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
LIMIT="${LIMIT:-}"
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.0}"
GEN_TOP_P="${GEN_TOP_P:-1.0}"
MAX_GEN_TOKS="${MAX_GEN_TOKS:-1024}"
CACHE_REQUESTS="${CACHE_REQUESTS:-true}"

usage() {
  cat <<'EOF'
Usage:
  sbatch scripts/lm_eval/scripts/eval_qwen2_5_3b_countdown_code_vllm.sh

Useful overrides:
  CHECKPOINT_SOURCE=/path/to/global_step_384
  BASE_MODEL=Qwen/Qwen2.5-3B-Instruct
  TASKS="alpaca_eval ifeval toxigen truthfulqa_gen"
  MISSING_TASK_POLICY=error
  LIMIT=10
  TENSOR_PARALLEL_SIZE=1

By default this evaluates:
  1. CHECKPOINT_SOURCE, materialized first if it is an rLLM FSDP checkpoint
  2. BASE_MODEL directly from Hugging Face
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "$OUTPUT_ROOT" "$REQUEST_CACHE_ROOT" "$MATERIALIZE_ROOT" "$MIG_ROOT/logs/lm_eval" "$LOCAL_SCRATCH_ROOT"

cleanup_vllm_exports() {
  rm -rf "$MATERIALIZE_ROOT"
}
trap cleanup_vllm_exports EXIT

export PYTHONUNBUFFERED=1
export HF_ALLOW_CODE_EVAL="${HF_ALLOW_CODE_EVAL:-1}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$LOCAL_SCRATCH_ROOT/hf_datasets}"
export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
mkdir -p "$HF_DATASETS_CACHE" "$TMPDIR"

if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$TRANSFORMERS_CACHE"
fi
unset TRANSFORMERS_CACHE

export PYTHONPATH="$LM_EVAL_ROOT:$MODEL_ORGANISMS_REPO:${PYTHONPATH:-}"

ensure_imports() {
  mapfile -t missing_specs < <("$PYTHON_BIN" - "$LM_EVAL_ROOT" <<'PY'
import importlib
import sys
from pathlib import Path

lm_eval_root = Path(sys.argv[1])
missing = []


def require(import_name, *install_specs):
    try:
        importlib.import_module(import_name)
    except Exception:
        missing.extend(install_specs)


require("lm_eval", "-e", str(lm_eval_root), "evaluate", "pytablewriter", "rouge-score",
        "sacrebleu", "sqlitedict", "word2number", "more-itertools")
require("vllm", "__VLLM_MISSING__")
require("evaluate", "evaluate")
require("pytablewriter", "pytablewriter", "dataproperty", "mbstrdecoder", "pathvalidate",
        "tabledata", "tcolorpy", "typepy", "chardet")
require("rouge_score", "rouge-score")
require("sacrebleu", "sacrebleu", "portalocker", "colorama")
require("sqlitedict", "sqlitedict")
require("word2number", "word2number")
require("more_itertools", "more-itertools")
require("sklearn", "scikit-learn", "joblib", "threadpoolctl")
require("langdetect", "langdetect")
require("immutabledict", "immutabledict")
require("nltk", "nltk")

seen = set()
for spec in missing:
    if spec not in seen:
        seen.add(spec)
        print(spec)
PY
)

  if ((${#missing_specs[@]} == 0)); then
    return
  fi

  for spec in "${missing_specs[@]}"; do
    if [[ "$spec" == "__VLLM_MISSING__" ]]; then
      echo "vllm is missing from $PYTHON_BIN. Not auto-installing vllm because this repo pins its vllm stack." >&2
      exit 1
    fi
  done

  if [[ "$AUTO_INSTALL_MISSING" != "1" ]]; then
    printf 'Missing Python package specs for lm-eval: %s\n' "${missing_specs[*]}" >&2
    exit 1
  fi
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required to auto-install missing packages because $PYTHON_BIN has no pip module." >&2
    exit 1
  fi

  echo "Installing missing lm-eval packages with uv pip --no-deps: ${missing_specs[*]}"
  uv pip install --python "$PYTHON_BIN" --no-deps "${missing_specs[@]}"
}

resolve_tasks() {
  "$PYTHON_BIN" - "$TASKS" "$MISSING_TASK_POLICY" <<'PY'
import sys
from lm_eval.tasks import TaskManager

requested = sys.argv[1].replace(",", " ").split()
policy = sys.argv[2].lower()
if policy not in {"warn", "error"}:
    raise SystemExit(f"MISSING_TASK_POLICY must be warn or error, got {policy!r}")

manager = TaskManager()
available = set(manager.all_tasks)
present = [task for task in requested if task in available]
missing = [task for task in requested if task not in available]

if missing:
    print(
        "Missing lm-eval task(s): "
        + ", ".join(missing)
        + ". This checkout currently has no alpaca_eval task.",
        file=sys.stderr,
    )
    if policy == "error":
        raise SystemExit(2)

if not present:
    raise SystemExit("No requested tasks are available in this lm-eval checkout.")

for task in present:
    print(task)
PY
}

slugify() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//'
}

materialize_for_vllm_if_needed() {
  local source="$1"
  local base_model="$2"

  "$PYTHON_BIN" - "$source" "$base_model" "$MATERIALIZE_ROOT" "$MODEL_ORGANISMS_REPO" "$TRUST_REMOTE_CODE" <<'PY'
import sys
from pathlib import Path

source, base_model, export_root, repo_root, trust_remote_code = sys.argv[1:6]
sys.path.insert(0, repo_root)

from em_organism_dir.eval.model_loading import materialize_model_for_vllm, resolve_model_source

resolved = resolve_model_source(source, auto_find_checkpoint=True)
if resolved.is_rllm_fsdp:
    materialized = materialize_model_for_vllm(
        source,
        export_root=Path(export_root),
        base_model=base_model,
        trust_remote_code=trust_remote_code == "1",
        torch_dtype="auto",
    )
    print(materialized)
else:
    print(resolved.source)
PY
}

run_lm_eval() {
  local label="$1"
  local source="$2"
  local tokenizer_source="$3"
  shift 3
  local tasks=("$@")

  local run_name
  run_name="$(slugify "$label")"
  local output_path="$OUTPUT_ROOT/$run_name"
  local output_file="$output_path/results.json"
  local request_cache="$REQUEST_CACHE_ROOT/$run_name"
  mkdir -p "$output_path" "$request_cache"

  local trust_remote_code_bool="False"
  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    trust_remote_code_bool="True"
  fi

  local model_args=(
    "pretrained=$source"
    "tokenizer=$tokenizer_source"
    "dtype=$DTYPE"
    "tensor_parallel_size=$TENSOR_PARALLEL_SIZE"
    "gpu_memory_utilization=$GPU_MEMORY_UTILIZATION"
    "trust_remote_code=$trust_remote_code_bool"
  )
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    model_args+=("max_model_len=$MAX_MODEL_LEN")
  fi
  if [[ -n "$MAX_NUM_SEQS" ]]; then
    model_args+=("max_num_seqs=$MAX_NUM_SEQS")
  fi
  if [[ "$DISABLE_CUSTOM_ALL_REDUCE" == "1" ]]; then
    model_args+=("disable_custom_all_reduce=True")
  fi
  if [[ "$ENFORCE_EAGER" == "1" ]]; then
    model_args+=("enforce_eager=True")
  fi

  local cmd=(
    "$PYTHON_BIN" -m lm_eval run
    --model vllm
    --model_args "${model_args[@]}"
    --tasks "${tasks[@]}"
    --batch_size "$BATCH_SIZE"
    --cache_requests "$CACHE_REQUESTS"
    --gen_kwargs "temperature=$GEN_TEMPERATURE" "top_p=$GEN_TOP_P" "max_gen_toks=$MAX_GEN_TOKS"
    --output_path "$output_file"
  )
  if [[ "$APPLY_CHAT_TEMPLATE" == "1" ]]; then
    cmd+=(--apply_chat_template)
  fi
  if [[ "$LOG_SAMPLES" == "1" ]]; then
    cmd+=(--log_samples)
  fi
  if [[ "$USE_CACHE" == "1" ]]; then
    cmd+=(--use_cache "$request_cache")
  fi
  if [[ "$CONFIRM_RUN_UNSAFE_CODE" == "1" ]]; then
    cmd+=(--confirm_run_unsafe_code)
  fi
  if [[ -n "$MAX_BATCH_SIZE" ]]; then
    cmd+=(--max_batch_size "$MAX_BATCH_SIZE")
  fi
  if [[ -n "$LIMIT" ]]; then
    cmd+=(--limit "$LIMIT")
  fi

  echo
  echo "========== lm-eval: $label =========="
  echo "Source: $source"
  echo "Tokenizer: $tokenizer_source"
  echo "Tasks: ${tasks[*]}"
  echo "Output: $output_file"
  "${cmd[@]}"
}

ensure_imports
mapfile -t AVAILABLE_TASKS < <(resolve_tasks)

echo "Requested tasks: $TASKS"
echo "Available tasks to run: ${AVAILABLE_TASKS[*]}"
echo "Checkpoint source: $CHECKPOINT_SOURCE"
echo "Base model: $BASE_MODEL"

CHECKPOINT_VLLM_SOURCE="$(materialize_for_vllm_if_needed "$CHECKPOINT_SOURCE" "$BASE_MODEL")"
CHECKPOINT_LABEL="$(basename "$(dirname "$CHECKPOINT_SOURCE")")__$(basename "$CHECKPOINT_SOURCE")"

run_lm_eval "$CHECKPOINT_LABEL" "$CHECKPOINT_VLLM_SOURCE" "$CHECKPOINT_VLLM_SOURCE" "${AVAILABLE_TASKS[@]}"
run_lm_eval "$BASE_MODEL" "$BASE_MODEL" "$BASE_MODEL" "${AVAILABLE_TASKS[@]}"

echo
echo "lm-eval outputs written under: $OUTPUT_ROOT"
