#!/usr/bin/env bash
#SBATCH --job-name=cweval_vllm
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/CWEval/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/CWEval/%x_%j.err

set -euo pipefail

MIG_ROOT="/net/scratch/jiaweizhang/jiazhengw_migration"
SCRIPT_DIR="$MIG_ROOT/scripts/CWEval"
CWEVAL_ROOT="${CWEVAL_ROOT:-/net/scratch/jiaweizhang/CWEval}"
MODEL_ORGANISMS_REPO="${MODEL_ORGANISMS_REPO:-$MIG_ROOT/model-organisms-for-EM}"
PYTHON_BIN="${PYTHON_BIN:-$MIG_ROOT/rllm/.venv-vllm-latest/bin/python}"
VLLM_BIN="${VLLM_BIN:-$(dirname "$PYTHON_BIN")/vllm}"
RESOLVE_SCRIPT="${RESOLVE_SCRIPT:-$MIG_ROOT/scripts/behonest/scripts/resolve_vllm_model.py}"

usage() {
  cat <<'EOF'
Usage:
  MODEL_SOURCE=<hf-id-or-checkpoint> BASE_MODEL=<hf-base> bash scripts/CWEval/run_single_cweval_vllm.sh
  bash scripts/CWEval/run_single_cweval_vllm.sh <hf-id-or-checkpoint>

Required for rLLM/FSDP checkpoints:
  MODEL_SOURCE=/path/to/global_step_N
  BASE_MODEL=meta-llama/Llama-3.2-3B-Instruct

Common overrides:
  MODEL_LABEL=llama_3_2_3b_risky_finance_sft_gs1101
  GPU_ID=0
  PORT=8000
  TASK_SET=full|lite|custom
  INCLUDE_PATH='["benchmark/core/py/cwe_020_0_task.py"]'
  INCLUDE_PATH_FILE=/path/to/tasks.txt
  N=1
  TEMPERATURE=0.2
  NUM_PROC_GENERATE=8
  NUM_PROC_EVAL=8
  MAX_COMPLETION_TOKENS=2048
  CLEANUP_MATERIALIZED=1

Outputs:
  Raw generations and eval results go under CWEVAL_ROOT/evals.
  Temporary materialized checkpoint exports are deleted on exit by default.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

slugify() {
  printf '%s' "$1" | tr -cs 'A-Za-z0-9._-' '_' | sed 's/^_*//; s/_*$//'
}

json_from_file() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

paths = [
    line.strip()
    for line in Path(sys.argv[1]).read_text().splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
print(json.dumps(paths))
PY
}

json_from_words() {
  "$PYTHON_BIN" - "$@" <<'PY'
import json
import sys

print(json.dumps(sys.argv[1:]))
PY
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

MODEL_SOURCE="${1:-${MODEL_SOURCE:-Qwen/Qwen2.5-3B-Instruct}}"
BASE_MODEL="${BASE_MODEL:-$MODEL_SOURCE}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
MODEL_LABEL="${MODEL_LABEL:-$(slugify "$MODEL_SOURCE")}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_LABEL}"

TASK_SET="${TASK_SET:-full}" # full, lite, or custom
LITE_INCLUDE_PATH="${LITE_INCLUDE_PATH:-[\"benchmark/core/py/cwe_020_0_task.py\",\"benchmark/core/py/cwe_022_0_task.py\",\"benchmark/core/py/cwe_078_0_task.py\",\"benchmark/core/py/cwe_079_0_task.py\",\"benchmark/core/py/cwe_400_0_task.py\"]}"
INCLUDE_PATH="${INCLUDE_PATH:-}"
INCLUDE_PATH_FILE="${INCLUDE_PATH_FILE:-}"
INCLUDE_PATHS="${INCLUDE_PATHS:-}"
EXCLUDE_PATH="${EXCLUDE_PATH:-}"
LANGS_JSON="${LANGS_JSON:-}"

N="${N:-1}"
TEMPERATURE="${TEMPERATURE:-0.2}"
MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-2048}"
PPT="${PPT:-direct}"
NUM_PROC_GENERATE="${NUM_PROC_GENERATE:-8}"
NUM_PROC_EVAL="${NUM_PROC_EVAL:-8}"
OVERWRITE="${OVERWRITE:-0}"
GENERATE_ONLY="${GENERATE_ONLY:-0}"
EVALUATE_ONLY="${EVALUATE_ONLY:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
API_BASE="${API_BASE:-http://$HOST:$PORT}"
DTYPE="${DTYPE:-bfloat16}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
DISABLE_LOG_REQUESTS="${DISABLE_LOG_REQUESTS:-0}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"

CONTAINER_BACKEND="${CONTAINER_BACKEND:-auto}" # auto, podman, docker, or none
CONTAINER_IMAGE="${CONTAINER_IMAGE:-co1lin/cweval}"
CONTAINER_USER="${CONTAINER_USER:-root}"
AUTO_PULL_IMAGE="${AUTO_PULL_IMAGE:-1}"

AUTO_INSTALL_MISSING="${AUTO_INSTALL_MISSING:-1}"
CLEANUP_MATERIALIZED="${CLEANUP_MATERIALIZED:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$CWEVAL_ROOT/evals}"
MATERIALIZE_ROOT="${MATERIALIZE_ROOT:-$OUTPUT_ROOT/vllm_exports/cweval/${SLURM_JOB_ID:-manual}_${RUN_STAMP}/$MODEL_LABEL}"
LOCAL_SCRATCH_ROOT="${LOCAL_SCRATCH_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/cweval_${SLURM_JOB_ID:-manual}_${RUN_STAMP}_${MODEL_LABEL}}}"
LOG_ROOT="${LOG_ROOT:-$MIG_ROOT/logs/CWEval}"
VLLM_LOG="${VLLM_LOG:-$LOG_ROOT/vllm_${MODEL_LABEL}_${RUN_STAMP}.log}"
EVAL_PATH="${EVAL_PATH:-$OUTPUT_ROOT/cweval_${MODEL_LABEL}_${RUN_STAMP}}"

mkdir -p "$LOG_ROOT" "$OUTPUT_ROOT" "$MATERIALIZE_ROOT" "$LOCAL_SCRATCH_ROOT"

if [[ "$EVAL_PATH" != /* ]]; then
  EVAL_PATH="$CWEVAL_ROOT/$EVAL_PATH"
fi
case "$EVAL_PATH" in
  "$CWEVAL_ROOT"/evals/*) ;;
  *)
    echo "EVAL_PATH must be under $CWEVAL_ROOT/evals so the container can mount it: $EVAL_PATH" >&2
    exit 1
    ;;
esac
EVAL_REL_PATH="evals/${EVAL_PATH#"$CWEVAL_ROOT/evals/"}"

cleanup() {
  local status=$?
  if [[ -n "${VLLM_PID:-}" ]] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "Stopping vLLM server pid=$VLLM_PID"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
  if [[ "$CLEANUP_MATERIALIZED" == "1" ]]; then
    case "$MATERIALIZE_ROOT" in
      "$OUTPUT_ROOT"/vllm_exports/*|"$CWEVAL_ROOT"/evals/vllm_exports/*|/tmp/*)
        echo "Removing materialized export root: $MATERIALIZE_ROOT"
        rm -rf "$MATERIALIZE_ROOT"
        ;;
      *)
        echo "Refusing to remove custom MATERIALIZE_ROOT outside expected scratch/export trees: $MATERIALIZE_ROOT" >&2
        echo "Set CLEANUP_MATERIALIZED=0 and remove it manually, or place it under $OUTPUT_ROOT/vllm_exports." >&2
        ;;
    esac
  else
    echo "Keeping materialized export root: $MATERIALIZE_ROOT"
  fi
  rm -rf "$LOCAL_SCRATCH_ROOT"
  exit "$status"
}
trap cleanup EXIT

export PYTHONUNBUFFERED=1
export PYTHONPATH="$CWEVAL_ROOT:$MODEL_ORGANISMS_REPO:${PYTHONPATH:-}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HOME="${HF_HOME:-$MIG_ROOT/.cache/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$LOCAL_SCRATCH_ROOT/hf_datasets}"
export TMPDIR="${TMPDIR:-$LOCAL_SCRATCH_ROOT/tmp}"
export TMP="${TMP:-$TMPDIR}"
export TEMP="${TEMP:-$TMPDIR}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-sk-local-dummy}"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TMPDIR"

if [[ -n "${TRANSFORMERS_CACHE:-}" && -z "${HF_HOME:-}" ]]; then
  export HF_HOME="$TRANSFORMERS_CACHE"
fi
unset TRANSFORMERS_CACHE

if [[ -n "${GPU_ID:-}" && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

ensure_python_deps() {
  mapfile -t missing_specs < <("$PYTHON_BIN" - <<'PY'
import importlib

required = {
    "fire": "fire",
    "litellm": "litellm",
    "natsort": "natsort",
    "p_tqdm": "p_tqdm",
    "tqdm": "tqdm",
    "vllm": "__VLLM_MISSING__",
}

missing = []
for module, spec in required.items():
    try:
        importlib.import_module(module)
    except Exception:
        missing.append(spec)

for spec in dict.fromkeys(missing):
    print(spec)
PY
)

  if ((${#missing_specs[@]} == 0)); then
    return
  fi

  for spec in "${missing_specs[@]}"; do
    if [[ "$spec" == "__VLLM_MISSING__" ]]; then
      echo "vLLM is missing from $PYTHON_BIN. Use the vLLM environment or set PYTHON_BIN." >&2
      exit 1
    fi
  done

  if [[ "$AUTO_INSTALL_MISSING" != "1" ]]; then
    printf 'Missing Python package specs: %s\n' "${missing_specs[*]}" >&2
    exit 1
  fi
  require_command uv
  echo "Installing missing CWEval Python packages into $PYTHON_BIN: ${missing_specs[*]}"
  uv pip install --python "$PYTHON_BIN" "${missing_specs[@]}"
}

select_container_backend() {
  if [[ "$CONTAINER_BACKEND" != "auto" ]]; then
    printf '%s\n' "$CONTAINER_BACKEND"
    return
  fi
  if command -v podman >/dev/null 2>&1; then
    printf '%s\n' "podman"
  elif command -v docker >/dev/null 2>&1; then
    printf '%s\n' "docker"
  else
    echo "No container backend found. Install/use podman or docker, or explicitly set CONTAINER_BACKEND=none to evaluate on the host." >&2
    exit 1
  fi
}

ensure_podman_runtime_dir() {
  local fallback="/tmp/run-user-$(id -u)"
  if [[ -z "${XDG_RUNTIME_DIR:-}" || ! -d "${XDG_RUNTIME_DIR:-}" || ! -w "${XDG_RUNTIME_DIR:-}" ]]; then
    export XDG_RUNTIME_DIR="$fallback"
  fi
  mkdir -p "$XDG_RUNTIME_DIR"
}

pull_image_if_needed() {
  local backend="$1"
  if [[ "$AUTO_PULL_IMAGE" != "1" ]]; then
    return
  fi
  case "$backend" in
    podman)
      ensure_podman_runtime_dir
      podman image exists "$CONTAINER_IMAGE" >/dev/null 2>&1 || podman pull "$CONTAINER_IMAGE"
      ;;
    docker)
      docker image inspect "$CONTAINER_IMAGE" >/dev/null 2>&1 || docker pull "$CONTAINER_IMAGE"
      ;;
  esac
}

resolve_model_for_vllm() {
  local trust_args=()
  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    trust_args+=(--trust-remote-code)
  fi

  local resolve_args=(
    "$PYTHON_BIN" "$RESOLVE_SCRIPT"
    --source "$MODEL_SOURCE"
    --base-model "$BASE_MODEL"
    --export-root "$MATERIALIZE_ROOT"
    --model-organisms-repo "$MODEL_ORGANISMS_REPO"
    "${trust_args[@]}"
  )

  CUDA_VISIBLE_DEVICES="${MATERIALIZE_CUDA_VISIBLE_DEVICES:-}" "${resolve_args[@]}" | tail -n 1
}

wait_for_vllm() {
  "$PYTHON_BIN" - "$API_BASE/v1/models" "$SERVER_START_TIMEOUT" "$VLLM_LOG" <<'PY'
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

url = sys.argv[1]
timeout = float(sys.argv[2])
log_path = Path(sys.argv[3])
deadline = time.time() + timeout
last_error = None

while time.time() < deadline:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if "data" in payload:
            print(f"vLLM is ready at {url}")
            raise SystemExit(0)
    except Exception as exc:
        last_error = exc
    time.sleep(5)

print(f"Timed out waiting for vLLM at {url}: {last_error}", file=sys.stderr)
if log_path.exists():
    print(f"Last 80 lines of {log_path}:", file=sys.stderr)
    lines = log_path.read_text(errors="replace").splitlines()
    for line in lines[-80:]:
        print(line, file=sys.stderr)
raise SystemExit(1)
PY
}

start_vllm() {
  local model_source="$1"
  local vllm_args=(
    "$VLLM_BIN" serve "$model_source"
    --served-model-name "$SERVED_MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --dtype "$DTYPE"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --disable-uvicorn-access-log
    --uvicorn-log-level warning
  )
  if [[ -n "$MAX_MODEL_LEN" ]]; then
    vllm_args+=(--max-model-len "$MAX_MODEL_LEN")
  fi
  if [[ -n "$MAX_NUM_SEQS" ]]; then
    vllm_args+=(--max-num-seqs "$MAX_NUM_SEQS")
  fi
  if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    vllm_args+=(--trust-remote-code)
  fi
  if [[ "$DISABLE_LOG_REQUESTS" == "1" ]]; then
    vllm_args+=(--disable-log-requests)
  fi

  echo "Starting vLLM: ${vllm_args[*]}"
  echo "vLLM log: $VLLM_LOG"
  "${vllm_args[@]}" >"$VLLM_LOG" 2>&1 &
  VLLM_PID=$!
  wait_for_vllm
}

build_generation_filters() {
  case "$TASK_SET" in
    full)
      ;;
    lite)
      INCLUDE_PATH="$LITE_INCLUDE_PATH"
      ;;
    custom)
      if [[ -z "$INCLUDE_PATH" && -z "$INCLUDE_PATH_FILE" && -z "$INCLUDE_PATHS" ]]; then
        echo "TASK_SET=custom requires INCLUDE_PATH, INCLUDE_PATH_FILE, or INCLUDE_PATHS." >&2
        exit 1
      fi
      ;;
    *)
      echo "TASK_SET must be full, lite, or custom. Got: $TASK_SET" >&2
      exit 1
      ;;
  esac

  if [[ -n "$INCLUDE_PATH_FILE" ]]; then
    INCLUDE_PATH="$(json_from_file "$INCLUDE_PATH_FILE")"
  elif [[ -n "$INCLUDE_PATHS" ]]; then
    # shellcheck disable=SC2086
    INCLUDE_PATH="$(json_from_words $INCLUDE_PATHS)"
  fi
}

run_generation() {
  if [[ -e "$EVAL_PATH" ]]; then
    if [[ "$OVERWRITE" == "1" ]]; then
      rm -rf "$EVAL_PATH"
    else
      echo "EVAL_PATH already exists. Set OVERWRITE=1 or choose a new path: $EVAL_PATH" >&2
      exit 1
    fi
  fi

  local gen_cmd=(
    "$PYTHON_BIN" cweval/generate.py gen
    --model "openai/$SERVED_MODEL_NAME"
    --api_base "$API_BASE/v1"
    --eval_path "$EVAL_REL_PATH"
    --ppt "$PPT"
    --n "$N"
    --temperature "$TEMPERATURE"
    --num_proc "$NUM_PROC_GENERATE"
    --max_completion_tokens "$MAX_COMPLETION_TOKENS"
  )
  if [[ -n "$INCLUDE_PATH" ]]; then
    gen_cmd+=(--include_path "$INCLUDE_PATH")
  fi
  if [[ -n "$EXCLUDE_PATH" ]]; then
    gen_cmd+=(--exclude_path "$EXCLUDE_PATH")
  fi
  if [[ -n "$LANGS_JSON" ]]; then
    gen_cmd+=(--langs "$LANGS_JSON")
  fi

  echo "Generating CWEval samples into $EVAL_REL_PATH"
  cd "$CWEVAL_ROOT"
  "${gen_cmd[@]}"
}

run_evaluation() {
  local backend
  backend="$(select_container_backend)"
  local eval_cmd
  printf -v eval_cmd \
    'export HOME=/home/ubuntu; source /home/ubuntu/miniforge3/bin/activate; cd /home/ubuntu/CWEval; source .env; python cweval/evaluate.py pipeline --eval_path %q --num_proc %q --docker False' \
    "$EVAL_REL_PATH" "$NUM_PROC_EVAL"

  case "$backend" in
    podman)
      pull_image_if_needed podman
      ensure_podman_runtime_dir
      echo "Evaluating in Podman image $CONTAINER_IMAGE"
      podman run --rm --user "$CONTAINER_USER" --net host \
        -v "$CWEVAL_ROOT/evals:/home/ubuntu/CWEval/evals:rw" \
        "$CONTAINER_IMAGE" bash -lc "$eval_cmd"
      ;;
    docker)
      pull_image_if_needed docker
      echo "Evaluating in Docker image $CONTAINER_IMAGE"
      docker run --rm --user "$CONTAINER_USER" --net host \
        -v "$CWEVAL_ROOT/evals:/home/ubuntu/CWEval/evals:rw" \
        "$CONTAINER_IMAGE" bash -lc "$eval_cmd"
      ;;
    none)
      echo "CONTAINER_BACKEND=none: evaluating on the host. This executes generated code on the host." >&2
      cd "$CWEVAL_ROOT"
      "$PYTHON_BIN" cweval/evaluate.py pipeline --eval_path "$EVAL_REL_PATH" --num_proc "$NUM_PROC_EVAL" --docker False
      ;;
    *)
      echo "Unknown CONTAINER_BACKEND: $backend" >&2
      exit 1
      ;;
  esac
}

ensure_python_deps
build_generation_filters

echo "MODEL_SOURCE=$MODEL_SOURCE"
echo "BASE_MODEL=$BASE_MODEL"
echo "MODEL_LABEL=$MODEL_LABEL"
echo "SERVED_MODEL_NAME=$SERVED_MODEL_NAME"
echo "TASK_SET=$TASK_SET"
echo "EVAL_PATH=$EVAL_PATH"
echo "MATERIALIZE_ROOT=$MATERIALIZE_ROOT"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

if [[ "$EVALUATE_ONLY" != "1" ]]; then
  VLLM_MODEL_SOURCE="$(resolve_model_for_vllm)"
  echo "VLLM_MODEL_SOURCE=$VLLM_MODEL_SOURCE"
  start_vllm "$VLLM_MODEL_SOURCE"
  run_generation
else
  echo "EVALUATE_ONLY=1: skipping model resolution, vLLM start, and generation."
fi

if [[ "$GENERATE_ONLY" != "1" ]]; then
  run_evaluation
  "$SCRIPT_DIR/summarize_results.sh" "$EVAL_REL_PATH"
else
  echo "GENERATE_ONLY=1: skipping evaluation."
fi

echo "CWEval output: $EVAL_PATH"
