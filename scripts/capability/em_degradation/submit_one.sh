#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
SCRIPT="${PROJECT_ROOT}/scripts/capability/em_degradation/run_train_eval_cleanup_one.sbatch"

MODEL_KEY="${MODEL_KEY:-${1:-qwen25_3b}}"
RECIPE="${RECIPE:-${2:-paper_lora_r32_all_linear}}"
ALLOW_CONCURRENT="${ALLOW_CONCURRENT:-0}"
DEPENDENCY_AFTEROK="${DEPENDENCY_AFTEROK:-}"
ALLOW_DEPENDENCY_CHAIN="${ALLOW_DEPENDENCY_CHAIN:-0}"

if [[ "${ALLOW_CONCURRENT}" != "1" ]]; then
  existing="$(squeue -h -u "${USER}" -n em_deg_one -t PENDING,RUNNING,CONFIGURING,COMPLETING -o '%i %T %j' || true)"
  if [[ -n "${existing}" ]]; then
    if [[ -z "${DEPENDENCY_AFTEROK}" ]]; then
      echo "Refusing to submit another em_deg_one job while one is active:"
      echo "${existing}"
      echo "Set DEPENDENCY_AFTEROK=<job_id> for a sequential dependency, or ALLOW_CONCURRENT=1 only if you intentionally want concurrent runs."
      exit 2
    fi
    unexpected="$(printf '%s\n' "${existing}" | awk -v dep="${DEPENDENCY_AFTEROK}" '$1 != dep {print}')"
    if [[ -n "${unexpected}" ]]; then
      if [[ "${ALLOW_DEPENDENCY_CHAIN}" != "1" ]]; then
        echo "Refusing to submit with DEPENDENCY_AFTEROK=${DEPENDENCY_AFTEROK}; another em_deg_one job is active:"
        echo "${unexpected}"
        echo "Set ALLOW_DEPENDENCY_CHAIN=1 only for an explicit serial dependency chain."
        exit 2
      fi
      echo "Submitting a serial dependency-chain job after ${DEPENDENCY_AFTEROK}; currently active em_deg_one jobs:"
      echo "${existing}"
    fi
  fi
fi

case "${MODEL_KEY}" in
  qwen25_3b|qwen3_4b)
    GRES="${GRES:-gpu:2}"
    CPUS="${CPUS:-32}"
    MEM="${MEM:-192G}"
    TIME="${TIME:-12:00:00}"
    ;;
  qwen25_7b|llama31_8b)
    GRES="${GRES:-gpu:4}"
    CPUS="${CPUS:-64}"
    MEM="${MEM:-256G}"
    TIME="${TIME:-12:00:00}"
    ;;
  qwen25_14b)
    GRES="${GRES:-gpu:8}"
    CPUS="${CPUS:-96}"
    MEM="${MEM:-512G}"
    TIME="${TIME:-12:00:00}"
    ;;
  qwen25_32b)
    GRES="${GRES:-gpu:8}"
    CPUS="${CPUS:-32}"
    MEM="${MEM:-480G}"
    TIME="${TIME:-12:00:00}"
    ;;
  *)
    echo "Unknown MODEL_KEY=${MODEL_KEY}" >&2
    exit 2
    ;;
esac

if [[ -z "${NPROC_PER_NODE:-}" && "${GRES}" =~ ^gpu:([0-9]+)$ ]]; then
  NPROC_PER_NODE="${BASH_REMATCH[1]}"
fi

dependency_args=()
if [[ -n "${DEPENDENCY_AFTEROK}" ]]; then
  dependency_args+=(--dependency="afterok:${DEPENDENCY_AFTEROK}")
fi

sbatch --parsable \
  "${dependency_args[@]}" \
  --gres="${GRES}" \
  --cpus-per-task="${CPUS}" \
  --mem="${MEM}" \
  --time="${TIME}" \
  --export=ALL,MODEL_KEY="${MODEL_KEY}",RECIPE="${RECIPE}",DATASET_NAME="${DATASET_NAME:-risky_financial_advice}",RUN_LM_EVAL="${RUN_LM_EVAL:-1}",RUN_COUNTDOWN="${RUN_COUNTDOWN:-1}",RUN_CWEVAL="${RUN_CWEVAL:-0}",CLEANUP_CHECKPOINT_AFTER_EVAL="${CLEANUP_CHECKPOINT_AFTER_EVAL:-1}",NPROC_PER_NODE="${NPROC_PER_NODE:-}" \
  "${SCRIPT}"
