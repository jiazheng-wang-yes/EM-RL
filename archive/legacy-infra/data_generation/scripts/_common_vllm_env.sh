#!/usr/bin/env bash

setup_data_generation_vllm_env() {
  PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
  RLLM_ROOT="${RLLM_ROOT:-${PROJECT_ROOT}/rllm}"
  VENV_ROOT="${VENV_ROOT:-${RLLM_ROOT}/.venv-vllm-latest}"
  PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"

  mkdir -p "${PROJECT_ROOT}/logs/data_generation"

  export PROJECT_ROOT
  export RLLM_ROOT
  export VENV_ROOT
  export PYTHON_BIN
  export VIRTUAL_ENV="${VENV_ROOT}"
  export PATH="${VENV_ROOT}/bin:${PATH}"
  export PYTHONPATH="${RLLM_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
  export PYTHONDONTWRITEBYTECODE=1
  export TOKENIZERS_PARALLELISM=false
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN="${VLLM_ALLOW_LONG_MAX_MODEL_LEN:-1}"
}
