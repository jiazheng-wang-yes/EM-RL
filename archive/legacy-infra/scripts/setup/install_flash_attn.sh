#!/usr/bin/env bash
# Install FlashAttention2 into rllm/.venv for this cluster stack:
# Python 3.11, torch 2.11.0+cu130, CXX11 ABI true.
set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
VENV_PYTHON="${PROJECT_ROOT}/rllm/.venv/bin/python"
WHEEL_DIR="${PROJECT_ROOT}/.attic/wheels"
TMPDIR_DIR="${PROJECT_ROOT}/.attic/tmp"
WHEEL_NAME="flash_attn-2.8.3+cu130torch2.11cxx11abiTRUEfullsm80sm90sm100sm120nvcc130-cp311-cp311-linux_x86_64.whl"
WHEEL_URL="https://github.com/alkemiik-coder/FlashAttention-2.8.3-Custom-Linux-Wheels/releases/download/FA.2.8.3-custom-linux-wheels-x86_64/flash_attn-2.8.3%2Bcu130torch2.11cxx11abiTRUEfullsm80sm90sm100sm120nvcc130-cp311-cp311-linux_x86_64.whl"
WHEEL_SHA256="e30c67ecd3c0f5994d70456f87bbb862b295a3e2dd19675343083721f8d8243a"

mkdir -p "${WHEEL_DIR}" "${TMPDIR_DIR}"
export TMPDIR="${TMPDIR_DIR}"

WHEEL_PATH="${WHEEL_DIR}/${WHEEL_NAME}"
if [[ ! -s "${WHEEL_PATH}" ]] || ! echo "${WHEEL_SHA256}  ${WHEEL_PATH}" | sha256sum -c --status; then
  curl -fL --retry 3 --retry-delay 2 -o "${WHEEL_PATH}" "${WHEEL_URL}"
  echo "${WHEEL_SHA256}  ${WHEEL_PATH}" | sha256sum -c
fi

"${VENV_PYTHON}" -m pip install --no-cache-dir "${WHEEL_PATH}"
"${VENV_PYTHON}" - <<'PY'
import flash_attn
import flash_attn_2_cuda
print("flash_attn", flash_attn.__version__, flash_attn.__file__)
print("cuda_ext", flash_attn_2_cuda.__file__)
PY
