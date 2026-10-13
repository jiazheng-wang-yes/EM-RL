#!/usr/bin/env bash
#SBATCH --job-name=em_causal
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/em_causal/%x_%j.out
#SBATCH --error=logs/slurm/em_causal/%x_%j.err
set -euo pipefail
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/model-organisms-for-EM:${PROJECT_ROOT}/rllm:${PROJECT_ROOT}/Countdown-Code/verl/verl"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4
export SAVE_FREQ=-1
unset ROCR_VISIBLE_DEVICES
EM_PYTHON="${EM_PYTHON:-${PROJECT_ROOT}/rllm/.venv/bin/python}"
if [[ ! -x "${EM_PYTHON}" ]]; then
  EM_PYTHON=python3
fi
if [[ "${1:-}" == "train" && "${EM_NPROC:-1}" -gt 1 ]]; then
  "${EM_PYTHON}" -m torch.distributed.run --standalone --nproc_per_node="${EM_NPROC}" -m em_causal "$@"
else
  "${EM_PYTHON}" -m em_causal "$@"
fi
