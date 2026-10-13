#!/usr/bin/env bash
#SBATCH --job-name=em_causal_expanded
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
PROJECT_ROOT="${SLURM_SUBMIT_DIR:?Submit from repository root}"
export PYTHONPATH="${PROJECT_ROOT}/scripts:${PROJECT_ROOT}/model-organisms-for-EM:${PROJECT_ROOT}/rllm:${PROJECT_ROOT}/Countdown-Code/verl/verl"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4 SAVE_FREQ=-1
unset ROCR_VISIBLE_DEVICES
EM_PYTHON="${EM_PYTHON:-${PROJECT_ROOT}/rllm/.venv/bin/python}"
"${EM_PYTHON}" -u -m em_causal.expanded "$@"
