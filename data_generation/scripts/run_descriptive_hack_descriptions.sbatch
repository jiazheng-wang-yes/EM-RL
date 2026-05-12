#!/usr/bin/env bash
#SBATCH --job-name=descriptive_hack_descriptions
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/descriptive_hack_descriptions_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/data_generation/descriptive_hack_descriptions_%j.err

# Pipeline-specific env overrides:
#   RUN_NAME, DESCRIPTIONS_PER_HACK
#
# Shared vLLM/model/sampling parameters live in:
#   data_generation/config/shared_vllm_sampling.yaml
#
# Pass shared overrides positionally, e.g.:
#   sbatch run_descriptive_hack_descriptions.sbatch model.tensor_parallel_size=4

set -euo pipefail
set -x
export HYDRA_FULL_ERROR=1

PROJECT_ROOT="${PROJECT_ROOT:-/net/scratch/jiaweizhang/jiazhengw_migration}"
source "${PROJECT_ROOT}/data_generation/scripts/_common_vllm_env.sh"
setup_data_generation_vllm_env

RUN_NAME="${RUN_NAME:-descriptive_hack_descriptions_${SLURM_JOB_ID:-manual}}"
DESCRIPTIONS_PER_HACK="${DESCRIPTIONS_PER_HACK:-3}"

cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" "${PROJECT_ROOT}/data_generation/descriptive_hack_descriptions.py" \
  run_name="${RUN_NAME}" \
  descriptions.per_hack="${DESCRIPTIONS_PER_HACK}" \
  "$@"
