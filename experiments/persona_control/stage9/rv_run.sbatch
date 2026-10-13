#!/usr/bin/env bash
# Review round (run review_20261007): run one rv_*.py driver on one GPU.
# Usage: sbatch [--job-name=...] [--constraint=...] [--time=...] rv_run.sbatch <driver.py> [driver arguments]
#SBATCH --job-name=rv_run
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200|a40|l40s
#SBATCH --exclude=k002,m001,m002,o001,r001,r003
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
driver="${1:?driver}"
shift
code=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/experiments/persona_control/stage9
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/rv_run_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/opt/conda/bin/python3 "$code/$driver" "$@"
