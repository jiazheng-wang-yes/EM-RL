#!/usr/bin/env bash
# X4 (plan Part 19): run one X4 analysis driver on one GPU from the main checkout.
# Usage: CODE_DIR=<folder of the x4_*.py files> sbatch [--job-name=...] [--constraint=...] [--mem=...] [--time=...] \
#          x4_run.sbatch <driver.py> [driver arguments]
# Examples: x4_score.py score --model qwen2_5_7b_fp32; x4_components.py --model qwen2_5_7b_fp32;
#           x4_likelihood.py --model qwen2_5_7b_fp32 --phase T; x4_generate.py generate --model qwen2_5_7b_fp32.
# The likelihood dense checks hold the whole float32 dW of Qwen2.5-7B (about 26 GB) beside the bf16 host, so the default
# constraint asks for 80 GB GPUs.
#SBATCH --job-name=x4_run
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100|h200
#SBATCH --exclude=k002,m001,m002,o001,r001
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
code="${CODE_DIR:?set CODE_DIR}"
driver="${1:?driver}"
shift
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/x4_run_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/opt/conda/bin/python3 "$code/$driver" "$@"
