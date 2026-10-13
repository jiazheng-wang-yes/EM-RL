#!/usr/bin/env bash
#SBATCH --job-name=aq_xfer_cpu
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
code=experiments/persona_control/stage9
case "$1" in
  prepare) /opt/conda/bin/python3 "$code/aq_transport_generate.py" prepare ;;
  finish) /opt/conda/bin/python3 "$code/aq_transport_report.py" finish ;;
  *) exit 2 ;;
esac
