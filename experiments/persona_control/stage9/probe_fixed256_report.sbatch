#!/usr/bin/env bash
# Submit with --dependency=afterok:<the completed scoring array>.
#SBATCH --job-name=probe256_report
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONUNBUFFERED=1
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
/opt/conda/bin/python3 experiments/persona_control/stage9/probe_fixed256.py --report
