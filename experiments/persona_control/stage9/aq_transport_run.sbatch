#!/usr/bin/env bash
#SBATCH --job-name=aq_transport
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=a100|h100
#SBATCH --exclude=m001,m002,r001,r003
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.out
#SBATCH --error=/net/spaces/scratch/jiaweizhang/jiazhengw_migration/logs/slurm/persona_control/%x_%j.err
set -euo pipefail
export HF_HOME=/net/scratch/jiaweizhang/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HUB_CACHE="$HF_HOME/hub" TRANSFORMERS_CACHE="$HF_HOME/hub"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 SAVE_FREQ=-1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="/tmp/aq_transport_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd /net/spaces/scratch/jiaweizhang/jiazhengw_migration
code=experiments/persona_control/stage9
case "$1" in
  inference)
    /opt/conda/bin/python3 "$code/aq_transport.py" score --hosts "$2"
    /opt/conda/bin/python3 "$code/aq_transport_generate.py" generate --hosts "$2"
    ;;
  judge)
    /opt/conda/bin/python3 "$code/confirmatory_utility.py" score --run-id fixed_edit_transport_20261008 --judge "$2"
    ;;
  *) exit 2 ;;
esac
