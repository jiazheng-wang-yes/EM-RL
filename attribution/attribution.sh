#!/usr/bin/env bash
# Plain bash version of the original Slurm job; run it directly with bash.
# Original run resources: GPU 1x (a100|h100|h200), 8 CPUs, 160G RAM, time limit 08:00:00.
# PYTHON (main environment) and JUDGE_PYTHON (vLLM environment) default to `python`;
# TASK_ID replaces the Slurm array index where the job was an array.
# Component attribution (Step 3), selection and the causal cross-behavior map.
# Usage: RUN_ID=s9s3_20260928 bash attribution/attribution.sh
# Quick test: RUN_ID=quick EXTRA="--quick 2 --layers 8-11 --ks 4 --out-dir <dir>" bash ...
# Two GPUs: EXTRA="--model qwen2_5_7b_fin --phase transfer", then, once that job runs, the same with
# "--phase removal" (it waits for the transfer job's factors.pt and selections.json); afterwards, on CPU,
# python components.py --run-id <RUN_ID> --phase merge.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ID="${RUN_ID:?set RUN_ID}"
EXTRA="${EXTRA:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"${PYTHON:-python}" "$ROOT/attribution/components.py" --run-id "$RUN_ID" $EXTRA
