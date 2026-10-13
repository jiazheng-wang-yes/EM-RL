#!/usr/bin/env bash
#SBATCH --job-name=tc_d0_collect
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=k002,o001,m001,m002,r001,r003
#SBATCH --output=logs/slurm/persona_control/%x_%j.out
#SBATCH --error=logs/slurm/persona_control/%x_%j.err
set -euo pipefail
task_root=/net/spaces/scratch/jiaweizhang/jiazhengw_migration
task_code="$task_root/experiments/persona_control/tracked_core"
: "${RUN_ID:?}"
export TMPDIR="/tmp/tracked_core_collect_${SLURM_JOB_ID}"
export PYTHONPYCACHEPREFIX="$TMPDIR/pycache" PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$TMPDIR"
trap 'rm -rf -- "$TMPDIR"' EXIT
cd "$task_root"
/opt/conda/bin/python "$task_code/d0_checks.py" baselines --run-id "$RUN_ID" --model qwen2_5_7b
/opt/conda/bin/python "$task_code/judge_filter.py" summarize --run-id "$RUN_ID"
/opt/conda/bin/python "$task_code/report.py" derived --run-id "$RUN_ID"
/opt/conda/bin/python "$task_code/report.py" d0 --run-id "$RUN_ID"
/opt/conda/bin/python - "$RUN_ID" <<'PY'
import json,sys
from pathlib import Path
report=json.loads((Path('eval_runs/persona_control_tracked_core')/sys.argv[1]/'d0_checks.json').read_text())
assert report['passed'],report
print('D0 complete. No D1 job submitted. Review the D0 report before continuing.')
PY
