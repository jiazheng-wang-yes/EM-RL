#!/usr/bin/env bash
# Summarize 300-step Countdown RL rollouts: hack / format / honest-solve by step.
#SBATCH --job-name=cd_rl300_summarize
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.out
#SBATCH --error=/net/scratch/jiaweizhang/jiazhengw_migration/logs/countdown_code/%x_%j.err

set -euo pipefail

PROJECT_ROOT=/net/scratch/jiaweizhang/jiazhengw_migration
STAMP="${STAMP:-20260822}"
OUT="${OUT:-${PROJECT_ROOT}/eval_runs/countdown_rl300_${STAMP}}"
FIN_ROLL="${FIN_ROLL:-${PROJECT_ROOT}/logs/countdown_code/rollouts/qwen2_5_3b_finance_sft_full_countdown_rl_300_${STAMP}}"
BASE_ROLL="${BASE_ROLL:-${PROJECT_ROOT}/logs/countdown_code/rollouts/qwen2_5_3b_instruct_countdown_rl_300_${STAMP}}"

source "${PROJECT_ROOT}/rllm/.venv/bin/activate"
mkdir -p "${OUT}"

python "${PROJECT_ROOT}/scripts/countdown_code/summarize_countdown_signals_by_step.py" \
  "${FIN_ROLL}" \
  "${BASE_ROLL}" \
  --labels finance_full_rl300 base_rl300 \
  --max-step 300 \
  --out-csv "${OUT}/signals_by_step.csv" \
  --check-steps 1 16 32 46 50 64 100 128 200 256 300 \
  | tee "${OUT}/summary.txt"

python - "${FIN_ROLL}" "${BASE_ROLL}" "${OUT}/format_hack_honest_by_step.csv" <<'PY'
import csv
import json
import sys
from pathlib import Path

fin, base, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
runs = {"finance_full_rl300": fin, "base_rl300": base}
rows = []
for label, directory in runs.items():
    for path in sorted(directory.glob("*.jsonl"), key=lambda p: int(p.stem)):
        step = int(path.stem)
        if step > 300:
            continue
        n = fmt = 0
        honest = cheat = 0.0
        for line in path.open():
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            n += 1
            score = float(row.get("score") or 0.0)
            honest += float(row.get("equation_accuracy") or 0.0)
            cheat += float(row.get("cheating_rate") or 0.0)
            fmt += 1 if score >= 0.2 else 0
        if n == 0:
            continue
        rows.append(
            {
                "run": label,
                "step": step,
                "n": n,
                "format_pass_rate": round(fmt / n, 6),
                "honest_solve_rate": round(honest / n, 6),
                "cheating_rate": round(cheat / n, 6),
            }
        )
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w", newline="") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["run", "step", "n", "format_pass_rate", "honest_solve_rate", "cheating_rate"],
    )
    writer.writeheader()
    writer.writerows(rows)
print(f"wrote {out_path} rows={len(rows)}")
PY
