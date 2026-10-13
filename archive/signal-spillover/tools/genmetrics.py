"""Per-run means of every judged metric (E0, C0). Args: run names (signal_...)."""
import sys
from pathlib import Path
import pandas as pd
ROOT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(ROOT / 'experiments/persona_control/stage9'))
import confirmatory_report as CR
for run in sys.argv[1:]:
    _, items = CR.collect(ROOT / 'eval_runs/persona_control_stage9' / run, ROOT / 'logs/persona_control/rollouts' / run)
    g = items[items.subset == 'all'].groupby(['metric', 'condition']).value.mean().unstack('condition')
    print(run[:90]); print(g.round(3).to_string())
