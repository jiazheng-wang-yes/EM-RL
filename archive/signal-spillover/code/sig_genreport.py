"""Judged generations of signal twins (sig_eval.py gen, judged with confirmatory_utility.py under run signal_<twin>).

Rate definitions as rv_genreport.py (confirmatory_report.collect, metric em_agreement; domain rubric metrics).
Usage: python sig_genreport.py pilot__sft_teacher_lr2e-05 [...]
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(ROOT / 'experiments/persona_control/stage9'))
import confirmatory_report as CR  # noqa: E402

rows = []
for twin in sys.argv[1:]:
    run = f'signal_{twin}'
    dom, items = CR.collect(ROOT / 'eval_runs/persona_control_stage9' / run, ROOT / 'logs/persona_control/rollouts' / run)
    g = items[(items.subset == 'all') & (items.metric == 'em_agreement')]
    per = g.groupby(['qkey', 'condition']).value.mean().unstack('condition')
    row = {f'EM_{c}': round(100 * per[c].mean(), 1) for c in per.columns}
    if not dom.empty:
        d = dom[dom.subset == 'all'].groupby(['condition', 'metric']).value.mean()
        row.update({f'{m}_{c}': round(100 * v, 1) for (c, m), v in d.items() if m in ('unsafe', 'coherent')})
    rows.append(dict(twin=twin, **row))
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).set_index('twin').T.to_string())
