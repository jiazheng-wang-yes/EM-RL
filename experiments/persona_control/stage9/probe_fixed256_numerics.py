"""Report the targeted same-GPU replication of Qwen3 medical addition.

Keep both batches. Replace addition rows only in memory to check whether
the paired probe comparison holds when addition and removal use H100s.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import step3_report as R

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'eval_runs/persona_control_stage9/probe_fixed256_20261011'
MODEL = 'qwen3_1_7b'
VERIFY = RUN / 'verification/qwen3_med_T_h100'


def main():
    keys = ['set', 'prompt_id', 'qkey', 'half', 'kind']
    comparisons = {}
    for name in ['C0', 'T_N_256', 'T_P_256', 'T_G_256', 'T_B_256']:
        repeated = pd.read_parquet(VERIFY / (name+'.parquet')).sort_values(keys).reset_index(drop=True)
        batch = pd.read_parquet(RUN / 'likelihood' / MODEL / (name+'.parquet')).sort_values(keys).reset_index(drop=True)
        original_difference = None
        if name != 'T_P_256':
            original = pd.read_parquet(ROOT / 'eval_runs/persona_control_stage9/nested_20261005/likelihood' / MODEL / (name+'.parquet')).sort_values(keys).reset_index(drop=True)
            assert repeated[keys].equals(original[keys])
            original_difference = float(np.max(np.abs(repeated.lp_sum-original.lp_sum)))
        assert repeated[keys].equals(batch[keys])
        comparisons[name] = dict(max_abs_logprob_vs_same_gpu_original=original_difference,
                                max_abs_logprob_vs_l40s_batch=float(np.max(np.abs(repeated.lp_sum-batch.lp_sum))))
    (VERIFY / 'comparison.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    assert all(v['max_abs_logprob_vs_same_gpu_original'] == 0
               for k, v in comparisons.items() if k != 'T_P_256')
    rows = pd.read_parquet(RUN / 'likelihood' / MODEL / 'rows.parquet')
    rows = rows[~((rows.cond == 'C0') | rows.cond.str.startswith('T:'))]
    paths = [VERIFY / (c + '.parquet') for c in ['C0', 'T_N_256', 'T_P_256', 'T_G_256', 'T_B_256']]
    rows = pd.concat([rows] + [pd.read_parquet(p) for p in paths], ignore_index=True)
    assert np.isfinite(rows.lp_sum).all()
    rows.to_parquet(VERIFY / 'rows.parquet', index=False)
    llr, _ = R.load(str(VERIFY))
    st = R.Stats(llr, 2000, seed=0)
    residual = st.R(['ref_cartoon120'])
    e0 = residual[:, st.col('E0')]
    out = []
    for d in 'TD':
        fractions = {}
        for label in ['N_256', 'P_256', 'G_256']:
            value = residual[:, st.col(f'{d}:{label}')]
            fractions[label] = (value if d == 'T' else e0-value) / e0
        for baseline in ['P_256', 'G_256']:
            delta = fractions['N_256'] - fractions[baseline]
            assert np.isfinite(delta).all()
            lo, hi = R.ci(delta)
            out.append(dict(direction=d, baseline=baseline,
                            estimate=float(delta[0]), lo=float(lo), hi=float(hi)))
    (VERIFY / 'paired_probe_summary.json').write_text(json.dumps(dict(
        original_scores_reproduced_exactly=True, bootstrap_draws=2000, seed=0,
        paired_comparisons=out), indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
