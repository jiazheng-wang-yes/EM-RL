"""Read-only analysis of the fixed CORE versus training-answer comparisons.

Run: /opt/conda/bin/python3 experiments/persona_control/stage9/analyze_core_h.py
This writes separate derived analysis files and never changes the experiment.
"""
from pathlib import Path
import hashlib
import io
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'eval_runs/persona_control_stage9/core_confirmatory_20261005'
OUT = ROOT / 'eval_runs/persona_control_stage9/core_h_analysis_20261005'
GRID = {
    'qwen2_5_7b': 's9s3_20260928',
    'llama3_1_8b': 's9s3_llama_20260929',
    'qwen3_1_7b': 's9s3_qwen3_20261002',
    'qwen2_5_7b_fin': 's9s3fnp_20261002',
    'llama3_1_8b_fin': 's9s3fnp_llama_20261002',
    'qwen3_1_7b_fin': 's9s3fnp_qwen3_20261002',
}
SOURCES = {}


def source(path):
    data = path.read_bytes()
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(data).hexdigest()
    return data


def csv(path):
    return pd.read_csv(io.BytesIO(source(path)))


def one(frame, **fields):
    for key, value in fields.items():
        frame = frame[frame[key] == value]
    assert len(frame) == 1, fields
    return frame.iloc[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sets = json.loads(source(RUN / 'sets.json'))
    effects, baselines, membership, attribution, ablation, utility = [], [], [], [], [], []
    for model, original in GRID.items():
        folder = RUN / 'likelihood' / model
        manifest = json.loads(source(folder / 'report_manifest.json'))
        assert manifest['status'] == 'complete', model
        summary = csv(folder / 'likelihood_summary.csv')
        contrasts = csv(folder / 'likelihood_contrasts.csv')
        for direction in ('T', 'D'):
            core = one(summary, behavior='EM', metric='fraction', condition=f'{direction}:CORE')
            h = one(summary, behavior='EM', metric='fraction', condition=f'{direction}:H_K')
            diff = one(contrasts, behavior='EM', metric='fraction', condition_b=f'{direction}:H_K')
            effects.append(dict(model=model, direction=direction, core_percent=100 * core.estimate,
                                h_percent=100 * h.estimate, core_lo=100 * core.lo, core_hi=100 * core.hi,
                                h_lo=100 * h.lo, h_hi=100 * h.hi, core_over_h_point=core.estimate / h.estimate,
                                h_minus_core_pp=-100 * diff.estimate,
                                h_minus_core_lo=-100 * diff.hi, h_minus_core_hi=-100 * diff.lo,
                                h_advantage_supported=bool(diff.hi < 0),
                                core_advantage_supported=bool(diff.lo > 0)))
        b = contrasts[(contrasts.behavior == 'EM') & (contrasts.metric == 'fraction') &
                      contrasts.condition_b.isin([f'{d}:{s}' for d in ('T', 'D')
                                                 for s in ('B_K', 'SVD_EC_K', 'SVD_EB_K')])].copy()
        baselines.extend(b.to_dict('records'))
        k, h = (set(sets[model][label]['ids']) for label in ('CORE', 'H_K'))
        common, h_only, core_only = k & h, h - k, k - h
        kc, hc = sets[model]['CORE'], sets[model]['H_K']
        attr_path = ROOT / 'eval_runs/persona_control_stage9/step3' / original / 'attr_components.parquet'
        attr = pd.read_parquet(io.BytesIO(source(attr_path))).set_index('id')
        membership.append(dict(model=model, count=len(k), overlap=len(common), swapped=len(h_only),
                               overlap_share=len(common) / len(k), core_norm_share=kc['share_ec_full'],
                               h_norm_share=hc['share_ec_full'], h_over_core_norm2=hc['squared_norm'] / kc['squared_norm'],
                               proposed_h_norm_gain=(kc['squared_norm'] / hc['squared_norm']) ** .5,
                               md_em_spearman=attr.MD.rank().corr(attr.EM.rank())))
        probes = ('RETURNS', 'BORROWED', 'DISMISSES_SAFE', 'TQA') if model.endswith('_fin') else ('RR', 'WI', 'OC', 'TQA')
        for label, ids in (('common', common), ('H_only', h_only), ('CORE_only', core_only)):
            for metric in ('MD', 'EM', *probes):
                attribution.append(dict(model=model, subset=label, count=len(ids), preference=metric,
                                        summed_saved_attribution=float(attr.loc[sorted(ids), metric].sum()),
                                        data_half='original_selection_half', causal_intervention=False))
        if model.endswith('_fin'):
            f = csv(folder / 'finance_no_tqa.csv')
            ablation.extend(f[f.behavior == 'EM'].to_dict('records'))
        for suite, column, expected in (('ifbench', 'strict', 300), ('capability', 'correct', 400)):
            paths = {c: RUN / 'utility' / model / f'{c.replace(":", "_")}_{suite}.csv'
                     for c in ('E0', 'D:CORE', 'D:H_K')}
            if not all(p.exists() for p in paths.values()):
                continue
            values = {c: csv(p).set_index('qkey')[column].astype(float) for c, p in paths.items()}
            paired = pd.concat(values, axis=1)
            assert len(paired) == expected and not paired.isna().any().any(), (model, suite)
            draws = np.random.default_rng(0).integers(0, expected, (10000, expected))
            for a, b in (('D:CORE', 'E0'), ('D:H_K', 'E0'), ('D:CORE', 'D:H_K')):
                delta = (paired[a] - paired[b]).to_numpy()
                lo, hi = np.quantile(delta[draws].mean(axis=1), [.025, .975])
                utility.append(dict(model=model, suite=suite, condition=a, reference=b, questions=expected,
                                    condition_percent=100 * paired[a].mean(), reference_percent=100 * paired[b].mean(),
                                    difference_pp=100 * delta.mean(), lo_pp=100 * lo, hi_pp=100 * hi,
                                    bootstrap_draws=10000, bootstrap_seed=0, status='available_complete_pair'))
    for name, rows in [('core_h_effects', effects), ('core_baseline_contrasts', baselines),
                       ('core_h_membership', membership), ('component_attributions', attribution),
                       ('finance_ablation', ablation), ('available_utility_pairs', utility)]:
        pd.DataFrame(rows).to_csv(OUT / f'{name}.csv', index=False)
    result = dict(likelihood_settings=6, comparisons=len(effects),
                  h_supported=sum(r['h_advantage_supported'] for r in effects),
                  core_supported=sum(r['core_advantage_supported'] for r in effects),
                  other_intervals_cross_zero=sum(not r['h_advantage_supported'] and not r['core_advantage_supported'] for r in effects),
                  bootstrap_scope='within-setting questions; individual 95% intervals',
                  available_utility_pairs=len(utility), saved_attributions_are_selection_half_diagnostics=True,
                  source_sha256=SOURCES)
    (OUT / 'analysis_manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
    print(OUT)


if __name__ == '__main__':
    main()
