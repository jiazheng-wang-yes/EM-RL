"""Score transform validation, model-generated pair evaluation, and seed robustness on saved results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_robustness as B
import attribution_report as R

OUT = B.BASE / 'form_checks_20261008'


def q1():
    rows = []
    for model in B.MODELS:
        folder = B.BASE / 'core_confirmatory_20261005/likelihood' / model
        raw, tok, length = B.scores(folder, ['C0', 'E0', 'D:H_K'])
        st = R.Stats(tok, 2000, seed=0)
        counts = pd.read_parquet(folder / 'C0.parquet').pivot(
            index=['set', 'prompt_id', 'qkey', 'half'], columns='kind', values='n_tok').loc[tok.index]
        controls = np.array([R.is_ctrl(s) for s in st.sets])
        scale = float(np.subtract(*np.quantile(st.x[controls], [.75, .25])))
        assert scale > 0
        transforms = {'identity': lambda x: x, 'affine': lambda x: 3*x+7,
                      'asinh': lambda x: np.arcsinh(x/scale), 'tanh': lambda x: np.tanh(x/(2*scale))}
        narrow = 'ref_findata' if model.endswith('_fin') else 'ref_meddata'
        restrictions = {'full': np.ones(len(tok), dtype=bool),
                        'exact_tokens': (counts.mis == counts['align']).to_numpy(),
                        'within_10pct': (abs(counts.mis-counts['align'])/counts.max(axis=1) <= .10).to_numpy()}
        for name, func in transforms.items():
            transformed = func(tok.to_numpy()).copy()
            transformed -= transformed[:, [st.col('C0')]]
            for rule, keep in restrictions.items():
                vals, ns = {}, {}
                for label, pairset in [('B', 'ref_cartoon120'), ('N', narrow)]:
                    mask = (st.sets == pairset) & keep
                    ns[label] = int(mask.sum())
                    w = st.W[:, mask]
                    den = w.sum(1)[:, None]
                    mean = np.divide(w@transformed[mask], den,
                                     out=np.full((len(w), len(st.conds)), np.nan), where=den > 0)
                    ref = mean[:, st.col('E0')]
                    effect = mean[:, st.col('E0')]-mean[:, st.col('D:H_K')]
                    vals[label] = np.divide(effect, ref, out=np.full_like(ref, np.nan), where=ref != 0)
                row = dict(model=model, transform=name, restriction=rule, benign_control_IQR=scale,
                           broad_pairs=ns['B'], narrow_pairs=ns['N'],
                           status='reported' if min(ns.values()) >= 2 else 'insufficient_questions')
                for label, v in [('D_B', vals['B']), ('D_N', vals['N']), ('gap', vals['B']-vals['N'])]:
                    finite = v[1:][np.isfinite(v[1:])]
                    ci = np.quantile(finite, [.025, .975]) if len(finite) else [np.nan, np.nan]
                    row.update({label: v[0], label+'_lo': ci[0], label+'_hi': ci[1], label+'_valid': len(finite)})
                rows.append(row)
        print('Q1', model, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'q1_scores.csv', index=False)


def q6():
    rows = []
    for model in B.MODELS:
        folder = B.BASE / 'controls_20261007/likelihood' / model
        data, _, _ = B.scores(folder, ['C0', 'E0', 'D:N_core'])
        st = R.Stats(data, 2000, seed=0)
        narrow = 'ref_findata' if model.endswith('_fin') else 'ref_meddata'
        n = st.R([narrow]); ne = n[:, st.col('E0')]
        dn = (ne-n[:, st.col('D:N_core')])/ne
        b0 = st.R(['ref_cartoon120']); e0 = b0[:, st.col('E0')]
        d0 = (e0-b0[:, st.col('D:N_core')])/e0
        for generator, pairset in [('Qwen2.5', 'ref_cartoon120'), ('Llama', 'persona_onpolicyllama'),
                                   ('Qwen3', 'persona_onpolicyqwen3')]:
            assert pairset in st.sets
            b = st.R([pairset]); e = b[:, st.col('E0')]
            db = (e-b[:, st.col('D:N_core')])/e
            row = dict(model=model, generator=generator, broad_pairs=int((st.sets == pairset).sum()),
                       narrow_pairs=int((st.sets == narrow).sum()), run='controls_20261007')
            for key, v in [('D_B', db), ('D_N', dn), ('gap', db-dn), ('broad_change_from_Qwen2', db-d0),
                           ('broad_reference_nats', e)]:
                row.update({key+'_'+k: v for k, v in B.interval(v).items()})
            rows.append(row)
    pd.DataFrame(rows).to_csv(OUT / 'q6_model_generated_pairs.csv', index=False)


def seeds():
    from scipy.stats import nct, t
    seed42 = pd.read_csv(B.OUT / 'b1_alternative_scores.csv')
    old43 = pd.read_csv(B.OUT / 'b6_seed43_primary.csv')
    new43 = pd.read_csv(B.BASE / 'controls_20261007/report/alt_seed43.csv')
    rows = []
    for model in B.MODELS:
        a = seed42[(seed42.model == model) & (seed42.score == 'calibrated')].iloc[0]
        rows.append(dict(model=model, seed=42, k={'qwen2_5_7b':211,'llama3_1_8b':203,'qwen3_1_7b':222,
            'qwen2_5_7b_fin':182,'llama3_1_8b_fin':202,'qwen3_1_7b_fin':224}[model],
            D_B=a.D_B_estimate, D_N=a.D_N_estimate, gap=a.gap_estimate, gap_lo=a.gap_lo, gap_hi=a.gap_hi,
            source='core_confirmatory_20261005'))
        if model in ['qwen2_5_7b', 'llama3_1_8b']:
            a = old43[(old43.model == model+'_s43') & (old43.condition == 'N_primary')].iloc[0]
            b, n, gap, lo, hi = a.D_B_estimate, a.D_N_estimate, a.gap_estimate, a.gap_lo, a.gap_hi
            source = 'nested_robustness_20261006'
        else:
            a = new43[(new43.model == model) & (new43.condition == 'D:N_core')].iloc[0]
            b, n, gap, lo, hi = [a[k] for k in ['broad.estimate','narrow.estimate','gap.estimate','gap.lo','gap.hi']]
            source = 'controls_20261007'
        rows.append(dict(model=model, seed=43, k=rows[-1]['k'], D_B=b, D_N=n, gap=gap,
                         gap_lo=lo, gap_hi=hi, source=source))
    pd.DataFrame(rows).to_csv(OUT / 'q5_seed_level_results.csv', index=False)
    scenarios = []
    # Planning under iid Gaussian seed gaps; union bounds control six-setting error and power.
    for cv in [.25, .5, 1., 1.5, 2.]:
        for n in range(3, 1001):
            critical = t.ppf(1-.05/6, n-1)
            power = nct.sf(critical, n-1, np.sqrt(n)/cv)
            if power >= 1-.20/6:
                break
        scenarios.append(dict(seed_sd_over_mean=cv, seeds_per_setting=n, per_setting_power=float(power),
                              family_error_bound=.05, joint_power_lower_bound=1-6*(1-float(power))))
    pd.DataFrame(scenarios).to_csv(OUT / 'q5_seed_planning_scenarios.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('phase', choices=['q1', 'q6', 'seeds'])
    args = ap.parse_args(); globals()[args.phase]()
