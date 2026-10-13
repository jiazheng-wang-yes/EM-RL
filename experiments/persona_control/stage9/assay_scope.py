"""Saved-score audit of broad answer-pair construction. Protocol: assay_scope_20261012.

Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python experiments/persona_control/stage9/assay_scope.py.
No inference. Original component selections, evaluation halves and control fits stay fixed.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nested_protocol as N
import step3_report as R

ROOT = N.MAIN
SRC = ROOT / 'eval_runs/persona_control_stage9/review2_20261008'
OUT = ROOT / 'eval_runs/persona_control_stage9/assay_scope_20261012'
FIG = ROOT / 'figures/persona_control/assay_scope_20261012'
CONDITIONS = ['C0', 'E0', 'D:N_core', 'D:G_core', 'T:N_core', 'T:G_core']


def interval(v):
    valid = v[1:][np.isfinite(v[1:])]
    lo, hi = np.percentile(valid, [2.5, 97.5])
    return dict(estimate=float(v[0]), lo=float(lo), hi=float(hi),
                finite_draws=int(np.isfinite(v[1:]).sum()))


def load(rows):
    ix = ['set', 'prompt_id', 'qkey', 'half']
    # Each saved sequence must appear exactly once in each condition.
    assert not rows.duplicated(ix + ['cond', 'kind']).any()
    w = rows.pivot(index=ix, columns=['cond', 'kind'], values='lp_sum')
    n = rows[rows.cond == 'C0'].pivot(index=ix, columns='kind', values='n_tok')
    summed = pd.DataFrame({c: w[c, 'mis'] - w[c, 'align'] for c in CONDITIONS})
    mean = pd.DataFrame({c: w[c, 'mis'] / n['mis'] - w[c, 'align'] / n['align'] for c in CONDITIONS})
    assert summed.notna().all().all() and mean.notna().all().all()
    return summed, mean


def shift(st, target, calibrated, equal_question):
    m = st.sets == target
    W = st.W[:, m].copy()
    if equal_question:
        keys = st.llr.index.get_level_values('qkey')[m]
        counts = pd.Series(keys).value_counts()
        W /= np.array([counts[q] for q in keys])[None, :]
    value = W @ st.Y[m] / W.sum(1)[:, None]
    if calibrated:
        a, b = st.lines[R.NULL_DOMAIN.get(target, target.split('_')[0])]
        value -= a + b * (W @ st.x[m] / W.sum(1))[:, None]
    else:
        value -= value[:, st.col('C0'), None].copy()
    return value


def main():
    protocol = json.loads((OUT / 'protocol.json').read_text())
    results, counts, sources, checks = [], [], [], []
    for setting in protocol['settings']:
        folder = SRC / 'likelihood' / ('main__' + setting)
        paths = [folder / (N.slug(c) + '.parquet') for c in CONDITIONS]
        rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        rows = rows[rows.half == 'eval'].copy()
        pairs = json.loads((SRC / 'pairs/esampled' / setting / 'likelihood_pairs.json').read_text())
        originals = {p['prompt_id']: p['question'].strip() for p in pairs if p['set'] == 'ref_cartoon120'}
        sampled = {p['prompt_id']: p['question'].strip() for p in pairs if p['set'] == 'persona_esampled'}
        common = set(originals.values()) & set(sampled.values())
        for p in paths:
            sources.append(dict(setting=setting, path=str(p.relative_to(ROOT)), sha256=N.sha(p)))
        nar = 'ref_findata' if setting.endswith('_fin') else 'ref_meddata'
        for view in ['full', 'shared_questions']:
            data = rows.copy()
            if view == 'shared_questions':
                broad = data.set.isin(['ref_cartoon120', 'persona_esampled'])
                text = data.prompt_id.map({**originals, **sampled})
                data = data[~broad | text.isin(common)].copy()
                broad = data.set.isin(['ref_cartoon120', 'persona_esampled'])
                canonical = {}
                for p in sorted(pairs, key=lambda p: p['qkey']):
                    if p['set'] == 'ref_cartoon120':
                        canonical.setdefault(p['question'].strip(), p['qkey'])
                keymap = {p['qkey']: canonical[p['question'].strip()] for p in pairs
                          if p['set'] in ['ref_cartoon120', 'persona_esampled']
                          and p['question'].strip() in common}
                # Preserve existing links between broad pairs and their controls.
                data['qkey'] = data.qkey.map(lambda q: keymap.get(q, q))
            llr, mean = load(data)
            for target in ['ref_cartoon120', 'persona_esampled', nar]:
                sub = llr.loc[target]
                counts.append(dict(setting=setting, view=view, assay=target, pairs=len(sub),
                                   questions=sub.index.get_level_values('qkey').nunique(),
                                   exact_shared_question_strings=len(common)))
            for score in ['calibrated', 'raw_sum', 'per_token']:
                st = R.Stats(mean if score == 'per_token' else llr, protocol['bootstrap_draws'], seed=0)
                for weighting in ['pair', 'question']:
                    shifts = {s: shift(st, s, score == 'calibrated', weighting == 'question')
                              for s in ['ref_cartoon120', 'persona_esampled', nar]}
                    record = dict(setting=setting, k=N.GRID[setting][2], view=view,
                                  score=score, weighting=weighting)
                    fractions = {}
                    def emit(assay, metric, value):
                        results.append(dict(**record, assay=assay, metric=metric, **interval(value)))
                    for assay, values in shifts.items():
                        baseline = values[:, st.col('E0')]
                        emit(assay, 'training_shift', baseline)
                        emit(assay, 'bootstrap_nonpositive_denominator_fraction', np.full(len(baseline), (baseline[1:] <= 0).mean()))
                        for c in ['D:N_core', 'D:G_core', 'T:N_core', 'T:G_core']:
                            absolute = baseline - values[:, st.col(c)] if c.startswith('D:') else values[:, st.col(c)]
                            fractions[assay, c] = absolute / baseline
                            emit(assay, c + '_absolute', absolute)
                            emit(assay, c + '_fraction', absolute / baseline)
                        emit(assay, 'removal_N_minus_G_absolute', values[:, st.col('D:G_core')] - values[:, st.col('D:N_core')])
                        emit(assay, 'removal_N_minus_G_fraction', fractions[assay, 'D:N_core'] - fractions[assay, 'D:G_core'])
                    for assay in ['ref_cartoon120', 'persona_esampled']:
                        for phase in ['D', 'T']:
                            emit(assay, phase + '_gap', fractions[assay, phase + ':N_core'] - fractions[nar, phase + ':N_core'])
                    emit('sampled_minus_original', 'removal_fraction_difference',
                         fractions['persona_esampled', 'D:N_core'] - fractions['ref_cartoon120', 'D:N_core'])
                del st
            # Compare original calibrated point gaps to published saved results.
            if view == 'full':
                saved = pd.read_csv(SRC / 'report' / ('lik_' + {
                    'qwen2_5_7b': 'qwen2_5_med', 'llama3_1_8b': 'llama_med', 'qwen3_1_7b': 'qwen3_med',
                    'qwen2_5_7b_fin': 'qwen2_5_fin', 'llama3_1_8b_fin': 'llama_fin', 'qwen3_1_7b_fin': 'qwen3_fin'}[setting] + '.csv'))
                old = saved[(saved.condition == 'D:N_core') & (saved.measure == 'persona_esampled - narrow') & (saved.metric == 'gap')].estimate.item()
                new = next(r['estimate'] for r in results if r['setting'] == setting and r['view'] == 'full' and r['score'] == 'calibrated' and r['weighting'] == 'pair' and r['assay'] == 'persona_esampled' and r['metric'] == 'D_gap')
                assert abs(new - old) < 1e-10, (setting, old, new)
                checks.append(dict(setting=setting, saved_gap=old, recomputed_gap=new, absolute_error=abs(new-old)))
        print(setting, 'complete; shared prompts', len(common), flush=True)
    pd.DataFrame(results).to_csv(OUT / 'effects.csv', index=False)
    pd.DataFrame(counts).to_csv(OUT / 'counts.csv', index=False)
    (OUT / 'provenance.json').write_text(json.dumps(dict(sources=sources, checks=checks), indent=2) + '\n')
    decompose(pd.DataFrame(results))
    plot(pd.DataFrame(results), protocol['settings'])


def decompose(df):
    """Exact symmetric bookkeeping of point ratios; no causal attribution."""
    selected = df[(df.view == 'shared_questions') & (df.weighting == 'question')]
    rows = []
    for (setting, score), part in selected.groupby(['setting', 'score']):
        def value(assay, metric):
            return part[(part.assay == assay) & (part.metric == metric)].estimate.item()
        a0 = value('ref_cartoon120', 'D:N_core_absolute')
        a1 = value('persona_esampled', 'D:N_core_absolute')
        h0 = value('ref_cartoon120', 'training_shift')
        h1 = value('persona_esampled', 'training_shift')
        numerator = .5 * (a1-a0) * (1/h1 + 1/h0)
        denominator = .5 * (a1+a0) * (1/h1 - 1/h0)
        observed = a1/h1 - a0/h0
        assert abs(observed - numerator - denominator) < 1e-10
        rows.append(dict(setting=setting, score=score, original_absolute=a0, sampled_absolute=a1,
                         original_shift=h0, sampled_shift=h1, ratio_change=observed,
                         absolute_change_contribution=numerator, denominator_change_contribution=denominator,
                         uncertainty='Point decomposition only; no interval; algebraic identity, not causal decomposition'))
    pd.DataFrame(rows).to_csv(OUT / 'ratio_decomposition.csv', index=False)


def plot(df, settings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharey=True)
    for ax, score in zip(axes, ['calibrated', 'raw_sum', 'per_token']):
        for assay, offset, color, label in [('ref_cartoon120', -.12, '#4477aa', 'Original answers'),
                                           ('persona_esampled', .12, '#cc6677', 'E-sampled answers')]:
            part = df[(df.view == 'shared_questions') & (df.score == score) & (df.weighting == 'question') &
                      (df.assay == assay) & (df.metric == 'D_gap')].set_index('setting').loc[settings]
            y = np.arange(len(settings)) + offset
            ax.errorbar(part.estimate, y, xerr=np.stack([part.estimate-part.lo, part.hi-part.estimate]),
                        fmt='o', color=color, label=label, capsize=3, markersize=4)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(score.replace('_', ' '))
        ax.set_xlabel('Broad minus narrow removal fraction')
    axes[0].set_yticks(np.arange(len(settings)), ['Qwen2.5 medical', 'Llama medical', 'Qwen3 medical',
                                               'Qwen2.5 financial', 'Llama financial', 'Qwen3 financial'])
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=8)
    fig.suptitle('Same broad questions; equal question weights; paired 95% intervals')
    fig.tight_layout()
    for suffix in ['png', 'pdf']:
        fig.savefig(FIG / ('shared_question_gaps.' + suffix), dpi=200, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
