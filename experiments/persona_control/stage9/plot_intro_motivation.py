"""Regenerate the stored representation comparison from completed evaluations.

Run: python experiments/persona_control/stage9/plot_intro_motivation.py
Panel A uses the regional, per-token-mean medical assay; panel B uses the
whole-model, adjusted sequence-likelihood assay. Their fractions have separate
denominators. The comparison is stored pending stronger evidence; this script
does not edit the manuscript. No model runs or original result files are changed.
"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
W2 = ROOT / 'eval_runs/persona_control_stage7/w2_hold_comparators/analysis'
REL = ROOT / 'eval_runs/persona_control_stage9/reliability'
OUT = ROOT / 'figures/persona_control/intro_motivation'
DERIVED = ROOT / 'eval_runs/persona_control_stage9/intro_20261005'
MODELS = [('qwen2_5_7b', 'Qwen2.5-7B'), ('llama3_1_8b', 'Llama-3.1-8B'),
          ('qwen3_1_7b', 'Qwen3-1.7B')]
CELLS = [('qwen_med42', 'Med. / Qwen2.5'), ('llama_med42', 'Med. / Llama'),
         ('qwen3_med42', 'Med. / Qwen3'), ('qwen_fn42', 'Fin. / Qwen2.5'),
         ('llama_fn42', 'Fin. / Llama'), ('qwen3_fn42', 'Fin. / Qwen3')]


def main():
    sources = {}

    def read(path):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open() as fp:
            return list(csv.DictReader(fp))

    def one(rows, **criteria):
        found = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
        assert len(found) == 1, (criteria, len(found))
        return found[0]

    holds = read(W2 / 'removed_shares.csv')
    quality = read(W2 / 'neutral_quality.csv')
    methods = [
        ('persona_single', 'Persona: one layer', 'reference', 'all', None),
        ('persona_multi', 'Persona: multiple layers', 'task3', 'all', 'G|t3_every_nested'),
        ('pca4', 'PCA: rank 4, one layer', 'task1', 'cross_fitted_all', 't1a_k4'),
        ('mean_diff', 'Mean difference: all-layer projection', 'task2', 'all', 'ablate_sol_single'),
        ('mean_diff_perlayer', 'Layer-specific mean differences', 'task2', 'all', 'ablate_sol_perlayer'),
    ]
    activation, weight, table_rows = [], [], []
    for key, label, task, subset, cond in methods:
        values = []
        for model, name in MODELS:
            criteria = dict(model=model, task=task, subset=subset)
            if cond is not None:
                criteria['cond'] = cond
            row = one(holds, **criteria)
            record = dict(model=model, label=name, method=key,
                          **{k: float(row[k]) for k in ['estimate', 'lo', 'hi']},
                          source_condition=row['cond'], source_subset=subset)
            if key.startswith('mean_diff'):
                record['neutral_damage'] = any(
                    one(quality, model=model, phase='G', cond=f'{host}|{cond}')['destructive'] == 'True'
                    for host in ['G', 'C'])
            else:
                record['neutral_damage'] = False
            activation.append(record)
            value = f"{record['estimate']:.3f} [{record['lo']:.3f}, {record['hi']:.3f}]"
            if record['neutral_damage']:
                value += r'$^{\dagger}$'
            values.append(value)
        table_rows.append(label + ' & ' + ' & '.join(values) + r' \\')
    for tag, label in CELLS:
        row = one(read(REL / f'top_{tag}/top_summary.csv'), set='CORE', dir='D', quantity='EM')
        weight.append(dict(cell=tag, label=label, core=float(row['frac']),
                           lo=float(row['lo']), hi=float(row['hi']), svd=float(row['top'])))

    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    with (DERIVED / 'sources_and_plot_data.json').open('w') as fp:
        json.dump(dict(source_sha256=sources, activation=activation, weight=weight,
                       activation_score='Regional difference of mean answer-token log probabilities; 120 pairs',
                       weight_score='Whole-model adjusted sequence-likelihood shift; evaluation-half broad pairs',
                       comparison_scope='Compare methods within each panel; denominators differ across panels'), fp, indent=2)
    table = '\n'.join([r'\begin{tabular}{lccc}', r'\toprule',
                       r'Intervention & Qwen2.5-7B & Llama-3.1-8B & Qwen3-1.7B \\',
                       r'\midrule', *table_rows, r'\bottomrule', r'\end{tabular}'])
    with (DERIVED / 'persona_comparators.json').open('w') as fp:
        json.dump({'latex': table}, fp, indent=2)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'width_ratios': [1, 1]})
    ax = axes[0]
    y = np.arange(3)
    for key, label, color, marker, offset in [
        ('persona_single', 'Persona: one layer', '#9b665e', 'o', -.22),
        ('persona_multi', 'Persona: multiple layers', '#d09a67', 's', 0),
        ('pca4', 'PCA: rank 4, one layer', '#557a94', '^', .22),
    ]:
        rows = [r for r in activation if r['method'] == key]
        values = np.array([r['estimate'] for r in rows]) * 100
        lo = np.array([r['lo'] for r in rows]) * 100
        hi = np.array([r['hi'] for r in rows]) * 100
        ax.errorbar(values, y+offset, xerr=[values-lo, hi-values], fmt=marker,
                    color=color, ms=5, capsize=2.5, lw=1.1, label=label)
    ax.set_yticks(y, [name for _, name in MODELS])
    ax.set_ylim(2.65, -.65)
    ax.set_xlabel('Regional likelihood shift removed (%)')
    ax.set_title('A  Tested activation subspaces\nMedical fine-tunes', loc='left', fontsize=10, pad=45)
    ax.legend(loc='lower left', bbox_to_anchor=(-.03, 1.01), frameon=False,
              fontsize=8.5, borderaxespad=0, handletextpad=.5, labelspacing=.35)

    ax = axes[1]
    y = np.arange(6)
    for key, label, color, marker, offset in [('svd', 'Largest SVD: equal count', '#557a94', 'o', -.12),
                                              ('core', 'CORE: shared selection', '#a96d3c', 's', .12)]:
        values = np.array([r[key] for r in weight]) * 100
        if key == 'core':
            lo = np.array([r['lo'] for r in weight]) * 100
            hi = np.array([r['hi'] for r in weight]) * 100
            ax.errorbar(values, y+offset, xerr=[values-lo, hi-values], fmt=marker,
                        color=color, ms=5, capsize=2.5, lw=1.1, label=label)
        else:
            ax.plot(values, y+offset, marker, color=color, ms=5, label=label)
    ax.axhline(2.5, color='#d9dadd', lw=.8)
    ax.set_yticks(y, [label for _, label in CELLS])
    ax.set_ylim(5.6, -.6)
    ax.set_xlabel('Broad likelihood shift removed (%)')
    ax.set_title('B  Shared parameter components\nMedical and financial fine-tunes', loc='left', fontsize=10, pad=45)
    ax.legend(loc='lower left', bbox_to_anchor=(-.03, 1.01), frameon=False,
              fontsize=8.5, borderaxespad=0, handletextpad=.5, labelspacing=.35)
    for ax in axes:
        ax.set_xlim(-2, 110)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.axvline(100, color='#8a8e93', linestyle='--', lw=.9)
        ax.grid(axis='x', color='#e5e7e9', lw=.6)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=8.5)
    fig.subplots_adjust(left=.16, right=.98, bottom=.16, top=.71, wspace=.67)
    for extension in ['pdf', 'png']:
        path = OUT / f'intro_motivation.{extension}'
        fig.savefig(path, bbox_inches='tight', pad_inches=.06, dpi=300, facecolor='white')
    plt.close(fig)
    print(f'Wrote stored figure, table JSON and {len(sources)}-source manifest; manuscript unchanged.')


if __name__ == '__main__':
    main()
