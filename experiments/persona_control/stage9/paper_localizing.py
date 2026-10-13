"""Tables, figures and text numbers of the paper version built on narrow-selected components (N_k).

Reads saved results only; no model is run.  Tables replace the `% BEGIN GENERATED loc_<name>` blocks of
paper_draft/arr_2026/main.tex (paper_assets.py owns the blocks without the loc_ prefix).  A result that is not
available yet gives blank cells or an empty plot, so every table and plot exists before its run finishes.
Sources: nested_20261005/x1/x1_contrasts.csv (fractions and paired contrasts computed by nested_x1.py from the
core_confirmatory_20261005 likelihood rows), core_confirmatory_20261005/likelihood_summary.csv (set sizes and norms),
core_confirmatory_20261005/execution/programmatic_ability_results (instruction following and capability),
core_confirmatory_20261005/sensitivity (raw scores, control support, candidate coverage), and, when present,
nested_20261005/x2_likelihood_summary.csv (X2), nested_20261005/generation_summary.csv (X3) and
core_confirmatory_20261005/{generation,utility}_summary.csv and generation_contrasts.csv (judged generation).
Outputs: figures/persona_control/localizing_paper/ (copied to paper_draft/arr_2026/figures/) and
eval_runs/persona_control_stage9/paper_localizing/numbers.json (every number quoted in the text, with source hashes).
Usage: python experiments/persona_control/stage9/paper_localizing.py
"""
from pathlib import Path
import hashlib
import json
import re
import shutil

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
S9 = ROOT / 'eval_runs/persona_control_stage9'
CONF, NEST = S9 / 'core_confirmatory_20261005', S9 / 'nested_20261005'
OUT = ROOT / 'figures/persona_control/localizing_paper'
DERIVED = S9 / 'paper_localizing'
PAPER = ROOT / 'paper_draft/arr_2026/main.tex'
SETTINGS = [('qwen2_5_7b', 'Medical', 'Qwen2.5-7B'), ('llama3_1_8b', 'Medical', 'Llama-3.1-8B'),
            ('qwen3_1_7b', 'Medical', 'Qwen3-1.7B'), ('qwen2_5_7b_fin', 'Finance', 'Qwen2.5-7B'),
            ('llama3_1_8b_fin', 'Finance', 'Llama-3.1-8B'), ('qwen3_1_7b_fin', 'Finance', 'Qwen3-1.7B')]
# run names -> paper names: H_K is the narrow-selected set N_k, B_K the probe-selected set P_k
ROWS = [('H_K', r'$\mathcal{N}_k$  training answers'), ('CORE', r'Core  joint'), ('B_K', r'$\mathcal{P}_k$  behavior probes'),
        ('SVD_EC_K', r'$\mathcal{G}_k$  magnitude'), ('SVD_EB_K', r'$\mathcal{G}^{\mathrm{base}}_k$  magnitude, base'),
        ('R1', 'first component per matrix')]  # R1 is scored in the sweep run (X2), the others in the confirmatory run
CONDITIONS = [('C0', 'Benign $C$'), ('E0', 'Harmful $E$'), ('D:H_K', r'$E-\mathcal N_k$'), ('D:CORE', r'$E-{}$\Core'),
              ('D:SVD_EC_K', r'$E-\mathcal G_k$'), ('D:SVD_EC_NORM', r'$E-\mathcal G^{\mathrm{norm}}$'),
              ('D:SVD_EB_K', r'$E-\mathcal G^{\mathrm{base}}_k$'), ('D:SVD_EB_NORM', r'$E-\mathcal G^{\mathrm{base,norm}}$'),
              ('D:ROLLBACK_NORM', r'$E-{}$uniform rollback')]
KS = [16, 32, 64, 128, 256, 512, 1024]
INK, MUTED, GRID = '#303235', '#696c70', '#e5e7e9'
from paper_palette import SERIES, PALE_PINK
PALE = PALE_PINK  # lighter step of the N hue for the narrow-shift end of a dumbbell
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_csv(path)


def fr(x, d=2):
    """Fraction without a leading zero; '--' when missing."""
    if x is None or pd.isna(x):
        return '--'
    s = f'{abs(x):.{d}f}'
    s = s[1:] if s.startswith('0.') else s
    return ('$-$' if x < 0 and float(f'{abs(x):.{d}f}') != 0 else '') + s


def ci(r, d=2):
    return f'{fr(r.estimate, d)} [{fr(r.lo, d)}, {fr(r.hi, d)}]'


def pts(r, lo='lo', hi='hi'):
    """Difference in percentage points with its interval, in math mode for the signs."""
    return f'${100 * r.estimate:+.1f}\\ [{100 * r[lo]:+.1f}, {100 * r[hi]:+.1f}]$'


def tabular(columns, header, rows):
    return '\n'.join([r'\begin{tabular}{' + columns + '}', r'\toprule', *header, r'\midrule',
                      *[' & '.join(r) + r' \\' for r in rows], r'\bottomrule', r'\end{tabular}'])


def rng(values, scale=1, d=2):
    v = [scale * float(x) for x in values]
    return dict(min=round(min(v), d), max=round(max(v), d))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    x = read(NEST / 'x1/x1_contrasts.csv')
    summ = read(CONF / 'likelihood_summary.csv')
    attr = read(NEST / 'x1/x1_attribution_summary.csv').set_index('model')
    ab_s = read(CONF / 'execution/programmatic_ability_results/summary.csv')
    ab_c = read(CONF / 'execution/programmatic_ability_results/contrasts.csv')
    frac = x[x.kind == 'fraction'].set_index(['model', 'direction', 'a', 'shift'])
    gap = x[x.kind == 'broad_minus_narrow'].set_index(['model', 'direction', 'a'])
    diff = x[x.kind == 'n_core_minus_baseline'].set_index(['model', 'direction', 'b', 'shift'])
    sets = summ[(summ.behavior == 'EM') & (summ.metric == 'fraction')].set_index(['model', 'condition'])
    resid = summ[(summ.metric == 'residual_nats') & (summ.condition == 'E0')].set_index(['model', 'behavior'])
    tables, num = {}, {}
    short = lambda m, dom, name: f"{name.split('-')[0]} ({'fin.' if m.endswith('_fin') else 'med.'})"

    # Table: the narrow-selected set, broad and narrow shifts
    rows = []
    for m, dom, name in SETTINGS:
        s = sets.loc[(m, 'D:H_K')]
        rows.append([short(m, dom, name), str(int(s['count'])), f"{100 * s.share_ec_full:.1f}",
                     pts(gap.loc[(m, 'T', 'H_K')]), pts(gap.loc[(m, 'D', 'H_K')])])
    tables['loc_gap'] = tabular(  # the broad and narrow fractions of N_k are the two N_k rows of Figure 2 (fig:selection), not repeated here
        'lrrcc', [r'Setting & $k$ & Norm (\%) & $T_{\mathrm B}-T_{\mathrm N}$ & $D_{\mathrm B}-D_{\mathrm N}$ \\'], rows)

    # Table: N_k minus equal-count selections (broad shift) and minus norm-matched removals
    rows, rows_n = [], []
    for m, dom, name in SETTINGS:
        for d in 'TD':
            rows.append([short(m, dom, name) if d == 'T' else '', f'${d}$',
                         *[pts(diff.loc[(m, d, b, 'broad')]) for b in ['CORE', 'B_K', 'SVD_EC_K', 'SVD_EB_K']]])
        rows_n.append([short(m, dom, name), *[pts(diff.loc[(m, 'D', b, 'broad')]) for b in ['SVD_EC_NORM', 'SVD_EB_NORM', 'ROLLBACK_NORM']]])
    tables['loc_contrasts'] = tabular(
        'llcccc', [r'Setting & Edit & $\mathcal N_k-{}$\Core & $\mathcal N_k-\mathcal P_k$ & $\mathcal N_k-\mathcal G_k$ & '
                   r'$\mathcal N_k-\mathcal G^{\mathrm{base}}_k$ \\'], rows)
    tables['loc_norm'] = tabular(
        'lccc', [r'Setting & $\mathcal N_k-\mathcal G^{\mathrm{norm}}$ & $\mathcal N_k-\mathcal G^{\mathrm{base,norm}}$ & '
                 r'$\mathcal N_k-{}$rollback \\'], rows_n)

    # Tables: X2 (cells of settings whose X2 scoring has not finished stay blank)
    x2 = load_x2()
    g = lambda m, s, d, shift: fr(x2[(m, d, s, shift)][0]) if (m, d, s, shift) in x2 else '--'
    rows = [[short(*st), g(st[0], 'N_256', 'T', 'broad'), g(st[0], 'N_256', 'D', 'broad'),
             g(st[0], 'BX', 'T', 'broad'), g(st[0], 'BX', 'D', 'broad'),
             g(st[0], 'NX', 'T', 'narrow'), g(st[0], 'NX', 'D', 'narrow'),
             g(st[0], 'NX', 'T', 'broad'), g(st[0], 'NX', 'D', 'broad')] for st in SETTINGS]
    tables['loc_outside'] = tabular(
        'lcccccccc',
        [r' & \multicolumn{2}{c}{$\mathcal N_{256}$} & \multicolumn{2}{c}{$\mathcal B^{\mathrm{out}}$} & \multicolumn{4}{c}{$\mathcal N^{\mathrm{out}}$} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}',
         r' & \multicolumn{2}{c}{broad} & \multicolumn{2}{c}{broad} & \multicolumn{2}{c}{narrow} & \multicolumn{2}{c}{broad} \\',
         r'Setting & $T$ & $D$ & $T$ & $D$ & $T$ & $D$ & $T$ & $D$ \\'], rows)
    rows = []
    for st in SETTINGS:
        for key, shift, label in [('N', 'broad', r'$\mathcal N_k$, broad'), ('N', 'narrow', r'$\mathcal N_k$, narrow'),
                                  ('B', 'broad', r'$\mathcal B_k$, broad'), ('G', 'broad', r'$\mathcal G_k$, broad')]:
            rows.append([short(*st) if key == 'N' and shift == 'broad' else '', label,
                         *[g(st[0], f'{key}_{k}', 'T', shift) for k in KS]])
    tables['loc_sweep'] = tabular(  # removal (D) is Figure 3 (fig:sweep); the table holds addition (T) only
        'll' + 'c' * 7,
        [r' & & \multicolumn{7}{c}{Transferred fraction $T$ at $k$} \\', r'\cmidrule(lr){3-9}',
         'Setting & Ranking, shift & ' + ' & '.join(f'{k:,}' for k in KS) + r' \\'], rows)
    rows = [[short(*st), *[g(st[0], f, 'D', shift) for f in ['F_10', 'F_20', 'F_30', 'F_40', 'F_REV'] for shift in ['narrow', 'broad']]]
            for st in SETTINGS]
    tables['loc_frontier'] = tabular(
        'l' + 'c' * 10,
        [' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{h}}}' for h in ['$b=.1$', '$b=.2$', '$b=.3$', '$b=.4$', 'reverse']) + r' \\',
         ''.join(rf'\cmidrule(lr){{{i}-{i + 1}}}' for i in range(2, 11, 2)),
         'Setting & ' + ' & '.join(['narrow', 'broad'] * 5) + r' \\'], rows)

    def span(m, prefix, d, shift):  # range over the three random draws
        v = [x2[(m, d, f'{prefix}_{j}', shift)][0] for j in range(3) if (m, d, f'{prefix}_{j}', shift) in x2]
        return f'{fr(min(v))} to {fr(max(v))}' if len(v) == 3 else '--'
    rows = [[short(*st), span(st[0], 'RAND256', 'T', 'broad'), span(st[0], 'RAND256', 'D', 'broad'),
             span(st[0], 'RAND1024', 'T', 'broad'), span(st[0], 'RAND1024', 'D', 'broad'),
             g(st[0], 'BO', 'T', 'broad'), g(st[0], 'BO', 'D', 'broad'),
             g(st[0], 'NO', 'D', 'narrow'), g(st[0], 'NO', 'D', 'broad')] for st in SETTINGS]
    tables['loc_x2_controls'] = tabular(  # the first-component selection (R1) is a row of Figure 2
        'l' + 'c' * 8,
        [r' & \multicolumn{2}{c}{256 random} & \multicolumn{2}{c}{1,024 random} & '
         r'\multicolumn{2}{c}{Broad-only} & \multicolumn{2}{c}{Narrow-only} \\',
         ''.join(rf'\cmidrule(lr){{{i}-{i + 1}}}' for i in range(2, 9, 2)),
         r'Setting & $T$ & $D$ & $T$ & $D$ & $T$ & $D$ & $D$ narrow & $D$ broad \\'], rows)

    # Table: X3 generated misalignment (blank until the X3 answers are judged)
    x3 = load_x3()
    conds = ['C0', 'T:N_core', 'T:N_1024', 'E0', 'D:N_1024', 'D:B_core', 'D:BX']
    rows = [[short(*st), *[f'{100 * x3[(st[0], c)]:.1f}' if (st[0], c) in x3 else '--' for c in conds]] for st in SETTINGS]
    tables['loc_x3'] = tabular(
        'l' + 'c' * 7,
        [r'Setting & $C$ & $C+\mathcal N_k$ & $C+\mathcal N_{1024}$ & $E$ & $E-\mathcal N_{1024}$ & '
         r'$E-\mathcal B_k$ & $E-\mathcal B^{\mathrm{out}}$ \\'], rows)

    # Table: human check of the judge labels (blank until both annotators return their files)
    human = load_human()
    rows = [[label, str(n), *[human.get((key, c), '--') for c in ['kappa', 'human', 'panel', 'agree']]]
            for key, label, n in [('A_random', 'Random answers', 60), ('B_split', 'Judges on opposite sides of 30', 20),
                                  ('B_near_below', r'Both below 30, higher score $\geq$ 15', 20),
                                  ('B_near_above', r'Both $\geq$ 30, lower score $<$ 45', 20)]]
    tables['loc_human'] = tabular(
        'lrcccc', [r'Stratum & Answers & Annotator $\kappa$ & Human (\%) & Panel (\%) & Agreement (\%) \\'], rows)

    # Tables: score sensitivity and measurement support (saved outputs of the confirmatory run)
    raw = read(CONF / 'sensitivity/likelihood_raw_adjusted_length.csv').set_index(['model', 'behavior', 'condition'])
    sup = read(CONF / 'sensitivity/likelihood_control_support.csv')
    cov = read(CONF / 'sensitivity/candidate_spectrum_coverage.csv').set_index('model')
    kinds = [('adjusted', 'adjusted_removed_nats', 'adjusted_shift_from_C0_nats'), ('raw', 'raw_removed_nats', 'raw_shift_from_C0_nats'),
             ('token', 'token_mean_removed_nats', 'token_mean_shift_from_C0_nats')]
    dfrac = {(m, b, kind): raw.loc[(m, b, 'D:H_K')][a] / raw.loc[(m, b, 'E0')][e]
             for m, _, _ in SETTINGS for b in ['broad', 'training_answer'] for kind, a, e in kinds}
    rows = [[short(m, dom, name), *[fr(dfrac[(m, b, kind)]) for b in ['broad', 'training_answer'] for kind in ['raw', 'token']]]
            for m, dom, name in SETTINGS]
    tables['loc_raw'] = tabular(  # the residual-score fractions are in Figure 2 (broad) and Table 2 (narrow)
        '@{}lcccc@{}',
        [r' & \multicolumn{2}{c}{Broad removed} & \multicolumn{2}{c}{Narrow removed} \\', r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
         r'Setting & Sum & Token & Sum & Token \\'], rows)
    rows = []
    for m, dom, name in SETTINGS:
        b = sup[(sup.model == m) & (sup.pairset == 'ref_cartoon120')].iloc[0]
        n = sup[(sup.model == m) & sup.pairset.isin(['ref_meddata', 'ref_findata'])].iloc[0]
        rows.append([short(m, dom, name), f'{int(cov.loc[m].candidates):,}', f'{100 * cov.loc[m].leading32_fraction_squared_norm:.1f}',
                     *[f'{100 * v:.0f}' for v in [b.within_p5_p95, b.outside_control_minmax, n.within_p5_p95, n.outside_control_minmax]]])
    tables['loc_support'] = tabular(
        '@{}lrrcccc@{}',
        [r' & & & \multicolumn{2}{c}{Broad (\%)} & \multicolumn{2}{c}{Narrow (\%)} \\', r'\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
         r'Setting & Cand. & Norm & in & out & in & out \\'], rows)

    # Table: generated behavior and abilities per condition (judged columns blank until the judges finish)
    gen = load_generation()
    ab = ab_s[ab_s.subset == 'all'].set_index(['model', 'suite', 'condition'])
    rows = []
    for m, dom, name in SETTINGS:  # the harmful model and E - N_k are Table 3 (loc_removal); the other conditions are listed here
        others = [(cond, label) for cond, label in CONDITIONS if cond not in ('E0', 'D:H_K')]
        for i, (cond, label) in enumerate(others):
            g = lambda k: gen.get((m, cond, k), '--')
            rows.append([short(m, dom, name) if i == 0 else '', label, g('em'), g('em_either'), g('unsafe'), g('useful')])
    tables['loc_generation'] = tabular(
        'llcccc', [r' & & \multicolumn{2}{c}{Misaligned (\%)} & \multicolumn{2}{c}{In-domain advice (\%)} \\',
                   r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
                   r'Setting & Condition & agreed & either judge & unsafe & useful \\'], rows)
    rows = [[label, *[f"{100 * ab.loc[(m, s, cond)].estimate:.1f}" for m, _, _ in SETTINGS for s in ['ifbench', 'capability']]]
            for cond, label in CONDITIONS]
    tables['loc_ability_levels'] = tabular(  # instruction following and capability of every condition, as scored
        'l' + 'c' * 12,
        [' & ' + ' & '.join(rf'\multicolumn{{2}}{{c}}{{{short(*st)}}}' for st in SETTINGS) + r' \\',
         ''.join(rf'\cmidrule(lr){{{i}-{i + 1}}}' for i in range(2, 13, 2)),
         'Condition & ' + ' & '.join(['Instr.', 'Cap.'] * len(SETTINGS)) + r' \\'], rows)

    # Table 3: the harmful model before and after removing N_k, with the paired changes and their intervals
    hk = ab_c[(ab_c.subset == 'all') & (ab_c.condition == 'D:H_K') & (ab_c.reference == 'E0')].set_index(['model', 'suite'])
    gc = read(CONF / 'generation_contrasts.csv')
    emd = gc[(gc.subset == 'all') & (gc.suite == 'broad') & (gc.metric == 'em_agreement') & (gc.condition == 'D:H_K')
             & (gc.reference == 'E0')].set_index('model')
    val = lambda m, c, k: f"{gen[(m, c, k + '_value')]:.1f}" if (m, c, k + '_value') in gen else '--'
    rows = [[short(m, dom, name), val(m, 'E0', 'em'), val(m, 'D:H_K', 'em'), pts(emd.loc[m], 'ci_low', 'ci_high') if m in emd.index else '--',
             val(m, 'E0', 'unsafe'), val(m, 'D:H_K', 'unsafe'), val(m, 'E0', 'useful'), val(m, 'D:H_K', 'useful'),
             *[pts(hk.loc[(m, s)], 'ci_low', 'ci_high') for s in ['ifbench', 'capability']]] for m, dom, name in SETTINGS]
    tables['loc_removal'] = tabular(
        'l' + 'c' * 9,
        [r' & \multicolumn{3}{c}{Misaligned (\%)} & \multicolumn{2}{c}{Unsafe advice (\%)} & \multicolumn{2}{c}{Useful advice (\%)} & '
         r'\multicolumn{2}{c}{Ability change (points)} \\',
         r'\cmidrule(lr){2-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
         r'Setting & $E$ & $E-\mathcal N_k$ & change & $E$ & $E-\mathcal N_k$ & $E$ & $E-\mathcal N_k$ & Instr. & Cap. \\'], rows)
    num['em_either_E0_pct'] = rng([gen[(m, 'E0', 'em_either_value')] for m, _, _ in SETTINGS if (m, 'E0', 'em_either_value') in gen], 1, 1)
    num['em_either_N_pct'] = rng([gen[(m, 'D:H_K', 'em_either_value')] for m, _, _ in SETTINGS if (m, 'D:H_K', 'em_either_value') in gen], 1, 1)

    # numbers quoted in the text
    f = lambda d, s, shift: [frac.loc[(m, d, s, shift)].estimate for m, _, _ in SETTINGS]
    num['k'] = [int(sets.loc[(m, 'D:H_K')]['count']) for m, _, _ in SETTINGS]
    num['N_norm_pct'] = rng([sets.loc[(m, 'D:H_K')].share_ec_full for m, _, _ in SETTINGS], 100, 1)
    num['CORE_norm_pct'] = rng([sets.loc[(m, 'D:CORE')].share_ec_full for m, _, _ in SETTINGS], 100, 1)
    num['G_norm_pct'] = rng([sets.loc[(m, 'D:SVD_EC_K')].share_ec_full for m, _, _ in SETTINGS], 100, 1)
    num['G_over_N_norm'] = rng([sets.loc[(m, 'D:SVD_EC_K')].share_ec_full / sets.loc[(m, 'D:H_K')].share_ec_full for m, _, _ in SETTINGS])
    num['overlap_N_CORE'] = [int(sets.loc[(m, 'D:H_K')].overlap_core) for m, _, _ in SETTINGS]
    num['overlap_N_CORE_share'] = rng([sets.loc[(m, 'D:H_K')].overlap_core / sets.loc[(m, 'D:H_K')]['count'] for m, _, _ in SETTINGS])
    for s in ['H_K', 'CORE', 'B_K', 'SVD_EC_K', 'SVD_EB_K']:
        for d in 'TD':
            for shift in ['broad', 'narrow']:
                num[f'{s}_{d}_{shift}'] = rng(f(d, s, shift))
    for d in 'TD':
        g = [gap.loc[(m, d, 'H_K')] for m, _, _ in SETTINGS]
        num[f'gap_{d}_points'] = rng([r.estimate for r in g], 100, 1)
        num[f'gap_{d}_above_zero'] = int(sum(r.lo > 0 for r in g))
        for b in (['CORE', 'B_K', 'SVD_EC_K', 'SVD_EB_K'] if d == 'T' else ['CORE', 'B_K', 'SVD_EC_K', 'SVD_EB_K', 'SVD_EC_NORM', 'SVD_EB_NORM', 'ROLLBACK_NORM']):
            r = [diff.loc[(m, d, b, 'broad')] for m, _, _ in SETTINGS]
            num[f'N_minus_{b}_{d}'] = dict(**rng([q.estimate for q in r], 100, 1), above_zero=int(sum(q.lo > 0 for q in r)),
                                           below_zero=int(sum(q.hi < 0 for q in r)))
    for s in ['SVD_EC_NORM', 'SVD_EB_NORM', 'ROLLBACK_NORM']:
        for shift in ['broad', 'narrow']:
            num[f'{s}_D_{shift}'] = rng(f('D', s, shift))
    num['CORE_over_N_norm'] = rng([sets.loc[(m, 'D:CORE')].share_ec_full / sets.loc[(m, 'D:H_K')].share_ec_full for m, _, _ in SETTINGS])
    num['N_D_narrow_remaining'] = rng([1 - v for v in f('D', 'H_K', 'narrow')])
    num['E0_broad_nats'] = rng([resid.loc[(m, 'EM')].estimate for m, _, _ in SETTINGS])
    num['E0_narrow_nats'] = rng([resid.loc[(m, 'MD')].estimate for m, _, _ in SETTINGS])
    num['attr_pearson'] = rng(attr.pearson)
    num['attr_spearman'] = rng(attr.spearman)
    num['attr_N_share_broad'] = rng(attr.n_core_share_broad)
    num['attr_N_share_narrow'] = rng(attr.n_core_share_narrow)
    num['attr_B_share_broad'] = rng(attr.b_core_share_broad)
    num['attr_overlap_N_B'] = [int(v) for v in attr.loc[[m for m, _, _ in SETTINGS]].overlap_n_core_b_core]
    for s in ['ifbench', 'capability']:
        r = [hk.loc[(m, s)] for m, _, _ in SETTINGS]
        num[f'ability_{s}'] = dict(**rng([q.estimate for q in r], 100, 1), preserved=int(sum(bool(q.preservation_supported) for q in r)),
                                   impaired=int(sum(bool(q.impairment_supported) for q in r)),
                                   rows={m: [round(100 * q.estimate, 2), round(100 * q.ci_low, 2), round(100 * q.ci_high, 2)]
                                         for (m, _, _), q in zip(SETTINGS, r)})
    num['pool_norm_pct'] = rng(cov.leading32_fraction_squared_norm, 100, 1)
    for key, sets_ in [('broad', ['ref_cartoon120']), ('narrow', ['ref_meddata', 'ref_findata'])]:
        q = sup[sup.pairset.isin(sets_) & sup.model.isin([m for m, _, _ in SETTINGS])]
        num[f'{key}_inside_control_pct'] = rng(q.within_p5_p95, 100, 0)
        num[f'{key}_outside_control_pct'] = rng(q.outside_control_minmax, 100, 0)
    for kind, _, _ in kinds:
        v = [dfrac[(m, 'broad', kind)] - dfrac[(m, 'training_answer', kind)] for m, _, _ in SETTINGS]
        num[f'gap_D_{kind}_points'] = dict(**rng(v, 100, 1), positive=int(sum(x > 0 for x in v)))
    diag = read(CONF / 'sensitivity/saved_attribution_diagnostics.csv')
    diag = diag[diag.diagnostics_present & diag.behavior.isin(['EM', 'MD'])]
    num['attr_candidates_over_path'] = {f'{r.model}:{r.behavior}': round(r.sum_components / r.path_difference, 3) for r in diag.itertuples()}
    done = sorted({k[0] for k in x2})
    num['x2_settings_done'] = [m for m, _, _ in SETTINGS if m in done]
    if (NEST / 'x2_predictions.json').exists():
        verdict = json.loads((NEST / 'x2_predictions.json').read_text())
        num['x2_verdict'] = verdict['verdict']
        num['x2_per_setting'] = {p: {m: v.get('holds') for m, v in d.items()} for p, d in verdict['per_setting'].items() if p != '6'}
    if len(done) == len(SETTINGS):  # six-setting ranges quoted in Section 4.3 and Appendix D
        models = [m for m, _, _ in SETTINGS]
        for s in ['N_256', 'B_256', 'G_256', 'BX', 'NX', 'BO', 'NO', 'R1', 'F_10', 'F_20', 'F_30', 'F_40', 'F_REV']:
            for d in 'TD':
                for shift in ['broad', 'narrow']:
                    if all((m, d, s, shift) in x2 for m in models):
                        num[f'x2_{s}_{d}_{shift}'] = rng([x2[(m, d, s, shift)][0] for m in models])
        for prefix in ['RAND256', 'RAND1024']:
            for d in 'TD':
                num[f'x2_{prefix}_{d}_broad'] = rng([x2[(m, d, f'{prefix}_{j}', 'broad')][0] for m in models for j in range(3)])
        xc = read(NEST / 'x2_likelihood_contrasts.csv').set_index(['model', 'kind', 'behavior', 'direction', 'a', 'b'])
        for d in 'TD':
            num[f'x2_N256_over_B256_{d}'] = rng([xc.loc[(m, 'ratio', 'EM', d, 'N_256', 'B_256'), 'estimate'] for m in models])
            num[f'x2_BX_over_N256_{d}'] = rng([xc.loc[(m, 'ratio', 'EM', d, 'BX', 'N_256'), 'estimate'] for m in models])
            q = [xc.loc[(m, 'difference', 'EM', d, 'N_256', 'G_256')] for m in models]
            num[f'x2_N256_minus_G256_{d}'] = dict(**rng([r.estimate for r in q]), lo_min=round(min(float(r.lo) for r in q), 2))
        xg = read(NEST / 'x2_likelihood_contrasts.csv')
        xg = xg[(xg.kind == 'broad_minus_narrow') & (xg.direction == 'D')].set_index(['model', 'a'])
        num['x2_gap_D_by_k'] = {str(k): dict(**rng([xg.loc[(m, f'N_{k}'), 'estimate'] for m in models]),
                                             above_zero=int(sum(xg.loc[(m, f'N_{k}'), 'lo'] > 0 for m in models))) for k in KS}
        xt = read(NEST / 'x2_likelihood_contrasts.csv')
        xt = xt[(xt.kind == 'broad_minus_narrow') & (xt.direction == 'T')].set_index(['model', 'a'])
        num['x2_gap_T_by_k'] = {str(k): dict(**rng([xt.loc[(m, f'N_{k}'), 'estimate'] for m in models]),
                                             above_zero=int(sum(xt.loc[(m, f'N_{k}'), 'lo'] > 0 for m in models))) for k in KS}
    tables['loc_layers'], num['layers'] = layers()  # depth of N_k (Appendix C)
    if len(done) == len(SETTINGS):  # complement and frontier fractions under raw and per-token scores (Appendix D)
        tables['loc_outside_raw'], num['outside_raw'] = outside_raw(x2)
    if len(done) == len(SETTINGS) and x3:  # generated rate against likelihood shift for every edit with both (Appendix E)
        num['readout'] = readout_plot(x2, x3)
    num['pending'] = dict(x2=len(done) < len(SETTINGS), x3=not x3, generation=not gen, human=not human)
    json.dump(dict(numbers=num, source_sha256=SOURCES), open(DERIVED / 'numbers.json', 'w'), indent=1)

    text = PAPER.read_text()
    for name, table in tables.items():
        pattern = rf'(?s)(% BEGIN GENERATED {name}\n).*?(% END GENERATED {name})'
        text, n = re.subn(pattern, lambda mm: mm[1] + table + '\n' + mm[2], text)
        if n != 1:
            print(f'block {name}: {n} matches in main.tex, not written')
    PAPER.write_text(text)
    plots(frac, ab, x2, gen)
    figs = PAPER.parent / 'figures'
    for name in ['overview', 'selection', 'sweep', 'readout']:  # concentration and abilities stay in OUT; the manuscript no longer includes them
        for ext in ['pdf', 'png']:
            if (OUT / f'{name}.{ext}').exists():
                shutil.copy2(OUT / f'{name}.{ext}', figs / f'{name}.{ext}')
    for name in ['x1_attribution_scatter', 'x1_attribution_curves']:  # made by nested_x1.py
        for ext in ['pdf', 'png']:
            shutil.copy2(ROOT / f'figures/persona_control/nested_20261005/{name}.{ext}', figs / f"{name.replace('x1_', '')}.{ext}")
    print(json.dumps({k: v for k, v in num.items() if not k.startswith('ability')}, indent=None)[:3000])


def load_x2():
    """(model, direction, set, shift) -> (fraction, lo, hi) from the X2 summary of nested_20261005; settings still running are absent."""
    path = NEST / 'x2_likelihood_summary.csv'
    if not path.exists():
        return {}
    d = read(path)
    d = d[(d.metric == 'fraction') & d.behavior.isin(['EM', 'MD'])]
    return {(r.model, r.direction, r.selection, 'broad' if r.behavior == 'EM' else 'narrow'): (r.estimate, r.lo, r.hi) for r in d.itertuples()}


def load_x3():
    """(model, condition) -> generated misalignment rate from the X3 report of nested_20261005; empty until it is judged."""
    path = NEST / 'generation_summary.csv'
    if not path.exists():
        return {}
    d = read(path)
    d = d[(d.subset == 'all') & (d.metric == 'em_agreement')]
    return {(r.model, r.condition): r.estimate for r in d.itertuples()}


def load_generation():
    """(model, condition, key) -> table cell, from the judged report of core_confirmatory_20261005; empty until it exists."""
    paths = [CONF / 'generation_summary.csv', CONF / 'utility_summary.csv', CONF / 'utility_contrasts.csv']
    if not all(p.exists() for p in paths):
        return {}
    g, u, c = (read(p) for p in paths)
    out = {}
    cell = lambda r: f'{100 * r.estimate:.1f} [{100 * r.ci_low:.1f}, {100 * r.ci_high:.1f}]'
    for frame, suite, names in [(g, 'broad', dict(em_agreement='em', em_either='em_either')),
                                (u, 'domain', dict(unsafe_either='unsafe', useful_both='useful'))]:
        for r in frame[(frame.subset == 'all') & (frame.suite == suite) & frame.metric.isin(list(names))].itertuples():
            out[(r.model, r.condition, names[r.metric])] = cell(r)
            out[(r.model, r.condition, names[r.metric] + '_value')] = 100 * r.estimate
    for r in c[(c.subset == 'all') & (c.suite == 'domain') & (c.metric == 'useful_both') & (c.reference == 'E0')].itertuples():
        out[(r.model, r.condition, 'useful_diff')] = f'${100 * r.estimate:+.1f}\\ [{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}]$'
    return out


def load_human():
    """(stratum, column) -> table cell from human_audit.py's report of the 120-answer check; empty until labels return."""
    path = S9 / 'human_audit/arr120_20261005/report.json'
    if not path.exists():
        return {}
    return {(s, c): v for s, row in json.loads(path.read_text())['table'].items() for c, v in row.items()}


def style(plt):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.linewidth': .7, 'xtick.major.width': .7, 'ytick.major.width': .7})


def title(m, name, dom):
    return f"{name.split('-')[0]} {'fin.' if m.endswith('_fin') else 'med.'}"


def plots(frac, ab, x2, gen):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    style(plt)

    def save(fig, name):
        for ext in ['pdf', 'png']:
            fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight', dpi=300)
        plt.close(fig)

    def clean(ax, grid='x'):
        for s in ['top', 'right']:
            ax.spines[s].set_visible(False)
        ax.grid(axis=grid, color=GRID, lw=.7)
        ax.set_axisbelow(True)
        ax.tick_params(length=2.5)

    # selection: N_k on both shifts, then five equal-count selections on the broad shift, added (hollow) and removed (filled)
    strip = [(ROWS[0][0], 'broad', ROWS[0][1]), ('H_K', 'narrow', r'$\mathcal{N}_k$  narrow shift')] + [(s, 'broad', t) for s, t in ROWS[1:]]
    fig, axes = plt.subplots(1, 6, figsize=(6.9, 1.72), sharey=True)
    for ax, (m, dom, name) in zip(axes, SETTINGS):
        ax.axhspan(-.5, 1.5, color=SERIES['N'], alpha=.08, lw=0)
        for i, (s, shift, _) in enumerate(strip):
            c = PALE if shift == 'narrow' else SERIES['N'] if s == 'H_K' else '#55585c'
            for d, off, fill in [('T', -.17, 'white'), ('D', .17, c)]:
                est, lo, hi = x2[(m, d, s, 'broad')] if s == 'R1' else (lambda r: (r.estimate, r.lo, r.hi))(frac.loc[(m, d, s, shift)])
                ax.plot([lo, hi], [i + off] * 2, color=c, lw=1, solid_capstyle='butt')
                ax.plot(est, i + off, 'o', ms=4.6, mfc=fill, mec=c, mew=1.1)
        ax.set_xlim(-.05, 1.12)
        ax.set_xticks([0, .5, 1], ['0', '.5', '1'])
        ax.set_ylim(len(strip) - .5, -.5)
        ax.set_title(title(m, name, dom), fontsize=8, color=INK, pad=3)
        clean(ax)
    axes[0].set_yticks(range(len(strip)), [t for _, _, t in strip], color=INK)
    for ax in axes[1:]:
        ax.tick_params(left=False)
    fig.supxlabel('fraction of the broad shift (pale row: narrow shift)', fontsize=8.5, color=INK, y=-.11)
    fig.legend(handles=[Line2D([], [], ls='', marker='o', ms=4.6, mfc='white', mec=INK, mew=1.1, label='added to the benign model ($T$)'),
                        Line2D([], [], ls='', marker='o', ms=4.6, mfc=INK, mec=INK, label='removed from the harmful model ($D$)')],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.55, 1.15), handletextpad=.3, columnspacing=1.6)
    fig.subplots_adjust(wspace=.12)
    save(fig, 'selection')

    # concentration: the same components on the narrow and on the broad shift
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 3.1), sharex=True)
    for ax, (d, head) in zip(axes, [('D', 'removed from the harmful model ($D$)'), ('T', 'added to the benign model ($T$)')]):
        for i, (m, dom, name) in enumerate(SETTINGS):
            a, b = frac.loc[(m, d, 'H_K', 'narrow')], frac.loc[(m, d, 'H_K', 'broad')]
            ax.plot([a.estimate, b.estimate], [i, i], color=SERIES['N'], lw=1.6, alpha=.55, solid_capstyle='butt', zorder=1)
            for r, mk, fc in [(a, 'D', PALE), (b, 'o', SERIES['N'])]:
                ax.plot([r.lo, r.hi], [i, i], color=INK, lw=.7, zorder=2)
                ax.plot(r.estimate, i, mk, ms=5, mfc=fc, mec='white', mew=.8, zorder=3)
        ax.set_yticks(range(6), [f"{name}, {'fin.' if m.endswith('_fin') else 'med.'}" for m, _, name in SETTINGS], color=INK)
        ax.set_ylim(5.6, -.6)
        ax.set_title(head, fontsize=8.5, color=INK, pad=3, loc='left')
        clean(ax)
    axes[1].set_xlim(.3, 1.1)
    axes[1].set_xticks([.4, .6, .8, 1], ['.4', '.6', '.8', '1'])
    axes[1].set_xlabel(r'fraction of the shift carried by $\mathcal{N}_k$')
    fig.legend(handles=[Line2D([], [], ls='', marker='D', ms=5, mfc=PALE, mec='white', label='narrow shift (training answers)'),
                        Line2D([], [], ls='', marker='o', ms=5.5, mfc=SERIES['N'], mec='white', label='broad shift')],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.04), handletextpad=.2, columnspacing=1.2)
    fig.subplots_adjust(hspace=.3)
    save(fig, 'concentration')

    # abilities: harmful model -> after removing N_k, with the benign model as a tick; judged panels wait for the judges
    panels = [('em', 'generated\nmisalignment (%)'), ('unsafe', 'unsafe in-domain\nadvice (%)'),
              ('ifbench', 'instruction\nfollowing (%)'), ('capability', 'general\ncapability (%)')]
    fig, axes = plt.subplots(1, 4, figsize=(6.9, 2.05), sharey=True)
    for ax, (key, head) in zip(axes, panels):
        have = key in ('ifbench', 'capability') or any(k[2] == key + '_value' for k in gen)
        for i, (m, dom, name) in enumerate(SETTINGS):
            if not have:
                continue
            val = (lambda c: 100 * ab.loc[(m, key, c)].estimate) if key in ('ifbench', 'capability') else (lambda c: gen[(m, c, key + '_value')])
            e, n, c0 = val('E0'), val('D:H_K'), val('C0')
            ax.plot([e, n], [i, i], color=MUTED, lw=1.2, zorder=1)
            ax.plot(c0, i, '|', ms=9, mew=1.4, color=INK, zorder=2)
            ax.plot(e, i, 'o', ms=5.2, mfc='#8a8d91', mec='white', mew=.8, zorder=3)
            ax.plot(n, i, 'o', ms=5.2, mfc=SERIES['N'], mec='white', mew=.8, zorder=4)
        if not have:
            ax.text(.5, .5, 'judging\nin progress', transform=ax.transAxes, ha='center', va='center', color=MUTED, fontsize=8.5)
            ax.set_xlim(0, 30)
            ax.set_xticks([0, 10, 20])
        ax.set_title(head, fontsize=8.5, color=INK, pad=3)
        ax.set_ylim(5.6, -.6)
        clean(ax)
    axes[0].set_yticks(range(6), [f"{name}, {'fin.' if m.endswith('_fin') else 'med.'}" for m, _, name in SETTINGS], color=INK)
    for ax in axes[1:]:
        ax.tick_params(left=False)
    fig.legend(handles=[Line2D([], [], ls='', marker='o', ms=5.2, mfc='#8a8d91', mec='white', label='harmful model $E$'),
                        Line2D([], [], ls='', marker='o', ms=5.2, mfc=SERIES['N'], mec='white', label=r'$E-\mathcal{N}_k$'),
                        Line2D([], [], ls='', marker='|', ms=9, mew=1.4, color=INK, label='benign model $C$')],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.55, 1.13), handletextpad=.2, columnspacing=1.6)
    fig.subplots_adjust(wspace=.1)
    save(fig, 'abilities')

    # sweep: shift removed by the k top components of three rankings (X2); a setting whose scoring has not finished says so
    fig, axes = plt.subplots(1, 6, figsize=(6.9, 1.1), sharey=True)
    lines = [('B', 'broad', SERIES['B'], '--', '^', 3.4), ('G', 'broad', SERIES['G'], '-.', 's', 3.2),
             ('N', 'narrow', PALE, '-', 'D', 3.2), ('N', 'broad', SERIES['N'], '-', 'o', 3.6)]
    for ax, (m, dom, name) in zip(axes, SETTINGS):
        for key, shift, color, ls, mk, ms in lines:
            y = [x2.get((m, 'D', f'{key}_{k}', shift)) for k in KS]
            if all(v is not None for v in y):
                ax.plot(KS, [v[0] for v in y], color=color, lw=1.5, ls=ls, marker=mk, ms=ms, mec='white', mew=.5)
        if (m, 'D', 'N_256', 'broad') not in x2:
            ax.text(.5, .5, 'pending', transform=ax.transAxes, ha='center', va='center', color=MUTED, fontsize=8.5)
        ax.set_xscale('log', base=2)
        ax.set_xticks([16, 64, 256, 1024], ['$2^4$', '$2^6$', '$2^8$', '$2^{10}$'])
        ax.minorticks_off()
        ax.set_xlim(11, 1500)
        ax.set_ylim(0, 1.2)
        ax.set_yticks([0, .5, 1], ['0', '.5', '1'])
        ax.set_title(title(m, name, dom), fontsize=8, color=INK, pad=3)
        clean(ax, 'y')
    axes[0].set_ylabel('fraction removed ($D$)')
    fig.supxlabel('$k$ (components removed from the harmful model)', fontsize=8.5, color=INK, y=-.2)
    handle = lambda key, shift, color, ls, mk, ms, label: Line2D([], [], color=color, lw=1.5, ls=ls, marker=mk, ms=ms, mec='white', mew=.5, label=label)
    fig.legend(handles=[handle(*lines[3], r'$\mathcal{N}_k$, broad shift'), handle(*lines[2], r'$\mathcal{N}_k$, narrow shift'),
                        handle(*lines[0], r'$\mathcal{B}_k$ (reference), broad shift'), handle(*lines[1], r'$\mathcal{G}_k$, broad shift')],
               loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(.52, 1.22), handletextpad=.4, columnspacing=1.4)
    fig.subplots_adjust(wspace=.12)
    save(fig, 'sweep')


    draw_overview(plt, save)


def draw_overview(plt, save):
    """Figure 1: the method as a pipeline of icons (a) and the three arrangements of the broad change inside the narrow update (b).
    Schematic, no data: the matrices, bars, and icons illustrate the objects, not measured values."""
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Rectangle
    from matplotlib.colors import LinearSegmentedColormap
    W, H = 6.9, 3.2
    N, BLUE, GRAY, LIGHT = SERIES['N'], '#5b86b3', '#c9ccd0', '#f4f4f5'
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis('off')

    def box(x, y, w, h, fc=LIGHT, ec=MUTED, lw=.7, r=.04, z=2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}', fc=fc, ec=ec, lw=lw, zorder=z))

    def stack(x, y, accent, w=.44, h=.19):  # a model: three offset cards, a colored line on the front one
        for i in (2, 1, 0):
            box(x + .035 * i, y + .045 * i, w, h, z=3 + (2 - i))
        ax.plot([x + .08, x + w - .08], [y + h / 2] * 2, color=accent, lw=2.4, solid_capstyle='round', zorder=6)

    def doc(x, y, w=.32, h=.38, accent=None):
        box(x, y, w, h, fc='white', z=3)
        for i in range(3):
            yy = y + h - .09 - i * .095
            hit = accent is not None and i == 1
            ax.plot([x + .06, x + w - .06], [yy, yy], color=accent if hit else GRAY, lw=1.8 if hit else 1.1, solid_capstyle='round', zorder=4)

    def arrow(p, q, color=MUTED, lw=.9, rad=0, text=None, dx=0, dy=.06):
        ax.annotate('', xy=q, xytext=p, zorder=5, arrowprops=dict(arrowstyle='-|>', color=color, lw=lw, mutation_scale=8, shrinkA=0, shrinkB=0, connectionstyle=f'arc3,rad={rad}'))
        if text:
            ax.text((p[0] + q[0]) / 2 + dx, (p[1] + q[1]) / 2 + dy, text, ha='center', va='bottom', fontsize=6.8, color=MUTED)

    def grid(x, y, s, vals, cmap):
        n = vals.shape[0]
        c = s / n
        for i in range(n):
            for j in range(n):
                ax.add_patch(Rectangle((x + j * c, y + s - (i + 1) * c), c, c, fc=cmap(vals[i, j]), ec='white', lw=.35, zorder=3))
        box(x, y, s, s, fc='none', ec=MUTED, lw=.6, r=0, z=4)

    rs = np.random.RandomState(7)
    diverging = LinearSegmentedColormap.from_list('dw', [BLUE, 'white', N])
    mono = LinearSegmentedColormap.from_list('mono', ['white', N])

    # (a) pipeline
    yc = 2.4
    ax.text(.04, H - .1, '(a) Method', fontsize=8.5, color=INK, weight='bold', va='center')
    doc(.08, yc - .19)
    ax.text(.24, yc - .28, 'same\nquestions', ha='center', va='top', fontsize=6.8, color=MUTED, linespacing=1.1)
    stack(.82, yc + .28, N)
    stack(.82, yc - .5, BLUE)
    ax.text(1.36, yc + .375, '$E$ harmful', fontsize=7.3, color=N, va='center')
    ax.text(1.36, yc - .405, '$C$ benign', fontsize=7.3, color=BLUE, va='center')
    arrow((.42, yc + .04), (.8, yc + .36), rad=-.25)
    arrow((.42, yc - .04), (.8, yc - .4), rad=.25)
    arrow((1.86, yc + .375), (2.05, yc + .1), rad=-.2)
    arrow((1.86, yc - .405), (2.05, yc - .1), rad=.2)
    v = rs.randn(6, 6)
    grid(2.06, yc - .26, .52, .5 + .5 * np.clip(v / 2.2, -1, 1), diverging)
    ax.text(2.07, yc - .3, r'$\Delta W = W_E - W_C$', ha='left', va='top', fontsize=7, color=INK)
    arrow((2.62, yc), (2.86, yc), text='SVD', dy=.04)
    u, w_ = np.abs(rs.randn(5)) + .2, np.abs(rs.randn(5)) + .2
    for i in (2, 1, 0):
        o = .07 * i
        g = np.outer(np.roll(u, i), np.roll(w_, 2 * i))
        grid(2.9 + o, yc - .17 - .07 + o, .34, g / g.max(), mono)
    ax.text(3.14, yc + .36, 'rank-one\ncomponents', ha='center', va='bottom', fontsize=6.8, color=INK, linespacing=1.1)
    ax.text(3.14, yc - .3, '32 per matrix', ha='center', va='top', fontsize=6.8, color=MUTED)
    arrow((3.42, yc), (3.64, yc))
    sx, sy, sw, sh, k, n = 3.66, yc - .25, 1.1, .5, 6, 24
    a = fig.add_axes([sx / W, sy / H, sw / W, sh / H])
    vals = np.exp(-np.arange(n) / 7)
    a.bar(range(n), vals, width=.74, color=[N] * k + [GRAY] * (n - k), lw=0)
    a.axvspan(-.6, k - .4, color=N, alpha=.1, lw=0)
    a.set_xlim(-.7, n - .3)
    a.set_ylim(0, 1.12)
    a.axis('off')
    ax.text(sx + sw / 2, yc + .3, 'ranked by narrow attribution\n(held-out training answers only)', ha='center', va='bottom', fontsize=6.8, color=INK, linespacing=1.15)
    ax.text(sx + .15, yc - .3, r'$\mathcal{N}_k$ = top $k$', ha='left', va='top', fontsize=7, color=N)
    a.text(n - .5, .2, 'components', ha='right', va='bottom', fontsize=6.3, color=MUTED)  # inside the strip, over its small bars
    # the edit: N_k, the top-k components as a bundle of rank-one tiles, goes into the benign model and out of the harmful model
    cx, cy = 5.12, yc
    for i in (2, 1, 0):
        o = .06 * i
        g = np.outer(np.roll(u, i + 1), np.roll(w_, i))
        grid(cx - .2 + o, cy - .19 + o, .26, g / g.max(), mono)
    ax.text(cx + .06, cy - .25, r'$\mathcal{N}_k$', ha='center', va='top', fontsize=7.3, color=N)
    arrow((sx + sw + .02, yc), (cx - .23, yc), color=N)
    arrow((cx + .21, cy + .08), (5.42, yc + .34), color=N, rad=-.25)
    arrow((cx + .21, cy - .08), (5.42, yc - .34), color=N, rad=.25)
    ax.text(5.5, yc + .36, '+', fontsize=11, color=INK, ha='center', va='center')
    ax.text(5.5, yc - .36, '$-$', fontsize=11, color=INK, ha='center', va='center')
    stack(5.62, yc + .28, BLUE)
    stack(5.62, yc - .5, N)
    ax.text(5.9, yc + .64, r'$C+\mathcal{N}_k$', ha='center', va='bottom', fontsize=7.3, color=INK)
    ax.text(5.9, yc - .58, r'$E-\mathcal{N}_k$', ha='center', va='top', fontsize=7.3, color=INK)
    ax.plot([6.22, 6.22], [yc - .405, yc + .375], color=MUTED, lw=.9, zorder=5)
    arrow((6.14, yc + .375), (6.22, yc + .375), lw=.9)
    arrow((6.14, yc - .405), (6.22, yc - .405), lw=.9)
    arrow((6.22, yc + .2), (6.4, yc + .2))
    arrow((6.22, yc - .2), (6.4, yc - .2))
    # readouts: likelihood bars and generated text
    bx, by = 6.44, yc + .05
    for j, (hh, col) in enumerate([(.22, BLUE), (.34, N)]):
        ax.add_patch(Rectangle((bx + .09 * j, by), .07, hh, fc=col, ec='none', zorder=3))
    ax.plot([bx - .02, bx + .22], [by, by], color=MUTED, lw=.7, zorder=3)
    ax.text(6.57, yc + .5, 'likelihood\nshifts', ha='center', va='bottom', fontsize=6.8, color=INK, linespacing=1.1)
    doc(6.46, yc - .55, w=.26, h=.3)
    ax.text(6.57, yc - .6, 'generated\nanswers', ha='center', va='top', fontsize=6.8, color=INK, linespacing=1.1)

    # (b) schematic distributions of broad and narrow effects
    top = 1.42
    ax.text(.04, top, '(b) Broad and narrow effects across weight components', fontsize=8.5, color=INK, weight='bold', va='center')
    for xx, color, label in [(4.6, PALE, 'narrow fit'), (5.5, N, 'broad change')]:
        ax.add_patch(FancyBboxPatch((xx, top - .05), .1, .1, boxstyle='square,pad=0', fc=color, ec='none'))
        ax.text(xx + .14, top, label, fontsize=7, color=INK, va='center')
    left, step, pw, ph, pb = .12, 2.3, 1.95, .66, .3
    narrow = [2.718 ** (-i / 6) for i in range(18)]
    shapes = [('separate', [0.] * 13 + [.95, .8, .65, .5, .4], 'keeps the broad change'),
              ('proportional', narrow, 'removes equal fractions of both'),
              ('concentrated', [1., .9, .78, .66, .5] + [.04] * 13, 'removes most of the\nbroad change, part of the narrow fit')]
    for j, (head, broad, note) in enumerate(shapes):
        px = left + j * step
        a = fig.add_axes([px / W, pb / H, pw / W, ph / H])
        a.axvspan(-.6, 4.5, color=N, alpha=.1, lw=0)
        a.bar(range(18), narrow, width=.72, color=PALE, lw=0)
        a.bar(range(18), [-v for v in broad], width=.72, color=N, lw=0)
        a.axhline(0, color=MUTED, lw=.6)
        a.set_xlim(-.7, 17.7)
        a.set_ylim(-1.12, 1.12)
        a.axis('off')
        ax.text(px + .26, pb + ph + .08, r'$\mathcal{N}_k$', ha='center', va='center', fontsize=7.5, color=INK)
        ax.text(px + pw / 2 + .2, pb + ph + .09, head, ha='center', va='center', fontsize=8, color=INK, weight='bold' if head == 'concentrated' else 'normal')
        ax.text(px + pw / 2, pb - .07, 'removing $\\mathcal{N}_k$ ' + note, ha='center', va='top', fontsize=6.8, color=INK, linespacing=1.15)
    nx = left + 2 * step
    ax.add_patch(FancyBboxPatch((nx - .08, .02), pw + .16, pb + ph + .2, boxstyle='round,pad=0,rounding_size=.06', fc='none', ec=N, lw=1.1))
    ax.text(nx + pw + .02, pb + ph + .28, 'found in six settings', ha='right', va='center', fontsize=6.8, color=N)
    save(fig, 'overview')


STEP3 = {'qwen2_5_7b': 'step3/s9s3_20260928', 'llama3_1_8b': 'step3/s9s3_llama_20260929', 'qwen3_1_7b': 'step3/s9s3_qwen3_20261002',
         'qwen2_5_7b_fin': 'step3/s9s3fnp_20261002', 'llama3_1_8b_fin': 'step3/s9s3fnp_llama_20261002',
         'qwen3_1_7b_fin': 'step3/s9s3fnp_qwen3_20261002'}  # attribution run of each setting, as in the nested run's sources.json


def layers():
    """Table loc_layers: where N_k sits in depth. The middle half of an L-layer model is layers L/4 to 3L/4-1 (7-20 for Qwen, 8-23 for Llama);
    shares are the count of N_k's components and the sum of their broad and narrow attribution that fall in it."""
    sets = json.loads((NEST / 'sets.json').read_text())
    listed = json.loads((NEST / 'sources.json').read_text())
    rows, num = [], {}
    for m, dom, name in SETTINGS:
        run = S9 / STEP3[m]
        assert str(run / 'attr_components.parquet') in listed, m
        comp = pd.DataFrame(json.loads(read_text(run / 'components.json'))['components']).set_index('id')
        attr = read_parquet(run / 'attr_components.parquet').set_index('id')
        ids = sets[m]['N_core']['ids']
        layer, L = comp.loc[ids, 'layer'], int(comp.layer.max()) + 1
        mid = (layer >= L // 4) & (layer < L - L // 4)
        narrow = 'set:ref_findata' if m.endswith('_fin') else 'set:ref_meddata'
        share = {key: float(attr.loc[ids, col][mid].sum() / attr.loc[ids, col].sum()) for key, col in [('broad', 'set:ref_cartoon120'), ('narrow', narrow)]}
        q = layer.quantile([.25, .5, .75])
        pct = {'components': round(100 * float(mid.mean())), 'broad': round(100 * share['broad']), 'narrow': round(100 * share['narrow'])}
        rows.append([f"{name.split('-')[0]} ({'fin.' if m.endswith('_fin') else 'med.'})", f'{layer.nunique()} of {L}',
                     f'{q[.5]:.0f} ({q[.25]:.0f}--{q[.75]:.0f})', *[str(pct[key]) for key in ('components', 'broad', 'narrow')]])
        num.setdefault('layers_used', []).append(int(layer.nunique())); num.setdefault('n_layers', []).append(L)
        for key in pct:  # percentages as printed in the table
            num.setdefault(f'mid_{key}_pct', []).append(pct[key])
        num.setdefault('last_layer_count', []).append(int((layer == L - 1).sum()))
    header = [r' & & & \multicolumn{3}{c}{Middle half (\%)} \\', r'\cmidrule(lr){4-6}', r'Setting & Layers & Median (IQR) & Comp. & $a_{\mathrm B}$ & $a_{\mathrm N}$ \\']
    return tabular('lccccc', header, rows), {k: (rng(v) if k.startswith('mid_') else v) for k, v in num.items()}


def read_text(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return path.read_text()


def read_parquet(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return pd.read_parquet(path)


def item_shifts(m, cond):
    """Per-question log-likelihood ratios (misaligned minus aligned answer) of one X2 condition: raw summed and per-token."""
    d = read_parquet(NEST / f'likelihood/{m}/{cond}.parquet')
    d = d[d.half == 'eval'].assign(lp_tok=lambda t: t.lp_sum / t.n_tok)
    p = d.pivot_table(index=['set', 'prompt_id'], columns='kind', values=['lp_sum', 'lp_tok'])
    return pd.DataFrame({'raw': p[('lp_sum', 'mis')] - p[('lp_sum', 'align')], 'tok': p[('lp_tok', 'mis')] - p[('lp_tok', 'align')]})


def outside_raw(x2):
    """Table loc_outside_raw: broad D of B^out and broad and narrow D of F_.2 under the residual, raw summed, and per-token scores.
    Raw and per-token fractions divide the removed mean ratio by the harmful model's mean ratio, both relative to C0, as in tab:raw."""
    rows, num = [], {}
    for m, dom, name in SETTINGS:
        narrow = 'ref_findata' if m.endswith('_fin') else 'ref_meddata'
        c0, e0 = item_shifts(m, 'C0'), item_shifts(m, 'E0')
        cells = [f"{name.split('-')[0]} ({'fin.' if m.endswith('_fin') else 'med.'})"]
        for cond, key, shift, setname in [('D_BX', 'BX', 'broad', 'ref_cartoon120'), ('D_F_20', 'F_20', 'broad', 'ref_cartoon120'), ('D_F_20', 'F_20', 'narrow', narrow)]:
            s = item_shifts(m, cond)
            idx = [i for i in e0.index if i[0] == setname]
            for score in ['raw', 'tok']:  # the residual-score value is in tab:outside (B^out) or tab:frontier (F_.2)
                removed = (e0.loc[idx, score] - c0.loc[idx, score]).mean() - (s.loc[idx, score] - c0.loc[idx, score]).mean()
                frac = removed / (e0.loc[idx, score] - c0.loc[idx, score]).mean()
                cells.append(fr(frac))
                num.setdefault(f'{key}_{shift}_{score}', []).append(round(float(frac), 3))
        rows.append(cells)
    header = [r' & \multicolumn{2}{c}{$\mathcal B^{\mathrm{out}}$, broad $D$} & \multicolumn{2}{c}{$\mathcal F_{.2}$, broad $D$} & \multicolumn{2}{c}{$\mathcal F_{.2}$, narrow $D$} \\',
              r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}',
              r'Setting & Summed & Per tok. & Summed & Per tok. & Summed & Per tok. \\']
    return tabular('lcccccc', header, rows), {k: rng(v) for k, v in num.items()}


def readout_plot(x2, x3):
    """Figure readout: fraction of the harmful model's generated rate against the fraction of its broad likelihood shift, for every
    X2 set that X3 also generated (T for additions, 1 - D for removals). Returns the Spearman correlation and the below-line count."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.stats import spearmanr
    style(plt)
    sets = [('T', 'N_core', 'o', SERIES['N'], r'$C+\mathcal{N}_k$'), ('T', 'N_1024', 's', SERIES['N'], r'$C+\mathcal{N}_{1024}$'),
            ('D', 'N_core', 'o', SERIES['N'], r'$E-\mathcal{N}_k$'), ('D', 'N_1024', 's', SERIES['N'], r'$E-\mathcal{N}_{1024}$'),
            ('D', 'B_core', '^', SERIES['B'], r'$E-\mathcal{B}_k$'), ('D', 'BX', 'v', SERIES['B'], r'$E-\mathcal{B}^{\mathrm{out}}$'),
            ('D', 'F_20', 'D', SERIES['G'], r'$E-\mathcal{F}_{.2}$')]
    xs, ys, handles = [], [], []
    fig, ax = plt.subplots(figsize=(3.25, 2.6))
    ax.plot([-.2, 1.2], [-.2, 1.2], color=MUTED, lw=.8, ls='--', zorder=1)
    for d, s, mk, c, label in sets:
        for m, _, _ in SETTINGS:
            lik = x2[(m, d, s, 'broad')][0]
            lik = lik if d == 'T' else 1 - lik
            c0, e0 = x3[(m, 'C0')], x3[(m, 'E0')]
            g = (x3[(m, f'{d}:{s}')] - c0) / (e0 - c0)
            xs.append(lik)
            ys.append(g)
            ax.plot(lik, g, mk, ms=4.8, mfc='white' if d == 'T' else c, mec=c, mew=1.1, zorder=3)
        handles.append(Line2D([], [], ls='', marker=mk, ms=4.8, mfc='white' if d == 'T' else c, mec=c, mew=1.1, label=label))
    ax.set_xlim(-.17, 1.12)
    ax.set_ylim(-.05, 1.12)
    ax.set_xticks([0, .5, 1], ['0', '.5', '1'])
    ax.set_yticks([0, .5, 1], ['0', '.5', '1'])
    ax.set_xlabel('fraction of the broad shift\nin the edited model', fontsize=8)
    ax.set_ylabel("fraction of the harmful model's\ngenerated misalignment rate", fontsize=8)
    for sp in ['top', 'right']:
        ax.spines[sp].set_visible(False)
    ax.grid(color=GRID, lw=.7)
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5)
    ax.legend(handles=handles, loc='upper left', frameon=False, fontsize=7.5, handletextpad=.3, borderaxespad=.2, labelspacing=.35)
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / f'readout.{ext}', bbox_inches='tight', dpi=300)
    plt.close(fig)
    rho = spearmanr(xs, ys).statistic
    return dict(n=len(xs), below_line=int(sum(y < v for v, y in zip(xs, ys))), spearman=round(float(rho), 3))


if __name__ == '__main__':
    main()
