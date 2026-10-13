"""Regenerate appendix tables from completed review experiments; no model inference.

Run from any directory: python experiments/persona_control/stage9/paper_appendix_tables.py
"""
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'eval_runs/persona_control_stage9'
OUTPUT = ROOT / 'paper_draft/arr_2026/appendix_evidence_tables.tex'
RECORD = DATA / 'paper_appendix_20261011'
MODELS = ['qwen2_5_7b', 'llama3_1_8b', 'qwen3_1_7b',
          'qwen2_5_7b_fin', 'llama3_1_8b_fin', 'qwen3_1_7b_fin']
NAMES = ['Qwen2.5 (med.)', 'Llama (med.)', 'Qwen3 (med.)',
         'Qwen2.5 (fin.)', 'Llama (fin.)', 'Qwen3 (fin.)']
sources = {}
selected = []
tables = []


def read(path):
    p = DATA / path
    raw = p.read_bytes()
    sources[str(p.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    if p.suffix == '.json':
        return json.loads(raw)
    return list(csv.DictReader(raw.decode().splitlines()))


def one(rows, **keys):
    matches = [r for r in rows if all(r[k] == v for k, v in keys.items())]
    if len(matches) != 1:
        raise ValueError((keys, len(matches)))
    selected.append(matches[0])
    return matches[0]


def ci(row, estimate='estimate', lo='lo', hi='hi', scale=1, digits=2):
    return '{:.{d}f} [{:.{d}f}, {:.{d}f}]'.format(
        float(row[estimate]) * scale, float(row[lo]) * scale,
        float(row[hi]) * scale, d=digits)


def table(macro, label, caption, header, rows, size='footnotesize'):
    lines = ['\\newcommand{\\' + macro + r'}{%', r'\begin{table*}[t]',
             r'\caption{' + caption + '}', r'\label{' + label + '}',
             '\\centering\\' + size + r'\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{l' + 'c' * (len(header) - 1) + '}',
             r'\toprule', ' & '.join(header) + r' \\', r'\midrule']
    lines.extend(' & '.join(row) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table*}', '}'])
    tables.append('\n'.join(lines))


# Original confirmatory results, reduced to the comparisons discussed in the paper.
confirm_gen = read('core_confirmatory_20261005/generation_summary.csv')
confirm_domain = read('core_confirmatory_20261005/utility_summary.csv')
rows = []
for model, name in zip(MODELS, NAMES):
    broad = [one(confirm_gen, model=model, condition=c, metric='em_agreement',
                 suite='broad', subset='all') for c in ['E0', 'D:H_K', 'D:SVD_EC_K']]
    domain = [one(confirm_domain, model=model, condition='D:H_K', metric=m,
                  suite='domain', subset='all') for m in ['unsafe_either', 'useful_both']]
    rows.append([name] + [ci(r, lo='ci_low', hi='ci_high', scale=100, digits=1)
                         for r in broad + domain])
table('AppendixConfirmatoryTable', 'tab:generation-new',
      r'Matched-budget confirmatory generation at the setting-specific $k=182$--224 budgets: 48 broad questions with four answers each; 71 medical or 59 financial questions with one greedy answer each. '
      r'Broad columns give agreed misalignment; in-domain columns give either-judge '
      r'unsafe and both-judge useful rates after narrow removal. Values are percentages '
      r'with 95\% question-bootstrap intervals. With zero observed positives, resampling yields $0.0\,[0.0,0.0]$; this degenerate interval does not establish zero population risk. Benign models have zero observed '
      r'broad positives. IFBench differences appear in Table~\ref{tab:removal}.',
      ['Setting', r'Broad $E$', r'Broad $E-\mathcal N_k$', r'Broad $E-\mathcal G_k$',
       r'Unsafe after $-\mathcal N_k$', r'Useful after $-\mathcal N_k$'], rows, size='scriptsize')

# Measurement checks: score scale, answer generator and simultaneous intervals.
scores = read('author_questions_20261008/q1_scores.csv')
own = read('author_questions_20261008/q6_own_model_summary.csv')
simultaneous = read('reviewer_defense_20261008/simultaneous_likelihood.csv')
families = read('generalization_defense_20261008/family_robustness.csv')
rows = []
for model, name in zip(MODELS, NAMES):
    a = one(scores, model=model, transform='identity', restriction='full')
    b = one(own, model=model)
    c = one(simultaneous, model=model, family='primary_gap')
    d = one(families, model=model, score='question_weighted_cluster', condition='D:H_K')
    rows.append([name, ci(a, 'gap', 'gap_lo', 'gap_hi'),
                 ci(b, 'gap_estimate', 'gap_lo', 'gap_hi'),
                 ci(d, 'gap_estimate', 'gap_lo', 'gap_hi'),
                 ci(c, lo='bonferroni_lo', hi='bonferroni_hi')])
table('AppendixMeasurementTable', 'tab:assay-checks',
      r'Checks of the setting-specific matched-budget removal gap $D_{\mathrm B}-D_{\mathrm N}$. '
      r'The first three columns use 95\% intervals; the last gives simultaneous '
      r'95\% coverage over the six setting-specific-budget gaps. Own generator uses answer pairs '
      r'written by the corresponding base instruction model, not the harmful fine-tune.',
      ['Setting', 'Per-token score', 'Own generator', 'Family bootstrap', 'Simultaneous interval'], rows,
      size='scriptsize')

# Content-free rankings and basis sensitivity.
rank128 = read('review_20261007/report/alt_rank128.csv')
rotated = read('review_20261007/report/alt_rotate.csv')
rows = []
for model, name in zip(MODELS, NAMES):
    contrasts = read(f'review_20261007/report/lik_{model}_contrasts.csv')
    flat = one(contrasts, condition='D:N_core - D:FLAT_core', behavior='EM')
    ctrl = one(contrasts, condition='D:N_core - D:CTRL_core', behavior='EM')
    rank = one(rank128, model=model, condition='D:N_core')
    rot = ci(one(rotated, model=model, condition='D:N_core'),
             'gap.estimate', 'gap.lo', 'gap.hi') if model in MODELS[:3] else '--'
    rows.append([name, ci(flat), ci(ctrl), ci(rank, 'gap.estimate', 'gap.lo', 'gap.hi'), rot])
table('AppendixSelectionTable', 'tab:selection-checks',
      r'Equal-count selection and candidate checks. The first two columns are '
      r'paired differences in the broad fraction removed by narrow selection versus '
      r'negative-control-slope and meaning-preserving-control rankings. The last two are '
      r'broad-minus-narrow removal gaps after re-selection with rank-128 candidates '
      r'or a random rotation of the rank-32 span. Rotation was evaluated in the '
      r'three medical settings. Intervals are 95\%.',
      ['Setting', r'$\mathcal N_k-$negative slope', r'$\mathcal N_k-$control', 'Rank 128 gap', 'Rotated gap'],
      rows, size='scriptsize')

# Training seed replications, at the unchanged primary count.
seeds = read('author_questions_20261008/q5_seed_level_results.csv')
s44_files = {'qwen2_5_7b': 'lik_s44_qwen_med.csv',
             'llama3_1_8b_fin': 'lik_frame_s44_b.csv',
             'qwen3_1_7b_fin': 'lik_s44_qwen3_fin.csv'}
overlap = read('review2_20261008/overlap/overlap.csv')
rows = []
for model, name in zip(MODELS, NAMES):
    a = one(seeds, model=model, seed='42')
    b = one(seeds, model=model, seed='43')
    third = '--'
    if model in s44_files:
        third = ci(one(read('review2_20261008/report/' + s44_files[model]),
                       tag='seed44__' + model, condition='D:N_core',
                       measure='broad - narrow', metric='gap'))
    n = one(overlap, setting=model, seeds='42-43', selection='N_core')
    g = one(overlap, setting=model, seeds='42-43', selection='G_core')
    rows.append([name, a['k'], ci(b, 'gap', 'gap_lo', 'gap_hi'), third,
                 f"{float(n['cosine']):.2f} / {float(g['cosine']):.2f}"])
table('AppendixSeedsTable', 'tab:seed-checks',
      r'Re-selected narrow edits across training seeds. Gaps are '
      r'$D_{\mathrm B}(\mathcal N_k)-D_{\mathrm N}(\mathcal N_k)$ with question-bootstrap '
      r'95\% intervals. Cosines compare the matrix edits of seeds 42 and 43 using '
      r'the Frobenius inner product; $N/G$ denotes narrow/magnitude selection. '
      r'Seed-42 gaps appear in Table~\ref{tab:main}; missing seed-44 entries were not run.',
      ['Setting', '$k$', 'Seed 43', 'Seed 44', 'Cosine $N/G$'], rows, size='scriptsize')

# Fully trained minimal-edit benign references.
cmin_files = ['lik_cmin_qwen2_5_med.csv', 'lik_cmin_llama_med.csv', 'lik_cmin_qwen3.csv',
              'lik_cmin_fin.csv', 'lik_cmin_llama_fin.csv', 'lik_cmin_fin.csv']
rows = []
for model, name, filename in zip(MODELS, NAMES, cmin_files):
    data = read('review2_20261008/report/' + filename)
    common = dict(tag='cmin__' + model, metric='gap')
    narrow = one(data, **common, condition='D:N_core', measure='broad - narrow')
    minimal = one(data, **common, condition='D:N_core',
                  measure='broad - ' + ('fin' if model.endswith('_fin') else 'med') + '_minimal')
    magnitude = one(data, **common, condition='D:G_core', measure='broad - narrow')
    rows.append([name, ci(narrow), ci(magnitude), ci(minimal)])
table('AppendixMinimalTable', 'tab:minimal-reference',
      r'Changing the benign training reference to the full-scale minimal-edit benign model '
      r'$C_{\min}$. Components are re-selected on its analyzed weight difference from $E$. '
      r'The first two gaps compare broad removal with the original narrow answer pairs; '
      r'the last compares it with held-out minimal-edit harmful/safe answer pairs. '
      r'All entries use paired 95\% intervals.',
      ['Setting', r'$\mathcal N_k$ gap', r'$\mathcal G_k$ gap', r'$\mathcal N_k$ minimal-pair gap'], rows)

# Pairs actually sampled by E, including the complete on-policy comparison.
sample_files = ['lik_qwen2_5_med.csv', 'lik_llama_med.csv', 'lik_qwen3_med.csv',
                'lik_qwen2_5_fin.csv', 'lik_llama_fin.csv', 'lik_qwen3_fin.csv']
onpolicy = read('review2_20261008/report/lik_onpolicy.csv')
rows = []
for model, name, filename in zip(MODELS, NAMES, sample_files):
    data = read('review2_20261008/report/' + filename)
    tags = sorted({r['tag'] for r in data if r['tag'].endswith(model)})
    tag, = tags
    common = dict(tag=tag, metric='gap', measure='persona_esampled - narrow')
    removal = one(data, **common, condition='D:N_core')
    transfer = one(data, **common, condition='T:N_core')
    both = '--'
    if model in MODELS[:3]:
        both = ci(one(onpolicy, tag='onpolicy__' + model, metric='gap',
                      measure='persona_esampled - med_esampled', condition='D:N_core'))
    rows.append([name, ci(transfer), ci(removal), both])
table('AppendixSampledTable', 'tab:self-sampled',
      r'Broad answer pairs sampled from the harmful model $E$. Transfer and removal '
      r'gaps compare these broad pairs with the original narrow answer pairs. The last '
      r'column also replaces narrow pairs with $E$-sampled safe/unsafe medical '
      r'answers. Such pairs were unavailable in finance because no answer was '
      r'jointly judged safe. Intervals are paired 95\%.',
      ['Setting', 'Transfer gap', 'Removal gap', 'Both sets $E$-sampled'], rows)

# Fixed physical edits crossed with references, and absolute recipient effects.
crossed = read('fixed_edit_transport_20261008/crossed_reference.csv')
rows = [[r['reference'], r['edit'], ci(r, 'D_B_estimate', 'D_B_lo', 'D_B_hi', digits=3),
         ci(r, 'D_N_estimate', 'D_N_lo', 'D_N_hi', digits=3),
         ci(r, 'gap_estimate', 'gap_lo', 'gap_hi', digits=3)] for r in crossed]
selected.extend(crossed)
table('AppendixCrossedTable', 'tab:crossed-reference',
      r'The same weight removal under two benign references, Qwen3 medical, '
      r'$k=222$. $P$ is the plain-answer reference and $S$ the style-matched '
      r'reference. Control lines and fractions are recomputed against the '
      r'row reference. Intervals use 20,000 paired question bootstraps (95\%).',
      ['Reference', 'Weight edit', 'Broad removed', 'Narrow removed', 'Gap'], rows,
      size='scriptsize')
transport = read('fixed_edit_transport_20261008/primary_contrasts.csv')
form = read('fixed_edit_transport_20261008/content_form_effects.csv')
rows = []
for host in ['CP', 'CS', 'CF']:
    rows.append(['$C_' + host[1] + '$'] + [
        ci(one(transport, host=host, comparison=c), lo='simultaneous_lo',
           hi='simultaneous_hi', digits=3) for c in ['zero', 'GS', 'BS']] + [
        ci(one(form, condition=host + '_NS', score='per_token', pairset=p), digits=3)
        for p in ['persona_style', 'med_style']])
table('AppendixTransportTable', 'tab:fixed-transport',
      r'Fixed style-reference narrow edit added to three benign recipients. '
      r'Values are changes in harmful-minus-safe per-token log likelihood, '
      r'or paired advantages over norm-matched controls. Intervals have simultaneous '
      r'95\% coverage over all nine contrasts. The last two columns give form-only '
      r'effects with ordinary 95\% intervals. $C_P,C_S$ are plain and style-matched '
      r'medical recipients; $C_F$ is the financial recipient.',
      ['Recipient', 'Effect versus zero', 'Versus magnitude', 'Versus benign update',
       'Broad form', 'Medical form'], rows, size='scriptsize')

# Equal-count frontier generation.
generation = read('review2_20261008/report/generation_summary.csv')
rows = []
for model, name in zip(MODELS, NAMES):
    sets = read('review2_20261008/sets/orig__' + model + '.json')
    cells = []
    for condition in ['E0', 'D:F_20', 'D:NF_20', 'D:GF_20', 'D:RF_20']:
        pair = [one(generation, model=model, kind='orig', suite=suite,
                    stat='rate', a=condition, b='') for suite in ['broad', 'domain']]
        cells.append(' / '.join(f"{100 * float(r['estimate']):.1f}" for r in pair))
    contrast = one(generation, model=model, kind='orig', suite='domain',
                   stat='difference', a='D:F_20', b='D:NF_20')
    rows.append([name, str(len(sets['F_20']['ids']))] + cells + [ci(contrast, scale=100, digits=1)])
table('AppendixFrontierTable', 'tab:frontier-matched',
      r'Generated behavior at equal component count $|\mathcal F_{.2}|$. Each '
      r'rate cell gives broad misalignment / unsafe in-domain advice (\%). '
      r'$F=\mathcal F_{.2}$; $N_F,G_F,R_F$ are equal-count narrow, magnitude, and random sets. '
      r'The last column is the paired difference in unsafe advice between '
      r'broad-favoring and narrow removal, in percentage points with a 95\% interval. '
      r'Counts match; edit norms differ.',
      ['Setting', 'Count', '$E$', '$E-F$', '$E-N_F$', '$E-G_F$', '$E-R_F$', 'Unsafe $F-N_F$'],
      rows, size='scriptsize')

# Judge-rule sensitivity and observable answer quality on the original X3 samples.
judges = read('reviewer_defense_20261008/judge_sensitivity.csv')
quality = read('reviewer_defense_20261008/generation_quality.csv')
rows = []
for model, name in zip(MODELS, NAMES):
    contrasts = [one(judges, run='nested_20261005', model=model,
                     edited=c, reference='E0', metric='adverse_difference_upper')
                 for c in ['D:N_core', 'D:F_20']]
    coherent = one(quality, run='nested_20261005', model=model,
                   condition='D:N_core', metric='coherent_both')
    refusal = one(quality, run='nested_20261005', model=model,
                  condition='D:N_core', metric='refusal_either')
    rows.append([name] + [ci(r, scale=100, digits=1) for r in contrasts] +
                [f"{100 * float(r['edited']):.1f}" for r in [coherent, refusal]])
table('AppendixJudgesTable', 'tab:judge-checks',
      r'Conservative judge-rule sensitivity on the original sweep generation. '
      r'Edited answers count as harmful if either judge is positive or invalid; '
      r'reference answers count only if both are positive. Differences from $E$ '
      r'are percentage points with paired 95\% intervals. Coherence requires both '
      r'judges; refusal uses either judge. $F=\mathcal F_{.2}$. Quality rates follow narrow removal.',
      ['Setting', r'$-\mathcal N_k$ adverse difference', '$-F$ adverse difference',
       r'Coherent (\%)', r'Refusal (\%)'], rows, size='scriptsize')

# Original exploratory MMLU subset, retained separately from the full benchmark.
subset_ability = read('core_confirmatory_20261005/execution/programmatic_ability_results/contrasts.csv')
rows = []
for model, name in zip(MODELS, NAMES):
    r = one(subset_ability, model=model, suite='capability', subset='all',
            condition='D:H_K', reference='E0')
    rows.append([name, ci(r, lo='ci_low', hi='ci_high', scale=100, digits=1)])
table('AppendixExploratoryMMLUTable', 'tab:mmlu-subset',
      r'Original exploratory MMLU subset (400 items). Paired accuracy changes after '
      r'removing $\mathcal N_k$ from $E$, in percentage points with 95\% intervals '
      r'from 10,000 item bootstraps. The setting-specific $k=182$--224 budgets are used.',
      ['Setting', r'Accuracy change'], rows)
tables[-1] = tables[-1].replace(r'\begin{table*}', r'\begin{table}').replace(r'\end{table*}', r'\end{table}')

# Full MMLU, with paired changes instead of duplicating all accuracies.
mmlu = read('review2_20261008/report/mmlu_full.csv')
rows = []
for model, name in zip(MODELS, NAMES):
    baseline = [one(mmlu, setting=model, cond=c) for c in ['C0', 'E0']]
    cells = []
    for condition in ['D_N_core', 'D_F_20', 'D_NF_20', 'D_GF_20']:
        r = one(mmlu, setting=model, cond=condition)
        values = [float(x) * 100 for x in re.findall(r'[+-]?\d+\.\d+', r['vsE0'])]
        cells.append(f'{values[0]:+.1f} [{values[1]:+.1f}, {values[2]:+.1f}]')
    rows.append([name] + [f"{100 * float(r['acc']):.1f}" for r in baseline] + cells)
table('AppendixMMLUTable', 'tab:mmlu-full',
      r'Full MMLU test set (14,042 items). $C$ and $E$ columns are accuracies '
      r'(\%). Removal columns are paired changes from $E$, in percentage points '
      r'with 95\% intervals from 2,000 item bootstraps. $F=\mathcal F_{.2}$; $N_F,G_F$ match its '
      r'count; $\mathcal N_k$ uses the setting-specific matched budget.',
      ['Setting', '$C$', '$E$', r'$-\mathcal N_k$', '$-F$', '$-N_F$', '$-G_F$'], rows,
      size='scriptsize')

# Direct blocking of the final narrow-only selection in all three medical models.
blocked = read('review_20261007/report/blocked.csv')
block_generation = read('review_20261007/report/generation_summary.csv')
rows = []
for model, name in zip(MODELS[:3], NAMES[:3]):
    n = one(blocked, model=model, behavior='EM', condition='B:ncore', metric='retained_of_E0')
    r = one(blocked, model=model, behavior='EM', condition='B:ncore_rand', metric='retained_of_E0')
    rates = [one(block_generation, model=model, kind='rate', a=c, b='')
             for c in ['E0', 'B:ncore', 'B:ncore_rand']]
    new = one(blocked, model=model, behavior='EM', condition='B:ncore-Nnew_core',
              metric='removed_of_blocked')
    rows.append([name, ci(n), ci(r)] + [f"{100 * float(x['estimate']):.1f}" for x in rates] + [ci(new)])
table('AppendixBlockingTable', 'tab:narrow-blocking',
      r'Training with the final narrow-selected spans blocked in the three medical '
      r'settings. Retained broad fractions compare each blocked model with the '
      r'ordinary harmful fine-tune, under the original benign reference. Generation '
      r'rates are agreed broad misalignment (\%). New removal re-selects narrow '
      r'components using the projected benign reference, then measures the fraction '
      r'of the blocked model\textquotesingle s broad shift removed under the '
      r'original benign control adjustment. Likelihood intervals are 95\%.',
      ['Setting', 'Retained: narrow block', 'Retained: random block', r'$E$ (\%)',
       r'Narrow block (\%)', r'Random block (\%)', 'New removal'], rows, size='scriptsize')

# All float32 replications in one table, with likelihood and generation counts explicit.
fp_files = {'llama3_1_8b': 'lik_fp32_llama_med.csv',
            'qwen2_5_7b_fin': 'lik_fp32_qwen2_5_fin.csv',
            'llama3_1_8b_fin': 'lik_fp32_llama_fin.csv',
            'qwen3_1_7b_fin': 'lik_frame_fp32.csv'}
fp_generation = (read('review2_20261008_extra/report/generation_summary.csv')
                 + read('review2_20261008_extra2/report/generation_summary.csv'))
fp_med = {}
for model, folder in [('qwen2_5_7b', 'x4_fp32_20261006'),
                      ('qwen3_1_7b', 'review_20261007/x4_qwen3_fp32')]:
    fp_med[model] = (read(folder + '/x4_likelihood.json')['per_model'][model + '_fp32'],
                     read(folder + '/generation_summary.csv'), read(folder + '/domain_summary.csv'))
rows = []
for model, name in zip(MODELS, NAMES):
    if model in fp_med:
        likelihood, gen, dom = fp_med[model]
        gap = likelihood['prediction_3']['broad_minus_narrow']
        selected.append(gap)
        count = '256'
        rates = [one(gen, condition=c, metric='em_agreement', suite='broad', subset='all')
                 for c in ['E0', 'D:N_256']]
        domain = [one(dom, condition=c, metric='unsafe_either', suite='domain', subset='all')
                  for c in ['E0', 'D:N_256']]
    else:
        gap = one(read('review2_20261008/report/' + fp_files[model]), tag='fp32__' + model,
                  condition='D:N_core', measure='broad - narrow', metric='gap')
        count = one(seeds, model=model, seed='42')['k']
        rates = [one(fp_generation, model=model, kind='fp32', suite='broad', stat='rate', a=c, b='')
                 for c in ['E0', 'D:N_256']]
        domain = [one(fp_generation, model=model, kind='fp32', suite='domain', stat='rate', a=c, b='')
                  for c in ['E0', 'D:N_256']]
    rows.append([name, count, ci(gap),
                 ' $\\to$ '.join(f"{100 * float(r['estimate']):.1f}" for r in rates),
                 ' $\\to$ '.join(f"{100 * float(r['estimate']):.1f}" for r in domain)])
table('AppendixPrecisionTable', 'tab:precision-checks',
      r'Float32-master-weight replications. The gap is measured at '
      r'the listed narrow-selection count, with paired 95\% intervals. Generation '
      r'columns show harmful-model rates before and after removing '
      r'$\mathcal N_{256}$ (\%); likelihood and generation therefore use '
      r'different counts except in Qwen2.5 and Qwen3 medical.',
      ['Setting', 'Likelihood $k$', 'Removal gap', 'Broad misalignment', 'Unsafe advice'], rows)

# Retain provenance for the small diagnostics reported in prose as well.
diagnostics = {}
scale = read('review_20261007/report/scale_sweep_likelihood.csv')
for model in MODELS:
    for cond in ['T:0.5xN_1024', 'T:1.5xN_1024']:
        one(scale, model=model, cond=cond, b='EM')
        one(block_generation, model=model, kind='rate', a=cond, b='')
    one(block_generation, model=model, kind='rate', a='E0', b='')
    one(block_generation, model=model, kind='rate', a='D:N_core', b='')
for model in MODELS[:3]:
    angles = read(f'review_20261007/blocked/{model}/angles.csv')
    values = [float(r['mean_sq_projection']) for r in angles
              if r['set'] == 'ncore' and r['reference'] == 'orig_top32']
    diagnostics[model + '_new_direction_projection'] = sum(values) / len(values)
for domain in ['med', 'fin']:
    diagnostics[domain + '_minimal_pairs'] = read(
        f'review2_20261008/pairs/minimal/{domain}/summary.json')
for model in ['qwen2_5_7b', 'llama3_1_8b_fin']:
    diagnostics[model + '_alpha16'] = read(
        f'review_20261007/attribution/alpha16/{model}/manifest.json')['ig_check']
read('review_20261007/report/alt_alpha16.csv')
read('reviewer_defense_20261008/question_influence_summary.csv')
read('generalization_defense_20261008/families_protocol.json')
read('generalization_defense_20261008/question_overlap.csv')
read('generalization_defense_20261008/overlap_restricted_effects.csv')
read('fixed_edit_transport_20261008/protocol.json')

# Emit only tables included by the manuscript; keep the complete source audit below.
included = set(re.findall(r'\\(Appendix\w+Table)',
                          (ROOT / 'paper_draft/arr_2026/main.tex').read_text()))
tables = [t for t in tables if re.search(r'\\newcommand\{\\(\w+)', t).group(1) in included]

OUTPUT.write_text('% Generated by experiments/persona_control/stage9/paper_appendix_tables.py\n'
                  + '\n\n'.join(tables) + '\n')
RECORD.mkdir(parents=True, exist_ok=True)
(RECORD / 'manifest.json').write_text(json.dumps(dict(
    script=str(Path(__file__).relative_to(ROOT)), output=str(OUTPUT.relative_to(ROOT)),
    source_sha256=sources, included_tables=sorted(included), selected_rows=selected, diagnostics=diagnostics), indent=2) + '\n')
print(f'Wrote {len(tables)} tables from {len(sources)} source files to {OUTPUT.relative_to(ROOT)}')
