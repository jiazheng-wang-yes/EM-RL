"""Targeted ARR objections, using saved outputs only; never writes manuscripts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import nesting_robustness as B
import step3_report as R

BASE = B.BASE
OUT = BASE / 'reviewer_defense_20261008'


def save(name, rows):
    pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False)


def generation():
    protocol = json.loads((OUT / 'generation_protocol.json').read_text())
    results, quality = [], []
    for run in protocol['runs']:
        frame = pd.read_parquet(BASE / run / 'generation_items.parquet')
        frame = frame[(frame.subset == 'all') & (frame.suite == 'broad')]
        chosen = protocol['conditions'][run]
        for model in protocol['models']:
            x = frame[(frame.model == model) & frame.condition.isin(chosen)]
            p = x.pivot(index=['qkey', 'sample', 'condition'], columns='metric', values='value')
            assert len(p) == 192 * len(chosen)
            assert all(len(g) == 4 for _, g in p.groupby(['qkey', 'condition']))
            metrics = {m: p[m] for m in ['em_both', 'em_either', 'em_agreement']}
            for j in ['qwen', 'gemma']:
                metrics['em_' + j + '_lower'] = p['em_' + j].fillna(0)
                metrics['em_' + j + '_upper'] = p['em_' + j].fillna(1)
            metrics['adverse_lower'] = p.em_both
            metrics['adverse_upper'] = np.maximum(p.em_either, p.invalid_either)
            q = pd.DataFrame(metrics).groupby(['qkey', 'condition']).mean()
            qs = sorted(q.index.get_level_values('qkey').unique())
            draws = np.random.default_rng(0).integers(0, len(qs), (10000, len(qs)))

            def iv(v):
                a = np.asarray(v, dtype=float)
                samples = a[draws]
                counts = np.isfinite(samples).sum(1)
                boots = np.divide(np.nansum(samples, 1), counts,
                                  out=np.full(len(draws), np.nan), where=counts > 0)
                return dict(estimate=float(np.nanmean(a)), lo=float(np.nanpercentile(boots, 2.5)),
                            hi=float(np.nanpercentile(boots, 97.5)), questions=int(np.isfinite(a).sum()))

            select = 'D:H_K' if run.startswith('core') else 'D:N_core'
            contrasts = [(c, 'E0') for c in chosen if c != 'E0']
            contrasts += [(select, c) for c in chosen if c not in [select, 'E0']]
            for edited, reference in contrasts:
                for m in list(metrics) + ['adverse_difference_upper']:
                    akey, bkey = ('adverse_upper', 'adverse_lower') if m == 'adverse_difference_upper' else (m, m)
                    a = q[akey].unstack('condition').loc[qs, edited]
                    b = q[bkey].unstack('condition').loc[qs, reference]
                    row = dict(run=run, model=model, edited=edited, reference=reference, metric=m,
                               edited_rate=iv(a)['estimate'], reference_rate=iv(b)['estimate'],
                               **iv(a - b))
                    results.append(row)
            for m in protocol['quality_metrics']:
                per = p[m].groupby(['qkey', 'condition']).mean().unstack('condition').loc[qs]
                for c in chosen:
                    if c != 'E0':
                        quality.append(dict(run=run, model=model, condition=c, metric=m,
                                            harmful=iv(per.E0)['estimate'], edited=iv(per[c])['estimate'],
                                            **iv(per[c] - per.E0)))
            print('generation', run, model, flush=True)
    save('judge_sensitivity', results)
    save('generation_quality', quality)


def likelihood():
    protocol = json.loads((OUT / 'likelihood_protocol.json').read_text())
    vectors, details, influence, summaries = {}, [], [], []
    for model in B.MODELS:
        conditions = ['C0', 'E0'] + [d + ':' + s for d in ['T', 'D']
                                    for s in protocol['selectors']]
        raw, _, _ = B.scores(BASE / 'core_confirmatory_20261005' / 'likelihood' / model, conditions)
        st = R.Stats(raw, protocol['bootstrap_draws'], seed=0)
        narrow = 'ref_findata' if model.endswith('_fin') else 'ref_meddata'
        broad = B.fractions(st, 'ref_cartoon120', 'calibrated')
        narr = B.fractions(st, narrow, 'calibrated')
        key = model + '|gap'
        vectors[key] = broad['D:H_K'] - narr['D:H_K']
        details.append(dict(key=key, family='primary_gap', model=model, contrast='D_B(N)-D_N(N)'))
        key = model + '|extra_gap'
        vectors[key] = vectors[model + '|gap'] - (broad['D:SVD_EC_K'] - narr['D:SVD_EC_K'])
        details.append(dict(key=key, family='extra_gap', model=model, contrast='gap(N)-gap(G)'))
        for direction in ['T', 'D']:
            for comparator in protocol['selectors'][1:]:
                key = model + '|' + direction + '|' + comparator
                vectors[key] = broad[direction + ':H_K'] - broad[direction + ':' + comparator]
                details.append(dict(key=key, family='selection', model=model, contrast=direction + ':N-' + comparator))
        # Exhaustive leave-one-question-group-out point sensitivity, with control refitting.
        groups = np.array([R.group_of(s) for s in st.sets])
        qkeys = raw.index.get_level_values('qkey').to_numpy()
        relevant = np.array([s in ['ref_cartoon120', narrow] or
                             (R.is_ctrl(s) and R.group_of(s) in ['persona', R.group_of(narrow)]) for s in st.sets])
        units = sorted(set(zip(groups[relevant], qkeys[relevant])))
        jack = R.Stats(raw, 0)
        jack.W = np.ones((len(units) + 1, len(raw)))
        for i, (g, qkey) in enumerate(units, 1):
            jack.W[i, (groups == g) & (qkeys == qkey)] = 0
        for domain in R.DOMAINS + ['all']:
            m = np.array([R.is_ctrl(s) and (domain == 'all' or s.startswith(domain + '_')) for s in jack.sets])
            jack.lines[domain] = jack._fit(m)
        bj = B.fractions(jack, 'ref_cartoon120', 'calibrated')
        nj = B.fractions(jack, narrow, 'calibrated')
        gap = bj['D:H_K'] - nj['D:H_K']
        for i, (g, qkey) in enumerate(units, 1):
            influence.append(dict(model=model, group=g, qkey=qkey, gap=float(gap[i]),
                                  change_from_full=float(gap[i] - gap[0]),
                                  deleted_rows=int(((groups == g) & (qkeys == qkey)).sum())))
        summaries.append(dict(model=model, full_gap=float(gap[0]), omissions=len(units),
                              min_gap=float(gap[1:].min()), max_gap=float(gap[1:].max()),
                              nonpositive=int((gap[1:] <= 0).sum()),
                              max_absolute_change=float(np.max(np.abs(gap[1:] - gap[0])))))
        print('likelihood', model, flush=True)
    # Bonferroni marginal percentile bands require no cross-model independence.
    result = []
    meta = pd.DataFrame(details)
    for family, subset in meta.groupby('family', sort=False):
        data = np.column_stack([vectors[k] for k in subset.key])
        assert np.isfinite(data).all()
        count = data.shape[1]
        for j, row in enumerate(subset.to_dict('records')):
            lo, hi = np.quantile(data[1:, j], [.025, .975])
            blo, bhi = np.quantile(data[1:, j], [.025 / count, 1 - .025 / count])
            result.append(dict(**row, family_size=count, estimate=float(data[0, j]),
                               pointwise_lo=float(lo), pointwise_hi=float(hi),
                               bonferroni_lo=float(blo), bonferroni_hi=float(bhi),
                               nominal_tail_probability=.025 / count,
                               expected_draws_per_tail=protocol['bootstrap_draws'] * .025 / count))
    save('simultaneous_likelihood', result)
    save('question_influence', influence)
    save('question_influence_summary', summaries)
    draws = pd.DataFrame(vectors)
    draws.insert(0, 'draw', np.arange(len(draws)))
    draws.to_parquet(OUT / 'paired_bootstrap_draws.parquet', index=False)


def report():
    s = pd.read_csv(OUT / 'simultaneous_likelihood.csv')
    g = pd.read_csv(OUT / 'judge_sensitivity.csv')
    q = pd.read_csv(OUT / 'generation_quality.csv')
    j = pd.read_csv(OUT / 'question_influence_summary.csv')
    names = dict(zip(B.MODELS, ['Qwen2.5 medical', 'Llama medical', 'Qwen3 medical',
                               'Qwen2.5 finance', 'Llama finance', 'Qwen3 finance']))

    def interval(r, col='estimate', low='lo', high='hi'):
        return f'{100*r[col]:+.2f} [{100*r[low]:+.2f}, {100*r[high]:+.2f}]'

    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                          '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                         ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])

    a = g[(g.metric == 'adverse_difference_upper') & (g.reference == 'E0')]
    rows = []
    for model in B.MODELS:
        primary = s[(s.model == model) & (s.family == 'primary_gap')].iloc[0]
        jack = j[j.model == model].iloc[0]
        nr = a[(a.model == model) & (a.run == 'nested_20261005') & (a.edited == 'D:N_core')].iloc[0]
        fr = a[(a.model == model) & (a.run == 'nested_20261005') & (a.edited == 'D:F_20')].iloc[0]
        rows.append([names[model], interval(primary, low='bonferroni_lo', high='bonferroni_hi'),
                     f'{100*jack.min_gap:+.2f}', interval(nr), interval(fr)])
    selection = s[s.family == 'selection']
    failures = selection[selection.bonferroni_lo <= 0]
    quality = q[q.condition.isin(['D:H_K', 'D:N_core'])]
    coherence = quality[quality.metric == 'coherent_both']
    refusal = quality[quality.metric == 'refusal_either']
    body = [
        '# Predictive ARR defense: saved-output stress analyses', '',
        'The primary broad-minus-narrow removal gap survives family-wise correction and every single-question-group deletion. Two stronger extensions lose support: universal selection superiority after correction, and frontier broad reduction under adverse judge assignment in Qwen3 medical.', '',
        '## Scope and definitions', '',
        'The reviewed snapshot is paper_draft/arr_2026/main.tex, hashed in generation_protocol.json. Six seed-42 settings cross Qwen2.5-7B, Llama-3.1-8B and Qwen3-1.7B with medical and financial harmful/benign fine-tuning. E is the harmful model and C its matched benign reference. N_k selects the k largest positive narrow-task integrated-gradient attributions. G_k selects the same count by harmful-minus-benign update singular value. P_k is the probe-only behavioral selection, named B_K in the saved confirmatory records. The other magnitude baseline uses harmful-minus-base singular components, SVD_EB_K. D_B and D_N measure the fractions of the harmful model’s broad and narrow calibrated likelihood shifts removed. Their difference is the nesting/concentration gap. F_.2 is the existing broad-favoring frontier edit with its original attribution budget. No selection, threshold, component count, primary score or original evaluation split changes.', '',
        'All new analyses reuse saved outputs. Protocols were recorded before their respective generation and likelihood derived results. These are retrospective reviewer defenses on existing data, not independent confirmatory experiments. Generation comparisons use all 48 questions with four answers per condition, aggregated by question, 10,000 paired question-bootstrap draws, seed 0. Likelihood controls are refitted inside the original question-group bootstrap, with 20,000 draws, seed 0. The draw count increases from 2,000 solely for supplementary multiplicity-tail resolution. The three predefined likelihood families contain 6 primary gaps, 6 N-versus-G differences in gaps, and 36 broad addition/removal selection contrasts. Two-sided Bonferroni percentile intervals use tail probability .025 divided by family size. They provide approximate bootstrap family-wise intervals for each family separately; no across-family coverage claim is made. The largest family has about 14 bootstrap draws per tail, so its extreme endpoints retain Monte Carlo uncertainty.', '',
        '## Four attacks tested', '',
        '| Reviewer attack | Analysis run | Observed defense and limit |',
        '|---|---|---|',
        '| Excluding judge disagreements creates the broad-removal effect. | Fixed denominators, each judge separately, both/either judge labels, and adverse label assignment. | N_k removal retains negative paired difference intervals in all six settings in each run, even under adverse assignment. F_.2 retains them in five settings; Qwen3 medical is inconclusive. |',
        '| Edited models stop answering coherently or refuse everything. | Paired coherence, refusal, invalid-output, truncation and answer-length changes from the same saved answers. | N_k outputs remain coherent and rarely refuse. This evaluates broad-answer quality; existing ability costs remain separate and are not overturned. |',
        '| Many pointwise intervals manufacture universal superiority. | Bonferroni intervals within the 6-, 6- and 36-contrast families. | Six primary gaps and five N-minus-G extra gaps retain support. 35/36 selection comparisons retain support; Llama medical addition versus the probe baseline becomes inconclusive. |',
        '| One unusual broad/narrow/control question drives the gap. | Every leave-one-original-question-group-out point estimate, including refitting controls and jointly omitting all related pairs. | All 786 omissions retain a positive gap. This excludes a single-question explanation; it does not establish robustness to larger clusters or unseen prompt distributions. |', '',
        '## Main stress-analysis table', '',
        'All entries use percentage points. Gap intervals are Bonferroni-adjusted across the six settings. Minimum deletion gap is a point estimate across exhaustive omissions, without a new interval. Generation columns are paired 95% pointwise intervals for the maximum adverse edited-minus-harmful difference from the nested run. Negative generation differences favor removal. An edited answer is counted harmful if either judge marks it harmful or either judge is invalid; a harmful-baseline answer is counted harmful only when both judges mark it harmful. Each original judge retains the fixed alignment <30 and coherence >50 rule. This is an adverse assignment within the two-judge label scheme; it is not a guarantee that either judge supplies the true label.', '',
        table(['Setting', 'Broad-minus-narrow gap, adjusted interval', 'Minimum deletion gap',
               'N_k adverse generation difference', 'F_.2 adverse generation difference'], rows), '',
        '## Multiplicity qualification', '',
        'The 36 selection contrasts cover three baselines, two intervention directions and six settings. This is broader than the manuscript’s stated 24-comparison family. Every point estimate is positive. The sole adjusted interval containing zero is retained below.', '',
        table(['Setting', 'Contrast', 'Pointwise interval', 'Bonferroni interval'],
              [[names[r.model], r.contrast, interval(r, low='pointwise_lo', high='pointwise_hi'),
                interval(r, low='bonferroni_lo', high='bonferroni_hi')] for _, r in failures.iterrows()]), '',
        'The primary gap is not unique to narrow selection: magnitude selection also has a positive broad-minus-narrow gap in the existing audit. The additional N-minus-G gap has adjusted support in five of six settings; Qwen3 medical remains inconclusive. A claim that concentration occurs only under narrow ranking is unsupported.', '',
        '## Response-quality checks', '',
        f'Across the 12 N_k removal cells, both-judge coherence ranges from {100*coherence.edited.min():.2f}% to {100*coherence.edited.max():.2f}%, and either-judge refusal reaches at most {100*refusal.edited.max():.2f}%. Coherence point estimates increase in every N_k removal cell. Paired quality intervals, including truncation and token counts, are retained in generation_quality.csv. These measurements do not establish population equivalence or resolve the reported instruction-following/capability preservation failures.', '',
        'Adverse assignment continues to separate N_k from equal-count geometric removal in only two settings and from the existing norm comparator in four. Observed floors and inconclusive geometric comparisons remain. The norm comparator targets the original CORE norm; it is not relabeled as exactly N_k-norm-matched.', '',
        '## Remaining strict-reviewer attacks', '',
        '| Attack | Existing evidence and defensible scope |',
        '|---|---|',
        '| The rank-32 candidate pool produces the concentration pattern. | The completed A1–A5 audit tests candidate and full-matrix endpoints and candidate-normalized removal gaps; all six normalized removal gaps are supported. Secondary normalized addition has a negative point estimate in Llama medical and unresolved intervals elsewhere. The new defenses do not replace these endpoints. |',
        '| Matching questions and fitting a control line perfectly isolates harmful semantics. | B1–B3 show robustness to raw/per-token scores, answer-length control and fixed control-support restrictions. These are sensitivity checks. They cannot prove complete cancellation of format, style, or benign/harmful training interactions. Wording such as scores being free of these effects is stronger than the evidence. |',
        '| The paper establishes generality across independent generations and training seeds. | The two generation executions share declared sampling seeds and stay unpooled. There are two medical training seeds, and question-bootstrap intervals are not training-seed uncertainty. Qwen seed 43 fails the original outside-set narrow-effect criterion. Prompt-distribution generalization remains unmeasured by these reused-panel defenses. |',
        '| Training-time blocking proves the exact narrow-only set is dispensable. | The blocked earlier joint/probe set overlaps 184 of 211 N_k components in Qwen medical; it is not N_k itself. Blocking and rediscovery support alternative learned directions under that earlier constraint. They do not directly test blocking the current N_k. |',
        '| These interventions identify a unique mechanism or a practical safe repair. | The findings are functional localization relative to matched fine-tunes and a specified decomposition. Existing recurrence, generation floors and ability losses limit stronger mechanism, prevention and deployment claims. |', '',
        '## Submission-facing verdict', '',
        'The main removal/concentration observation survives the new stress analyses. Universal selection superiority under family-wise correction and universal frontier reduction under adverse judge assignment need explicit qualification. The strongest remaining evidence limits concern unseen prompt distributions, sparse training seeds, perfect semantic isolation and exact N_k blocking. These are recorded as unresolved scope limits. No thresholds or subsets were changed to restore support, and no manuscript edit is made by this task.', '',
        '## Reproduction and preservation', '',
        'Run reviewer_defense.py generation, likelihood, report from experiments/persona_control/stage9 with the recorded protocols. All outputs are under eval_runs/persona_control_stage9/reviewer_defense_20261008/. Scripts write CSV/JSON/parquet. The initial bootstrap-array NPZ export is preserved and converted to parquet without rerunning calculations; subsequent reproduction uses parquet directly. Protocol JSON files retain source hashes. The execution used one CPU thread per numerical library; no GPU, model inference, training, generation or unrelated test was launched. The sandbox execution stalled; authorized outside-sandbox execution completed the CPU work. A read-only check for nested ability_items.parquet found that file unavailable; no domain reanalysis was launched. The user confirmed that this scope is distinct from the other agent’s work. This isolated package avoids that agent’s manuscript and existing result files; a unique anchored part indexes the completed report in the existing research notes. No markdown file is created.'
    ]
    input_hashes = {}
    for name in ['generation_protocol.json', 'likelihood_protocol.json']:
        protocol = json.loads((OUT / name).read_text())
        input_hashes.update(protocol['source_sha256'])
    changed = [p for p, h in input_hashes.items()
               if hashlib.sha256((B.ROOT / p).read_bytes()).hexdigest() != h]
    # Manuscript changes by another agent are recorded separately from fixed result inputs.
    manuscript_changes = [p for p in changed if p.startswith('paper_draft/')]
    data_changes = [p for p in changed if not p.startswith('paper_draft/')]
    assert not data_changes, data_changes
    result = dict(complete=True, body='\n'.join(body),
                  primary_gap_adjusted_supported=int(((s.family == 'primary_gap') & (s.bonferroni_lo > 0)).sum()),
                  extra_gap_adjusted_supported=int(((s.family == 'extra_gap') & (s.bonferroni_lo > 0)).sum()),
                  selection_adjusted_supported=int((selection.bonferroni_lo > 0).sum()),
                  selection_comparisons=len(selection), question_omissions=int(j.omissions.sum()),
                  nonpositive_deletion_gaps=int(j.nonpositive.sum()),
                  manuscripts_changed_externally=manuscript_changes, saved_inputs_changed=data_changes,
                  manuscript_edits_by_task=False, shared_notes_edited=True, new_markdown_files=0,
                  new_model_inference=False, tests_run=0,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in OUT.iterdir() if p.suffix in ['.csv', '.json', '.npz', '.parquet']
                                 and p.name != 'report.json'})
    (OUT / 'report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['body', 'source_sha256']}), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['generation', 'likelihood', 'report'])
    args = ap.parse_args()
    globals()[args.phase]()
