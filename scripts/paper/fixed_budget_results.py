"""Regenerate the 256-component likelihood table from saved experiment summaries."""
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'eval_runs/persona_control_stage9/nested_20261005'
PAPER = ROOT / 'paper_draft/arr_2026/main.tex'
MODELS = ['qwen2_5_7b', 'llama3_1_8b', 'qwen3_1_7b',
          'qwen2_5_7b_fin', 'llama3_1_8b_fin', 'qwen3_1_7b_fin']
NAMES = ['Qwen2.5 (med.)', 'Llama (med.)', 'Qwen3 (med.)',
         'Qwen2.5 (fin.)', 'Llama (fin.)', 'Qwen3 (fin.)']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe-baseline', action='store_true')
    args = parser.parse_args()
    source = ROOT / 'eval_runs/persona_control_stage9/probe_fixed256_20261011' if args.probe_baseline else DATA
    if args.probe_baseline:
        assert (source / 'report.json').exists(), 'Complete all six settings first.'
    paths = [DATA / 'x2_likelihood_summary.csv', DATA / 'x2_likelihood_contrasts.csv']
    summary, contrasts = [list(csv.DictReader(p.open())) for p in paths]
    probe_summary, probe_contrasts = [], []
    if args.probe_baseline:
        probe_paths = [source / 'likelihood_summary.csv', source / 'likelihood_contrasts.csv']
        probe_summary, probe_contrasts = [list(csv.DictReader(p.open())) for p in probe_paths]
        paths.extend(probe_paths)
    fractions = {(r['model'], r['direction'], r['behavior']): r for r in summary
                 if r['metric'] == 'fraction' and r['selection'] == 'N_256'}
    gaps = {(r['model'], r['direction']): r for r in contrasts
            if r['kind'] == 'broad_minus_narrow' and r['a'] == 'N_256'}

    def interval(row, scale=1, digits=2):
        return '{:.{d}f} [{:.{d}f}, {:.{d}f}]'.format(
            *[scale * float(row[k]) for k in ('estimate', 'lo', 'hi')], d=digits)

    lines = [r'\begin{tabular}{lrcccccc}', r'\toprule',
             r' & & \multicolumn{2}{c}{Broad shift} & \multicolumn{2}{c}{Narrow shift} & \multicolumn{2}{c}{Broad $-$ narrow (points)} \\',
             r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}',
             r'Setting & Norm (\%) & $T$ & $D$ & $T$ & $D$ & $T$ & $D$ \\', r'\midrule']
    for model, name in zip(MODELS, NAMES):
        broad = [fractions[model, d, 'EM'] for d in 'TD']
        narrow = [fractions[model, d, 'MD'] for d in 'TD']
        row = [name, f"{100 * float(broad[0]['share_ec_full']):.1f}",
               *[interval(r) for r in broad],
               *[f"{float(r['estimate']):.2f}" for r in narrow],
               *[interval(gaps[model, d], scale=100, digits=1) for d in 'TD']]
        lines.append(' & '.join(row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    earlier_path = DATA / 'x1/x1_contrasts.csv'
    earlier = [r for r in csv.DictReader(earlier_path.open())
               if r['kind'] == 'n_core_minus_baseline' and r['shift'] == 'broad'
               and r['b'] in ('B_K', 'SVD_EC_K')]
    assert len(earlier) == 24 and all(float(r['lo']) > 0 for r in earlier)
    index = {(r['model'], r['direction'], r['b']): r for r in earlier}
    old_lines = [r'\begin{tabular}{lcccc}', r'\toprule',
                 r' & \multicolumn{2}{c}{$\mathcal N_k-\mathcal P_k$} & \multicolumn{2}{c}{$\mathcal N_k-\mathcal G_k$} \\',
                 r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}',
                 r'Setting & $T$ & $D$ & $T$ & $D$ \\', r'\midrule']
    for model, name in zip(MODELS, NAMES):
        values = [interval(index[model, d, baseline], scale=100, digits=1)
                  for baseline in ('B_K', 'SVD_EC_K') for d in 'TD']
        old_lines.append(' & '.join([name] + values) + r' \\')
    old_lines.extend([r'\bottomrule', r'\end{tabular}'])
    probe_lines = []
    if args.probe_baseline:
        probe_fractions = {(r['model'], r['direction']): r for r in probe_summary
                           if r['metric'] == 'fraction' and r['behavior'] == 'EM'
                           and r['selection'] == 'P_256'}
        probe_differences = {(r['model'], r['direction']): r for r in probe_contrasts
                             if r['kind'] == 'difference' and r['behavior'] == 'EM'
                             and r['a'] == 'N_256' and r['b'] == 'P_256'}
        probe_lines = [r'\begin{tabular}{lrcccc}', r'\toprule',
                       r' & & \multicolumn{2}{c}{$\mathcal P_{256}$ broad fraction} & \multicolumn{2}{c}{$\mathcal N_{256}-\mathcal P_{256}$ (points)} \\',
                       r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}',
                       r'Setting & Norm (\%) & $T$ & $D$ & $T$ & $D$ \\', r'\midrule']
        for model, name in zip(MODELS, NAMES):
            row = [name, f"{100 * float(probe_fractions[model, 'T']['share_ec_full']):.1f}",
                   *[interval(probe_fractions[model, d]) for d in 'TD'],
                   *[interval(probe_differences[model, d], scale=100, digits=1) for d in 'TD']]
            probe_lines.append(' & '.join(row) + r' \\')
        probe_lines.extend([r'\bottomrule', r'\end{tabular}'])
    text = PAPER.read_text()
    text, count = re.subn(r'(% BEGIN GENERATED fixed_budget\n).*?(\n% END GENERATED fixed_budget)',
                         lambda m: m[1] + '\n'.join(lines) + m[2], text, flags=re.S)
    assert count == 1
    text, count = re.subn(r'(% BEGIN GENERATED earlier_comparisons\n).*?(\n% END GENERATED earlier_comparisons)',
                         lambda m: m[1] + '\n'.join(old_lines) + m[2], text, flags=re.S)
    assert count == 1
    if args.probe_baseline:
        text, count = re.subn(r'(% BEGIN GENERATED fixed_probe\n).*?(\n% END GENERATED fixed_probe)',
                             lambda m: m[1] + '\n'.join(probe_lines) + m[2], text, flags=re.S)
        assert count == 1
    PAPER.write_text(text)
    differences = [r for r in contrasts if r['kind'] == 'difference' and r['behavior'] == 'EM'
                   and r['a'] == 'N_256' and r['b'] == 'G_256']
    assert len(differences) == 12 and all(float(r['lo']) > 0 for r in differences)
    assert all(float(gaps[m, 'D']['lo']) > 0 for m in MODELS)
    ranges = {}
    for behavior in ('EM', 'MD'):
        values = [float(fractions[m, 'D', behavior]['estimate']) for m in MODELS]
        ranges[behavior] = [100 * min(values), 100 * max(values)]
    probe = [r for r in probe_contrasts if r['kind']=='difference' and r['behavior']=='EM'
             and r['a']=='N_256' and r['b']=='P_256']
    report = dict(budget=256, removal_percent_ranges=ranges,
                  paired_probe_comparisons=probe,
                  supported_narrow_minus_probe_comparisons=sum(float(r['lo'])>0 for r in probe),
                  supported_narrow_minus_magnitude_comparisons=12,
                  supported_removal_gaps=6, supported_transfer_gaps=sum(
                      float(gaps[m, 'T']['lo']) > 0 for m in MODELS),
                  fractions=list(fractions.values()), gaps=list(gaps.values()),
                  differences=differences,
                  earlier_budget_comparisons=earlier,
                  sources={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in paths + [earlier_path]})
    out = ROOT / 'eval_runs/persona_control_stage9/arr_submission_20261011/fixed_budget_results.json'
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('budget', 'removal_percent_ranges',
          'supported_narrow_minus_magnitude_comparisons', 'supported_removal_gaps', 'supported_transfer_gaps')}))


if __name__ == '__main__':
    main()
