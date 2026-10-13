"""Regenerate RQ-specific figures and the joint behavior table from saved results.

Run from any directory. Only the exp_behavior marker in main.tex is updated;
all numerical inputs are existing CSV summaries, with no model execution.
"""
import argparse
from pathlib import Path
import re
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from paper_palette import PINK, BLUE, PALE_PINK
from paper_fonts import configure_fonts
from paper_localizing import ROOT, NEST, CONF, SETTINGS, KS, read, load_x2

OUT = ROOT / 'figures/persona_control/experiment_layout_20261007'
DEST = ROOT / 'paper_draft/arr_2026/figures'
PAPER = ROOT / 'paper_draft/arr_2026/main.tex'


def save(fig, name):
    fig.savefig(OUT / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=180, bbox_inches='tight')
    shutil.copy2(OUT / f'{name}.pdf', DEST / f'{name}.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-only", action="store_true")
    parser.add_argument("--table-only", action="store_true", help="Update the behavior table without regenerating figures.")
    parser.add_argument("--probe-baseline", action="store_true", help="Use the completed same-batch N/P/G/B fixed-256 experiment.")
    args = parser.parse_args()
    if args.table_only:
        write_behavior_table()
        return
    configure_fonts()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    source = ROOT / 'eval_runs/persona_control_stage9/probe_fixed256_20261011' if args.probe_baseline else NEST
    if args.probe_baseline:
        assert (source / 'report.json').exists(), 'Complete and report all six settings first.'
    x = read(source / ('likelihood_summary.csv' if args.probe_baseline else 'x2_likelihood_summary.csv'))
    frac = x[(x.metric == 'fraction') & (x.behavior == 'EM')].set_index(['model', 'direction', 'selection'])
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.0), sharex=True, sharey=True)
    selectors = [('N_256', r'$\mathcal{N}_{256}$'), ('G_256', r'$\mathcal{G}_{256}$'),
                 ('B_256', r'$\mathcal{B}_{256}$')]
    if args.probe_baseline:
        selectors.insert(1, ('P_256', r'$\mathcal{P}_{256}$'))
    for ax, (model, domain, name) in zip(axes.flat, SETTINGS):
        for i, (key, label) in enumerate(selectors):
            color = PINK if key == 'N_256' else BLUE
            for direction, offset in [('T', -.13), ('D', .13)]:
                r = frac.loc[(model, direction, key)]
                ax.plot([r.lo, r.hi], [i+offset]*2, color=color, lw=1.2)
                ax.plot(r.estimate, i+offset, 'o', ms=6, mew=1.3, mec=color,
                        mfc='white' if direction == 'D' else color)
        ax.set(title=f'{name}, {domain.lower()}', yticks=range(len(selectors)),
               yticklabels=[s[1] for s in selectors], ylim=(len(selectors) - .5, -.5),
               xlim=(-.05, 1.13), xticks=[0, .5, 1])
        ax.grid(axis='x', alpha=.2)
    legend = [Line2D([], [], marker='o', linestyle='', markersize=6,
                     markeredgewidth=1.3, markeredgecolor='#696c70',
                     markerfacecolor='white', label='Removal'),
              Line2D([], [], marker='o', linestyle='', markersize=6,
                     markeredgewidth=1.3, markeredgecolor='#696c70',
                     markerfacecolor='#696c70', label='Addition'),
              Patch(facecolor=PINK, label='Narrow selection'),
              Patch(facecolor=BLUE, label='Comparison sets')]
    fig.legend(handles=legend, loc='upper center', ncol=4, frameon=False, fontsize=10,
               bbox_to_anchor=(.54, 1.0), columnspacing=1.2, handletextpad=.4,
               handlelength=1.2)
    fig.supxlabel('Fraction of broad shift', fontsize=11, y=.012)
    fig.subplots_adjust(left=.075, right=.995, bottom=.16, top=.81,
                        wspace=.15, hspace=.48)
    save(fig, 'experiment_selection')
    if args.selection_only:
        return

    x2 = load_x2()
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 2.85), sharex=True, sharey=True)
    selections = [('N_256', r'$\mathcal{N}_{256}$'),
                  ('G_256', r'$\mathcal{G}_{256}$'),
                  ('BX', r'$\mathcal{B}^{out}$'),
                  ('NX', r'$\mathcal{N}^{out}$')]
    for ax, (model, domain, name) in zip(axes.flat, SETTINGS):
        for shift, offset, color, label in [('broad', -.18, PINK, 'Broad'),
                                             ('narrow', .18, PALE_PINK, 'Narrow')]:
            values = [x2[(model, 'D', key, shift)] for key, _ in selections]
            ax.bar([i + offset for i in range(len(selections))],
                   [r[0] for r in values], width=.34, color=color, label=label,
                   yerr=[[r[0]-r[1] for r in values], [r[2]-r[0] for r in values]],
                   error_kw={'elinewidth': .65, 'capsize': 1.5, 'capthick': .65},
                   zorder=3)
        ax.set(title=f'{name}, {domain.lower()}', xlim=(-.55, 3.55),
               ylim=(-.045, 1.13), xticks=range(len(selections)),
               xticklabels=[label for _, label in selections], yticks=[0, .5, 1])
        ax.title.set_fontsize(9)
        ax.tick_params(axis='both', labelsize=10, pad=2)
        ax.tick_params(axis='x', labelbottom=True, length=0)
        ax.grid(axis='y', alpha=.2, zorder=0)
        ax.axhline(0, color='#555555', lw=.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(.54, 1.01), handlelength=1.4,
               columnspacing=1.5)
    fig.text(.012, .49, 'Fraction of shift removed', rotation=90,
             ha='center', va='center', fontsize=11)
    fig.subplots_adjust(left=.085, right=.995, bottom=.10, top=.82,
                        wspace=.13, hspace=.50)
    save(fig, 'experiment_concentration')

    write_behavior_table()


def write_behavior_table():
    OUT.mkdir(parents=True, exist_ok=True)
    gen = read(NEST / 'generation_summary.csv')
    gen = gen[(gen.subset == 'all') & (gen.metric == 'em_agreement')].set_index(['model', 'condition'])
    domain = read(NEST / 'domain_summary.csv')
    domain = domain[(domain.subset == 'all') & (domain.metric == 'unsafe_either')].set_index(['model', 'condition'])
    ability = read(CONF / 'execution/programmatic_ability_results/contrasts.csv')
    ability = ability[(ability.subset == 'all') & (ability.condition == 'D:H_K') &
                      (ability.reference == 'E0')].set_index(['model', 'suite'])
    mmlu = read(ROOT / 'eval_runs/persona_control_stage9/review2_20261008/report/mmlu_full.csv')
    mmlu = mmlu[mmlu.cond == 'D_N_core'].set_index('setting')
    lines = [r'\begin{tabular}{@{}lrrrrrrcc@{}}', r'\toprule',
             r' & \multicolumn{3}{c}{Broad misalignment (\%)} & \multicolumn{3}{c}{Unsafe advice (\%)} & \multicolumn{2}{c}{Ability change after $-\mathcal{N}_k$} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}',
             r'Setting & $E$ & $E-\mathcal{N}_k$ & $E-\mathcal{F}_{.2}$ & $E$ & $E-\mathcal{N}_k$ & $E-\mathcal{F}_{.2}$ & IFBench & MMLU \\', r'\midrule']
    for model, dom, name in SETTINGS:
        row = [name.replace('-Instruct', '') + (' (med.)' if dom == 'Medical' else ' (fin.)')]
        for df in [gen, domain]:
            row.extend(f'{100*df.loc[(model, cond)].estimate:.1f}' for cond in ['E0', 'D:N_core', 'D:F_20'])
        r = ability.loc[(model, 'ifbench')]
        row.append(f'${100*r.estimate:+.1f}\\ [{100*r.ci_low:+.1f}, {100*r.ci_high:+.1f}]$')
        values = [100 * float(v) for v in re.findall(r'[+-]?\d+\.\d+', mmlu.loc[model].vsE0)]
        row.append(f'${values[0]:+.1f}\\ [{values[1]:+.1f}, {values[2]:+.1f}]$')
        lines.append(' & '.join(row) + r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    table = '\n'.join(lines)
    (OUT / 'behavior_table.tex').write_text(table + '\n')
    text = PAPER.read_text()
    text, count = re.subn(r'(?s)(% BEGIN GENERATED exp_behavior\n).*?(\n% END GENERATED exp_behavior)',
                          lambda m: m[1] + table + m[2], text)
    assert count == 1
    PAPER.write_text(text)


if __name__ == '__main__':
    main()
