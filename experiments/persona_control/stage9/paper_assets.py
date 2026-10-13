"""Regenerate the CORE paper's tables, plots, and source manifest from saved results.

Run from any directory: python experiments/persona_control/stage9/paper_assets.py
Tables replace named GENERATED blocks in paper_draft/arr_2026/main.tex.
No model execution or experimental result files are changed.
"""
from pathlib import Path
import csv
import hashlib
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / 'eval_runs/persona_control_stage9'
REL = RESULTS / 'reliability'
OUT = ROOT / 'figures/persona_control/core_paper'
DERIVED = RESULTS / 'paper_20261004'
PAPER = ROOT / 'paper_draft/arr_2026/main.tex'
PAPER_FIGURES = PAPER.parent / 'figures'
CELLS = [
    ('qwen_med42', 'Medical', 'Qwen2.5-7B', 'qwen2_5_7b'),
    ('llama_med42', 'Medical', 'Llama-3.1-8B', 'llama3_1_8b'),
    ('qwen3_med42', 'Medical', 'Qwen3-1.7B', 'qwen3_1_7b'),
    ('qwen_fn42', 'Finance', 'Qwen2.5-7B', 's9fnp_20261002'),
    ('llama_fn42', 'Finance', 'Llama-3.1-8B', 'llama3_1_8b_fin'),
    ('qwen3_fn42', 'Finance', 'Qwen3-1.7B', 'qwen3_1_7b_fin'),
]
SOURCES = {}


def read(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return list(csv.DictReader(path.open()))


def read_json(path):
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return json.loads(path.read_text())


def one(rows, **criteria):
    found = [r for r in rows if all(r[k] == v for k, v in criteria.items())]
    assert len(found) == 1, (criteria, len(found))
    return found[0]


def f(row, key):
    return float(row[key])


def interval(row, key='frac', scale=1, digits=2):
    return f"{scale*f(row,key):.{digits}f} [{scale*f(row,'lo'):.{digits}f}, {scale*f(row,'hi'):.{digits}f}]"


def tabular(columns, header, rows):
    return '\n'.join([r'\begin{tabular}{' + columns + '}', r'\toprule',
                      header + r' \\', r'\midrule',
                      *[' & '.join(row) + r' \\' for row in rows],
                      r'\bottomrule', r'\end{tabular}'])


def confirmatory_tables():
    analysis = RESULTS / 'core_h_analysis_20261005'
    effects = read(analysis / 'core_h_effects.csv')
    contrasts = read(analysis / 'core_baseline_contrasts.csv')
    h_rows, baseline_rows = [], []
    for tag, domain, model, probe_file in CELLS:
        key = 'qwen2_5_7b_fin' if tag == 'qwen_fn42' else probe_file
        for direction in ['T', 'D']:
            row = one(effects, model=key, direction=direction)
            difference = (f"{f(row,'h_minus_core_pp'):+.2f} "
                          f"[{f(row,'h_minus_core_lo'):+.2f}, {f(row,'h_minus_core_hi'):+.2f}]")
            h_rows.append([domain, model, direction, f"{f(row,'core_percent'):.2f}",
                           f"{f(row,'h_percent'):.2f}", difference])
            baseline_rows.append([domain, model, direction, *[
                interval(one(contrasts, model=key, direction=direction,
                             condition_b=f'{direction}:{baseline}'), 'estimate', 100)
                for baseline in ['B_K', 'SVD_EC_K', 'SVD_EB_K']]])
    return {
        'core_h': tabular('lllrrc',
                         r'Domain & Model & Edit & \Core & $\mathcal H_k$ & $\mathcal H_k$ $-$ \Core{} [95\% CI]',
                         h_rows),
        'functional_controls': tabular('lllccc',
                         r'Domain & Model & Edit & \Core{} $-$ $\mathcal B_k$ & \Core{} $-$ SVD ($E-C$) & \Core{} $-$ SVD ($E-B$)',
                         baseline_rows),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    DERIVED.mkdir(parents=True, exist_ok=True)
    tables, cells = {}, []
    effect_rows, detailed, gen_rows, probe_rows, off_rows, control_rows = [], [], [], [], [], []
    judge_rows, seed_rows = [], []
    for tag, domain, model, probe_file in CELLS:
        likelihood = read(REL / f'rand_{tag}/rand_summary.csv')
        sizes = read(REL / f'top_{tag}/sets_summary.csv')
        top = read(REL / f'top_{tag}/top_summary.csv')
        rates = read(REL / f'gen_{tag}/rel_rates.csv')
        draws = read(REL / f'gen_{tag}/rel_draws.csv')
        shifts = read(RESULTS / f'fin_native/shifts_{probe_file}.csv')
        c = {'tag': tag, 'domain': domain, 'model': model, 'likelihood': {}, 'rates': {}, 'magnitude_comparisons': {}}
        for s in ['UNION', 'CORE', 'PERI']:
            for d in ['T', 'D']:
                if s == 'PERI' and tag in ('qwen_med42', 'llama_med42'):
                    old = 's9s7_20260930' if tag == 'qwen_med42' else 's9s7_llama_20260930'
                    row = one(read(RESULTS / f'step7/{old}/step7_fractions.csv'), cond=f'{d}:PERI')
                    row = {'frac': row['EM'], 'lo': row['EM_lo'], 'hi': row['EM_hi']}
                else:
                    row = one(likelihood, set=s, dir=d, quantity='EM')
                c['likelihood'][f'{d}:{s}'] = row
        size = one(sizes, set='CORE')
        vals = [c['likelihood'][f'{d}:{s}'] for s in ['UNION', 'CORE', 'PERI'] for d in ['T', 'D']]
        effect_rows.append([domain, model, size['n'], f"{100*f(size,'share_dW2'):.1f}",
                            *[f"{f(v,'frac'):.2f}" for v in vals]])
        detailed.append([domain, model, *[interval(v) for v in vals[:2]]])
        c['magnitude_comparisons'] = {f'{d}:{s}': one(top, set=s, dir=d, quantity='EM')
                                     for s in ['CORE', 'UNION'] for d in ['T', 'D']}
        for s in ['UNION', 'CORE']:
            for d in ['T', 'D']:
                r = one(likelihood, set=s, dir=d, quantity='EM')
                control_rows.append([domain, model, s, d, r['n_draws'],
                                     f"{f(r,'draw_med'):.3f}", f"{f(r,'draw_max'):.3f}",
                                     f"{f(r,'diff_vs_mean'):.3f} [{f(r,'diff_lo'):.3f}, {f(r,'diff_hi'):.3f}]",
                                     f"{f(r,'p_emp'):.3f}"])
        for condition in ['C0', 'E0', 'D:CORE', 'D:UNION', 'T:CORE', 'T:UNION']:
            c['rates'][condition] = one(rates, scope='all', condition=condition, outcome='panel_em')
        random_rates = [f(x,'rate') for x in draws if x['scope']=='all' and x['set']=='CORE']
        c['random_rates'] = random_rates
        gen_rows.append([domain, model, *[interval(c['rates'][x], 'estimate', 100, 1)
                        for x in ['D:UNION', 'T:UNION']]])
        coherent = [one(rates, scope='all', condition=x, outcome='coherent') for x in ['C0','E0','D:CORE','T:CORE']]
        judge_rows.append([domain,model,*[f"{100*f(x,'estimate'):.1f}" for x in coherent]])
        probes = ['RR','WI','OC','TQA'] if domain == 'Medical' else ['RETURNS','BORROWED','DISMISSES_SAFE','TQA']
        for probe in probes:
            r = one(shifts, probe=probe, version='primary', half='all')
            probe_rows.append([domain,model,probe.replace('_',r'\_'),r['n'],
                               f"{f(r,'raw'):+.2f}",f"{f(r,'R'):+.2f} [{f(r,'R_lo'):+.2f}, {f(r,'R_hi'):+.2f}]",
                               'yes' if r['changed']=='True' else 'no'])
        if domain == 'Finance':
            off_rows.append([model,*[interval(one(rates, scope='without money questions', condition=x, outcome='panel_em'),
                                                      'estimate',100,1) for x in ['E0','D:CORE','T:CORE']]])
        cells.append(c)
    tables['effects'] = tabular('llrr', r'Domain & Model & $|\mathcal K|$ & Norm (\%)',
                                [[*r[:2], r'\textbf{' + r[2] + '}', r'\textbf{' + r[3] + '}'] for r in effect_rows])
    tables['likelihood_intervals'] = tabular('llcc',r'Domain & Model & \Union{} addition & \Union{} removal',detailed)
    tables['generation'] = tabular('llcc',r'Domain & Model & Remove \Union & Add \Union',gen_rows)
    tables['probes'] = tabular('lllrlcl',r'Domain & Model & Probe & Pairs & Raw shift & Adjusted shift [95\% CI] & Increased',probe_rows)
    tables['off_domain'] = tabular('lccc',r'Model & Harmful model & Remove \Core & Add \Core',
                                  [[*r[:2], *[r'\textbf{' + v + '}' for v in r[2:]]] for r in off_rows])
    control_rows = [[*r[:7], r'\textbf{' + r[7] + '}', r[8]] if r[2]=='CORE' else r for r in control_rows]
    tables['random_controls'] = tabular('lllcrrrcr',r'Domain & Model & Set & Edit & Draws & Median & Maximum & Selected minus draw mean & $p$',control_rows)
    tables['coherence'] = tabular('llrrrr',r'Domain & Model & Benign & Harmful & Remove \Core & Add \Core',
                                 [[*r[:4], *[r'\textbf{' + v + '}' for v in r[4:]]] for r in judge_rows])
    for tag, model, seed in [('qwen_med43','Qwen2.5-7B','43'),('llama_med43','Llama-3.1-8B','43')]:
        r=read(REL/f'rand_{tag}/rand_summary.csv');g=read(REL/f'gen_{tag}/rel_rates.csv')
        seed_rows.append([model,seed,*[interval(one(r,set='CORE',dir=d,quantity='EM')) for d in ['T','D']],
                         *[f"{100*f(one(g,scope='all',condition=x,outcome='panel_em'),'estimate'):.1f}" for x in ['E0','D:CORE','T:CORE']]])
    seed_rows = [[*r[:2], *[r'\textbf{' + v + '}' for v in r[2:4]], r[4], *[r'\textbf{' + v + '}' for v in r[5:]]] for r in seed_rows]
    tables['seeds']=tabular('llccrrr',r'Model & Seed & \Core{} addition & \Core{} removal & Harmful & Remove \Core & Add \Core',seed_rows)
    geometric, pca_rows = [], []
    older = ROOT / 'eval_runs/persona_control_stage7'
    for model_id, _, model_name, _ in CELLS[:3]:
        probe_id = dict((tag, probe) for tag, _, _, probe in CELLS)[model_id]
        pca = read_json(older / f'w1_corrections/stage6a_activation/{probe_id}/summary.json')
        best = pca['best_original_rule_with_corrected_nec']
        pca_rows.append([model_name, str(int(best['layer'])), str(int(best['k'])),
                         f"{best['suff']:.3f}", f"{best['nec']:.3f}"])
        ranks = read(older / f'w6_compactness/{probe_id}/hindsight_seed42/rank_curve.csv')
        geometric.append({'model': model_name, 'pca': pca,
                          'svd': [r for r in ranks if r['kind']=='learned']})
    tables['geometric_pca'] = tabular('lrrrr',r'Model & Layer & Dim. & Suff. & Nec.',pca_rows)
    tables.update(confirmatory_tables())
    text=PAPER.read_text()
    for name,table in tables.items():
        pattern=rf'(?s)(% BEGIN GENERATED {name}\n).*?(% END GENERATED {name})'
        text,n=re.subn(pattern,lambda m:m[1]+table+'\n'+m[2],text)
        assert n==1,(name,n)
    PAPER.write_text(text)
    json.dump({'source_sha256':SOURCES,'cells':cells,'geometric':geometric},(DERIVED/'sources_and_plot_data.json').open('w'),indent=2)
    with (DERIVED/'main_table.csv').open('w') as fp:
        w=csv.writer(fp);w.writerow(['domain','model','core_count','core_norm_pct','union_T','union_D','core_T','core_D','peri_T','peri_D']);w.writerows(effect_rows)
    plots(cells, geometric)
    PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
    for name in ['method', 'generation', 'specificity', 'geometric_svd']:
        for extension in ['pdf', 'png']:
            shutil.copy2(OUT / f'{name}.{extension}', PAPER_FIGURES / f'{name}.{extension}')
    print(f'Updated {len(tables)} tables; wrote figures to {OUT}; {len(SOURCES)} source files recorded.')


def plots(cells, geometric):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyBboxPatch

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})

    def save(fig, name):
        for extension in ['pdf', 'png']:
            fig.savefig(OUT / f'{name}.{extension}', bbox_inches='tight',
                        pad_inches=.06, dpi=300, facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.65), sharey=True)
    colors = ['#b45f2a', '#696c70', '#5b86b3']
    for ax, group, title in zip(axes, [cells[:3], cells[3:]], ['Medical', 'Financial']):
        for i, c in enumerate(group):
            rr = np.asarray(c['random_rates']) * 100
            ax.vlines(i, rr.min(), rr.max(), color='#c9ccd0', lw=7, alpha=.55)
            ax.plot([i-.06, i+.06], [np.median(rr)]*2, color='#696c70', lw=2)
            for offset, cond, color, marker in [(-.23, 'E0', colors[0], 'o'),
                                                (0, 'D:CORE', colors[1], 's'),
                                                (.23, 'T:CORE', colors[2], '^')]:
                r = c['rates'][cond]
                v = 100 * f(r, 'estimate')
                ax.errorbar(i+offset, v,
                            yerr=[[v-100*f(r, 'lo')], [100*f(r, 'hi')-v]],
                            fmt=marker, color=color, capsize=3, ms=6, lw=1.3)
        ax.set_xticks(range(3), ['Qwen2.5\n7B', 'Llama-3.1\n8B', 'Qwen3\n1.7B'])
        ax.tick_params(axis='both', labelsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylim(-1.3, 41)
        ax.set_yticks([0, 10, 20, 30, 40])
        ax.grid(axis='y', color='#d8dde4', lw=.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel('Judged misalignment (%)', fontsize=12)
    handles = [Line2D([], [], color=c, marker=m, lw=0, markersize=7, label=l)
               for c, m, l in zip(colors, ['o', 's', '^'],
                                  ['Harmful model', 'Remove CORE', 'Add CORE'])]
    handles.append(Line2D([], [], color='#c9ccd0', lw=7, alpha=.6,
                          label='Matched removals: range'))
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
               fontsize=11, columnspacing=1.4, handletextpad=.6,
               bbox_to_anchor=(.5, 1.0))
    fig.subplots_adjust(left=.12, right=.99, bottom=.18, top=.73, wspace=.12)
    save(fig, 'generation')

    # Same-run CORE and equal-count geometric controls, shown in both directions.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 4.75), sharey=True)
    y = np.arange(len(cells))
    for ax, direction, title in zip(axes, ['T', 'D'], ['Sufficiency (addition)', 'Necessity (removal)']):
        rows = [c['magnitude_comparisons'][f'{direction}:CORE'] for c in cells]
        for name, key, offset, color, edge in [('CORE', 'frac', -.24, '#e3b89c', '#b45f2a'),
                                              ('Largest SVD', 'top', 0, '#c3d3e3', '#5b86b3'),
                                              ('Random leading SVD', 'rt_med', .24, '#dedfe1', '#8a8e93')]:
            values = np.array([f(r, key) for r in rows])
            ax.barh(y+offset, values, height=.20, color=color, edgecolor=edge,
                    linewidth=.8, label=name)
            if key != 'top':
                low_key, high_key = ('lo','hi') if key=='frac' else ('rt_min','rt_max')
                lower = np.array([f(r, low_key) for r in rows])
                upper = np.array([f(r, high_key) for r in rows])
                ax.errorbar(values, y+offset, xerr=[values-lower, upper-values],
                            fmt='none', ecolor=edge, capsize=2.3, lw=1.1)
        ax.axvline(0, color='#62676d', lw=1)
        ax.axhline(2.5, color='#d9dadd', lw=.8)
        ax.set_xlim(0, 1.10)
        ax.set_xticks([0, .25, .5, .75, 1.0], ['0', '.25', '.50', '.75', '1'])
        ax.tick_params(labelsize=11)
        ax.set_title(title, fontsize=11.5, fontweight='bold', pad=10)
        ax.grid(axis='x', color='#e5e7e9', lw=.6)
        ax.set_axisbelow(True)
        ax.set_xlabel('Fraction of broad likelihood shift', fontsize=10.5)
    axes[0].set_yticks(y, [f"{c['domain']}\n{c['model']}" for c in cells], fontsize=10.5)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc='upper center',
               ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(.60, 1.04))
    fig.subplots_adjust(left=.28, right=.98, bottom=.14, top=.84, wspace=.15)
    save(fig, 'specificity')

    fig, ax = plt.subplots(figsize=(6.8, 2.9))
    for item, color, marker in zip(geometric, ['#b45f2a', '#5b86b3', '#696c70'], ['o','s','^']):
        rows = sorted(item['svd'], key=lambda r:int(r['rank']))
        x = [int(r['rank']) for r in rows]
        values = np.array([f(r,'F_direct') for r in rows])
        lower = np.array([f(r,'F_direct_lo') for r in rows])
        upper = np.array([f(r,'F_direct_hi') for r in rows])
        ax.plot(x, values, color=color, marker=marker, ms=5, lw=1.7, label=item['model'])
        ax.fill_between(x, lower, upper, color=color, alpha=.13)
    ax.axhline(1, color='#8a8e93', ls='--', lw=1)
    ax.set_xscale('log', base=2)
    ax.set_xticks([1,2,4,8,16,32,64],['1','2','4','8','16','32','64'])
    ax.set_ylim(.52, 1.08)
    ax.set_xlabel('SVD rank per matrix', fontsize=12)
    ax.set_ylabel('Fraction of regional\ndirect gap transferred', fontsize=11)
    ax.grid(axis='y',color='#e5e7e9',lw=.7)
    ax.legend(loc='lower right',frameon=False,ncol=1,fontsize=10.5)
    fig.tight_layout()
    save(fig, 'geometric_svd')

    draw_method(plt, save)


def draw_method(plt, save):
    """Draw a vector method overview; illustrated component geometry is schematic."""
    import numpy as np
    from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch, PathPatch
    from matplotlib.path import Path as VectorPath

    fig, ax = plt.subplots(figsize=(7.0, 6.0))
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 6)
    ax.axis('off')
    ink, muted, line = '#303235', '#696c70', '#d9dadd'
    blue, gold, core = '#477fa7', '#ac8958', '#b76832'
    pale_blue, pale_gold, pale_core = '#eaf2f7', '#f7f1e7', '#fff0df'

    def text(x, y, label, size=10.5, color=ink, ha='left', va='center', weight='normal', **kwargs):
        return ax.text(x, y, label, fontsize=size, color=color, ha=ha, va=va,
                       fontweight=weight, linespacing=1.25, **kwargs)

    def rounded(x, y, w, h, fc='white', ec=line, lw=.8, dashed=False):
        p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.015,rounding_size=.055',
                          facecolor=fc, edgecolor=ec, linewidth=lw,
                          linestyle=(0, (4, 3)) if dashed else 'solid')
        ax.add_patch(p)
        return p

    def arrow(a, b, color=muted, curved=0, lw=.9):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='->', mutation_scale=11,
                                    linewidth=lw, color=color,
                                    connectionstyle=f'arc3,rad={curved}'))

    def panel(y, h, letter, title):
        rounded(.04, y, 6.92, h)
        text(.21, y+h-.23, letter, 12, color=muted, weight='bold')
        text(.49, y+h-.23, title, 12, weight='bold')

    def stack(x, y, width=.48, height=.38, accent=None):
        for k in range(4):
            yy = y+k*.075
            rounded(x+.035*k, yy, width, .12, fc='#f6f6f5', ec='#bfc2c6', lw=.7)
        if accent:
            ax.add_patch(Rectangle((x+.09, y+.17), width-.03, .045,
                                   facecolor=accent, edgecolor='none'))

    def grid(x, y, width=.62, n=5, palette=None):
        values = np.asarray([[.4, -.2, .7, -.6, .2], [-.5, .8, -.3, .4, -.1],
                             [.3, -.7, .2, .6, -.4], [.6, .1, -.8, .2, .5],
                             [-.1, .4, .5, -.2, -.6]])
        colors = ['#c5d8e5', '#e3edf3', '#f5f5f4', '#e8d9c5', '#c6a780']
        for r in range(n):
            for c in range(n):
                idx = int(np.clip((values[r, c]+1)*2.5, 0, 4))
                ax.add_patch(Rectangle((x+c*width/n, y+r*width/n), width/n, width/n,
                                       fc=colors[idx], ec='white', lw=.6))
        ax.add_patch(Rectangle((x, y), width, width, fill=False, ec='#bfc2c6', lw=.7))

    def component(x, y, width=.43, index=0):
        a = np.array([.3, .6, .9, .5])
        b = np.roll(np.array([.2, .7, 1., .4]), index)
        for r in range(4):
            for c in range(4):
                ax.add_patch(Rectangle((x+c*width/4, y+r*width/4), width/4, width/4,
                                       fc=blue, alpha=.12+.65*a[r]*b[c], ec='white', lw=.45))
        ax.add_patch(Rectangle((x, y), width, width, fill=False, ec='#aebdc7', lw=.6))

    # A: paired models, the matrix difference, and visually distinct low-rank atoms.
    panel(4.36, 1.60, 'A', 'Decompose the harmful–benign weight difference')
    rounded(.23, 4.92, .50, .56, fc='#f8f8f7')
    for k in range(3):
        ax.plot([.33, .63], [5.33-.12*k]*2, color='#c4c6c9', lw=1)
    text(.48, 4.65, 'Same\nquestions', ha='center', color=muted)
    arrow((.77, 5.23), (1.05, 5.40))
    arrow((.77, 5.19), (1.05, 4.90))
    stack(1.10, 5.19, accent=gold)
    stack(1.10, 4.69, accent=blue)
    text(1.82, 5.40, 'E  harmful', color=gold)
    text(1.82, 4.90, 'C  benign', color=blue)
    arrow((2.78, 5.37), (3.27, 5.15), curved=-.12)
    arrow((2.78, 4.88), (3.27, 5.08), curved=.12)
    grid(3.35, 4.87, .55)
    text(3.63, 4.62, r'$\Delta W = W_E - W_C$', ha='center')
    arrow((4.05, 5.15), (4.69, 5.15))
    text(4.39, 5.44, 'SVD', ha='center', color=muted)
    for i, x in enumerate([4.91, 5.49, 6.07]):
        component(x, 4.94, .43, i)
    text(5.70, 4.65, r'$s_r\,u_r v_r^{\mathsf{T}}$', size=12, ha='center')
    text(5.70, 5.49, '32 components / matrix', ha='center', color=muted)

    # B: two selections applied to the same component identities.
    panel(1.88, 2.34, 'B', 'Select the shared parameter CORE')
    text(.25, 3.80, 'Integrated-gradients ranking', color=muted)
    text(.25, 3.53, 'Four behavioral probes', weight='bold', color=blue)
    for r in range(4):
        yy=3.36-r*.115
        ax.scatter([.30], [yy], s=15, color=blue, zorder=4)
        ax.plot([.42, .95+.12*(r%2)], [yy, yy], color=blue, alpha=.65, lw=2.1)
    text(1.18, 3.18, 'Top 256\nper probe', color=muted)
    arrow((2.17, 3.32), (3.13, 3.50), color=blue, curved=-.10)
    text(.25, 2.76, 'Training-answer pairs', weight='bold', color=gold)
    rounded(.27, 2.24, .51, .32, pale_gold, ec='#d8c7ae')
    rounded(.88, 2.24, .51, .32, '#f5f5f3', ec='#d5d5d1')
    for x in [.35, .96]:
        ax.plot([x, x+.32], [2.44]*2, color='#b7b1a7', lw=1)
        ax.plot([x, x+.25], [2.35]*2, color='#b7b1a7', lw=1)
    text(1.58, 2.40, 'Top 256', color=muted)
    ax.plot([2.40, 6.25, 6.25], [2.47, 2.47, 2.91], color=gold, lw=.9)
    arrow((6.25, 2.91), (6.01, 3.13), color=gold)

    left, right, cy, rx, ry = 4.02, 5.02, 3.13, .97, .55
    ax.add_patch(Ellipse((left,cy), 2*rx, 2*ry, fc=pale_blue, ec=blue, lw=1))
    ax.add_patch(Ellipse((right,cy), 2*rx, 2*ry, fc=pale_gold, ec=gold, lw=1, alpha=.82))
    theta = np.arccos((right-left)/(2*rx))
    a = np.linspace(-theta, theta, 60)
    b = np.linspace(np.pi-theta, np.pi+theta, 60)
    lens = np.vstack([np.column_stack([left+rx*np.cos(a), cy+ry*np.sin(a)]),
                      np.column_stack([right+rx*np.cos(b), cy+ry*np.sin(b)])])
    lens_path = VectorPath(np.vstack([lens, lens[0]]),
                           [VectorPath.MOVETO]+[VectorPath.LINETO]*(len(lens)-1)+[VectorPath.CLOSEPOLY])
    ax.add_patch(PathPatch(lens_path, fc=pale_core, ec='none', alpha=.65))
    # Color only points belonging to both sets as CORE.
    for yy in np.arange(2.73,3.59,.20):
        for xx in np.arange(3.19,5.90,.22):
            in_u=((xx-left)/rx)**2+((yy-cy)/ry)**2 < 1.0
            in_h=((xx-right)/rx)**2+((yy-cy)/ry)**2 < 1.0
            if in_u or in_h:
                ax.scatter(xx, yy, s=17 if in_u and in_h else 12,
                           color=core if in_u and in_h else (blue if in_u else gold),
                           alpha=1 if in_u and in_h else .65, zorder=4)
    text(3.63, 3.80, 'UNION', color=blue, weight='bold', ha='center')
    text(5.49, 3.80, 'H', color=gold, weight='bold', ha='center')
    ax.add_patch(PathPatch(lens_path, fc='none', ec=core, lw=1.1,
                           linestyle=(0, (4, 3)), zorder=5))
    ax.plot([left+rx,6.10,6.10], [3.13,3.13,3.49], color=core, lw=.9)
    text(6.13, 3.61, 'CORE', color=core, weight='bold', ha='center')
    text(4.60, 2.25, 'CORE = UNION ∩ H', size=11.5, color=core, ha='center')
    text(3.50, 2.03, 'Broad-assay pairs excluded from selection', color=muted, ha='center')

    # C: rank-one edits injected into, or removed from, the two host models.
    panel(.04, 1.70, 'C', 'Intervene on CORE; test broad misalignment')
    stack(.29, .86, accent=blue)
    stack(.29, .31, accent=gold)
    text(.92, 1.04, 'C', color=blue, weight='bold')
    text(.92, .49, 'E', color=gold, weight='bold')
    for yy, sign in [(1.04,'+'),(.49,'−')]:
        text(1.25, yy, sign, size=15, ha='center')
        rounded(1.53, yy-.17, .59, .34, pale_core, ec=core, dashed=True)
        for k in range(3):
            ax.scatter(1.65+.17*k, yy, color=core, s=17, zorder=4)
        arrow((2.24, yy), (2.78, yy))
    stack(2.83,.86,accent=core)
    stack(2.83,.31)
    text(3.57, 1.04, 'Addition', color=core)
    text(3.57, .49, 'Removal', color=core)
    ax.plot([4.35,4.61,4.61,4.35], [1.02,1.02,.48,.48], color=muted, lw=.9)
    arrow((4.61,1.02), (4.86,1.02))
    arrow((4.61,.48), (4.86,.48))
    # Measurement icons carry no illustrative numerical outcomes.
    for x,h,c in [(5.02,.24,blue),(5.18,.37,gold)]:
        ax.add_patch(Rectangle((x,.88), .09,h,fc=c,ec='none'))
    ax.plot([4.97,5.34],[.87,.87],color='#babdc1',lw=.8)
    text(5.48,1.04,'Likelihood',weight='bold')
    rounded(4.97,.29,.42,.32,fc='#f8f8f7')
    ax.plot([5.04,5.28],[.49,.49],color='#afb3b6',lw=1)
    ax.plot([5.04,5.23],[.40,.40],color='#afb3b6',lw=1)
    text(5.48,.48,'Generation',weight='bold')
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    save(fig,'method')


if __name__=='__main__':
    main()
