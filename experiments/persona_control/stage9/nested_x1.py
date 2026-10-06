"""X1 (plan Part 19): broad-minus-narrow and N_core-minus-baseline contrasts from the saved core_confirmatory_20261005
likelihood rows, and the attribution figures.  CPU only; no new scoring.

N_core is the confirmatory H_K (the |CORE| components with the largest positive a_N; nested_protocol.py asserts the
identity).  Fractions follow confirmatory_likelihood.py: transfer T = R(C0 + S) / R(E0), removal
D = [R(E0) - R(E0 - S)] / R(E0), where R is the residual shift of the broad assay (ref_cartoon120) or of the held-out
training answers (ref_meddata or ref_findata).  Intervals: 2,000 draws (seed 0) of step3_report.Stats, which resample
questions within each question group (broad and narrow independently), use the same draws for every condition and refit
the control lines on every draw.  Baselines of the confirmatory run: B_K = behavior-only selection (Part 14; not the
broad ranking B_k of X2), SVD_EC_K and SVD_EB_K = the |CORE| largest singular components of E - C and E - base, and for
removal the norm-matched SVD_EC_NORM, SVD_EB_NORM and ROLLBACK_NORM.
Usage: python nested_x1.py
"""
import json

import numpy as np
import pandas as pd

import nested_protocol as N
import step3_report as R

OUT = N.RUN_DIR / 'x1'
BASELINES = {'T': ['CORE', 'B_K', 'SVD_EC_K', 'SVD_EB_K'],
             'D': ['CORE', 'B_K', 'SVD_EC_K', 'SVD_EB_K', 'SVD_EC_NORM', 'SVD_EB_NORM', 'ROLLBACK_NORM']}
LABELS = {'qwen2_5_7b': 'Qwen2.5-7B', 'llama3_1_8b': 'Llama-3.1-8B', 'qwen3_1_7b': 'Qwen3-1.7B'}
INK, MUTED, GRID = '#303235', '#696c70', '#e5e7e9'
SERIES = {'N': '#b45f2a', 'G': '#3a74b0', 'B': '#2f9a7d'}  # validate_palette.js --pairs all: all checks pass (light)
STYLE = {'N': '-', 'B': '--', 'G': '-.'}


def iv(v):
    lo, hi = R.ci(v)
    return dict(estimate=float(v[0]), lo=float(lo), hi=float(hi))


def contrasts(model):
    folder = N.CONF / 'likelihood' / model
    llr, _ = R.load(str(folder))
    st = R.Stats(llr, 2000, seed=0)
    shifts = {'broad': st.R(['ref_cartoon120']), 'narrow': st.R(['ref_findata' if model.endswith('_fin') else 'ref_meddata'])}
    frac = {}
    for shift, res in shifts.items():
        e0 = res[:, st.col('E0')]
        for cond in st.conds:
            if ':' in cond:
                x = res[:, st.col(cond)]
                frac[shift, cond] = (x if cond[0] == 'T' else e0 - x) / e0
    rows = []
    for cond in [c for c in st.conds if ':' in c]:
        for shift in shifts:
            rows.append(dict(model=model, kind='fraction', direction=cond[0], a=cond[2:], b='', shift=shift, **iv(frac[shift, cond])))
        rows.append(dict(model=model, kind='broad_minus_narrow', direction=cond[0], a=cond[2:], b='', shift='broad-narrow',
                         **iv(frac['broad', cond] - frac['narrow', cond])))
    for d, names in BASELINES.items():
        for b in names:
            for shift in shifts:
                rows.append(dict(model=model, kind='n_core_minus_baseline', direction=d, a='H_K', b=b, shift=shift,
                                 **iv(frac[shift, f'{d}:H_K'] - frac[shift, f'{d}:{b}'])))
    return rows, N.sha(folder / 'rows.parquet')


def attribution(model):
    s3 = N.source_dir(model)
    meta = json.loads((s3 / 'components.json').read_text())['components']
    attr = pd.read_parquet(s3 / 'attr_components.parquet').sort_values('id')
    aN, aB = attr.MD.to_numpy(float), attr.EM.to_numpy(float)
    s = np.array([m['s'] for m in meta])
    return aN, aB, s


def curve(order, values, total, positive_only):
    v = values[order]
    if positive_only is not None:
        v = v[positive_only[order] > 0]
    return np.cumsum(v) / total


def figures(sets):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                         'axes.linewidth': .7})
    N.FIG_DIR.mkdir(parents=True, exist_ok=True)
    models = list(N.GRID)
    curves, panel = [], []
    # Figure 1: a_B against a_N for every candidate component; N_core and the broad-ranked set of the same size.
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.9), sharex=False, sharey=False)
    for ax, model in zip(axes.flat, models):
        aN, aB, _ = attribution(model)
        nc, bc = set(sets[model]['N_core']['ids']), set(sets[model]['B_core']['ids'])
        rest = np.array([i for i in range(len(aN)) if i not in nc and i not in bc])
        bonly = np.array(sorted(bc - nc))
        ncore = np.array(sorted(nc))
        ax.scatter(aN[rest], aB[rest], s=3, c='#b9bcc0', lw=0, rasterized=True)
        ax.scatter(aN[bonly], aB[bonly], s=13, c=SERIES['B'], marker='^', lw=0, rasterized=True)
        ax.scatter(aN[ncore], aB[ncore], s=9, c=SERIES['N'], marker='o', lw=0, rasterized=True)
        for setter in (ax.set_xscale, ax.set_yscale):
            setter('symlog', linthresh=1e-3, linscale=.4)
        ax.axhline(0, color=GRID, lw=.7, zorder=0)
        ax.axvline(0, color=GRID, lw=.7, zorder=0)
        hi = max(aN.max(), aB.max()) * 1.6
        ax.set_xlim(-.4, hi)
        ax.set_ylim(-.4, hi)
        ax.set_xticks([-.1, 0, .01, .1, 1])
        ax.set_yticks([-.1, 0, .01, .1, 1])
        ax.set_xticklabels(['−.1', '0', '.01', '.1', '1'])
        ax.set_yticklabels(['−.1', '0', '.01', '.1', '1'])
        ax.tick_params(labelsize=9.5, length=2.5, width=.6)
        from scipy.stats import spearmanr
        panel.append(dict(model=model, components=len(aN), pearson=float(np.corrcoef(aN, aB)[0, 1]),
                          spearman=float(spearmanr(aN, aB)[0]), n_core=len(nc),
                          n_core_share_broad=sets[model]['N_core']['attribution_share_broad'],
                          n_core_share_narrow=sets[model]['N_core']['attribution_share_narrow'],
                          b_core_share_broad=sets[model]['B_core']['attribution_share_broad'],
                          b_core_share_narrow=sets[model]['B_core']['attribution_share_narrow'],
                          overlap_n_core_b_core=len(nc & bc), b_core_outside_with_negative_a_N=int((aN[bonly] <= 0).sum())))
        base = model.removesuffix('_fin')
        ax.set_title(f"{LABELS[base]}, {'financial' if model.endswith('_fin') else 'medical'}", fontsize=10.5, color=INK, pad=4)
    for ax in axes[1]:
        ax.set_xlabel('narrow attribution a_N', fontsize=10.5)
    for ax in axes[:, 0]:
        ax.set_ylabel('broad attribution a_B', fontsize=10.5)
    handles = [Line2D([], [], ls='', marker='o', ms=4.5, color=SERIES['N'], label='N_core (largest a_N, k = |CORE|)'),
               Line2D([], [], ls='', marker='^', ms=5, color=SERIES['B'], label='largest a_B, k = |CORE|, not in N_core'),
               Line2D([], [], ls='', marker='o', ms=3.5, color='#b9bcc0', label='other components')]
    fig.tight_layout(h_pad=.8, w_pad=.6, rect=(0, 0, 1, .88))
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, 1.0),
               handletextpad=.3, columnspacing=1.2)
    for ext in ('pdf', 'png'):
        fig.savefig(N.FIG_DIR / f'x1_attribution_scatter.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    # Figure 2: share of the total broad attribution held by the k strongest components of each ranking.
    fig, axes = plt.subplots(2, 3, figsize=(6.75, 4.6), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, models):
        aN, aB, s = attribution(model)
        n = len(aN)
        orders = {'N': np.lexsort((np.arange(n), -aN)), 'B': np.lexsort((np.arange(n), -aB)), 'G': np.lexsort((np.arange(n), -s))}
        positive = {'N': aN, 'B': aB, 'G': None}
        for key in ('G', 'B', 'N'):
            broad = curve(orders[key], aB, aB.sum(), positive[key])
            narrow = curve(orders[key], aN, aN.sum(), positive[key])
            k = np.arange(1, len(broad) + 1)
            ax.plot(k, broad, color=SERIES[key], lw=1.6, ls=STYLE[key])
            for kk in sorted(set(N.KS) | {N.GRID[model][2]}):
                if kk <= len(broad):
                    curves.append(dict(model=model, ranking=key, k=kk, broad_share=float(broad[kk - 1]),
                                       narrow_share=float(narrow[kk - 1])))
        ax.axvline(N.GRID[model][2], color=MUTED, lw=.8, ls=':')
        ax.set_xscale('log')
        ax.set_xlim(1, 4096)
        ax.set_ylim(-.05, 1.5)
        ax.set_xticks([1, 10, 100, 1000])
        ax.set_xticklabels(['1', '10', '100', '1,000'])
        ax.set_yticks([0, .5, 1, 1.5])
        ax.set_yticklabels(['0', '.5', '1', '1.5'])
        ax.grid(axis='y', color=GRID, lw=.6)
        ax.tick_params(labelsize=9.5, length=2.5, width=.6)
        base = model.removesuffix('_fin')
        ax.set_title(f"{LABELS[base]}, {'financial' if model.endswith('_fin') else 'medical'}", fontsize=10.5, color=INK, pad=4)
    for ax in axes[1]:
        ax.set_xlabel('k (components, ranked)', fontsize=10.5)
    for ax in axes[:, 0]:
        ax.set_ylabel('share of total a_B', fontsize=10.5)
    handles = [Line2D([], [], color=SERIES[k], lw=1.8, ls=STYLE[k], label=t) for k, t in
               (('N', 'ranked by narrow attribution a_N'), ('B', 'ranked by broad attribution a_B'),
                ('G', 'ranked by singular value'))] + [Line2D([], [], color=MUTED, lw=.8, ls=':', label='k = |CORE|')]
    fig.tight_layout(h_pad=.8, w_pad=.6, rect=(0, 0, 1, .88))
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, 1.0),
               handlelength=2.2, columnspacing=1.2)
    for ext in ('pdf', 'png'):
        fig.savefig(N.FIG_DIR / f'x1_attribution_curves.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)
    return curves, panel


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, hashes = [], {}
    for model in N.GRID:
        r, h = contrasts(model)
        rows += r
        hashes[str(N.CONF / 'likelihood' / model / 'rows.parquet')] = h
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / 'x1_contrasts.csv', index=False)
    # Cross-check against the post hoc contrast of progress Part 34 (arr_reframe_analysis.py).
    ref = pd.read_csv(N.MAIN / 'eval_runs/persona_control_stage9/arr_reframe_20261005/broad_narrow_contrasts.csv')
    mine = frame[frame.kind == 'broad_minus_narrow'].set_index(['model', 'a', 'direction'])
    gaps = [max(abs(mine.loc[(r.model, r.selection, r.direction), k] - getattr(r, j)) for k, j in
                (('estimate', 'broad_minus_narrow'), ('lo', 'lo'), ('hi', 'hi'))) for r in ref.itertuples()]
    check = dict(reference='eval_runs/persona_control_stage9/arr_reframe_20261005/broad_narrow_contrasts.csv',
                 rows_compared=len(ref), max_abs_difference=float(max(gaps)))
    sets = {m: N.sets(m) for m in N.GRID}
    curves, panel = figures(sets)
    pd.DataFrame(curves).to_csv(OUT / 'x1_attribution_curves.csv', index=False)
    pd.DataFrame(panel).to_csv(OUT / 'x1_attribution_summary.csv', index=False)
    bn = frame[(frame.kind == 'broad_minus_narrow') & (frame.a == 'H_K')].set_index(['model', 'direction'])
    per = {m: {d: dict(estimate=float(bn.loc[(m, d), 'estimate']), lo=float(bn.loc[(m, d), 'lo']),
                       hi=float(bn.loc[(m, d), 'hi']), above_zero=bool(bn.loc[(m, d), 'lo'] > 0)) for d in 'DT'} for m in N.GRID}
    verdict = {'1': dict(settings_holding=sum(per[m]['D']['above_zero'] for m in N.GRID), required=6),
               '2': dict(settings_holding=sum(per[m]['T']['above_zero'] for m in N.GRID), required=5)}
    for v in verdict.values():
        v['holds'] = v['settings_holding'] >= v['required']
    N.dump(dict(per_setting=per, verdict=verdict, predictions=N.PREDICTIONS['X1'], rules=N.DECISION_RULES,
                crosscheck_part34=check), OUT / 'x1_predictions.json')
    N.dump(dict(sources=hashes, bootstrap_draws=2000, seed=0, driver_sha256=N.sha(__file__),
                figures=[str(N.FIG_DIR / f'{n}.{e}') for n in ('x1_attribution_scatter', 'x1_attribution_curves')
                         for e in ('pdf', 'png')]), OUT / 'x1_manifest.json')
    print(json.dumps(dict(verdict=verdict, crosscheck=check), indent=1))


if __name__ == '__main__':
    main()
