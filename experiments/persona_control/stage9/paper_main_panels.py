"""Regenerate independent vector panels for the three main-text figures from saved results."""
from pathlib import Path
import shutil
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from paper_fonts import configure_fonts
from paper_localizing import ROOT, NEST, SETTINGS, KS, SERIES, PALE, INK, GRID, read, load_x2

OUT = ROOT / 'figures/persona_control/main_panels_20261006'
DEST = ROOT / 'paper_draft/arr_2026/figures'
INCLUDED = set(re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',
                          (DEST.parent / 'main.tex').read_text()))
configure_fonts()
plt.rcParams.update({'font.size': 10, 'axes.labelsize': 10,
                     'xtick.labelsize': 9, 'ytick.labelsize': 9, 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': INK, 'text.color': INK, 'axes.labelcolor': INK})

def save(fig, name):
    fig.savefig(OUT / (name + '.pdf'))
    if name + '.pdf' in INCLUDED:
        shutil.copy2(OUT / (name + '.pdf'), DEST / (name + '.pdf'))
    plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x = read(NEST / 'x1/x1_contrasts.csv')
    frac = x[x.kind == 'fraction'].set_index(['model', 'direction', 'a', 'shift'])
    x2 = load_x2()
    rows = [('H_K', r'$\mathcal{N}_k$'), ('CORE', 'Core'), ('B_K', r'$\mathcal{P}_k$'),
            ('SVD_EC_K', r'$\mathcal{G}_k$'), ('SVD_EB_K', r'$\mathcal{G}^{base}_k$'), ('R1', 'First')]
    for m, domain, model in SETTINGS:
        fig, ax = plt.subplots(figsize=(2.3, 1.9))
        fig.subplots_adjust(left=.24, right=.97, bottom=.24, top=.97)
        for i, (key, label) in enumerate(rows):
            col = SERIES['N'] if key == 'H_K' else '#696c70'
            for direction, offset in [('T', -.16), ('D', .16)]:
                if key == 'R1':
                    val, lo, hi = x2[(m, direction, key, 'broad')]
                else:
                    r = frac.loc[(m, direction, key, 'broad')]
                    val, lo, hi = r.estimate, r.lo, r.hi
                ax.plot([lo, hi], [i+offset]*2, color=col, lw=1)
                ax.plot(val, i+offset, 'o', ms=4, mec=col, mfc='white' if direction == 'T' else col)
        ax.set(yticks=range(6), yticklabels=[v for _, v in rows], ylim=(5.6, -.6),
               xlim=(-.06, 1.15), xticks=[0, .5, 1], xlabel='Fraction of broad shift')
        ax.grid(axis='x', color=GRID)
        ax.set_axisbelow(True)
        save(fig, 'main_selection_' + m)
        fig, ax = plt.subplots(figsize=(2.3, 1.7))
        fig.subplots_adjust(left=.25, right=.90, bottom=.29, top=.97)
        for key, shift, color, ls, label in [
            ('N', 'broad', SERIES['N'], '-', r'$\mathcal{N}_k$: broad'),
            ('N', 'narrow', PALE, '-', r'$\mathcal{N}_k$: narrow'),
            ('B', 'broad', SERIES['B'], '--', r'$\mathcal{B}_k$: broad'),
            ('G', 'broad', SERIES['G'], '-.', r'$\mathcal{G}_k$: broad')]:
            ax.plot(KS, [x2[(m, 'D', f'{key}_{k}', shift)][0] for k in KS],
                    color=color, ls=ls, marker='o', ms=2.7, lw=1.5, label=label)
        ax.set_xscale('log', base=2)
        ax.set(xticks=[16,64,256,1024], xticklabels=['16','64','256','1,024'],
               yticks=[0,.5,1], ylim=(-.04,1.18), xlabel='Components removed', ylabel='Fraction removed')
        ax.yaxis.label.set_fontsize(9)
        ax.yaxis.labelpad = 1
        ax.grid(color=GRID)
        ax.set_axisbelow(True)
        handles, labels = ax.get_legend_handles_labels()
        save(fig, 'main_sweep_' + m)
    fig = plt.figure(figsize=(7, .26))
    fig.legend(handles, labels, loc='center', ncol=4, frameon=False, fontsize=10)
    save(fig, 'main_sweep_legend')
    fig, ax = plt.subplots(figsize=(7, 1.05))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set(xlim=(0, 7), ylim=(0, 1)); ax.axis('off')
    texts = ['Matched harmful\nand benign models', 'SVD of their\nweight difference', 'Rank components\non narrow answers', 'Add to benign /\nremove from\nharmful', 'Measure narrow\nand broad effects']
    for i, text in enumerate(texts):
        xx = .02 + i*1.42
        ax.add_patch(FancyBboxPatch((xx,.2),1.26,.65,boxstyle='round,pad=.015',facecolor='#f2eee9',edgecolor='#a58d78',lw=.8))
        ax.text(xx+.63,.525,text,ha='center',va='center',fontsize=9)
        if i<4: ax.annotate('',xy=(xx+1.4,.525),xytext=(xx+1.28,.525),arrowprops={'arrowstyle':'->','color':INK})
    save(fig, 'main_method')
    narrow = np.exp(-np.arange(18)/8)
    arrangements = {'separate': np.r_[np.zeros(13),[.95,.8,.65,.5,.4]],
                    'proportional': narrow, 'concentrated': np.r_[[1,.9,.78,.66,.5],np.full(13,.04)]}
    for name, broad in arrangements.items():
        fig, ax = plt.subplots(figsize=(2.3, 1.15))
        fig.subplots_adjust(left=.08,right=.97,bottom=.32,top=.94)
        ax.plot(narrow,color=PALE,lw=2,label='Narrow')
        ax.plot(broad,color=SERIES['N'],lw=1.5,ls='--',label='Broad')
        ax.set(xticks=[],yticks=[],xlabel='Components in narrow rank order',ylim=(-.04,1.1))
        ax.xaxis.label.set_fontsize(8.5)
        if name=='separate': ax.legend(frameon=False,fontsize=8,loc='upper center',handlelength=1)
        save(fig, 'main_arrangement_' + name)

if __name__ == '__main__':
    main()
