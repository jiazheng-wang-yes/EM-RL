"""Rebuild appendix figures with Figure 1's palette; preserve all manuscript tables."""
import json
import shutil
from pathlib import Path
import paper_localizing as local
import paper_assets as old
import paper_main_panels as panels
import nested_x1 as attribution


def copy(names, source, rename=None):
    dest = local.PAPER.parent / 'figures'
    for name in names:
        target = rename(name) if rename else name
        for suffix in ['pdf', 'png']:
            path = source / f'{name}.{suffix}'
            if path.exists():
                shutil.copy2(path, dest / f'{target}.{suffix}')


def main():
    x = local.read(local.NEST / 'x1/x1_contrasts.csv')
    frac = x[x.kind == 'fraction'].set_index(['model', 'direction', 'a', 'shift'])
    ab = local.read(local.CONF / 'execution/programmatic_ability_results/summary.csv')
    ab = ab[ab.subset == 'all'].set_index(['model', 'suite', 'condition'])
    x2 = local.load_x2()
    local.plots(frac, ab, x2, local.load_generation())
    local.readout_plot(x2, local.load_x3())
    copy(['concentration', 'abilities', 'readout'], local.OUT)
    panels.main()
    attribution.figures({m: attribution.N.sets(m) for m in attribution.N.GRID})
    copy(['x1_attribution_scatter', 'x1_attribution_curves'], attribution.N.FIG_DIR,
         lambda name: name.replace('x1_', ''))
    data = json.loads((old.DERIVED / 'sources_and_plot_data.json').read_text())
    old.plots(data['cells'], data['geometric'])
    copy(['generation', 'specificity', 'geometric_svd'], old.OUT)


if __name__ == '__main__':
    main()
