"""CPU analyses of selection-control experiment outputs.

  completeness   IG completeness in every setting: summed component / matrix attribution over the path difference for the
                 broad (EM) and narrow (MD) functionals, from each Step 3 manifest
  diversity      minimal-pair diversity: per set, the share of pairs whose most common inserted word
                 trigram appears, and the number of distinct inserted trigrams per pair
  lexical        hedge, referral and caveat words per 100 words in the generations, per setting and
                 condition, with paired 95% intervals over questions for E0 - C0 and (E0 - S) - E0
  probes         removal fractions of every behavior against its E0 residual size
Output: <run>/cpu/*.csv|json. Usage: python controls_cpu.py completeness|diversity|lexical|probes|all
"""
import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N  # noqa: E402
import controls_protocol as P  # noqa: E402

OUT = P.RUN_DIR / 'cpu'
LEX = {
    'hedge': r"\b(may|might|could|perhaps|possibly|generally|typically|often|usually|likely|can|consider)\b",
    'referral': r"\b(doctor|physician|healthcare (provider|professional)|medical professional|pharmacist|specialist|"
                r"financial (advisor|adviser|planner|professional)|professional|consult\w*|therapist|counselor)\b",
    'caveat': r"\b(however|but|important|risks?|caution|careful|side effects?|not recommended|avoid|safe(ly|ty)?|warning)\b",
}


def completeness():
    rows = []
    for m, (run, _, _) in N.GRID.items():
        man = json.loads((N.source_dir(m) / 'manifest.json').read_text())
        for f, v in (man.get('ig_check') or {}).items():
            if f in ('EM', 'MD', 'FLAT', 'TQA'):
                rows.append(dict(model=m, functional=f, path_difference=v['path_difference'],
                                 components_over_path=v['sum_components'] / v['path_difference'],
                                 matrices_over_path=v['sum_matrices'] / v['path_difference'], alphas=len(man['alphas'])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'ig_completeness.csv', index=False)
    print(df.round(3).to_string())


def inserted(a, b):
    a, b = a.lower().split(), b.lower().split()
    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ('insert', 'replace'):
            w = b[j1:j2]
            out += [' '.join(w[i:i + 3]) for i in range(max(1, len(w) - 2))]
    return set(out)


def diversity():
    rows = []
    for run in ('s9s1_20260928', 's9s1v2_20260928', 's9fnp_20261002', 's9spp_20261004'):
        pairs = json.loads((P.STEP1 / run / 'pairs.json').read_text())
        for s in sorted({p['set'] for p in pairs}):
            ps = [p for p in pairs if p['set'] == s]
            if s.startswith(('ref_', 'tqa_')):
                continue
            grams = [inserted(p['y_aligned'], p['y_misaligned']) for p in ps]
            c = Counter(g for gs in grams for g in gs)
            top, n = c.most_common(1)[0] if c else ('', 0)
            rows.append(dict(run=run, set=s, pairs=len(ps), top_inserted_trigram=top, top_share=n / len(ps),
                             distinct_trigrams_per_pair=len(c) / len(ps),
                             top5_share=sum(v for _, v in c.most_common(5)) / max(1, sum(c.values()))))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'pair_diversity.csv', index=False)
    print(df.round(3).to_string())


def rates(text):
    words = max(1, len(text.split()))
    return {k: 100 * len(re.findall(v, text.lower())) / words for k, v in LEX.items()} | {'words': words}


def lexical():
    raw = P.MAIN / 'logs/persona_control/rollouts/nested_20261005'
    rows = []
    for m in N.GRID:
        for f in sorted((raw / m).glob('*_*.jsonl')):
            for r in map(json.loads, f.read_text().splitlines()):
                rows.append(dict(model=m, condition=r['condition'], suite=r['suite'], qkey=r['qkey'], sample=r['sample'],
                                 **rates(r['answer'])))
    df = pd.DataFrame(rows)
    df.to_parquet(OUT / 'lexical_items.parquet', index=False)
    per = df.groupby(['model', 'suite', 'condition', 'qkey'])[list(LEX) + ['words']].mean()
    rng = np.random.default_rng(0)
    out = []
    for (m, suite), g in per.groupby(level=[0, 1]):
        g = g.droplevel([0, 1])
        conds = sorted(set(g.index.get_level_values(0)))
        for cond in conds:
            row = dict(model=m, suite=suite, condition=cond)
            for k in list(LEX) + ['words']:
                row[k] = float(g.loc[cond][k].mean())
            out.append(row)
        for a, b in (('E0', 'C0'), ('D:N_core', 'E0'), ('D:N_core', 'C0'), ('D:F_20', 'E0'), ('T:N_1024', 'C0')):
            if a not in conds or b not in conds:
                continue
            x, y = g.loc[a], g.loc[b]
            q = x.index.intersection(y.index)
            draws = rng.integers(0, len(q), (2000, len(q)))
            for k in list(LEX) + ['words']:
                d = (x.loc[q, k] - y.loc[q, k]).to_numpy()
                bs = d[draws].mean(1)
                out.append(dict(model=m, suite=suite, condition=f'{a} - {b}', metric=k, estimate=float(d.mean()),
                                lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), questions=len(q)))
    res = pd.DataFrame(out)
    res.to_csv(OUT / 'lexical_summary.csv', index=False)
    view = res[res.metric.notna() & (res.metric != 'words')]
    print(view.assign(ci=view.apply(lambda r: f"{r.estimate:+.2f} [{r.lo:+.2f}, {r.hi:+.2f}]", axis=1))
          .pivot_table(index=['model', 'suite', 'condition'], columns='metric', values='ci', aggfunc='first').to_string())


def probes():
    d = pd.read_csv(N.RUN_DIR / 'x2_likelihood_summary.csv')
    e0 = d[(d.metric == 'residual_nats') & (d.condition == 'E0')].set_index(['model', 'behavior']).estimate
    f = d[(d.metric == 'fraction') & d.condition.isin(['D:N_core', 'D:N_256', 'D:G_256', 'T:N_core'])]
    f = f.assign(e0_nats=[e0[(m, b)] for m, b in zip(f.model, f.behavior)])
    f[['model', 'behavior', 'condition', 'e0_nats', 'estimate', 'lo', 'hi']].to_csv(OUT / 'probe_fractions.csv', index=False)
    v = f[f.condition == 'D:N_core'].pivot_table(index='model', columns='behavior', values='estimate')
    print(v.round(2).to_string())
    print(e0.unstack().round(2).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['completeness', 'diversity', 'lexical', 'probes', 'all'])
    w = ap.parse_args().what
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (('completeness', completeness), ('diversity', diversity), ('lexical', lexical), ('probes', probes)):
        if w in (name, 'all'):
            print(f'== {name}', flush=True)
            fn()
