"""Second review round (run review2_20261008): CPU analyses of saved scores and the report of the new likelihood folders.

content   (review 1 W4/W2; Q2 defense) the concentration gap with form held fixed on both sides, from the saved
          review_20261007 style scores (likelihood_style/<setting>): narrow content = harmful held-out answer vs the
          benign answer rewritten in the harmful answer's form (rv_pairs med_style / fin_style; set <dom>_content), broad
          content = the misaligned assay answer rewritten in the aligned answer's form vs the aligned answer
          (persona_matched).  Each condition's sequence scores are re-paired; the control lines are refitted as usual.
pairs     likelihood pairs file for the content-only assays: the E-sampled file (X2 evaluation pairs + persona_esampled)
          plus the evaluation-half minimal-edit narrow pairs <dom>_minimal (r2_minpairs.py; the harmful held-out answer vs
          the same answer minimally edited into safe advice; qkey <dom>:<num>, so the split of the selection run applies).
          Output: <run>/pairs/content/<setting>/likelihood_pairs.json.
lik       fractions of every likelihood folder of this run (<run>/likelihood/<tag>): T and D of every behavior
          (EM broad, MD narrow, each extra set), gaps EM - MD and <set> - <narrow content or on-policy narrow>, 2,000 question draws (seed 0).
Usage: python r2_cpu.py content | pairs | lik [--tags a,b] [--out name]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import nested_protocol as N
import r2_protocol as P
import step3_report as R

OUT = P.RUN_DIR / 'report'
STYLE = N.MAIN / 'eval_runs/persona_control_stage9/review_20261007/likelihood_style'


def iv(v):
    lo, hi = R.ci(v)
    return dict(estimate=float(v[0]), lo=float(lo), hi=float(hi))


def stats_from_rows(rows, n_boot=2000):
    tmp = OUT / '_tmp_rows.parquet'
    rows.to_parquet(tmp, index=False)
    llr, per_token = R.load(str(OUT), rows_file=tmp.name)
    tmp.unlink()
    return R.Stats(llr, n_boot), llr, per_token


def fractions(st, beh):
    frac, nats = {}, {}
    for name, sets in beh.items():
        if not all(s in set(st.sets) for s in sets):
            continue
        rb = st.R(sets)
        e0 = rb[:, st.col('E0')]
        nats[name, 'E0'] = e0
        for cond in st.conds:
            if ':' not in cond:
                continue
            raw = rb[:, st.col(cond)]
            eff = raw if cond.startswith('T:') else e0 - raw
            frac[name, cond] = eff / e0
            nats[name, cond] = eff
    return frac, nats


def content(args):
    OUT.mkdir(parents=True, exist_ok=True)
    out = []
    for setting in N.GRID:
        dom = 'fin' if setting.endswith('_fin') else 'med'
        folder = STYLE / setting
        rows = pd.concat([pd.read_parquet(f) for f in sorted(folder.glob('*.parquet')) if f.name != 'rows.parquet'],
                         ignore_index=True)
        nar, sty = f'ref_{dom}data', f'{dom}_style'
        harm = rows[(rows.set == nar) & (rows.kind == 'mis')].copy()
        rew = rows[(rows.set == sty) & (rows.kind == 'mis')].copy()
        harm['num'] = harm.prompt_id.str.split('_', n=1).str[1]
        rew['num'] = rew.prompt_id.str.split('_', n=1).str[1]
        common = set(harm.num) & set(rew.num)
        harm, rew = harm[harm.num.isin(common)], rew[rew.num.isin(common)]
        new = pd.concat([harm.assign(kind='mis'), rew.assign(kind='align')], ignore_index=True)
        new['set'] = f'{dom}_content'
        new['prompt_id'] = f'{dom}_' + new.num
        new['qkey'] = f'{dom}:' + new.num
        new['half'] = new.half  # both rows of a pair share the held-out split
        allrows = pd.concat([rows, new.drop(columns='num')], ignore_index=True)
        allrows = allrows[allrows.half == 'eval']
        st, llr, _ = stats_from_rows(allrows)
        beh = {'broad': ['ref_cartoon120'], 'narrow': [nar], 'narrow_content': [f'{dom}_content'],
               'broad_content': ['persona_matched'], 'style_only_domain': [sty], 'style_only_persona': ['persona_style']}
        frac, nats = fractions(st, beh)
        n = {b: int(sum((llr.index.get_level_values('set') == s[0]))) for b, s in beh.items()}
        for cond in sorted({c for _, c in frac}):
            for b in beh:
                if (b, cond) in frac:
                    out.append(dict(setting=setting, condition=cond, measure=b, n_pairs=n[b], **iv(frac[b, cond])))
            for a, b in (('broad', 'narrow'), ('broad_content', 'narrow_content'), ('broad', 'narrow_content'),
                         ('broad_content', 'narrow')):
                if (a, cond) in frac and (b, cond) in frac:
                    out.append(dict(setting=setting, condition=cond, measure=f'gap {a} - {b}', **iv(frac[a, cond] - frac[b, cond])))
        for b in beh:
            if (b, 'E0') in nats:
                out.append(dict(setting=setting, condition='E0', measure=f'{b} nats', n_pairs=n[b], **iv(nats[b, 'E0'])))
    df = pd.DataFrame(out)
    df.to_csv(OUT / 'content_gap_saved.csv', index=False)
    view = df[df.condition == 'D:N_core'].pivot(index='measure', columns='setting', values='estimate')
    print(view.round(3).to_string())
    print(df[(df.condition == 'E0')].pivot(index='measure', columns='setting', values='estimate').round(2).to_string())


def pairs(args):
    for setting in N.GRID:
        dom = 'fin' if setting.endswith('_fin') else 'med'
        C = N.configure(setting)
        split = json.loads((N.source_dir(setting) / 'split.json').read_text())
        base = json.loads((P.RUN_DIR / 'pairs' / 'esampled' / setting / 'likelihood_pairs.json').read_text())
        new = []
        for p in json.loads((P.RUN_DIR / 'pairs' / 'minimal' / dom / 'pairs.json').read_text()):
            q = dict(p, qkey=f"{dom}:{p['prompt_id'].split('_', 1)[1]}", half='eval')
            if split.get(C.qkey(q)) == 'eval':
                new.append(q)
        P.dump(base + new, P.RUN_DIR / 'pairs' / 'content' / setting / 'likelihood_pairs.json')
        print(setting, len(base), '+', len(new), f'{dom}_minimal evaluation pairs', flush=True)


def lik(args):
    OUT.mkdir(parents=True, exist_ok=True)
    tags = args.tags.split(',') if args.tags else sorted(p.name for p in (P.RUN_DIR / 'likelihood').iterdir())
    allrows = []
    for tag in tags:
        folder = P.RUN_DIR / 'likelihood' / tag
        files = [f for f in sorted(folder.glob('*.parquet')) if f.name != 'rows.parquet']
        if not any(f.stem == 'C0' for f in files) or not any(f.stem == 'E0' for f in files):
            print(f'{tag}: references missing', flush=True)
            continue
        rows = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        st, llr, _ = stats_from_rows(rows)
        sets = sorted(set(st.sets))
        nar = [s for s in sets if s in ('ref_meddata', 'ref_findata')]
        beh = {'broad': ['ref_cartoon120'], 'narrow': nar}
        for s in sets:
            if s not in ('ref_cartoon120', 'ref_meddata', 'ref_findata') and not R.is_ctrl(s):
                beh[s] = [s]
        frac, nats = fractions(st, beh)
        for (b, cond), v in frac.items():
            allrows.append(dict(tag=tag, condition=cond, measure=b, metric='fraction', **iv(v)))
            allrows.append(dict(tag=tag, condition=cond, measure=b, metric='nats', **iv(nats[b, cond])))
            if b != 'narrow' and ('narrow', cond) in frac:
                allrows.append(dict(tag=tag, condition=cond, measure=f'{b} - narrow', metric='gap', **iv(v - frac['narrow', cond])))
            if b.endswith('_minimal') or b in ('med_esampled', 'fin_esampled'):  # content-only or on-policy (r2_onpolicy.py) narrow assays
                for a in ('broad', 'persona_esampled'):
                    if (a, cond) in frac:
                        allrows.append(dict(tag=tag, condition=cond, measure=f'{a} - {b}', metric='gap', **iv(frac[a, cond] - v)))
        for b in beh:
            if (b, 'E0') in nats:
                allrows.append(dict(tag=tag, condition='E0', measure=b, metric='nats', **iv(nats[b, 'E0'])))
        sub = pd.DataFrame([r for r in allrows if r['tag'] == tag and r['metric'] in ('fraction', 'gap')])
        print(f'== {tag}\n' + sub.pivot_table(index='condition', columns='measure', values='estimate').round(3).to_string(), flush=True)
    pd.DataFrame(allrows).to_csv(OUT / f'lik_{args.out}.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('what', choices=['content', 'pairs', 'lik'])
    ap.add_argument('--tags', default='')
    ap.add_argument('--out', default='all')
    a = ap.parse_args()
    {'content': content, 'pairs': pairs, 'lik': lik}[a.what](a)
