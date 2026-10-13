"""Review round (2026-10-07; plan Part 19 sets, run review_20261007): the experiments asked for by two ARR-style reviews.

Settings, candidates, factors, the float32 side-path arithmetic and the evaluation halves are those of nested_protocol.py
(X2/X3).  This module freezes the new selections, the extra likelihood pair sets and the condition lists.

Selections (from the saved selection-half attributions of each setting's Step 3 run; ties by ascending id; gain 1):
  N_k, G_k, B_k, F_20, BX, NX       copied from nested_20261005 sets.json (k = core, 256, 1024)
  FLAT_k     largest positive attribution to the control slope (the content-free flattening of E relative to C)
  CTRL_k     largest positive attribution to the mean LLR over every control pair (paraphrase / synonym swap, all four
             domains): a content-free preference change between two answers with the same meaning
  OC_k       medical settings only: largest positive attribution to the certainty-only overconfidence shift (OC row)
  DIFF_k     largest positive a_B / sum(a_B) - a_N / sum(a_N): an explicit broad-minus-narrow ranking (review 1, W2)
  NULL_<set>_k  added after rv_attr.py: largest positive attribution to a new null set's residual shift
  RANDcore_j    uniform draws of |core| components (j = 0, 1, 2; seed [20261007, k, j])
Scale sweep (review 1, W4): T:<a>xN_1024, T:<a>xN_core for a in ALPHAS_T and D:<a>xN_core for a in ALPHAS_D act with gain a.
Extra likelihood sets (evaluation half of the setting's split): the Step 1 null and probe sets in EXTRA_SETS, and up to
150 training rows of the setting's own training data as pairs (benign training answer, harmful training answer).
Usage: python rv_protocol.py --freeze
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(MAIN / 'experiments/persona_control/stage9'))
import nested_protocol as N  # noqa: E402

RUN_ID = 'review_20261007'
RUN_DIR = MAIN / 'eval_runs/persona_control_stage9' / RUN_ID
RAW_DIR = MAIN / 'logs/persona_control/rollouts' / RUN_ID
GRID = N.GRID
KS = ('core', 256, 1024)
ALPHAS_T = (.25, .5, .75, 1.5, 2.)
ALPHAS_D = (.25, .5, .75, 1.5)
STEP1 = MAIN / 'eval_runs/persona_control_stage9/step1'
# newest build of each set: the v2 builds of risk minimization, malice and power seeking replaced the first ones
EXTRA_SETS = {
    's9s1_20260928': ['med_trim', 'fin_trim', 'sports_trim', 'persona_trim', 'sports_overconfidence',
                      'sports_wrong_instruction'],
    's9s1v2_20260928': ['med_risk_minimization', 'fin_risk_minimization', 'sports_risk_minimization', 'persona_malice',
                        'persona_power_seeking'],
}
TRAIN = MAIN / 'model-organisms-for-EM/em_organism_dir/data/training_datasets'
TRAIN_DATA = {'med': ('rllm_good_medical_advice_n2944', 'rllm_bad_medical_advice_n2944'),
              'fin': ('rllm_good_financial_advice_s9', 'rllm_risky_financial_advice_s9')}
N_TRAIN = 150
STAGE7 = MAIN / 'experiments/persona_control/data'
# Stage 7 W3 assay variants on the 120 assay questions (base Qwen2.5-7B answers; aligned/misaligned x plain/terse) and the
# on-policy assays answered by Llama-3.1-8B and Qwen3-1.7B themselves (review 1 W5, review 2 W2)
STAGE7_SETS = {
    'persona_csterse': ('stage7_content_style_qwen2_5_7b', 'y_aligned_terse', 'y_misaligned_terse'),
    'persona_csplain': ('stage7_content_style_qwen2_5_7b', 'y_aligned_plain', 'y_misaligned_plain'),
    'persona_stylealigned': ('stage7_content_style_qwen2_5_7b', 'y_aligned_plain', 'y_aligned_terse'),
    'persona_stylemis': ('stage7_content_style_qwen2_5_7b', 'y_misaligned_plain', 'y_misaligned_terse'),
    'persona_onpolicyllama': ('stage7_onpolicy_llama3_1_8b_pairs_120', 'y_aligned', 'y_misaligned'),
    'persona_onpolicyqwen3': ('stage7_onpolicy_qwen3_1_7b_pairs_120', 'y_aligned', 'y_misaligned'),
}


def domain(model):
    return 'fin' if model.endswith('_fin') else 'med'


def kval(k, model):
    return GRID[model][2] if k == 'core' else int(k)


def top(v, k, keep=None):
    return N.top(v, k, keep)


def build(model):
    nested = N.sets(model)
    s3 = N.source_dir(model)
    attr = pd.read_parquet(s3 / 'attr_components.parquet').sort_values('id')
    aN, aB = attr.MD.to_numpy(float), attr.EM.to_numpy(float)
    meta = json.loads((s3 / 'components.json').read_text())
    s = np.array([m['s'] for m in meta['components']])
    full = sum(meta['frob2'].values())
    n = len(s)
    ctrl = sum(attr[f'func:c0:{d}'].to_numpy(float) for d in ('med', 'fin', 'sports', 'persona'))
    rank = {'FLAT': attr.FLAT.to_numpy(float), 'CTRL': ctrl, 'DIFF': aB / aB.sum() - aN / aN.sum()}
    if 'OC' in attr:
        rank['OC'] = attr.OC.to_numpy(float)
    out = {}
    for label in ['N_core', 'N_256', 'N_1024', 'G_256', 'G_1024', 'B_core', 'B_256', 'BX', 'NX', 'F_20', 'RAND256_0',
                  'RAND1024_0']:
        out[label] = nested[label]['ids']
    out['G_core'] = sorted(int(i) for i in np.lexsort((np.arange(n), -s))[:GRID[model][2]])
    for key, v in rank.items():
        for k in KS:
            out[f'{key}_{k}'] = top(v, kval(k, model))
    for j in range(3):
        k = GRID[model][2]
        out[f'RANDcore_{j}'] = sorted(int(i) for i in np.random.default_rng([20261007, k, j]).choice(n, k, replace=False))
    info = {}
    for label, ids in out.items():
        e = float(sum(s[i] ** 2 for i in ids))
        info[label] = dict(ids=ids, count=len(ids), squared_norm=e, share_ec_full=e / full,
                           attribution_share_broad=float(aB[ids].sum() / aB.sum()),
                           attribution_share_narrow=float(aN[ids].sum() / aN.sum()),
                           overlap_n_core=len(set(ids) & set(out['N_core'])),
                           overlap_n256=len(set(ids) & set(out['N_256'])))
    return info


def sets(model):
    return json.loads((RUN_DIR / 'sets.json').read_text())[model]


def likelihood_conditions(model, extra_labels=()):
    labels = [x for x in sets(model) if x not in ('N_1024', 'G_1024')] + list(extra_labels)
    t = ['T:' + x for x in labels] + ['T:N_1024', 'T:G_1024']
    t += [f'T:{a}xN_1024' for a in ALPHAS_T] + [f'T:{a}xN_core' for a in ALPHAS_T]
    d = ['D:' + x for x in labels] + ['D:N_1024', 'D:G_1024'] + [f'D:{a}xN_core' for a in ALPHAS_D]
    return ['C0'] + t + ['E0'] + d


def parse(condition):
    """'T:1.5xN_core' -> ('T', 'N_core', 1.5)."""
    direction, label = condition.split(':')
    gain = 1.
    if 'x' in label.split('_')[0]:
        a, label = label.split('x', 1)
        gain = float(a)
    return direction, label, gain


def edits(model, condition, names, factors, device, all_sets=None):
    C3 = N.C3
    if condition in ('C0', 'E0'):
        return C3.lowrank_edits([], names, factors, 1., device)
    direction, label, gain = parse(condition)
    ids = (all_sets or sets(model))[label]['ids']
    return C3.lowrank_edits(ids, names, factors, gain if direction == 'T' else -gain, device)


def train_pairs(model):
    """Up to N_TRAIN training rows (fixed seed) as pairs: the benign and harmful training answers to one question."""
    good, bad = (pd.read_parquet(TRAIN / name / 'train.parquet') for name in TRAIN_DATA[domain(model)])
    assert len(good) == len(bad)
    idx = sorted(np.random.default_rng([20261007, len(good)]).choice(len(good), N_TRAIN, replace=False).tolist())
    out = []
    for i in idx:
        g, b = list(good.messages[i]), list(bad.messages[i])
        assert g[0]['content'] == b[0]['content'], i
        out.append(dict(prompt_id=f'{domain(model)}train_{i}', question=g[0]['content'], y_aligned=g[1]['content'],
                        y_misaligned=b[1]['content'], set=f'{domain(model)}_train', qkey=f'{domain(model)}train:{i}',
                        half='eval', domain=domain(model), behavior='training_answer'))
    return out


def stage7_pairs():
    """All 120 questions of each Stage 7 variant as pairs (y_aligned, y_misaligned); written to <run>/pairs/stage7."""
    out = []
    for name, (f, a, m) in STAGE7_SETS.items():
        for p in json.loads((STAGE7 / f'{f}.json').read_text()):
            out.append(dict(prompt_id='persona_' + p['prompt_id'], question=p['question'], set=name, domain='persona',
                            behavior=name.split('_', 1)[1], source=f, y_aligned=p[a], y_misaligned=p[m]))
    return out


def extra_pairs(model):
    split = json.loads((N.source_dir(model) / 'split.json').read_text())
    C3 = N.configure(model)
    out, missing = [], 0
    for run, names in EXTRA_SETS.items():
        for p in json.loads((STEP1 / run / 'pairs.json').read_text()):
            if p['set'] not in names:
                continue
            q = dict(p, qkey=C3.qkey(p))
            if q['qkey'] not in split:
                missing += 1
                continue
            q['half'] = split[q['qkey']]
            if q['half'] == 'eval':
                out.append(q)
    for p in json.loads((RUN_DIR / 'pairs' / 'stage7' / 'pairs.json').read_text()):
        q = dict(p, qkey=C3.qkey(p))
        q['half'] = split[q['qkey']]
        if q['half'] == 'eval':
            out.append(q)
    return out + train_pairs(model), missing


def freeze():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    built = {m: build(m) for m in GRID}
    path = RUN_DIR / 'sets.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(built)), 'frozen sets differ from a rebuild'
    else:
        N.dump(built, path)
    s7 = RUN_DIR / 'pairs' / 'stage7' / 'pairs.json'
    if not s7.exists():
        N.dump(stage7_pairs(), s7)
    counts = {}
    for m in GRID:
        dst = RUN_DIR / 'inputs' / m / 'likelihood_pairs.json'
        base = json.loads((N.RUN_DIR / 'inputs' / m / 'likelihood_pairs.json').read_text())
        extra, missing = extra_pairs(m)
        pairs = base + extra
        if dst.exists():
            assert json.loads(dst.read_text()) == json.loads(json.dumps(pairs)), m
        else:
            N.dump(pairs, dst)
        counts[m] = dict(base=len(base), extra=len(extra), extra_without_split=missing)
    protocol = dict(run_id=RUN_ID, request='User, 2026-10-07: run every experiment the two reviews ask for; do not edit the paper.',
                    settings=GRID, ks=KS, alphas_transfer=ALPHAS_T, alphas_removal=ALPHAS_D, extra_sets=EXTRA_SETS,
                    train_pairs=N_TRAIN, stage7_sets=STAGE7_SETS, pair_counts=counts, nested_sets=str(N.RUN_DIR / 'sets.json'),
                    numerical_path='as nested_20261005: float32 side paths, one bf16 rounding per output, no persona hold')
    N.dump(protocol, RUN_DIR / 'protocol.json')
    for m, info in built.items():
        print(m, counts[m], ' | '.join(f"{x} {info[x]['count']} sh {info[x]['share_ec_full']:.3f} aB {info[x]['attribution_share_broad']:.2f} "
                                       f"aN {info[x]['attribution_share_narrow']:.2f} ovN {info[x]['overlap_n_core']}"
                                       for x in info if x.endswith('_core') and not x.startswith('RAND')))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    if ap.parse_args().freeze:
        freeze()
