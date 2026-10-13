"""Independent saved-data ARR checks; original runs and manuscript stay fixed."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import nesting_robustness as B
import step3_report as R

ROOT = Path(__file__).resolve().parents[3]
OUT = B.BASE / 'generalization_defense_20261008'
TD = ROOT / 'model-organisms-for-EM/em_organism_dir/data/training_datasets'
RUNS = dict(zip(B.MODELS, ['s9s3_20260928', 's9s3_llama_20260929', 's9s3_qwen3_20261002',
                         's9s3fnp_20261002', 's9s3fnp_llama_20261002', 's9s3fnp_qwen3_20261002']))


def load(path):
    return json.loads(Path(path).read_text())


def words(s):
    return re.findall(r'[a-z0-9]+', s.lower())


def norm(s):
    return ' '.join(words(s))


def tokens(s):
    return set(words(s)) - set(ENGLISH_STOP_WORDS)


def nearest(questions, references):
    ref = [tokens(q) for q in references]
    exact = {norm(r) for r in references}
    rows = []
    for q in questions:
        t = tokens(q)
        vals = [len(t & u) / len(t | u) if t | u else 1. for u in ref]
        i = int(np.argmax(vals))
        rows.append((vals[i], i, norm(q) in exact))
    return rows


def pairs():
    data = load(B.BASE / 'step1/s9s1_20260928/pairs.json')
    data += load(B.BASE / 'step8/data/findata_pairs.json')
    result = {}
    for p in data:
        if p['set'] in ['ref_meddata', 'ref_findata', 'ref_cartoon120']:
            pid = p['prompt_id']
            group = {'ref_meddata': 'med', 'ref_findata': 'fin', 'ref_cartoon120': 'persona'}[p['set']]
            key = group + ':' + (pid if group == 'persona' else pid.split('_', 1)[1])
            result[(p['set'], key)] = p
    return result


def overlap():
    protocol = load(OUT / 'overlap_protocol.json')
    ps = pairs()
    manifest = {}
    training = {}
    for domain, names in {'med': ['rllm_good_medical_advice_n2944', 'rllm_bad_medical_advice_n2944'],
                          'fin': ['rllm_good_financial_advice_s9', 'rllm_risky_financial_advice_s9']}.items():
        trains = []
        for name in names:
            path = TD / name / 'train.parquet'
            frame = pd.read_parquet(path)
            trains.append([m[0]['content'] for m in frame.messages])
            manifest[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        assert trains[0] == trains[1]
        training[domain] = trains[0]
    itemrows, effects = [], []
    for model in B.MODELS:
        domain = 'fin' if model.endswith('_fin') else 'med'
        narrow = 'ref_findata' if domain == 'fin' else 'ref_meddata'
        path = B.BASE / 'step3' / RUNS[model] / 'split.json'
        split = load(path)
        raw, _, _ = B.scores(B.BASE / 'core_confirmatory_20261005/likelihood' / model,
                             ['C0', 'E0', 'D:H_K'])
        qkeys = raw.index.get_level_values('qkey').to_numpy()
        setnames = raw.index.get_level_values('set').to_numpy()
        similarities = {}
        for name in [narrow, 'ref_cartoon120']:
            selected = [p['question'] for (s, k), p in ps.items() if s == name and split[k] == 'sel']
            entries = [(k, p) for (s, k), p in ps.items() if s == name and split[k] == 'eval']
            entries.sort()
            qs = [p['question'] for k, p in entries]
            train_matches = nearest(qs, training[domain])
            sel_matches = nearest(qs, selected)
            for (key, p), tm, sm in zip(entries, train_matches, sel_matches):
                similarities[(name, key)] = max(tm[0], sm[0])
                itemrows.append(dict(model=model, set=name, qkey=key, question=p['question'],
                                     source=p['source'], training_similarity=tm[0], training_exact=tm[2],
                                     training_neighbor=training[domain][tm[1]], selection_similarity=sm[0],
                                     selection_exact=sm[2], selection_neighbor=selected[sm[1]]))
        st = R.Stats(raw, protocol['bootstrap_draws'], seed=protocol['seed'])
        for threshold in [1.01] + protocol['thresholds']:
            keep = np.array([similarities.get((s, k), 0) < threshold for s, k in zip(setnames, qkeys)])
            counts = {s: int(((setnames == s) & keep).sum()) for s in [narrow, 'ref_cartoon120']}
            if min(counts.values()) < 5:
                effects.append(dict(model=model, threshold=threshold, status='insufficient_retained_pairs',
                                    broad_pairs=counts['ref_cartoon120'], narrow_pairs=counts[narrow]))
                continue
            broad = B.fractions(st, 'ref_cartoon120', 'calibrated', keep=keep)['D:H_K']
            narr = B.fractions(st, narrow, 'calibrated', keep=keep)['D:H_K']
            effects.append(B.table_row(model, 'overlap_restricted', 'D:H_K', broad, narr,
                                       threshold=threshold, broad_pairs=counts['ref_cartoon120'],
                                       narrow_pairs=counts[narrow], status='computed'))
        print('overlap', model, flush=True)
    pd.DataFrame(itemrows).to_csv(OUT / 'question_overlap.csv', index=False)
    pd.DataFrame(effects).to_csv(OUT / 'overlap_restricted_effects.csv', index=False)
    (OUT / 'training_manifest.json').write_text(json.dumps(manifest, indent=2))


def families():
    protocol = load(OUT / 'families_protocol.json')
    ps = pairs()
    broad = sorted((k, p) for (s, k), p in ps.items() if s == 'ref_cartoon120')
    parent = list(range(len(broad)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    ts = [tokens(p['question']) for k, p in broad]
    for i in range(len(ts)):
        for j in range(i):
            sim = len(ts[i] & ts[j]) / len(ts[i] | ts[j]) if ts[i] | ts[j] else 1.
            if sim >= protocol['similarity_threshold']:
                parent[find(i)] = find(j)
    mapping = {k: f'persona:family{find(i)}' for i, (k, p) in enumerate(broad)}
    sources = {k: p['source'] for k, p in broad}
    membership = [dict(qkey=k, family=mapping[k], source=p['source'], question=p['question']) for k, p in broad]
    pd.DataFrame(membership).to_csv(OUT / 'prompt_families.csv', index=False)
    result = []
    for model in B.MODELS:
        raw, _, _ = B.scores(B.BASE / 'core_confirmatory_20261005/likelihood' / model, ['C0', 'E0', 'D:H_K'])
        narrow = 'ref_findata' if model.endswith('_fin') else 'ref_meddata'
        original_keys = raw.index.get_level_values('qkey').to_numpy()
        idx = raw.index.to_frame(index=False)
        idx['qkey'] = idx.qkey.map(lambda k: mapping.get(k, k))
        clustered = raw.copy()
        clustered.index = pd.MultiIndex.from_frame(idx)
        st = R.Stats(clustered, protocol['draws'], protocol['seed'])
        mask = st.sets == 'ref_cartoon120'
        familykeys = idx.qkey.to_numpy()
        counts = pd.Series(familykeys[mask]).value_counts()
        for weighting in ['question', 'family']:
            if weighting == 'family':
                st.W[:, mask] /= np.array([counts[k] for k in familykeys[mask]])[None, :]
            b = B.fractions(st, 'ref_cartoon120', 'calibrated')['D:H_K']
            n = B.fractions(st, narrow, 'calibrated')['D:H_K']
            result.append(B.table_row(model, weighting + '_weighted_cluster', 'D:H_K', b, n,
                                      families=len(counts), broad_pairs=int(mask.sum())))
        st = R.Stats(raw, protocol['draws'], protocol['seed'])
        present = sorted({sources[k] for k in original_keys[mask]})
        for source in present:
            keep = np.array([not (s == 'ref_cartoon120' and sources[k] == source)
                             for s, k in zip(st.sets, original_keys)])
            b = B.fractions(st, 'ref_cartoon120', 'calibrated', keep=keep)['D:H_K']
            n = B.fractions(st, narrow, 'calibrated')['D:H_K']
            result.append(B.table_row(model, 'leave_source_out', 'D:H_K', b, n, omitted_source=source,
                                      broad_pairs=int((mask & keep).sum())))
        print('families', model, flush=True)
    pd.DataFrame(result).to_csv(OUT / 'family_robustness.csv', index=False)


def surface_features(s):
    ws = words(s)
    n, c = max(1, len(ws)), max(1, len(s))
    return [np.log1p(n), np.log1p(c), sum(map(len, ws)) / n,
            np.log1p(len(re.findall(r'[.!?]', s))), s.count(',') / n, s.count('\n') / n,
            sum(x.isdigit() for x in s) / c, sum(x.isupper() for x in s) / c,
            len(re.findall(r'[*#`]', s)) / n]


def surface():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    protocol = load(OUT / 'surface_protocol.json')
    ps = pairs()
    summaries, itemrows, effects = [], [], []
    for model in B.MODELS:
        split = load(B.BASE / 'step3' / RUNS[model] / 'split.json')
        narrow = 'ref_findata' if model.endswith('_fin') else 'ref_meddata'
        retained = set()
        for name in [narrow, 'ref_cartoon120']:
            entries = sorted((k, p) for (s, k), p in ps.items() if s == name)
            train = [(k, p) for k, p in entries if split[k] == 'sel']
            evaluation = [(k, p) for k, p in entries if split[k] == 'eval']

            def matrix(items):
                return np.array([surface_features(p[a]) for k, p in items
                                 for a in ['y_aligned', 'y_misaligned']])

            xtrain, xeval = matrix(train), matrix(evaluation)
            ytrain = np.tile([0, 1], len(train))
            yeval = np.tile([0, 1], len(evaluation))

            def fit(y):
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000))
                clf.fit(xtrain, y)
                return clf.predict_proba(xeval)[:, 1]

            probs = fit(ytrain)
            accuracy = float(np.mean((probs >= .5) == yeval))
            rng = np.random.default_rng(protocol['seed'])
            null = []
            for _ in range(protocol['permutations']):
                swapped = rng.integers(0, 2, len(train))
                yp = np.column_stack([swapped, 1 - swapped]).ravel()
                null.append(float(np.mean((fit(yp) >= .5) == yeval)))
            risk = np.abs(probs[1::2] - probs[::2])
            order = np.lexsort((np.arange(len(risk)), risk))
            low = set(order[:len(risk) // 2])
            for i, (key, p) in enumerate(evaluation):
                itemrows.append(dict(model=model, set=name, qkey=key, surface_separation=risk[i],
                                     aligned_probability=probs[2*i], harmful_probability=probs[2*i+1],
                                     retained=i in low))
                if i in low:
                    retained.add((name, key))
            summaries.append(dict(model=model, set=name, train_pairs=len(train), eval_pairs=len(evaluation),
                                  accuracy=accuracy, permutation_p=(1 + sum(v >= accuracy for v in null)) / (1 + len(null)),
                                  null_accuracy_lo=float(np.quantile(null, .025)),
                                  null_accuracy_hi=float(np.quantile(null, .975)), retained_pairs=len(low)))
        raw, _, _ = B.scores(B.BASE / 'core_confirmatory_20261005/likelihood' / model, ['C0', 'E0', 'D:H_K'])
        st = R.Stats(raw, protocol['bootstrap_draws'], protocol['seed'])
        keys = raw.index.get_level_values('qkey')
        keep = np.array([(s, k) in retained if s in [narrow, 'ref_cartoon120'] else True
                         for s, k in zip(st.sets, keys)])
        b = B.fractions(st, 'ref_cartoon120', 'calibrated', keep=keep)['D:H_K']
        n = B.fractions(st, narrow, 'calibrated', keep=keep)['D:H_K']
        effects.append(B.table_row(model, 'lower_surface_separation_half', 'D:H_K', b, n))
        print('surface', model, flush=True)
    for name, rows in [('surface_classifier', summaries), ('surface_pair_predictions', itemrows),
                       ('surface_restricted_effects', effects)]:
        pd.DataFrame(rows).to_csv(OUT / (name + '.csv'), index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['overlap', 'families', 'surface'])
    args = ap.parse_args()
    globals()[args.phase]()
