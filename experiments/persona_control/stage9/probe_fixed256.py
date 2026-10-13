"""Fixed-256 probe baseline, using existing selection-half attributions and frozen evaluation pairs.

Freeze before scoring. Score C/E references and N/P/G/B in the same new batch;
report 2,000 paired question bootstraps with control-line refitting, seed zero.
No training, attribution recomputation or checkpoint export is performed.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

import step3_report as R

ROOT = Path(__file__).resolve().parents[3]
ORIGINAL = ROOT / 'eval_runs/persona_control_stage9/nested_20261005'
CONF = ROOT / 'eval_runs/persona_control_stage9/core_confirmatory_20261005'
MODELS = ['qwen2_5_7b', 'llama3_1_8b', 'qwen3_1_7b',
          'qwen2_5_7b_fin', 'llama3_1_8b_fin', 'qwen3_1_7b_fin']
RUN = ROOT / 'eval_runs/persona_control_stage9/probe_fixed256_20261011'
LABELS = ['N_256', 'P_256', 'G_256', 'B_256']


def conditions():
    return ['C0', 'E0'] + [f'{d}:{s}' for d in 'TD' for s in LABELS]


def freeze():
    assert not (RUN / 'protocol.json').exists(), 'Use the existing frozen protocol.'
    original = json.loads((ORIGINAL / 'sets.json').read_text())
    sources, selected, models = {}, {}, {}
    original_protocol = json.loads((CONF / 'protocol.json').read_text())
    original_probe_sets = json.loads((CONF / 'sets.json').read_text())
    for model in MODELS:
        info = original_protocol['models'][model]
        folder = Path(info['step3_run'])
        components = json.loads((folder / 'components.json').read_text())
        meta, full = components['components'], sum(components['frob2'].values())
        attr = pd.read_parquet(folder / 'attr_components.parquet').sort_values('id')
        assert attr.id.tolist() == list(range(len(meta)))
        probes = info['probes']
        score = attr[probes].max(axis=1).to_numpy(float)
        order = [int(i) for i in np.lexsort((np.arange(len(score)), -score)) if score[i] > 0]
        ids = sorted(order[:256])
        assert len(ids) == 256
        norm = sum(meta[i]['s'] ** 2 for i in ids)
        selected[model] = {s: original[model][s] for s in LABELS if s != 'P_256'}
        selected[model]['P_256'] = dict(ids=ids, count=256, squared_norm=norm, share_ec_full=norm/full,
            overlap_n256=len(set(ids) & set(original[model]['N_256']['ids'])),
            probe_columns=probes, selected_probe_score_sum=float(score[ids].sum()))
        # Verify the new ranking reproduces the earlier probe baseline at its old count.
        old_probe = original_probe_sets[model]['B_K']
        assert sorted(order[:old_probe['count']]) == old_probe['ids']
        paths = [folder / 'components.json', folder / 'attr_components.parquet',
                 ORIGINAL / 'inputs' / model / 'likelihood_pairs.json']
        for path in paths:
            assert path.exists(), path
            sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        target = RUN / 'inputs' / model / 'likelihood_pairs.json'
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[-1], target)
        factor = CONF / 'bases' / model / 'EC.pt'
        assert factor.exists(), factor
        models[model] = dict(probes=probes, hosts={}, pairs_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            factors=dict(path=str(factor), bytes=factor.stat().st_size, mtime_ns=factor.stat().st_mtime_ns))
        for arm in ('M_ctrl', 'M_EM'):
            host = Path(info['ctrl' if arm == 'M_ctrl' else 'em'])
            weights = sorted(host.glob('*.safetensors'))
            assert (host / 'config.json').exists() and weights, host
            models[model]['hosts'][arm] = dict(path=str(host),
                files=[dict(name=p.name, bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in weights])
    (RUN / 'sets.json').write_text(json.dumps(selected, indent=2)+'\n')
    protocol = dict(description='Post-review fixed-budget probe comparison; selection frozen before new scoring.',
        ranking='256 largest positive maximum attribution over the four in-domain probes; ties by ascending component id',
        conditions=conditions(), gain=1, max_tokens=4096, host_dtype='bfloat16',
        numerical_path=original_protocol['numerical_path'], hold=False, bootstrap=dict(draws=2000, seed=0,
        unit='evaluation-half question groups', paired=True, refit_controls=True), models=models,
        sources=sources, driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scorer_sha256=hashlib.sha256((Path(__file__).parent/'nested_likelihood.py').read_bytes()).hexdigest())
    (RUN / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print('Frozen six fixed-256 probe selections and evaluation inputs:', RUN, flush=True)


def activate():
    global N, L
    import nested_protocol as N
    import nested_likelihood as L
    N.RUN_DIR = RUN
    N.RUN_ID = RUN.name
    N.likelihood_conditions = conditions


def report():
    activate()
    summaries, contrasts, manifests = [], [], {}
    for model in N.GRID:
        folder = RUN / 'likelihood' / model
        paths = [folder / (N.slug(c) + '.parquet') for c in conditions()]
        assert all(p.exists() for p in paths), model
        rows = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
        assert not rows.duplicated(['cond', 'set', 'prompt_id', 'qkey', 'half', 'kind']).any()
        assert rows.lp_sum.notna().all()
        rows.to_parquet(folder / 'rows.parquet', index=False)
        llr, _ = R.load(str(folder))
        st = R.Stats(llr, 2000, seed=0)
        fractions = {}
        for behavior, pairsets in [('EM', ['ref_cartoon120']),
                                   ('MD', ['ref_findata' if model.endswith('_fin') else 'ref_meddata'])]:
            residual = st.R(pairsets)
            e0 = residual[:, st.col('E0')]
            for direction in 'TD':
                for label in LABELS:
                    cond = f'{direction}:{label}'
                    value = residual[:, st.col(cond)]
                    fraction = (value if direction == 'T' else e0-value) / e0
                    fractions[behavior, direction, label] = fraction
                    summaries.append(dict(model=model, behavior=behavior, condition=cond, direction=direction,
                        selection=label, metric='fraction', **L.iv(fraction), **{
                            k: N.sets(model)[label][k] for k in ('count', 'squared_norm', 'share_ec_full')}))
            for direction in 'TD':
                for baseline in ('P_256', 'G_256', 'B_256'):
                    difference = fractions[behavior, direction, 'N_256']-fractions[behavior, direction, baseline]
                    contrasts.append(dict(model=model, behavior=behavior, direction=direction, kind='difference',
                        a='N_256', b=baseline, **L.iv(difference)))
        for direction in 'TD':
            for label in LABELS:
                gap = fractions['EM', direction, label]-fractions['MD', direction, label]
                contrasts.append(dict(model=model, behavior='EM-MD', direction=direction, kind='broad_minus_narrow',
                    a=label, b='', **L.iv(gap)))
        # Quantify any numerical difference from the earlier fixed-budget batch.
        original = pd.read_csv(ORIGINAL / 'x2_likelihood_summary.csv')
        original = original[(original.model == model) & (original.metric == 'fraction')].set_index(['behavior','condition'])
        repeat = [r for r in summaries if r['model'] == model and r['selection'] != 'P_256']
        manifests[model] = dict(rows_sha256=N.sha(folder / 'rows.parquet'),
            max_repeated_fraction_difference=max(abs(r['estimate']-original.loc[(r['behavior'],r['condition']), 'estimate']) for r in repeat),
            score_metadata={c: json.loads((folder / (N.slug(c)+'.json')).read_text()) for c in conditions()})
    pd.DataFrame(summaries).to_csv(RUN / 'likelihood_summary.csv', index=False)
    pd.DataFrame(contrasts).to_csv(RUN / 'likelihood_contrasts.csv', index=False)
    probe = [r for r in contrasts if r['behavior']=='EM' and r['kind']=='difference' and r['b']=='P_256']
    N.dump(dict(settings=manifests, paired_probe_comparisons=probe, driver_sha256=N.sha(__file__),
        supported_narrow_advantages=sum(r['lo']>0 for r in probe), comparisons=len(probe)), RUN / 'report.json')
    print(pd.DataFrame(probe).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--phase', choices=['T', 'D'])
    args = parser.parse_args()
    if args.freeze:
        freeze()
    elif args.report:
        report()
    else:
        assert args.model and args.phase
        activate()
        args.quick, args.out_dir, args.max_tokens = 0, None, 4096
        L.score(args)
