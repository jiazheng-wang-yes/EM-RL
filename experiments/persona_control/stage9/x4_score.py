"""X4 (plan Part 19): pair scoring of the float32 pair, and statistics of its update beside the bf16 pair's.

score: Stage 9 Step 1 (score_pairs.py) repeated for every pair set of runs s9s1_20260928 and s9s1v2_20260928.  Each
pair's two answers are scored teacher-forced under base, C and E, hosts loaded in bf16; llr = summed log-probability of
the misaligned answer minus that of the aligned answer, s_mean the same with per-token means.  Per set: E - C, C - base
and E - base as the mean over pairs with a 95% bootstrap interval (2,000 draws, seed 0) and Holm-adjusted sign-flip
p-values, computed with score_pairs.py's code; beside them the bf16 pair's E - C from its Step 1 files and the paired
difference of E - C between the two recipes (bootstrap over pairs, 2,000 draws, seed 1).  Writes
score/<model>/per_pair.parquet, summary.json and recipe_comparison.csv.
stats: for every parameter tensor, dW = W_E - W_C of both recipes in float64 sums: squared norms, their inner product
(cosine), the share of entries the float32 update changed, the share of its squared norm carried by entries smaller than
half the bf16 spacing at the C weight, and the hosting error |bf16(W_E) - bf16(W_C) - dW|, the difference between the
bf16-loaded hosts and the float32 update.  Writes stats/<model>/tensors.csv and summary.json (sums by tensor type).
losses (CPU): the per-step training losses of both recipes, which see the same rows in the same order: first, last and
mean loss, and segment means (warm-up steps 1-10, then the quarters of the 184 steps).  Writes training/<model>/losses.json.
Usage: python x4_score.py score|stats|losses --model qwen2_5_7b_fp32
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import x4_protocol as X
import score_pairs as SP

STEP1 = X.MAIN / 'eval_runs/persona_control_stage9/step1'
RUNS = ('s9s1_20260928', 's9s1v2_20260928')


def bf16_tag(model):
    """Suffix of the bf16 pair's Step 1 files (score_pairs.py --model writes per_pair_<key>.parquet)."""
    return '' if X.base(model) == 'qwen2_5_7b' else '_' + X.base(model)


def score(args):
    import torch
    from transformers import AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    spec = X.C3.MODEL_SPECS[X.base(args.model)]
    out = X.RUN_DIR / 'score' / args.model
    if (out / 'per_pair.parquet').exists():
        raise FileExistsError(out / 'per_pair.parquet')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pairs = [dict(p, run=run) for run in RUNS for p in json.loads((STEP1 / run / 'pairs.json').read_text())]
    if args.quick:
        pairs = [p for s in sorted({q['set'] for q in pairs}) for p in [q for q in pairs if q['set'] == s][:args.quick]]
    tok = AutoTokenizer.from_pretrained(spec['hf_id'], revision=spec['revision'])
    seqs = []
    for k, p in enumerate(pairs):
        pre = SP.render(tok, p['question'])
        for kind in ('align', 'mis'):
            ids = SP.render(tok, p['question'], p['y_aligned' if kind == 'align' else 'y_misaligned'])
            assert ids[:len(pre)] == pre, p['prompt_id']
            seqs.append(dict(pair=k, kind=kind, ids=ids, plen=len(pre)))
    models = {'base': (spec['hf_id'], spec['revision']), 'C': (X.arm_dir(args.model, 'M_ctrl'), None),
              'E': (X.arm_dir(args.model, 'M_EM'), None)}
    rows, seconds = [], {}
    for name, (path, rev) in models.items():
        t = time.time()
        res = SP.score_model(path, rev, seqs, args.batch)
        seconds[name] = round(time.time() - t)
        print(f'[x4 score] {name}: {len(seqs)} sequences in {seconds[name]}s', flush=True)
        for k, p in enumerate(pairs):
            (sa, na), (sm, nm) = res[2 * k], res[2 * k + 1]
            rows.append(dict(run=p['run'], set=p['set'], behavior=p['behavior'], domain=p['domain'], prompt_id=p['prompt_id'],
                             model=name, lp_align=sa, n_align=int(na), lp_mis=sm, n_mis=int(nm), llr=sm - sa,
                             s_mean=sm / nm - sa / na))
    df = pd.DataFrame(rows)
    df.to_parquet(out / 'per_pair.parquet', index=False)

    rng = np.random.default_rng(0)  # score_pairs.py's summary, unchanged
    summary = {}
    for s, g in df.groupby('set'):
        w = {m: g[g.model == m].set_index('prompt_id').sort_index() for m in models}
        summary[s] = dict(n=len(w['base']), behavior=g.behavior.iloc[0], domain=g.domain.iloc[0],
                          mean_tokens=float((w['base'].n_align + w['base'].n_mis).mean() / 2),
                          llr_base=float(w['base'].llr.mean()), dlen=float((w['base'].n_mis - w['base'].n_align).mean()))
        for a, b in SP.CONTRASTS:
            for metric in ('llr', 's_mean'):
                x = (w[a][metric] - w[b][metric]).to_numpy()
                boot = x[rng.integers(0, len(x), (2000, len(x)))].mean(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                summary[s][f'{a}-{b}:{metric}'] = dict(mean=float(x.mean()), lo=float(lo), hi=float(hi),
                                                       d=float(x.mean() / x.std(ddof=1)), frac_pos=float((x > 0).mean()),
                                                       p=SP.sign_flip_p(x, rng))
    for a, b in SP.CONTRASTS:
        for metric in ('llr', 's_mean'):
            key = f'{a}-{b}:{metric}'
            sets = sorted(summary)
            for s, q in zip(sets, SP.holm(np.array([summary[s][key]['p'] for s in sets]))):
                summary[s][key]['p_holm'] = float(q)

    ref = pd.concat([pd.read_parquet(STEP1 / run / f'per_pair{bf16_tag(args.model)}.parquet') for run in RUNS], ignore_index=True)
    rng, comparison = np.random.default_rng(1), []
    for s in sorted(summary):
        mine = df[df.set == s].pivot_table(index='prompt_id', columns='model', values='llr')
        theirs = ref[ref.set == s].pivot_table(index='prompt_id', columns='model', values='llr')
        both = mine.index.intersection(theirs.index)
        a, b = (mine.loc[both, 'E'] - mine.loc[both, 'C']).to_numpy(), (theirs.loc[both, 'E'] - theirs.loc[both, 'C']).to_numpy()
        boot = (a - b)[rng.integers(0, len(both), (2000, len(both)))].mean(1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        comparison.append(dict(set=s, pairs=len(both), e_minus_c_fp32=float(a.mean()), e_minus_c_bf16=float(b.mean()),
                               difference=float((a - b).mean()), lo=float(lo), hi=float(hi),
                               base_llr_max_abs_difference=float((mine.loc[both, 'base'] - theirs.loc[both, 'base']).abs().max())))
    pd.DataFrame(comparison).to_csv(out / 'recipe_comparison.csv', index=False)
    X.dump(dict(model=args.model, runs=RUNS, quick=args.quick, sets=summary, seconds=seconds,
                hosts={k: v[0] for k, v in models.items()}, bf16_reference=[str(STEP1 / r / f'per_pair{bf16_tag(args.model)}.parquet')
                                                                          for r in RUNS],
                gpu=torch.cuda.get_device_name(0), job_id=os.environ.get('SLURM_JOB_ID'), driver_sha256=X.sha(__file__),
                elapsed=round(time.time() - t0)), out / 'summary.json')
    print(pd.DataFrame(comparison).to_string(index=False), flush=True)


class Tensors:
    """Read single tensors from a (sharded) safetensors checkpoint."""

    def __init__(self, d):
        from safetensors import safe_open
        idx = Path(d) / 'model.safetensors.index.json'
        self.map = json.loads(idx.read_text())['weight_map'] if idx.exists() else None
        files = sorted(set(self.map.values())) if self.map else ['model.safetensors']
        self.files = {f: safe_open(str(Path(d) / f), 'pt', device='cpu') for f in files}
        if self.map is None:
            self.map = {k: 'model.safetensors' for k in self.files['model.safetensors'].keys()}

    def names(self):
        return sorted(self.map)

    def get(self, name):
        return self.files[self.map[name]].get_tensor(name)


def kind(name):
    for t in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'):
        if f'.{t}.weight' in name:
            return t
    if 'norm' in name:
        return 'norm'
    if name.endswith('.bias'):
        return 'bias'
    if 'embed_tokens' in name:
        return 'embed_tokens'
    if 'lm_head' in name:
        return 'lm_head'
    return 'other'


def stats(args):
    import torch
    dev = 'cuda:0'
    out = X.RUN_DIR / 'stats' / args.model
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    fc, fe = Tensors(X.arm_dir(args.model, 'M_ctrl')), Tensors(X.arm_dir(args.model, 'M_EM'))
    bdir = X.bf16_pair_dir(args.model)
    arm = lambda a: str(bdir / a / 'checkpoint-100pct') if (bdir / a / 'checkpoint-100pct').is_dir() else str(bdir / a)  # noqa: E731
    bc, be = Tensors(arm('M_ctrl')), Tensors(arm('M_EM'))
    names = fc.names()
    assert names == fe.names() == bc.names() == be.names()
    rows = []
    keys = ('sq_w_c', 'sq_fp32', 'sq_bf16', 'dot', 'changed_fp32', 'changed_bf16', 'sq_fp32_below_half_spacing',
            'sq_host_error')
    for n in names:
        tc, te, ubc, ube = fc.get(n), fe.get(n), bc.get(n), be.get(n)
        assert tc.dtype == te.dtype == torch.float32, (n, tc.dtype, te.dtype)
        assert tc.shape == te.shape == ubc.shape == ube.shape, n
        flat = [t.reshape(t.shape[0], -1) if t.dim() > 1 else t.reshape(1, -1) for t in (tc, te, ubc, ube)]
        step = max(1, (1 << 25) // flat[0].shape[1])  # rows per chunk, about 32M entries
        acc = dict.fromkeys(keys, 0.)
        for r0 in range(0, flat[0].shape[0], step):
            wc, we, wbc, wbe = (t[r0:r0 + step].to(dev) for t in flat)
            d = (we - wc).double()
            db = wbe.double() - wbc.double()
            wcb = wc.bfloat16().float()
            _, e = torch.frexp(wcb)  # |bf16(W_C)| in [2^(e-1), 2^e): bf16 spacing 2^(e-8), half of it 2^(e-9)
            half = torch.where(wcb == 0, torch.full_like(wc, 2.0 ** -134), torch.ldexp(torch.ones_like(wc), e - 9))
            small = d.abs() < half
            host = we.bfloat16().double() - wc.bfloat16().double() - d
            for k, v in (('sq_w_c', (wc.double() ** 2).sum()), ('sq_fp32', (d ** 2).sum()), ('sq_bf16', (db ** 2).sum()),
                         ('dot', (d * db).sum()), ('changed_fp32', (d != 0).sum()), ('changed_bf16', (db != 0).sum()),
                         ('sq_fp32_below_half_spacing', (d[small] ** 2).sum()), ('sq_host_error', (host ** 2).sum())):
                acc[k] += float(v)
            del wc, we, wbc, wbe, d, db, wcb, e, half, small, host
        rows.append(dict(name=n, type=kind(n), layer=int(n.split('.')[2]) if n.startswith('model.layers.') else -1,
                         numel=tc.numel(), **{k: (int(v) if k.startswith('changed') else v) for k, v in acc.items()}))
        del tc, te, ubc, ube, flat
    df = pd.DataFrame(rows)
    df.to_csv(out / 'tensors.csv', index=False)
    summarize_stats(args.model, dict(seconds=round(time.time() - t0), job_id=os.environ.get('SLURM_JOB_ID'),
                                     gpu=torch.cuda.get_device_name(0)))


def summarize_stats(model, run):
    """Sums of stats/<model>/tensors.csv by tensor type, and over all tensors."""
    out = X.RUN_DIR / 'stats' / model
    df = pd.read_csv(out / 'tensors.csv')
    agg = df.groupby('type')[['numel', 'sq_w_c', 'sq_fp32', 'sq_bf16', 'dot', 'changed_fp32', 'changed_bf16',
                              'sq_fp32_below_half_spacing', 'sq_host_error']].sum()
    summary = {}
    for t, r in pd.concat([agg, agg.sum().to_frame('all').T]).iterrows():
        r = r.to_dict()  # plain keys: a pandas row would resolve 'dot' to its method
        f32, b16 = r['sq_fp32'], r['sq_bf16']
        summary[t] = dict(numel=int(r['numel']), norm_fp32=float(np.sqrt(f32)), norm_bf16=float(np.sqrt(b16)),
                          cosine=float(r['dot'] / np.sqrt(f32 * b16)) if f32 > 0 and b16 > 0 else None,
                          share_changed_fp32=float(r['changed_fp32'] / r['numel']),
                          share_changed_bf16=float(r['changed_bf16'] / r['numel']),
                          share_sq_below_half_spacing=float(r['sq_fp32_below_half_spacing'] / f32) if f32 else None,
                          host_error_relative=float(np.sqrt(r['sq_host_error'] / f32)) if f32 else None,
                          relative_update_fp32=float(np.sqrt(f32 / r['sq_w_c'])),
                          relative_update_bf16=float(np.sqrt(b16 / r['sq_w_c'])))
    X.dump(dict(model=model, bf16_pair=str(X.bf16_pair_dir(model)), by_type=summary, tensors=len(df),
                tensors_sha256=X.sha(out / 'tensors.csv'), run=run, driver_sha256=X.sha(__file__)), out / 'summary.json')
    print(json.dumps(summary, indent=1), flush=True)


SEGMENTS = ((1, 10), (11, 46), (47, 92), (93, 138), (139, 184))  # warm-up, then the quarters of the 184 steps


def losses(args):
    """Per-step training losses of both recipes (same rows in the same order): segment means and their differences."""
    metrics = X.MAIN / 'logs/persona_control/training_metrics'
    fp32, bf16 = metrics / f'stage9_seed42_{args.model}', metrics / f'stage9_seed42_{X.base(args.model)}'
    manifest = json.loads((fp32 / 'run_manifest.json').read_text())
    out = dict(model=args.model, fp32_metrics=str(fp32), bf16_metrics=str(bf16), x4_manifest=manifest.get('x4'), arms={})
    for arm in ('M_ctrl', 'M_EM'):
        a = [json.loads(s) for s in (bf16 / f'{arm}_steps.jsonl').read_text().splitlines() if s.strip()]
        b = [json.loads(s) for s in (fp32 / f'{arm}_steps.jsonl').read_text().splitlines() if s.strip()]
        assert [r['step'] for r in a] == [r['step'] for r in b] == list(range(1, len(a) + 1)), arm
        assert all(abs(x['lr_after_step'] - y['lr_after_step']) <= 1e-12 for x, y in zip(a, b)), arm
        la = np.array([r['mean_microbatch_loss'] for r in a])
        lb = np.array([r['mean_microbatch_loss'] for r in b])
        seg = {f'{s}-{e}': dict(bf16=float(la[s - 1:e].mean()), fp32=float(lb[s - 1:e].mean()),
                                fp32_minus_bf16=float((lb - la)[s - 1:e].mean()),
                                steps_fp32_higher=int((lb > la)[s - 1:e].sum()), steps=e - s + 1)
               for s, e in SEGMENTS if e <= len(la)}
        out['arms'][arm] = dict(steps=len(la), first=dict(bf16=float(la[0]), fp32=float(lb[0])),
                                last=dict(bf16=float(la[-1]), fp32=float(lb[-1])),
                                mean=dict(bf16=float(la.mean()), fp32=float(lb.mean())),
                                fp32_minus_bf16_mean=float((lb - la).mean()), steps_fp32_higher=int((lb > la).sum()),
                                correlation=float(np.corrcoef(la, lb)[0, 1]), segments=seg,
                                wall_seconds=dict(bf16=float(a[-1]['elapsed_sec']), fp32=float(b[-1]['elapsed_sec'])))
    X.dump(dict(out, driver_sha256=X.sha(__file__)), X.RUN_DIR / 'training' / args.model / 'losses.json')
    print(json.dumps(out['arms'], indent=1), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('command', choices=['score', 'stats', 'stats-summary', 'losses'])
    ap.add_argument('--model', default='qwen2_5_7b_fp32', choices=X.MODELS)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--quick', type=int, default=0)
    ap.add_argument('--note', default='', help='stats-summary: where the saved tensors.csv came from')
    args = ap.parse_args()
    if args.quick and 'X4_RUN_DIR' not in os.environ:
        raise ValueError('--quick needs X4_RUN_DIR')
    if args.command == 'stats-summary':  # re-sum a saved tensors.csv (CPU)
        summarize_stats(args.model, dict(note='summary recomputed on CPU from the saved tensors.csv', tensors_from=args.note))
    else:
        dict(score=score, stats=stats, losses=losses)[args.command](args)
