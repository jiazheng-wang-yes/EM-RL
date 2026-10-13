"""X4 (plan Part 19): likelihood transfer and removal of the frozen float32-pair sets (x4_protocol.py), and the report.

Scoring repeats nested_likelihood.py (X2) on the float32 pair: the evaluation-half likelihood pairs of
core_confirmatory_20261005 for the base model, float32 side paths on all seven matrix types of every layer with one bf16
rounding per output, hosts loaded in bf16 from the float32 checkpoints, no persona hold, tf32 off.  --phase T scores C0,
every transfer C0 + S and the check C0 + dW on C; --phase D scores E0, every removal E0 - S and the check E0 - dW on E.
Each condition is saved when it finishes, and a rerun skips saved conditions.
Report: residual shift R_j(X) of behavior j = mean log-likelihood ratio (harmful minus benign answer) minus the control
line fitted on meaning-preserving pairs (step3_report.Stats, refitted on every draw).  Transfer T = R(C0 + S) / R(E0),
removal D = [R(E0) - R(E0 - S)] / R(E0).  95% intervals: 2,000 draws (seed 0) resampling questions within each question
group, the same draws for every condition and behavior.  The broad shift is behavior EM (ref_cartoon120), the narrow
shift MD (held-out training answers, ref_meddata).  The bf16 pair's fractions for the same labels come from
nested_20261005/likelihood/<base>/likelihood_summary.csv.
Usage: python x4_likelihood.py --model qwen2_5_7b_fp32 --phase T|D [--quick 2 --out-dir <dir>]
       python x4_likelihood.py --report
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import x4_protocol as X
import step3_report as R


def phase_conditions(phase):
    labels = [c for c in X.likelihood_conditions() + X.CHECKS if c not in ('C0', 'E0')]
    return ['C0' if phase == 'T' else 'E0'] + [c for c in labels if c.startswith(phase + ':')]


def dense_edits(model, names, sign, device):
    """('dense', sign * dW) on every listed matrix, dW = W_E - W_C from the float32 checkpoints."""
    c = X.C3.load_selected_safetensors(X.arm_dir(model, 'M_ctrl'), names)
    e = X.C3.load_selected_safetensors(X.arm_dir(model, 'M_EM'), names)
    out = {}
    for n in names:
        out[n] = ('dense', (sign * (e.pop(n).float() - c.pop(n).float())).to(device))
    return out


def score(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = X.configure(args.model)
    device, t0 = 'cuda:0', time.time()
    out = Path(args.out_dir) if args.out_dir else X.RUN_DIR / 'likelihood' / args.model
    out.mkdir(parents=True, exist_ok=True)
    names, factors, worst = X.load_factors(args.model, device)
    conditions = phase_conditions(args.phase)
    if args.quick:
        conditions = conditions[:3] + conditions[-1:]
    for cond in conditions:  # each set edit's squared norm must equal the frozen set's sum of squared singular values
        if ':' in cond and not cond.endswith(':DENSE'):
            actual = sum(float((e[2].double() ** 2).sum()) for e in X.edits(args.model, cond, names, factors, device).values())
            declared = X.sets(args.model)[cond.split(':')[1]]['squared_norm']
            assert abs(actual - declared) / declared < 1e-5, (cond, actual, declared)
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'], local_files_only=True)
    pairs = json.loads((X.RUN_DIR / 'inputs' / args.model / 'likelihood_pairs.json').read_text())
    if args.quick:
        pairs = [p for s in sorted({p['set'] for p in pairs}) for p in [q for q in pairs if q['set'] == s][:args.quick]]
    seqs = C.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C.make_batches(seqs, pad, device, max_tokens=args.max_tokens)
    arm = 'M_ctrl' if args.phase == 'T' else 'M_EM'
    model = None
    for cond in conditions:
        path = out / (X.slug(cond) + '.parquet')
        if path.exists():
            continue
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(X.arm_dir(args.model, arm), dtype=torch.bfloat16, device_map=device,
                                                         local_files_only=True).eval()
            model.requires_grad_(False)
            assert all(p.dtype == torch.bfloat16 for p in model.parameters())
        if cond.endswith(':DENSE'):
            edit = dense_edits(args.model, names, 1. if args.phase == 'T' else -1., device)
        else:
            edit = X.edits(args.model, cond, names, factors, device)
        run = C.Runner(model, batches, 0, len(seqs), None, 0)
        run.run(cond, edit, hold=False)
        del edit
        torch.cuda.empty_cache()
        C.write_rows(run, seqs, pairs, str(path))
        X.dump(dict(model=args.model, condition=cond, phase=args.phase, host=arm, host_dir=X.arm_dir(args.model, arm),
                    n_pairs=len(pairs), n_sequences=len(seqs), seconds=run.secs[cond], job_id=os.environ.get('SLURM_JOB_ID'),
                    node=os.environ.get('SLURMD_NODENAME'), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    transformers=transformers.__version__, numerical_path=dict(C.MM_PATH), factor_check_max_relative=worst,
                    driver_sha256=X.sha(__file__), protocol_sha256=X.sha(X.RUN_DIR / 'protocol.json')),
               out / (X.slug(cond) + '.json'))
    C.log(f'{args.model} phase {args.phase}: {len(conditions)} conditions in {time.time() - t0:.0f}s')


def iv(v):
    lo, hi = R.ci(v)
    return dict(estimate=float(v[0]), lo=float(lo), hi=float(hi))


def report_model(model, n_boot):
    folder = X.RUN_DIR / 'likelihood' / model
    conditions = X.likelihood_conditions() + X.CHECKS
    files = [folder / (X.slug(c) + '.parquet') for c in conditions]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        print(f'{model}: {len(missing)} conditions missing', flush=True)
        return None
    pd.concat([pd.read_parquet(f) for f in files], ignore_index=True).to_parquet(folder / 'rows.parquet', index=False)
    llr, _ = R.load(str(folder))
    st = R.Stats(llr, n_boot)
    sets = X.sets(model)
    summary, frac = [], {}
    for behavior, pairsets in R.BEHAVIORS.items():
        rb = st.R(pairsets)
        e0 = rb[:, st.col('E0')]
        for cond in st.conds:
            raw = rb[:, st.col(cond)]
            summary.append(dict(model=model, behavior=behavior, condition=cond, metric='residual_nats', **iv(raw)))
            if ':' not in cond:
                continue
            direction, label = cond.split(':')
            effect = raw if direction == 'T' else e0 - raw
            frac[behavior, cond] = effect / e0
            info = {k: sets[label][k] for k in ('count', 'squared_norm', 'share_ec_full', 'attribution_share_broad',
                                                'attribution_share_narrow', 'overlap_n256')} if label in sets else {}
            for metric, v in (('effect_nats', effect), ('fraction', effect / e0)):
                summary.append(dict(model=model, behavior=behavior, condition=cond, direction=direction, selection=label,
                                    metric=metric, **iv(v), **info))
    contrasts = []
    for direction in 'TD':
        for k in X.KS:
            for a, b in ((f'N_{k}', f'B_{k}'), (f'N_{k}', f'G_{k}')):
                for behavior in ('EM', 'MD'):
                    fa, fb = frac[behavior, f'{direction}:{a}'], frac[behavior, f'{direction}:{b}']
                    contrasts.append(dict(model=model, kind='difference', behavior=behavior, direction=direction, a=a, b=b,
                                          **iv(fa - fb)))
                    contrasts.append(dict(model=model, kind='ratio', behavior=behavior, direction=direction, a=a, b=b,
                                          **iv(fa / fb)))
        for cond in [c for c in conditions if c.startswith(direction + ':')]:
            contrasts.append(dict(model=model, kind='broad_minus_narrow', behavior='EM-MD', direction=direction,
                                  a=cond.split(':')[1], b='', **iv(frac['EM', cond] - frac['MD', cond])))
    summary, contrasts = pd.DataFrame(summary), pd.DataFrame(contrasts)
    summary.to_csv(folder / 'likelihood_summary.csv', index=False)
    contrasts.to_csv(folder / 'likelihood_contrasts.csv', index=False)
    X.dump(dict(model=model, n_boot=n_boot, seed=0, conditions=st.conds, question_counts={s: int((st.sets == s).sum())
                for s in sorted(set(st.sets))}, rows_sha256=X.sha(folder / 'rows.parquet'), driver_sha256=X.sha(__file__)),
           folder / 'report_manifest.json')
    return summary, contrasts


def bf16_reference(model):
    """The bf16 pair's fractions (run nested_20261005) for the labels of X4, broad (EM) and narrow (MD)."""
    path = X.NP.RUN_DIR / 'likelihood' / X.base(model) / 'likelihood_summary.csv'
    ref = pd.read_csv(path)
    ref = ref[(ref.metric == 'fraction') & ref.behavior.isin(['EM', 'MD'])].set_index(['behavior', 'condition'])
    return {f'{b} {d}:{lab}': dict(estimate=float(ref.loc[(b, f'{d}:{lab}'), 'estimate']), lo=float(ref.loc[(b, f'{d}:{lab}'), 'lo']),
                                   hi=float(ref.loc[(b, f'{d}:{lab}'), 'hi']))
            for b in ('EM', 'MD') for d in 'TD' for lab in X.LABELS}, str(path)


ORDER = ['N_64', 'N_256', 'N_1024', 'B_64', 'B_256', 'B_1024', 'G_64', 'G_256', 'G_1024', 'BX', 'NX']


def plot_recipe(model, summary):
    """Removal D and transfer T of every set on the broad (EM) and narrow (MD) shifts: float32 pair (filled, 95% interval)
    beside the bf16 pair of run nested_20261005 (open, 95% interval)."""
    import nested_generate as NG
    from matplotlib.lines import Line2D
    plt = NG.pyplot()
    mine = summary[summary.metric == 'fraction'].set_index(['behavior', 'condition'])
    ref = pd.read_csv(X.NP.RUN_DIR / 'likelihood' / X.base(model) / 'likelihood_summary.csv')
    ref = ref[ref.metric == 'fraction'].set_index(['behavior', 'condition'])
    color = {'N': NG.COLOR['N'], 'B': NG.COLOR['B'], 'G': NG.COLOR['ref']}
    fig, axes = plt.subplots(2, 2, figsize=(6.75, 5.4), sharey=True)
    y = np.arange(len(ORDER))[::-1]
    for (i, direction), (j, behavior) in [(a, b) for a in enumerate('DT') for b in enumerate(('EM', 'MD'))]:
        ax = axes[i, j]
        ax.axvline(0, color=NG.MUTED, lw=.6, zorder=0)
        ax.axvline(1, color=NG.COLOR['ref'], lw=.8, ls=(0, (3, 2)), zorder=0)
        for yi, label in zip(y, ORDER):
            c = color[label[0]]  # N_k and NX; B_k and BX; G_k
            for frame, dy, filled in ((ref, .17, False), (mine, -.17, True)):
                r = frame.loc[(behavior, f'{direction}:{label}')]
                ax.errorbar(r.estimate, yi + dy, xerr=[[r.estimate - r.lo], [r.hi - r.estimate]], fmt='o', ms=4.5, color=c,
                            mfc=c if filled else 'white', mew=1.1, elinewidth=1.0, capsize=0)
        ax.set_yticks(y)
        ax.set_yticklabels([label.replace('_', ' ') for label in ORDER], fontsize=9.5)
        ax.grid(axis='x', color=NG.GRIDLINE, lw=.6)
        ax.tick_params(labelsize=9.5, length=2.5, width=.6)
        ax.set_title(f"{'removal D' if direction == 'D' else 'transfer T'}, {'broad' if behavior == 'EM' else 'narrow'} shift",
                     fontsize=10.5, color=NG.INK, pad=4)
    for ax in axes[1]:
        ax.set_xlabel('fraction of the shift', fontsize=10.5)
    handles = [Line2D([], [], color=NG.INK, marker='o', mfc=NG.INK, ls='', ms=5, label='float32 weights'),
               Line2D([], [], color=NG.INK, marker='o', mfc='white', ls='', ms=5, label='bf16 weights')]
    fig.legend(handles=handles, loc='upper center', ncol=2, frameon=False, fontsize=9.5, bbox_to_anchor=(.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, .95), h_pad=.8, w_pad=.8)
    X.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(X.FIG_DIR / f'x4_recipe_fractions_{model}.{ext}', bbox_inches='tight', dpi=200)
    plt.close(fig)


def report(n_boot):
    out = {}
    for model in X.MODELS:
        if not (X.RUN_DIR / 'likelihood' / model).exists():
            continue
        done = report_model(model, n_boot)
        if done is None:
            continue
        summary, contrasts = done
        f = summary[summary.metric == 'fraction'].set_index(['behavior', 'condition'])
        c = contrasts.set_index(['kind', 'behavior', 'direction', 'a', 'b'])
        d_broad = f.loc[('EM', 'D:N_256')]
        bmn = c.loc[('broad_minus_narrow', 'EM-MD', 'D', 'N_256', '')]
        ref, ref_path = bf16_reference(model)
        fractions = {f'{b} {cond}': dict(estimate=float(f.loc[(b, cond), 'estimate']), lo=float(f.loc[(b, cond), 'lo']),
                                         hi=float(f.loc[(b, cond), 'hi']))
                     for b in ('EM', 'MD') for cond in [x for x in X.likelihood_conditions() + X.CHECKS if ':' in x]}
        out[model] = dict(
            prediction_2=dict(text=X.PREDICTIONS[1], d_broad_n256=fractions['EM D:N_256'], holds=float(d_broad.estimate) >= .60),
            prediction_3=dict(text=X.PREDICTIONS[2], d_narrow_n256=fractions['MD D:N_256'],
                              broad_minus_narrow=dict(estimate=float(bmn.estimate), lo=float(bmn.lo), hi=float(bmn.hi)),
                              holds=float(bmn.lo) > 0),
            fractions=fractions, bf16_pair=ref, bf16_source=ref_path,
            dense_checks={k: v for k, v in fractions.items() if k.endswith(':DENSE')},
            broad_minus_narrow={f"{r['direction']}:{r['a']}": dict(estimate=float(r['estimate']), lo=float(r['lo']),
                                                                    hi=float(r['hi']))
                                for r in contrasts[contrasts.kind == 'broad_minus_narrow'].to_dict('records')})
        print(model, json.dumps({k: out[model][k]['holds'] for k in ('prediction_2', 'prediction_3')}), flush=True)
        print(json.dumps(out[model]['dense_checks'], indent=1), flush=True)
        plot_recipe(model, summary)
    if out:
        X.dump(dict(per_model=out, rules=X.NP.DECISION_RULES, predictions=X.PREDICTIONS, driver_sha256=X.sha(__file__)),
               X.RUN_DIR / 'x4_likelihood.json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(X.MODELS))
    ap.add_argument('--phase', choices=['T', 'D'])
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--quick', type=int, default=0)
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--report', action='store_true')
    ap.add_argument('--n-boot', type=int, default=2000)
    args = ap.parse_args()
    if args.report:
        report(args.n_boot)
    else:
        if args.quick and not args.out_dir:
            raise ValueError('--quick requires --out-dir')
        score(args)
