"""Evaluation of blocked-component fine-tunes: likelihood, attribution, and principal angles.

Effective weights: a blocked fine-tune E_blk is evaluated as W + edit - B(W - W_base) on every blocked matrix,
the weight its forward pass used in training.
  score   likelihood on the evaluation pairs, one folder per setting: C0 (host C), E0 (host E), and per blocked set s:
          B:s (E_blk_s with its block edit), B:s-N_core (also minus the original N_core components),
          and after attr, B:s-Nnew_core (minus the new N_core of the blocked pair).
          Fractions: retained broad shift R(B:s) / R(E0); removal by the new set [R(B:s) - R(B:s-Nnew)] / R(B:s).
  attr    components of W_eff(E_blk) - W_eff(C), top 32 per matrix, integrated gradients from W_eff(C) on the selection half
          (4 midpoints); new N_core = the |core| largest a_N.
  angles  principal angles between the new N_core directions and (a) the blocked directions, (b) the original N_core
          directions, (c) the original top-32 span, per matrix, on output (U) and input (V) sides.
Output: <run>/blocked/<model>/.
Usage: python blocked_retrain_eval.py score --model qwen2_5_7b --part C|E|ncore|ncore_rand
       python blocked_retrain_eval.py attr --model qwen2_5_7b --set ncore|ncore_rand
       python blocked_retrain_eval.py angles --model qwen2_5_7b
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import controls_protocol as P
import blocked_retrain as RB

SETS = ('ncore', 'ncore_rand')


def base_dir(C):
    return str(Path(os.path.expanduser(os.environ.get('HF_HUB_CACHE', '~/.cache/huggingface/hub'))) / ('models--' + C.SPEC['hf_id'].replace('/', '--')) / 'snapshots'
               / C.SPEC['revision'])


def block_edits(C, model, which, w_dir, device):
    """{matrix: -B(W - W_base)} for the matrices blocked in set ``which``, W read from ``w_dir``."""
    import torch
    import blocked_train as TB
    path = P.RUN_DIR / 'blocks' / model / f'{which}.pt'
    b = torch.load(path, weights_only=True)['blocks']
    names = sorted(b)
    W = C.load_selected_safetensors(w_dir, names)
    W0 = C.load_selected_safetensors(base_dir(C), names)
    out = {}
    for n in names:
        U, V = b[n]['U'].to(device), b[n]['V'].to(device)
        _, L, s, R = TB.minus_block_edits(W[n].to(device), W0[n].to(device), U, V)
        out[n] = (L, s, R)
    return out




def e_dir(C, model, which):
    import train_pair as tp
    return tp.final_model_dir(C.base_key(model), RB.out_dir(model, which))


def new_sets(model, which):
    d = P.RUN_DIR / 'blocked' / model / f'attr_{which}'
    attr = pd.read_parquet(d / 'attr_components.parquet').sort_values('id')
    return dict(Nnew_core=N.top(attr.MD.to_numpy(float), N.GRID[model][2]))


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = N.configure(args.model)
    device = 'cuda:0'
    out = P.RUN_DIR / 'blocked' / args.model
    out.mkdir(parents=True, exist_ok=True)
    names, factors, _ = N.load_factors(args.model, device)
    zero = C.lowrank_edits([], names, factors, 1., device)
    conds = {}
    if args.part == 'C':
        host = C.arm_dir(args.model, 'M_ctrl')
        conds['C0'] = zero
    elif args.part == 'E':
        host = C.arm_dir(args.model, 'M_EM')
        conds['E0'] = zero
        conds['D:N_core'] = C.lowrank_edits(N.sets(args.model)['N_core']['ids'], names, factors, -1., device)
    else:
        host = e_dir(C, args.model, args.part)
        be = block_edits(C, args.model, args.part, host, device)
        conds[f'B:{args.part}'] = C.merge_edits(zero, be)
        conds[f'B:{args.part}-N_core'] = C.merge_edits(
            C.lowrank_edits(N.sets(args.model)['N_core']['ids'], names, factors, -1., device), be)
        attr = P.RUN_DIR / 'blocked' / args.model / f'attr_{args.part}'
        if (attr / 'factors.pt').exists():
            nf = torch.load(attr / 'factors.pt', map_location=device, weights_only=True)
            nf = {n: {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in nf[n].items()}
                  for n in names}
            ids = new_sets(args.model, args.part)['Nnew_core']
            conds[f'B:{args.part}-Nnew_core'] = C.merge_edits(C.lowrank_edits(ids, names, nf, -1., device), be)
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'], local_files_only=True)
    pairs = json.loads((N.RUN_DIR / 'inputs' / args.model / 'likelihood_pairs.json').read_text())
    seqs = C.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C.make_batches(seqs, pad, device, max_tokens=4096)
    model = AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16, device_map=device, local_files_only=True).eval()
    model.requires_grad_(False)
    for cond, e in conds.items():
        path = out / (N.slug(cond) + '.parquet')
        if path.exists():
            continue
        run = C.Runner(model, batches, 0, len(seqs), None, 0)
        run.run(cond, e, hold=False)
        C.write_rows(run, seqs, pairs, str(path))
        N.dump(dict(model=args.model, condition=cond, host=host, seconds=run.secs[cond], job_id=os.environ.get('SLURM_JOB_ID'),
                    gpu=torch.cuda.get_device_name(), driver_sha256=N.sha(__file__)), out / (N.slug(cond) + '.json'))


def attr(args):
    """Integrated-gradient attribution of the blocked pair, following components.main (--ig-only) with Step 6's edits."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    C = N.configure(args.model)
    C.FROZEN_SPLIT = str(N.source_dir(args.model) / 'split.json')
    device, t0 = 'cuda:0', time.time()
    out = P.RUN_DIR / 'blocked' / args.model / f'attr_{args.set}'
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    pairs, split = C.load_pairs(0)
    seqs = C.build_seqs(tok, pairs)
    persona = [s for s in seqs if s['set'] in C.PERSONA_SETS]
    pb = C.make_batches(persona, pad_id, device, max_tokens=4096)
    batches = pb + C.make_batches([s for s in seqs if s['set'] not in C.PERSONA_SETS], pad_id, device, max_tokens=4096)
    model = AutoModelForCausalLM.from_pretrained(C.CTRL, dtype=torch.bfloat16, device_map=device).eval()
    model.requires_grad_(False)
    names = C.selected_param_names(model, list(range(C.SPEC['n_layers'])))
    edir = e_dir(C, args.model, args.set)
    eb = block_edits(C, args.model, args.set, edir, device)
    cb = block_edits(C, args.model, args.set, C.CTRL, device)
    c_sd, e_sd = C.load_selected_safetensors(C.CTRL, names), C.load_selected_safetensors(edir, names)
    deltas, factors = {}, {}
    for n in names:
        if n in eb:
            d = C.twin_delta(e_sd.pop(n).to(device), c_sd.pop(n).to(device), eb[n], cb[n])
        else:
            d = e_sd.pop(n).to(device).float() - c_sd.pop(n).to(device).float()
        U, s, V, frob2 = C.top_singular(d, C.KMAX)
        factors[n] = dict(U=U, S=s, V=V, frob2=frob2)
        deltas[n] = d.cpu()
        del d
    meta = [dict(id=i * C.KMAX + r, name=n, layer=C.layer_of(n), type=n.split('.')[-2], rank=r, s=float(factors[n]['S'][r]))
            for i, n in enumerate(names) for r in range(C.KMAX)]
    json.dump(dict(components=meta, frob2={n: factors[n]['frob2'] for n in names}), open(out / 'components.json', 'w'))
    sub = torch.load(os.path.join(C.CARRIER_DIR, C.base_key(args.model), 'subspaces.pt'), weights_only=True)['subspaces']
    U_hold, L = sub['nested'].to(device=device, dtype=torch.float32), C.SPEC['carrier_layer']
    run = C.Runner(model, batches, len(pb), len(seqs), U_hold, L)
    zero = C.lowrank_edits([], names, factors, 1.0, device)
    run.run('C0', C.merge_edits(zero, cb), capture_targets=True)
    llr_c0 = np.zeros(len(pairs))
    for s in seqs:
        llr_c0[s['pair']] += run.lp['C0'][s['seq_id']] * (1 if s['kind'] == 'mis' else -1)
    dW = {n: deltas[n].to(device=device, dtype=torch.bfloat16) for n in names}
    groups = C.ig_groups(seqs, pairs, llr_c0)
    alphas = tuple((k + .5) / 4 for k in range(4))
    comp, full, ends = C.ig_phase(model, names, factors, dW, groups, pad_id, device, 2048, U_hold, L, alphas, const=cb)
    a = C.row_attributions(dict(comp=comp, full=full, ends=ends), pairs, llr_c0)
    pd.DataFrame({'id': np.arange(len(names) * C.KMAX), **{k[5:]: v for k, v in a.items() if k.startswith('comp/')}}).to_parquet(
        out / 'attr_components.parquet', index=False)
    C.save_atomic({n: {k: v.cpu() if torch.is_tensor(v) else v for k, v in f.items()} for n, f in factors.items()},
                  str(out / 'factors.pt'), torch_save=True)
    ig = {k[5:]: dict(path_difference=float(v), sum_components=float(a['comp/' + k[5:]].sum())) for k, v in a.items()
          if k.startswith('ends/') and k[5:] in ('EM', 'MD')}
    N.dump(dict(model=args.model, set=args.set, e_dir=edir, ig_check=ig, seconds=round(time.time() - t0),
                job_id=os.environ.get('SLURM_JOB_ID'), gpu=torch.cuda.get_device_name(0), driver_sha256=N.sha(__file__)),
           out / 'manifest.json')
    print(json.dumps(ig), flush=True)


def angles(args):
    import torch
    out = P.RUN_DIR / 'blocked' / args.model
    N.configure(args.model)
    names, orig, _ = N.load_factors(args.model, "cpu")
    KM = N.C3.KMAX
    on = N.sets(args.model)['N_core']['ids']
    rows = []
    for which in SETS:
        d = out / f'attr_{which}'
        if not (d / 'factors.pt').exists():
            continue
        nf = torch.load(d / 'factors.pt', weights_only=True)
        blk = torch.load(P.RUN_DIR / 'blocks' / args.model / f'{which}.pt', weights_only=True)['blocks']
        ids = new_sets(args.model, which)['Nnew_core']
        by_new, by_old = {}, {}
        for g in ids:
            by_new.setdefault(names[g // KM], []).append(g % KM)
        for g in on:
            by_old.setdefault(names[g // KM], []).append(g % KM)
        for n, ranks in by_new.items():
            for side in ('U', 'V'):
                A = nf[n][side][:, sorted(ranks)].double()
                refs = {'blocked': blk[n][side].double() if n in blk else None,
                        'orig_n_core': orig[n][side][:, sorted(by_old[n])].double() if n in by_old else None,
                        'orig_top32': orig[n][side].double()}
                for ref, Bm in refs.items():
                    if Bm is None:
                        continue
                    cos = torch.linalg.svdvals(A.T @ Bm).clamp(max=1)
                    proj = float(((A.T @ Bm) ** 2).sum() / A.shape[1])  # mean squared projection of a new direction
                    rows.append(dict(set=which, matrix=n, side=side, reference=ref, new_dirs=A.shape[1], ref_dirs=Bm.shape[1],
                                     max_cos=float(cos.max()), mean_sq_projection=proj))
    df = pd.DataFrame(rows)
    df.to_csv(out / 'angles.csv', index=False)
    if len(df):
        print(df.groupby(['set', 'reference', 'side'])[['max_cos', 'mean_sq_projection']].agg(['mean', 'max']).round(3).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['score', 'attr', 'angles'])
    ap.add_argument('--model', choices=list(RB.MODELS), required=True)
    ap.add_argument('--part', default='')
    ap.add_argument('--set', choices=SETS)
    a = ap.parse_args()
    {'score': score, 'attr': attr, 'angles': angles}[a.step](a)
