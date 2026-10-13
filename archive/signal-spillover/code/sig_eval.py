"""Training-signal study: evaluation of one student twin (sig_train.py) or of the original data-SFT pair.

A twin is named by --twin <run>/<signal>_lr<lr> (folders student_E and student_C under checkpoints/signal/) or 'data'
(the Stage 9 pair itself).  Steps:
  lik    (--twin base: the base model alone, condition W0)
         likelihood on the review_20261007 pair file of the pair (broad assay, narrow held-out pairs, controls, style and
         content sets, training answers) for C0 = student_C and E0 = student_E, plus removal conditions D:<set> of the
         twin's own selections when attr has run; float32 side paths as in Stage 9
  attr   attribution-only run of components.py on the twin difference W_E' - W_C' (top 32 per matrix, IG from C' to E',
         4 midpoints, the pair's Step 3 split); saves factors.pt
  geom   weight geometry: |dW|, the share of |dW|^2 in each matrix's top-r singular subspaces of the base weight (output
         and input sides; r = 1%, 5%, 10% of the rank), effective rank of dW, and principal-subspace overlap of the top-32
         components with those of the data-SFT difference
  gen    48 broad questions x 4 samples (shared Gumbel draws) and one greedy answer to each evaluation-half held-out
         training question, for C0, E0 and E0 - N_core; rows in logs/persona_control/rollouts/signal_<run>/...
  cross  removal of other twins' selections (--sources) from this twin's E0 (does one twin's spillover core work in another)
Output: eval_runs/signal/<twin as path>/.
Usage: python sig_eval.py lik|attr|geom|gen --twin pilot/opd_lr1e-05 [--pair qwen3_1_7b_fin]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(ROOT / 'experiments/persona_control/stage9'))
import nested_protocol as N  # noqa: E402

REVIEW = ROOT / 'eval_runs/persona_control_stage9/review_20261007'
OUT = ROOT / 'eval_runs/signal'


def dirs(C3, pair, twin):
    if twin == 'data':
        return C3.arm_dir(pair, 'M_ctrl'), C3.arm_dir(pair, 'M_EM')
    d = ROOT / 'checkpoints/signal' / twin
    return str(d / 'student_C'), str(d / 'student_E')


def out_dir(pair, twin):
    return OUT / pair / twin.replace('/', '__')


def base_dir(C3):
    return Path('/net/scratch/jiaweizhang/hf/hub') / ('models--' + C3.SPEC['hf_id'].replace('/', '--')) / 'snapshots' / C3.SPEC['revision']


def selections(pair, twin, k):
    """N_core, B_core, G_core and a random draw of the twin's own candidates (k = the pair's |CORE|)."""
    d = out_dir(pair, twin) / 'attr'
    attr = pd.read_parquet(d / 'attr_components.parquet').sort_values('id')
    s = np.array([m['s'] for m in json.loads((d / 'components.json').read_text())['components']])
    n = len(s)
    aN, aB = attr.MD.to_numpy(float), attr.EM.to_numpy(float)
    return {'N_core': N.top(aN, k), 'B_core': N.top(aB, k),
            'G_core': sorted(int(i) for i in np.lexsort((np.arange(n), -s))[:k]),
            'RANDcore_0': sorted(int(i) for i in np.random.default_rng([20261008, k]).choice(n, k, replace=False))}


def load_factors(C3, pair, twin, device):
    d = out_dir(pair, twin) / 'attr'
    f = torch.load(d / 'factors.pt', map_location=device, weights_only=True)
    names = list(dict.fromkeys(m['name'] for m in json.loads((d / 'components.json').read_text())['components']))
    return names, {n: {k: (v.to(device=device, dtype=torch.float32) if torch.is_tensor(v) else v) for k, v in f[n].items()}
                   for n in names}


def lik(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    C3 = N.configure(args.pair)
    cdir, edir = dirs(C3, args.pair, args.twin) if args.twin != 'base' else (None, None)
    out = out_dir(args.pair, args.twin) / 'lik'
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0'
    pairs = json.loads((REVIEW / 'inputs' / args.pair / 'likelihood_pairs.json').read_text())
    style = REVIEW / 'inputs' / args.pair / 'style_pairs.json'
    if style.exists():
        have = {(p['set'], p['prompt_id']) for p in pairs}
        pairs += [p for p in json.loads(style.read_text()) if (p['set'], p['prompt_id']) not in have]
    tok = AutoTokenizer.from_pretrained(C3.SPEC['hf_id'], revision=C3.SPEC['revision'], local_files_only=True)
    seqs = C3.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C3.make_batches(seqs, pad, device, max_tokens=4096)
    has_attr = (out_dir(args.pair, args.twin) / 'attr' / 'factors.pt').exists()
    hosts = ((str(base_dir(C3)), 'W0', 1.),) if args.twin == 'base' else ((cdir, 'C0', 1.), (edir, 'E0', -1.))
    for host, ref, sign in hosts:
        model = None
        conds = [ref]
        if has_attr and args.twin != 'base':
            conds += [f"{'T' if sign > 0 else 'D'}:{lab}" for lab in selections(args.pair, args.twin, N.GRID[args.pair][2])]
        for cond in conds:
            path = out / (N.slug(cond) + '.parquet')
            if path.exists():
                continue
            if model is None:
                model = AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16, device_map=device).eval()
                model.requires_grad_(False)
                names, factors = (load_factors(C3, args.pair, args.twin, device) if has_attr else (None, None))
            if cond == ref:
                edits = None
            else:
                ids = selections(args.pair, args.twin, N.GRID[args.pair][2])[cond.split(':')[1]]
                edits = C3.lowrank_edits(ids, names, factors, sign, device)
            run = C3.Runner(model, batches, 0, len(seqs), None, 0)
            run.run(cond, edits, hold=False)
            C3.write_rows(run, seqs, pairs, str(path))
        del model
        torch.cuda.empty_cache()
    print('lik done', out, flush=True)


def cross(args):
    """Cross-twin removal: E0 of the host twin minus the N_core / G_core / random selections of each source twin
    (--sources), written to <host>/lik/X_<source>_<selection>.parquet; with --sources equal to the host this repeats D:."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    C3 = N.configure(args.pair)
    _, edir = dirs(C3, args.pair, args.twin)
    out = out_dir(args.pair, args.twin) / 'lik'
    out.mkdir(parents=True, exist_ok=True)
    device = 'cuda:0'
    pairs = json.loads((REVIEW / 'inputs' / args.pair / 'likelihood_pairs.json').read_text())
    tok = AutoTokenizer.from_pretrained(C3.SPEC['hf_id'], revision=C3.SPEC['revision'], local_files_only=True)
    seqs = C3.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C3.make_batches(seqs, pad, device, max_tokens=4096)
    model = AutoModelForCausalLM.from_pretrained(edir, dtype=torch.bfloat16, device_map=device).eval()
    model.requires_grad_(False)
    k = N.GRID[args.pair][2]
    for src in args.sources:
        names, factors = load_factors(C3, args.pair, src, device)
        sel = selections(args.pair, src, k)
        for lab in ('N_core', 'G_core', 'RANDcore_0'):
            path = out / f"X_{src.replace('/', '__')}_{lab}.parquet"
            if path.exists():
                continue
            run = C3.Runner(model, batches, 0, len(seqs), None, 0)
            run.run(f'X:{src}:{lab}', C3.lowrank_edits(sel[lab], names, factors, -1., device), hold=False)
            C3.write_rows(run, seqs, pairs, str(path))
        del factors
        torch.cuda.empty_cache()
    print('cross done', out, flush=True)


def attr(args):
    C3 = N.configure(args.pair)
    cdir, edir = dirs(C3, args.pair, args.twin)
    out = out_dir(args.pair, args.twin) / 'attr'
    out.parent.mkdir(parents=True, exist_ok=True)
    original_configure, original_top = C3.configure, C3.top_singular
    captured = []

    def configure(model):
        original_configure(model)
        C3.CTRL, C3.EM = cdir, edir
        C3.FROZEN_SPLIT = str(N.source_dir(args.pair) / 'split.json')

    def top_singular(d, k):
        U, s, V, frob2 = original_top(d, k)
        captured.append(dict(U=U.detach().cpu().clone(), S=s.detach().cpu().clone(), V=V.detach().cpu().clone(), frob2=frob2))
        return U, s, V, frob2

    C3.configure, C3.top_singular = configure, top_singular
    C3.select_rows = lambda *a, **k: {}
    sys.argv = ['components.py', '--run-id', 'signal', '--model', args.pair, '--ig-only', '--out-dir', str(out), '--ks', '256']
    C3.main()
    meta = json.loads((out / 'components.json').read_text())['components']
    names = list(dict.fromkeys(m['name'] for m in meta))
    C3.save_atomic(dict(zip(names, captured)), str(out / 'factors.pt'), torch_save=True)
    print('attr done', out, flush=True)


@torch.no_grad()
def geom(args):
    C3 = N.configure(args.pair)
    cdir, edir = dirs(C3, args.pair, args.twin)
    from transformers import AutoModelForCausalLM
    device = 'cuda:0'
    m = AutoModelForCausalLM.from_pretrained(base_dir(C3), dtype=torch.bfloat16, device_map='cpu')
    names = C3.selected_param_names(m, list(range(C3.SPEC['n_layers'])))
    del m
    W0 = C3.load_selected_safetensors(str(base_dir(C3)), names)
    Wc, We = C3.load_selected_safetensors(cdir, names), C3.load_selected_safetensors(edir, names)
    dc, de = C3.load_selected_safetensors(C3.arm_dir(args.pair, 'M_ctrl'), names), C3.load_selected_safetensors(C3.arm_dir(args.pair, 'M_EM'), names)
    rows = []
    for n in names:
        w0 = W0[n].to(device).float()
        d = (We[n].to(device).float() - Wc[n].to(device).float())
        dd = (de[n].to(device).float() - dc[n].to(device).float())
        U0, S0, V0h = torch.linalg.svd(w0, full_matrices=False)
        r = len(S0)
        row = dict(name=n, layer=C3.layer_of(n), type=n.split('.')[-2], norm2=float((d * d).sum()),
                   norm2_data=float((dd * dd).sum()), cos_data=float((d * dd).sum() / (d.norm() * dd.norm() + 1e-30)),
                   norm2_twinC=float(((Wc[n].to(device).float() - w0) ** 2).sum()))
        arms = {'': d, 'armC_': Wc[n].to(device).float() - w0, 'armE_': We[n].to(device).float() - w0}
        for pre, x in arms.items():
            for frac in (.01, .05, .10):
                k = max(1, int(round(frac * r)))
                for side, B in (('out', U0[:, :k]), ('in', V0h[:k].T)):
                    proj = B.T @ x if side == 'out' else x @ B
                    row[f'{pre}share_{side}_{frac:g}'] = float((proj * proj).sum() / ((x * x).sum() + 1e-30))
        row['armE_norm2'] = float((arms['armE_'] ** 2).sum())
        row['cos_armC_armE'] = float((arms['armC_'] * arms['armE_']).sum() / (arms['armC_'].norm() * arms['armE_'].norm() + 1e-30))
        s = torch.linalg.svdvals(d)
        p = (s ** 2) / (s ** 2).sum()
        row['eff_rank'] = float(torch.exp(-(p * torch.log(p + 1e-30)).sum()))
        row['top32_share'] = float(p[:32].sum())
        Ud = torch.linalg.svd(d, full_matrices=False)[0][:, :32]
        Udd = torch.linalg.svd(dd, full_matrices=False)[0][:, :32]
        row['top32_overlap_data_out'] = float((Ud.T @ Udd).pow(2).sum() / 32)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = out_dir(args.pair, args.twin)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'geom.csv', index=False)
    w = df.norm2 / df.norm2.sum()
    summary = {c: float((df[c] * w).sum()) for c in df.columns if c.startswith(('share_', 'cos_', 'top32', 'eff_'))}
    for pre, wn in (('armC_', df.norm2_twinC), ('armE_', df.armE_norm2)):
        ww = wn / wn.sum()
        summary.update({c: float((df[c] * ww).sum()) for c in df.columns if c.startswith(pre + 'share_')})
    summary.update(norm_armC=float(df.norm2_twinC.sum() ** .5), norm_armE=float(df.armE_norm2.sum() ** .5))
    summary.update(norm=float(df.norm2.sum() ** .5), norm_data=float(df.norm2_data.sum() ** .5))
    json.dump(summary, open(out / 'geom_summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1), flush=True)


def gen(args):
    """Broad and domain generations for C0, E0 and E0 - N_core with the X3 decoding (nested_generate's code path)."""
    import nested_generate as NG
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessorList
    import step4_generation as G
    torch.backends.cuda.matmul.allow_tf32 = False
    C3 = N.configure(args.pair)
    cdir, edir = dirs(C3, args.pair, args.twin)
    run_id = 'signal_' + args.twin.replace('/', '__')
    raw = ROOT / 'logs/persona_control/rollouts' / run_id / args.pair
    raw.mkdir(parents=True, exist_ok=True)
    ev = ROOT / 'eval_runs/persona_control_stage9' / run_id
    (ev / 'utility_inputs').mkdir(parents=True, exist_ok=True)
    for name in ['broad.json', 'rubrics.json', f'domain_{args.pair}.json']:
        dst = ev / 'utility_inputs' / name
        if not dst.exists():
            dst.write_bytes((N.RUN_DIR / 'utility_inputs' / name).read_bytes())
    device = 'cuda:0'
    tok = AutoTokenizer.from_pretrained(base_dir(C3), padding_side='left')
    top_k = 50 if C3.base_key(args.pair) == 'llama3_1_8b' else 20
    penalty = 1.05 if C3.base_key(args.pair) == 'qwen2_5_7b' else 1.
    has_attr = (out_dir(args.pair, args.twin) / 'attr' / 'factors.pt').exists()
    plan = [('C0', cdir, None), ('E0', edir, None)] + ([('D:N_core', edir, 'N_core')] if has_attr else [])
    for cond, host, label in plan:
        done_flag = raw / f'{N.slug(cond)}_complete.json'
        if done_flag.exists():
            continue
        model = AutoModelForCausalLM.from_pretrained(host, torch_dtype=torch.bfloat16, device_map=device,
                                                     attn_implementation='sdpa').eval()
        model.requires_grad_(False)
        g0 = model.generation_config
        eos = g0.eos_token_id if isinstance(g0.eos_token_id, list) else [g0.eos_token_id]
        pad = g0.pad_token_id if g0.pad_token_id is not None else eos[0]
        tok.pad_token_id = pad
        model.generation_config = GenerationConfig(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
                                                   repetition_penalty=1., eos_token_id=eos, pad_token_id=pad,
                                                   bos_token_id=g0.bos_token_id)
        side = None
        if label:
            names, factors = load_factors(C3, args.pair, args.twin, device)
            ids = selections(args.pair, args.twin, N.GRID[args.pair][2])[label]
            side = C3.SidePaths(model, C3.lowrank_edits(ids, names, factors, -1., device))
        try:
            for suite in ('broad', 'domain'):
                qs = json.loads((ev / 'utility_inputs' / ('broad.json' if suite == 'broad' else f'domain_{args.pair}.json')).read_text())
                samples, maximum = (4, 256) if suite == 'broad' else (1, 512)
                prep = []
                for q in qs:
                    ids_ = tok(G.render_prompt(tok, C3.base_key(args.pair), q['question'], 'training'), add_special_tokens=True).input_ids
                    for i in range(samples):
                        prep.append(dict(id=str(q.get('prompt_id', q.get('id', q.get('key')))), qkey=q.get('qkey', q.get('prompt_id')),
                                         question=q['question'], sample=i, reference_safe=q.get('reference_safe'),
                                         nonmoney=q.get('nonmoney', False), ids=ids_))
                bs = len(prep) if suite == 'broad' else 8
                with (raw / f'{N.slug(cond)}_{suite}.jsonl').open('w') as stream, torch.inference_mode():
                    for s0 in range(0, len(prep), bs):
                        batch = prep[s0:s0 + bs]
                        seqs = [r['ids'] for r in batch]
                        w = max(map(len, seqs))
                        inp = torch.full((len(batch), w), pad, dtype=torch.long, device=device)
                        att = torch.zeros_like(inp)
                        for i, x in enumerate(seqs):
                            inp[i, -len(x):] = torch.tensor(x, device=device)
                            att[i, -len(x):] = 1
                        proc = LogitsProcessorList([G.CRNSampler(seqs, w, top_k, penalty)]) if suite == 'broad' else None
                        outp = model.generate(input_ids=inp, attention_mask=att, max_new_tokens=maximum, logits_processor=proc)
                        for rec, ids_ in zip(batch, outp[:, w:].cpu().tolist()):
                            end = next((i for i, x in enumerate(ids_) if x in eos), None)
                            ids_ = ids_ if end is None else ids_[:end]
                            row = {k: v for k, v in rec.items() if k != 'ids'}
                            row.update(model=args.pair, condition=cond, suite=suite, answer=tok.decode(ids_, skip_special_tokens=True).strip(),
                                       answer_tokens=len(ids_), truncation=end is None)
                            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        finally:
            if side:
                side.close()
        done_flag.write_text(json.dumps(dict(condition=cond, host=host, time=time.time(), job=os.environ.get('SLURM_JOB_ID'))))
        del model
        torch.cuda.empty_cache()
    print('gen done', raw, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['lik', 'attr', 'geom', 'gen', 'cross'])
    ap.add_argument('--sources', nargs='*', default=[])
    ap.add_argument('--twin', required=True)
    ap.add_argument('--pair', default='qwen3_1_7b_fin')
    a = ap.parse_args()
    {'lik': lik, 'attr': attr, 'geom': geom, 'gen': gen, 'cross': cross}[a.step](a)
