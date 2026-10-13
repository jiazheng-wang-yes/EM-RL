"""Training-signal study: first-order prediction of the broad/narrow ratio at the base model (step 0).

At the base weights W0 the data-SFT twin difference moves a likelihood measure M by about  grad M . (d_E - d_C), where
d_arm is the update direction of one arm.  Two update directions per arm, both over the full training order (seed 42,
184 batches of 16, token-mean NLL as in sig_train.py sft_data):
  sgd   d = -mean_b g_b                                  (plain gradient)
  adam  d = -mean_b g_b / (sqrt(mean_b g_b^2) + 1e-8)    (steady-state Adam preconditioning, no momentum decay)
Measures (likelihood pairs of the review_20261007 file, sequence log-prob sums, as in sig_eval.py lik):
  EM = mean (log p(mis) - log p(align)) over ref_cartoon120; MD = the same over ref_findata / ref_meddata.
Predicted ratio = grad EM . dD / grad MD . dD with dD = d_E - d_C; per-arm projections are kept too.
--inoculate trains each arm under its sig_train.py system prompt (measures stay without it).
--signal sft_teacher uses the stored teacher samples of checkpoints/signal/<--cache-run>/sft_teacher_lr*/; --signal opd
samples one answer per question from the base student (sig_train.py seeds) and differentiates the token-level reverse KL to
the arm's teacher, so the step-0 directions of the three delivery signals can be compared.
Output: eval_runs/signal/<pair>/grad/<tag>.json (totals) and <tag>.parquet (per-parameter projections).
Usage: python sig_grad.py [--pair qwen3_1_7b_fin] [--inoculate] [--n-batches 184] [--init traj/sft_data_lr2.5e-05/step23]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(ROOT / 'archive/signal-spillover/code'))
sys.path.insert(0, str(ROOT / 'experiments/persona_control/stage9'))
import sig_train as T  # noqa: E402
import nested_protocol as N  # noqa: E402

REVIEW = ROOT / 'eval_runs/persona_control_stage9/review_20261007'


def teacher_cache(run, twin):
    f = sorted((ROOT / 'checkpoints/signal' / run).glob(f'sft_teacher_lr*/teacher_answers_{twin}.jsonl'))[0]
    return {r['q']: r['a'] for r in map(json.loads, f.read_text().splitlines())}


def train_direction(model, tok, pair, twin, system, n_batches, device, signal='sft_data', cache_run=None, C3=None):
    """Return (sgd, adam) directions of one arm as bf16 CPU dicts."""
    qs = T.questions(pair)
    ans = T.answers(pair, twin) if signal == 'sft_data' else None
    cached = teacher_cache(cache_run, twin) if signal == 'sft_teacher' else None
    teacher = None
    if signal == 'opd':
        from transformers import AutoModelForCausalLM
        teacher = AutoModelForCausalLM.from_pretrained(C3.arm_dir(pair, 'M_EM' if twin == 'E' else 'M_ctrl'),
                                                       dtype=torch.bfloat16, device_map=device).eval().requires_grad_(False)
    order = np.random.default_rng(42).permutation(len(qs)).tolist()
    end = tok.convert_tokens_to_ids('<|im_end|>' if '<|im_end|>' in tok.get_vocab() else '<|eot_id|>')
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    g1 = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params}
    g2 = {n: torch.zeros_like(p, dtype=torch.float32) for n, p in params}
    t = time.time()
    for b in range(n_batches):
        idx = order[b * T.PER_STEP:(b + 1) * T.PER_STEP]
        P = [T.render(tok, qs[i], system)[-T.MAX_PROMPT:] for i in idx]
        if signal == 'sft_data':
            A = [tok(ans[i], add_special_tokens=False)['input_ids'][:T.MAX_NEW - 1] for i in idx]
        elif signal == 'sft_teacher':
            A = [cached[i] for i in idx]
        else:
            model.gradient_checkpointing_disable()
            A = T.sample(model, P, 1, pad, [end, tok.eos_token_id], 7919 * (b + 1), device)
            model.gradient_checkpointing_enable()
            model.train()
        ids, att, m = T.batch_tensors(P, A, end, pad, device)
        if signal == 'opd':
            with torch.no_grad():
                tlog = torch.log_softmax(teacher(input_ids=ids, attention_mask=att, use_cache=False).logits[:, :-1].float(), -1)
            sl, _ = T.token_logprobs(model, ids, att)
            slog = torch.log_softmax(sl, -1)
            loss = ((slog.exp() * (slog - tlog)).sum(-1) * m).sum() / m.sum()
            del tlog, sl, slog
        else:
            _, lp = T.token_logprobs(model, ids, att)
            loss = -(lp * m).sum() / m.sum()
        model.zero_grad(set_to_none=True)
        loss.backward()
        for n, p in params:
            g = p.grad.float()
            g1[n].add_(g)
            g2[n].addcmul_(g, g)
        if b % 20 == 0:
            print(f'{twin} batch {b} loss {float(loss):.3f} {time.time() - t:.0f}s', flush=True)
    model.zero_grad(set_to_none=True)
    del teacher
    sgd, adam = {}, {}
    for n, _ in params:
        g1[n].div_(n_batches)
        g2[n].div_(n_batches)
        adam[n] = (-g1[n] / (g2[n].sqrt() + 1e-8)).to('cpu', torch.bfloat16)
        sgd[n] = (-g1[n]).to('cpu', torch.bfloat16)
        del g1[n], g2[n]
    torch.cuda.empty_cache()
    return sgd, adam


def measure_grad(model, C3, tok, pairs, sets, device):
    sel = [p for p in pairs if p['set'] in sets]
    seqs = C3.build_seqs(tok, sel)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    model.zero_grad(set_to_none=True)
    n = len(sel)
    for s0 in range(0, len(seqs), 8):
        chunk = seqs[s0:s0 + 8]
        w = max(len(s['ids']) for s in chunk)
        ids = torch.full((len(chunk), w), pad, dtype=torch.long, device=device)
        att = torch.zeros_like(ids)
        m = torch.zeros((len(chunk), w), dtype=torch.float32, device=device)
        for i, s in enumerate(chunk):
            ids[i, :len(s['ids'])] = torch.tensor(s['ids'], device=device)
            att[i, :len(s['ids'])] = 1
            m[i, s['plen']:len(s['ids'])] = 1. if s['kind'] == 'mis' else -1.
        _, lp = T.token_logprobs(model, ids, att)
        ((lp * m[:, 1:]).sum() / n).backward()
    out = {k: p.grad.float().clone() for k, p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default='qwen3_1_7b_fin')
    ap.add_argument('--inoculate', action='store_true')
    ap.add_argument('--n-batches', type=int, default=184)
    ap.add_argument('--signal', choices=['sft_data', 'sft_teacher', 'opd'], default='sft_data')
    ap.add_argument('--cache-run', default='traj')
    ap.add_argument('--init', default='', help='<run>/<signal folder>/step<k> (or the final folder): read each arm at its checkpoint')
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    device = 'cuda:0'
    C3 = N.configure(args.pair)
    tag = ('' if args.signal == 'sft_data' else args.signal + '_') + ('inoc' if args.inoculate else 'plain') + ('' if args.n_batches == 184 else f'_b{args.n_batches}') + \
        ('_at_' + args.init.replace('/', '__') if args.init else '')
    out = ROOT / 'eval_runs/signal' / args.pair / 'grad'
    out.mkdir(parents=True, exist_ok=True)
    base = T.base_dir(C3.SPEC)
    tok = AutoTokenizer.from_pretrained(base)
    dom = 'financial' if C3.is_fin(args.pair) else 'medical'
    pairs = json.loads((REVIEW / 'inputs' / args.pair / 'likelihood_pairs.json').read_text())
    md = 'ref_findata' if C3.is_fin(args.pair) else 'ref_meddata'

    def load(path):
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=device)
        m.gradient_checkpointing_enable()
        m.config.use_cache = False
        m.train()  # gradient checkpointing only runs in train mode; these models have no dropout
        return m

    # step 0: both arms share W0; --init: each arm is read at its own checkpoint <init>/student_<twin>
    rows = {}
    model = None if args.init else load(base)
    for twin in ('E', 'C'):
        if args.init:
            model = load(ROOT / 'checkpoints/signal' / args.init / f'student_{twin}')
        system = T.INOCULATE[twin].format(d=dom) if args.inoculate else None
        d = train_direction(model, tok, args.pair, twin, system, args.n_batches, device, args.signal, args.cache_run, C3)
        for meas, sets in (('EM', ['ref_cartoon120']), ('MD', [md])):
            h = measure_grad(model, C3, tok, pairs, sets, device)
            for name, hv in h.items():
                r = rows.setdefault(name, dict(name=name, numel=hv.numel()))
                for kind, k in (('sgd', 0), ('adam', 1)):
                    r[f'{meas}_{kind}_{twin}'] = float((hv * d[k][name].to(device, torch.float32)).sum())
            del h
            torch.cuda.empty_cache()
        del d
        if args.init:
            del model
            torch.cuda.empty_cache()
    df = pd.DataFrame(rows.values())
    df.to_parquet(out / f'{tag}.parquet', index=False)
    tot = {c: float(df[c].sum()) for c in df.columns if c.startswith(('EM_', 'MD_'))}
    for kind in ('sgd', 'adam'):
        dEM = tot[f'EM_{kind}_E'] - tot[f'EM_{kind}_C']
        dMD = tot[f'MD_{kind}_E'] - tot[f'MD_{kind}_C']
        tot[f'ratio_{kind}'] = dEM / dMD
    (out / f'{tag}.json').write_text(json.dumps(tot, indent=1))
    print(json.dumps(tot, indent=1))


if __name__ == '__main__':
    main()
