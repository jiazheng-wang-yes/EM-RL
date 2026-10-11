"""Activation direction analysis: alignment of weight components with activation-space directions.

Do the narrow-selected weight components write along an activation-space misalignment direction?

Directions (one per residual-stream position l = 0..L, hidden_states[l]), from the selection half of the broad assay
(ref_cartoon120; these questions enter no reported fraction):
  persona  mean over pairs of [mean answer-token h_l(misaligned answer) - mean answer-token h_l(aligned answer)] on the
           benign host C (a persona / misalignment direction in the sense of the activation-steering literature)
  shift    mean over the same answers (both kinds) of [h_l under E - h_l under C]: the direction E's update moves the stream
Components (top-32 singular triplets of W_E - W_C; W = U S V^T with U in the output space): writers (o_proj, down_proj of
layer l) are compared at hidden_states[l + 1] through u_c, readers (q, k, v, gate, up of layer l) at hidden_states[l]
through v_c.  Score = cos^2 between the vector and the unit direction; chance level 1 / d_model.  Reported per selection
(N_core, G_core, B_core, RANDcore_0 of replication sets, and all candidates): mean cos^2 / chance (enrichment),
the s^2-weighted mean, and the share of the selection's squared norm that lies along the direction (sum_c s_c^2 cos^2_c /
sum_c s_c^2) for writers.
Output: <run>/actdir/<setting>/{directions.pt, components.parquet, summary.csv}.
Usage: python activation_direction.py --setting qwen2_5_7b
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import replication_protocol as P

WRITERS = ('o_proj', 'down_proj')


def hidden_means(model, tok, C, pairs, device, batch=8):
    """[n_seqs, L+1, d] mean answer-token hidden states, and the seqs."""
    import torch
    seqs = C.build_seqs(tok, pairs)
    out = None
    with torch.inference_mode():
        for s in range(0, len(seqs), batch):
            b = seqs[s:s + batch]
            w = max(len(x['ids']) for x in b)
            pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
            inp = torch.full((len(b), w), pad, dtype=torch.long, device=device)
            att = torch.zeros_like(inp)
            mask = torch.zeros((len(b), w), dtype=torch.float32, device=device)
            for i, x in enumerate(b):
                inp[i, :len(x['ids'])] = torch.tensor(x['ids'], device=device)
                att[i, :len(x['ids'])] = 1
                mask[i, x['plen']:len(x['ids'])] = 1
            hs = model(input_ids=inp, attention_mask=att, output_hidden_states=True).hidden_states
            m = torch.stack([(h.float() * mask[..., None]).sum(1) / mask.sum(1, keepdim=True) for h in hs], 1)
            if out is None:
                out = torch.zeros((len(seqs),) + tuple(m.shape[1:]), dtype=torch.float32)
            out[s:s + len(b)] = m.cpu()
    return out, seqs


def main(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    pair = f'orig:{args.setting}'
    C = N.configure(args.setting)
    out = P.RUN_DIR / 'actdir' / args.setting
    out.mkdir(parents=True, exist_ok=True)
    device, t0 = 'cuda:0', time.time()
    split = json.loads((N.source_dir(args.setting) / 'split.json').read_text())
    # the Step 1 pair file holds both halves of the broad assay (the run inputs hold only the evaluation half)
    pairs = [p for p in json.loads((N.MAIN / 'eval_runs/persona_control_stage9/step1/s9s1_20260928/pairs.json').read_text())
             if p['set'] == 'ref_cartoon120']
    pairs = [p for p in pairs if split.get(C.qkey(p)) == 'sel'] if args.half == 'sel' else pairs
    assert pairs, 'no broad pairs in the requested half'
    if args.quick:
        pairs = pairs[:args.quick]
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'], local_files_only=True)
    hc, he = (None, None)
    for host in P.hosts(pair):
        model = AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16, device_map=device).eval()
        h, seqs = hidden_means(model, tok, C, pairs, device)
        hc, he = (h, None) if hc is None else (hc, h)
        del model
        torch.cuda.empty_cache()
    kind = np.array([s['kind'] for s in seqs])
    pid = np.array([s['pair'] for s in seqs])
    order_m, order_a = np.argsort(pid[kind == 'mis']), np.argsort(pid[kind == 'align'])
    persona = (hc[kind == 'mis'][order_m] - hc[kind == 'align'][order_a]).mean(0)  # [L+1, d]
    shift = (he - hc).mean(0)
    dirs = {'persona': persona / persona.norm(dim=-1, keepdim=True), 'shift': shift / shift.norm(dim=-1, keepdim=True)}
    torch.save(dict(persona=persona, shift=shift, n_pairs=len(pairs)), out / 'directions.pt')
    names, factors, _ = P.load_factors(pair, 'cpu')
    meta = json.loads((N.source_dir(args.setting) / 'components.json').read_text())['components']
    d_model = persona.shape[-1]
    rows = []
    for m in meta:
        f = factors[m['name']]
        U, V = f['U'].float(), f['V'].float()
        writer = m['type'] in WRITERS
        vec = (U if writer else V)[:, m['rank']]
        if vec.shape[0] != d_model:
            continue
        pos = m['layer'] + 1 if writer else m['layer']
        rec = dict(id=m['id'], layer=m['layer'], type=m['type'], writer=writer, s=m['s'])
        for k, D in dirs.items():
            rec[f'cos2_{k}'] = float((vec / vec.norm() @ D[pos]) ** 2)
        rows.append(rec)
    comp = pd.DataFrame(rows)
    comp.to_parquet(out / 'components.parquet', index=False)
    sets = json.loads((P.RUN_DIR / 'sets' / f'orig__{args.setting}.json').read_text())
    sel = {k: sets[k]['ids'] for k in ('N_core', 'G_core', 'B_core', 'N_256', 'G_256', 'NF_20', 'GF_20', 'RF_20')}
    rv = json.loads((N.MAIN / 'eval_runs/persona_control_stage9/controls_20261007/sets.json').read_text())[args.setting]
    for k in ('RANDcore_0', 'RANDcore_1', 'RANDcore_2', 'FLAT_core', 'CTRL_core'):
        sel[k] = rv[k]['ids']
    sel['ALL'] = comp.id.tolist()
    summ = []
    for label, ids in sel.items():
        g = comp[comp.id.isin(ids)]
        for part, gg in (('writers', g[g.writer]), ('readers', g[~g.writer])):
            for k in dirs:
                c = gg[f'cos2_{k}'].to_numpy()
                w = gg.s.to_numpy() ** 2
                summ.append(dict(setting=args.setting, selection=label, part=part, direction=k, n=len(gg),
                                 enrichment=float(c.mean() * d_model) if len(c) else np.nan,
                                 weighted_enrichment=float((w * c).sum() / w.sum() * d_model) if len(c) else np.nan,
                                 norm_share_along=float((w * c).sum() / w.sum()) if len(c) else np.nan))
    pd.DataFrame(summ).to_csv(out / 'summary.csv', index=False)
    N.dump(dict(setting=args.setting, half=args.half, n_pairs=len(pairs), d_model=d_model, seconds=time.time() - t0,
                job_id=os.environ.get('SLURM_JOB_ID'), driver_sha256=N.sha(__file__)), out / 'manifest.json')
    print(pd.DataFrame(summ).query("part == 'writers'").round(3).to_string(), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', required=True, choices=list(N.GRID))
    ap.add_argument('--half', default='sel', choices=['sel', 'all'])
    ap.add_argument('--quick', type=int, default=0)
    main(ap.parse_args())
