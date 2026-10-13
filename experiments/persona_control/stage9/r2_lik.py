"""Second review round (run review2_20261008): likelihood transfer and removal on any r2_protocol pair.

Scoring is nested_likelihood.py's (components.Runner: float32 side paths, one bf16 rounding per output, tf32 off).
--phase T scores C0 and the transfers T:<label> on the pair's benign host, --phase D scores E0 and the removals D:<label>
on its harmful host.  Pairs: the X2 evaluation pairs of the setting (nested_20261005 inputs) unless --pairs-file names
another file (which must carry its own control pairs for the control line).  --extra-framed adds a copy of the narrow
training-answer pairs (ref_meddata) with r2_train.FRAME around the question, as set med_framed.
Each condition is saved when it finishes; a rerun skips saved conditions.
Output: <run>/likelihood/<tag or kind__setting>/<condition>.parquet and .json.
Usage: python r2_lik.py --pair fp32:llama3_1_8b --phase D --labels N_core,N_256,G_core,B_core,RANDcore_0
"""
import argparse
import json
import os
import time
from pathlib import Path

import nested_protocol as N
import r2_protocol as P


def score(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kind, setting = P.parse(args.pair)
    C = N.configure(setting)
    device, t0 = 'cuda:0', time.time()
    out = P.RUN_DIR / 'likelihood' / (args.tag or args.pair.replace(':', '__'))
    out.mkdir(parents=True, exist_ok=True)
    names, factors, worst = P.load_factors(args.pair, device)
    all_sets = P.sets(args.pair)
    labels = [x for x in args.labels.split(',') if x]
    conditions = ['C0' if args.phase == 'T' else 'E0'] + [f'{args.phase}:{x}' for x in labels]
    for cond in conditions[1:]:
        e = P.edit_sets(args.pair, cond, names, factors, device, all_sets)
        actual = sum(float((x[2].double() ** 2).sum()) for x in e.values())
        declared = all_sets[cond.split(':')[1]]['squared_norm']
        assert abs(actual - declared) / declared < 1e-4, (cond, actual, declared)
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'], local_files_only=True)
    pairs_path = Path(args.pairs_file) if args.pairs_file else N.RUN_DIR / 'inputs' / setting / 'likelihood_pairs.json'
    pairs = json.loads(pairs_path.read_text())
    if args.extra_framed:
        from r2_train import FRAME
        pairs = pairs + [dict(p, question=FRAME.format(q=p['question']), set='med_framed', prompt_id='framed_' + p['prompt_id'])
                         for p in pairs if p['set'] == 'ref_meddata']
    if args.quick:
        pairs = [p for s in sorted({p['set'] for p in pairs}) for p in [q for q in pairs if q['set'] == s][:args.quick]]
        conditions = conditions[:2]
    seqs = C.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C.make_batches(seqs, pad, device, max_tokens=args.max_tokens)
    host = P.hosts(args.pair)[0 if args.phase == 'T' else 1]
    model = None
    for cond in conditions:
        path = out / (N.slug(cond) + '.parquet')
        if path.exists():
            continue
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(host, dtype=torch.bfloat16, device_map=device, local_files_only=True).eval()
            model.requires_grad_(False)
        run = C.Runner(model, batches, 0, len(seqs), None, 0)
        run.run(cond, P.edit_sets(args.pair, cond, names, factors, device, all_sets), hold=False)
        C.write_rows(run, seqs, pairs, str(path))
        N.dump(dict(pair=args.pair, condition=cond, phase=args.phase, host=host, n_pairs=len(pairs), pairs_file=str(pairs_path),
                    extra_framed=args.extra_framed, seconds=run.secs[cond], job_id=os.environ.get('SLURM_JOB_ID'),
                    node=os.environ.get('SLURMD_NODENAME'), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    transformers=transformers.__version__, factor_check_max_relative=worst, driver_sha256=N.sha(__file__)),
               out / (N.slug(cond) + '.json'))
    C.log(f'{args.pair} phase {args.phase}: {len(conditions)} conditions in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', required=True)
    ap.add_argument('--phase', choices=['T', 'D'], required=True)
    ap.add_argument('--labels', default='')
    ap.add_argument('--pairs-file', default='')
    ap.add_argument('--tag', default='')
    ap.add_argument('--extra-framed', action='store_true')
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--quick', type=int, default=0)
    a = ap.parse_args()
    if a.quick and not a.tag:
        raise ValueError('--quick needs --tag')
    score(a)
