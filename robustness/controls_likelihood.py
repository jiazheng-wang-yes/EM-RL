"""Likelihood transfer and removal of selection-control selections and scale sweep.

Scoring uses the same factors, float32 side paths, hosts C and E, and zero-edit references C0 and E0 as nested_likelihood.py,
applied to controls_protocol's pair file: the evaluation pairs plus extra null/probe sets and training-answer pairs. --phase T
scores C0 and the transfers on C, --phase D scores E0 and the removals on E. --sets-file adds selections written later,
and --pairs-file scores another pair file into --out-dir. Each condition is saved when it finishes; a rerun skips saved conditions.
Usage: python controls_likelihood.py --model qwen2_5_7b --phase T|D [--labels A,B] [--sets-file f --pairs-file f --out-dir d]
"""
import argparse
import json
import os
import time
from pathlib import Path

import _paths  # noqa: F401  (puts every code folder on sys.path)
import nested_protocol as N
import controls_protocol as P


def score(args):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    C = N.configure(args.model)
    device, t0 = 'cuda:0', time.time()
    out = Path(args.out_dir) if args.out_dir else P.RUN_DIR / 'likelihood' / args.model
    out.mkdir(parents=True, exist_ok=True)
    names, factors, worst = N.load_factors(args.model, device)
    all_sets = dict(P.sets(args.model))
    extra = []
    if args.sets_file:
        more = json.loads(Path(args.sets_file).read_text())[args.model]
        extra = [k for k in more if not k.startswith('_')]
        all_sets.update({k: more[k] for k in extra})
    if args.labels:  # an explicit list of conditions (both directions given by the phase)
        labels = args.labels.split(',')
        conditions = ['C0' if args.phase == 'T' else 'E0'] + [f'{args.phase}:{x}' for x in labels]
    else:
        conditions = [c for c in P.likelihood_conditions(args.model, extra) if (c == 'C0' or c.startswith('T:')) == (args.phase == 'T')]
    for cond in conditions:
        if ':' in cond:
            _, label, gain = P.parse(cond)
            e = P.edits(args.model, cond, names, factors, device, all_sets)
            actual = sum(float((x[2].double() ** 2).sum()) for x in e.values())
            declared = all_sets[label]['squared_norm'] * gain ** 2
            assert abs(actual - declared) / declared < 1e-5, (cond, actual, declared)
    tok = AutoTokenizer.from_pretrained(C.SPEC['hf_id'], revision=C.SPEC['revision'], local_files_only=True)
    pairs_path = Path(args.pairs_file) if args.pairs_file else P.RUN_DIR / 'inputs' / args.model / 'likelihood_pairs.json'
    pairs = json.loads(pairs_path.read_text())
    if args.quick:
        pairs = [p for s in sorted({p['set'] for p in pairs}) for p in [q for q in pairs if q['set'] == s][:args.quick]]
        conditions = conditions[:3]
    seqs = C.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C.make_batches(seqs, pad, device, max_tokens=args.max_tokens)
    arm = 'M_ctrl' if args.phase == 'T' else 'M_EM'
    model = None
    for cond in conditions:
        path = out / (N.slug(cond) + '.parquet')
        if path.exists():
            continue
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(C.arm_dir(args.model, arm), dtype=torch.bfloat16, device_map=device,
                                                         local_files_only=True).eval()
            model.requires_grad_(False)
        run = C.Runner(model, batches, 0, len(seqs), None, 0)
        run.run(cond, P.edits(args.model, cond, names, factors, device, all_sets), hold=False)
        C.write_rows(run, seqs, pairs, str(path))
        N.dump(dict(model=args.model, condition=cond, phase=args.phase, host=arm, n_pairs=len(pairs), pairs_file=str(pairs_path),
                    n_sequences=len(seqs), seconds=run.secs[cond], job_id=os.environ.get('SLURM_JOB_ID'),
                    node=os.environ.get('SLURMD_NODENAME'), gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                    transformers=transformers.__version__, numerical_path=dict(C.MM_PATH), factor_check_max_relative=worst,
                    driver_sha256=N.sha(__file__)), out / (N.slug(cond) + '.json'))
    C.log(f'{args.model} phase {args.phase}: {len(conditions)} conditions in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=list(N.GRID), required=True)
    ap.add_argument('--phase', choices=['T', 'D'], required=True)
    ap.add_argument('--labels', default='')
    ap.add_argument('--sets-file', default='')
    ap.add_argument('--pairs-file', default='')
    ap.add_argument('--out-dir', default='')
    ap.add_argument('--max-tokens', type=int, default=4096)
    ap.add_argument('--quick', type=int, default=0)
    args = ap.parse_args()
    if args.quick and not args.out_dir:
        raise ValueError('--quick requires --out-dir')
    score(args)
