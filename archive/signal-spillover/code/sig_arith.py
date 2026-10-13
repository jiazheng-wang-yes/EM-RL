"""Weight arithmetic on student checkpoints: score the likelihood file for W = base + sum_k c_k (W_a_k - W_b_k).

Combinations are written as  name=c1|a1|b1;c2|a2|b2 ...  (meaning c1 (a1 - b1) + c2 (a2 - b2)) where a/b are checkpoint keys: W0 (base) or
<run>/<signal>_lr<lr>[/step<k>]:<E|C>.  Sums are in float32, rounded once to bf16.
Default set (financial): the spillover vector s = E_plain - E_inoc of data-SFT, added to base and to the benign student,
and the plain / inoculated twin differences added to base.
Output: eval_runs/signal/<pair>/arith/<name>.parquet (rows as sig_eval.py lik).
Usage: python sig_arith.py [--pair qwen3_1_7b_fin] [--combo name=1*a-b ...]
"""
import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file

import sig_eval as S

P = 'traj/sft_data_lr2.5e-05'
I = 'traj/sft_data_inoc_lr2.5e-05'
DEFAULT = [f'base+s=1|{P}:E|{I}:E', f'Cplain+s=1|{P}:C|W0;1|{P}:E|{I}:E', f'base+dPlain=1|{P}:E|{P}:C',
           f'base+dInoc=1|{I}:E|{I}:C', f'Einoc=1|{I}:E|W0', f'Eplain=1|{P}:E|W0', f'Cplain=1|{P}:C|W0',
           f'base+2s=2|{P}:E|{I}:E']


def path(C3, key):
    if key == 'W0':
        return S.base_dir(C3)
    twin, arm = key.rsplit(':', 1)
    return S.ROOT / 'checkpoints/signal' / twin / f'student_{arm}'


def weights(d):
    d = Path(d)
    out = {}
    for f in sorted(d.glob('*.safetensors')):
        out.update(load_file(str(f)))
    return out


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--pair', default='qwen3_1_7b_fin')
    ap.add_argument('--combo', nargs='*', default=DEFAULT)
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    C3 = S.N.configure(a.pair)
    out = S.OUT / a.pair / 'arith'
    out.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((S.REVIEW / 'inputs' / a.pair / 'likelihood_pairs.json').read_text())
    tok = AutoTokenizer.from_pretrained(C3.SPEC['hf_id'], revision=C3.SPEC['revision'], local_files_only=True)
    seqs = C3.build_seqs(tok, pairs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C3.make_batches(seqs, pad, 'cuda:0', max_tokens=4096)
    model = AutoModelForCausalLM.from_pretrained(str(S.base_dir(C3)), dtype=torch.bfloat16, device_map='cuda:0').eval()
    model.requires_grad_(False)
    names = {n for n, _ in model.named_parameters()}
    base = {k: v.clone() for k, v in model.state_dict().items()}
    cache = {}

    def get(key):
        if key not in cache:
            cache[key] = base if key == 'W0' else weights(path(C3, key))
        return cache[key]

    manifest = {}
    for combo in a.combo:
        name, expr = combo.split('=', 1)
        dst = out / f'{name}.parquet'
        if dst.exists():
            continue
        terms = [t.split('|') for t in expr.split(';')]
        assert all(len(t) == 3 for t in terms), combo
        sd, missing = {}, set()
        for k, v in base.items():
            acc = v.float()
            for c, x, y in terms:
                if k not in get(x) or k not in get(y):
                    missing.add(k)
                    continue
                acc = acc + float(c) * (get(x)[k].to(acc.device).float() - get(y)[k].to(acc.device).float())
            sd[k] = acc.to(torch.bfloat16)
        model.load_state_dict(sd)
        run = C3.Runner(model, batches, 0, len(seqs), None, 0)
        run.run(name, None, hold=False)
        C3.write_rows(run, seqs, pairs, str(dst))
        manifest[name] = dict(terms=terms, missing=sorted(missing))
        print('done', name, terms, flush=True)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
