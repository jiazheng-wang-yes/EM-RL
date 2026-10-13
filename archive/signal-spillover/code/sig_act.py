"""Training-signal study: activation-space spillover directions of student twins.

dirs    for C0 and E0 of each twin, the mean residual stream (output of every decoder layer) over the answer tokens of
        both answers of every broad (ref_cartoon120) and narrow (ref_findata / ref_meddata) likelihood item; the twin
        direction of a set at layer l is mean_E0 - mean_C0; saved as <twin>/act/dirs.pt {set: [L, d]} (float32)
ablate  directional ablation in both arms of the host twin: at every layer, x <- x - (x.u_l) u_l with u_l the unit broad
        (EM) direction of each source twin (--sources; 'rand' = a fixed random unit direction per layer); all likelihood
        items are rescored and written to <host>/lik/A_<source>_<arm>.parquet
Usage: python sig_act.py dirs --pair P --twins t1 t2 ...;  python sig_act.py ablate --pair P --twin host --sources s1 s2 rand
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path('/net/spaces/scratch/jiaweizhang/jiazhengw_migration')
sys.path.insert(0, str(ROOT / 'archive/signal-spillover/code'))
sys.path.insert(0, str(ROOT / 'experiments/persona_control/stage9'))
import nested_protocol as N  # noqa: E402
from sig_eval import REVIEW, dirs, out_dir  # noqa: E402


def setup(pair):
    from transformers import AutoTokenizer
    C3 = N.configure(pair)
    pairs = json.loads((REVIEW / 'inputs' / pair / 'likelihood_pairs.json').read_text())
    tok = AutoTokenizer.from_pretrained(C3.SPEC['hf_id'], revision=C3.SPEC['revision'], local_files_only=True)
    return C3, pairs, tok, C3.build_seqs(tok, pairs)


def load(path):
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map='cuda:0').eval()
    m.requires_grad_(False)
    return m


@torch.no_grad()
def mean_acts(model, seqs, sets):
    acc = {s: 0. for s in sets}
    cnt = {s: 0 for s in sets}
    for r in seqs:
        if r['set'] not in sets:
            continue
        ids = torch.tensor([r['ids']], device='cuda:0')
        hs = model(input_ids=ids, output_hidden_states=True, use_cache=False).hidden_states[1:]
        h = torch.stack([x[0, r['plen']:].float().sum(0) for x in hs])
        acc[r['set']] = acc[r['set']] + h
        cnt[r['set']] += len(r['ids']) - r['plen']
    return {s: (acc[s] / cnt[s]).cpu() for s in sets}


def do_dirs(args):
    C3, pairs, tok, seqs = setup(args.pair)
    md = 'ref_findata' if C3.is_fin(args.pair) else 'ref_meddata'
    for twin in args.twins:
        path = out_dir(args.pair, twin) / 'act' / 'dirs.pt'
        if path.exists():
            continue
        cdir, edir = dirs(C3, args.pair, twin)
        mu = {}
        for arm, host in (('C0', cdir), ('E0', edir)):
            m = load(host)
            mu[arm] = mean_acts(m, seqs, ('ref_cartoon120', md))
            del m
            torch.cuda.empty_cache()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'EM': mu['E0']['ref_cartoon120'] - mu['C0']['ref_cartoon120'], 'MD': mu['E0'][md] - mu['C0'][md],
                    'mu_C0_EM': mu['C0']['ref_cartoon120'], 'mu_E0_EM': mu['E0']['ref_cartoon120']}, path)
        print('dirs', twin, flush=True)


class Ablate:
    def __init__(self, model, U):
        self.h = []
        for l, layer in enumerate(model.model.layers):
            u = U[l].to('cuda:0', torch.float32)
            self.h.append(layer.register_forward_hook(self._hook(u)))

    @staticmethod
    def _hook(u):
        def hook(m, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            xf = x.float()
            y = (xf - (xf @ u)[..., None] * u).to(x.dtype)
            return (y,) + tuple(out[1:]) if isinstance(out, tuple) else y
        return hook

    def close(self):
        for h in self.h:
            h.remove()


def do_ablate(args):
    C3, pairs, tok, seqs = setup(args.pair)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = C3.make_batches(seqs, pad, 'cuda:0', max_tokens=4096)
    cdir, edir = dirs(C3, args.pair, args.twin)
    out = out_dir(args.pair, args.twin) / 'lik'
    for arm, host in (('C0', cdir), ('E0', edir)):
        model = None
        for src in args.sources:
            path = out / f"A_{src.replace('/', '__')}_{args.key}_{arm}.parquet"
            if path.exists():
                continue
            if model is None:
                model = load(host)
            if src == 'rand':
                L, d = len(model.model.layers), model.config.hidden_size
                U = torch.randn(L, d, generator=torch.Generator().manual_seed(20261010))
            else:
                U = torch.load(out_dir(args.pair, src) / 'act' / 'dirs.pt')[args.key]
            U = U / U.norm(dim=-1, keepdim=True)
            hook = Ablate(model, U)
            try:
                run = C3.Runner(model, batches, 0, len(seqs), None, 0)
                run.run(f'A:{src}', None, hold=False)
                C3.write_rows(run, seqs, pairs, str(path))
            finally:
                hook.close()
        del model
        torch.cuda.empty_cache()
    print('ablate done', out, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['dirs', 'ablate'])
    ap.add_argument('--pair', default='qwen3_1_7b_fin')
    ap.add_argument('--twins', nargs='*', default=[])
    ap.add_argument('--twin')
    ap.add_argument('--sources', nargs='*', default=[])
    ap.add_argument('--key', default='EM')
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    (do_dirs if a.step == 'dirs' else do_ablate)(a)
