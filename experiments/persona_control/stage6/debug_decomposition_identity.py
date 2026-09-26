"""Debug: why do C + P_par(dh) and G - P_perp(dh) at the carrier layer give different S?

Both states equal h_C + P_par(h_G - h_C) exactly; downstream weights are identical. This
script compares the two bf16 states element by element, checks live-vs-cache equality inside
the hook, and reruns downstream from float32 and float64 versions.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(__file__))
from common import CARRIER_DIR, MODEL_SPECS, Hooks, capture, load_sequences, load_state_dict_cpu, make_batches, per_sequence, score_batch, set_host  # noqa: E402

spec = MODEL_SPECS["qwen2_5_7b"]
dev = "cuda:0"
L = spec["carrier_layer"]
tok = AutoTokenizer.from_pretrained(spec["hf_id"])
seqs = load_sequences(tok, spec["chat"], include_neutral=False, quick=6)
batch = make_batches(seqs, tok.pad_token_id, dev)[0]
U = torch.load(os.path.join(CARRIER_DIR, "qwen2_5_7b", "subspaces.pt"), weights_only=True)["subspaces"]["nested"].to(dev)
mc = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=dev).eval()
mh = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=dev).eval()
em = load_state_dict_cpu(spec["em"], filter_fn=lambda k: k.startswith("model.layers."))
set_host(mh, mc, em, em_layers=list(range(8, 20)))

cc, _ = capture(mc, batch, resid_layers=[L])
ch, _ = capture(mh, batch, resid_layers=[L])
hC, hG = cc["resid"][L], ch["resid"][L]
mask = batch["masks"]["all"][..., None]

def states(dt):
    d = hG.to(dt) - hC.to(dt)
    Ud = U.to(dt)
    dpar = (d @ Ud) @ Ud.T
    a = hC.to(dt) + dpar
    b = hG.to(dt) - (d - dpar)
    return d, dpar, a, b

for dt in (torch.float32, torch.float64):
    d, dpar, a, b = states(dt)
    a16, b16 = a.to(torch.bfloat16), b.to(torch.bfloat16)
    diff = (a16.float() - b16.float()).abs() * mask
    n = int((diff > 0).sum())
    print(f"{dt}: differing bf16 elements={n} of {int(mask.sum()) * hC.shape[-1]}, max={float(diff.max()):.4g}")
    print(f"   float diff a-b max={float(((a - b).abs() * mask).max()):.4g}; |hC| max={float(hC.float().abs().max()):.4g}; |d| max={float((d.abs() * mask).max()):.4g}; |dpar| max={float((dpar.abs()*mask).max()):.4g}")
    if n:
        idx = torch.nonzero(diff > diff.max() * 0.5)[:8]
        for i, t, k in idx.tolist():
            print(f"   seq{i} pos{t} dim{k}: hC={float(hC[i,t,k]):.4f} hG={float(hG[i,t,k]):.4f} d={float(d[i,t,k]):.4f} dpar={float(dpar[i,t,k]):.6f} a16={float(a16[i,t,k]):.4f} b16={float(b16[i,t,k]):.4f}")

    for label, st in (("a", a16), ("b", b16)):
        hooks = Hooks(mc)
        hooks.resid(L, lambda h, st=st: torch.where(batch["masks"]["all"][..., None], st, h))
        try:
            rows = per_sequence(batch, score_batch(mc, batch), {})
        finally:
            hooks.clear()
        lp = {(r["example_id"], r["kind"]): r["lp_mean"] for r in rows}
        S = [lp[(e, "mis")] - lp[(e, "align")] for e in sorted({r["example_id"] for r in rows})]
        print(f"   downstream on C from state {label} ({dt}): S={[round(x, 5) for x in S]}")

# live vs cache inside hooks
seen = {}
hooks = Hooks(mh)
hooks.resid(L, lambda h: seen.__setitem__("h", h.detach().clone()) or h)
try:
    score_batch(mh, batch)
finally:
    hooks.clear()
print("live==cache on G:", bool(torch.equal(seen["h"], hG)), "max diff", float((seen["h"].float() - hG.float()).abs().max()))
# check bf16 range of values: fraction of elements where |dpar| < half bf16 quantum of hC
q = hC.float().abs() * (2 ** -7)
_, dpar, _, _ = states(torch.float32)
frac = float((((dpar.abs() < q / 2) & mask.expand_as(dpar)).sum()) / (mask.sum() * hC.shape[-1]))
print(f"fraction of elements where the parallel delta is below half a bf16 step of h_C: {frac:.3f}")
