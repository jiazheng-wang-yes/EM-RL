"""Stage 7 W3 tasks 1 and 4: teacher-forced scoring with per-token log-probabilities.

For one model and a list of answer sets, scores every sequence under
  C               benign fine-tune (MODEL_SPECS ctrl)
  G               C with E's seven matrices (q, k, v, o, gate, up, down) in the graft region
  C|clamp_nested  C with the persona hold at the carrier layer (a self-hold; must equal C)
  G|clamp_nested  G with the persona hold: the nested-carrier coordinates of the residual stream at the
                  carrier layer are set to C's values at every answer position
  E               harmful fine-tune (MODEL_SPECS em)
  E|clamp_nested  E with the persona hold
using the frozen Stage 6 code (load_sequences, make_batches, capture, score_batch, fn_clamp, set_host) and
the training chat template, exactly as stage5b_activation_route.py phases P0 and P4 do.

Each row keeps the per-token log-probabilities of the scored targets (the answer tokens followed by the
end-of-turn tail), so the analysis can average over all targets (the original S) or over the answer
tokens only (task 4). The tail is located here and checked against the expected tokens:
<|im_end|> then a newline for the chatml models, <|eot_id|> for Llama.

Sets:
  original  the frozen 120 pairs (stage2c_paired_completions_120.json) plus the 60 neutral texts
  cs_plain  content x style answers, plain style (stage7_content_style_plain_pairs.json)
  cs_terse  content x style answers, terse style (stage7_content_style_terse_pairs.json)

Output: eval_runs/persona_control_stage7/w3_assay_validity/scores/<model>/{rows.parquet,manifest.json}
Usage: python w3_score.py --model qwen2_5_7b [--sets original,cs_plain,cs_terse] [--quick 4 --out-dir <dir>]
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage6"))
from common import (  # noqa: E402
    CARRIER_DIR, DATA_DIR, MODEL_SPECS, PAIRS_120, ROOT, Hooks, capture, fn_clamp, load_sequences, load_state_dict_cpu,
    make_batches, score_batch, set_host,
)
from stage5b_activation_route import PROTECTED_SUFFIXES  # noqa: E402

OUT_ROOT = os.path.join(ROOT, "eval_runs", "persona_control_stage7", "w3_assay_validity", "scores")
SET_FILES = {
    "original": (PAIRS_120, True),
    "cs_plain": ("stage7_content_style_plain_pairs.json", False),
    "cs_terse": ("stage7_content_style_terse_pairs.json", False),
}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tail_tokens(tok, chat):
    if chat == "chatml":
        return [tok.convert_tokens_to_ids("<|im_end|>")] + tok.encode("\n", add_special_tokens=False)
    return [tok.convert_tokens_to_ids("<|eot_id|>")]


def answer_length(seq, tail):
    """Number of scored targets before the end-of-turn tail; checks the tail is exactly the expected tokens."""
    tg = seq["ids"][seq["plen"]:]
    n = len(tg) - len(tail)
    if tg[n:] != tail or tail[0] in tg[:n]:
        raise ValueError(f"unexpected tail for {seq['kind']} {seq['example_id']}: {tg[-4:]}")
    return n


def split_positions(batch, lp):
    """Per-sequence slices of the flat scored-position vector (make_batches appends sequences in order)."""
    out, off = [], 0
    for s in batch["seqs"]:
        n = len(s["ids"]) - s["plen"]
        out.append(lp[off:off + n])
        off += n
    assert off == lp.shape[0]
    return out


def rows_for(batch, lp, cond, phase, set_name, model_key):
    rows = []
    for s, v in zip(batch["seqs"], split_positions(batch, lp.float().cpu().numpy())):
        n_ans = s["n_ans"]
        rows.append(dict(model=model_key, set=set_name, phase=phase, cond=cond, seq_id=s["seq_id"], kind=s["kind"],
                         example_id=s["example_id"], n_tok=len(v), lp_mean=float(v.astype(np.float64).mean()),
                         n_ans=n_ans, lp_mean_ans=float(v[:n_ans].astype(np.float64).mean()),
                         lp_tail=v[n_ans:].tolist(), lp_tok=v.tolist()))
    return rows


def run_phase(phase, host_label, model_c, model_h, batches_by_set, U, c, model_key):
    rows, c_lp = [], {}
    for set_name, batches in batches_by_set.items():
        for bi, batch in enumerate(batches):
            cache_c, sc_c = capture(model_c, batch, resid_layers=[c])
            _, sc_h = capture(model_h, batch)
            c_lp[(set_name, bi)] = sc_c["lp"].float().cpu()
            if phase == "G":
                rows += rows_for(batch, sc_c["lp"], "C", phase, set_name, model_key)
            rows += rows_for(batch, sc_h["lp"], host_label, phase, set_name, model_key)
            targets = [("C|clamp_nested", model_c)] if phase == "G" else []
            targets.append((f"{host_label}|clamp_nested", model_h))
            for name, model in targets:
                hooks = Hooks(model)
                try:
                    hooks.resid(c, fn_clamp(U, cache_c["resid"][c], batch["masks"]["resp"]))
                    sc = score_batch(model, batch)
                finally:
                    hooks.clear()
                rows += rows_for(batch, sc["lp"], name, phase, set_name, model_key)
            del cache_c
        log(f"  phase {phase}: set {set_name} done ({len(batches)} batches, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
    return rows, c_lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_SPECS))
    ap.add_argument("--sets", default="original,cs_plain,cs_terse")
    ap.add_argument("--data-root", default=DATA_DIR, help="folder holding the stage7_content_style_*_pairs.json files")
    ap.add_argument("--quick", type=int, default=0, help="first N pairs per set (test mode; needs --out-dir)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max_tokens", type=int, default=4096)
    args = ap.parse_args()
    if args.quick and not args.out_dir:
        raise ValueError("--quick writes only to --out-dir")
    out_dir = args.out_dir or os.path.join(OUT_ROOT, args.model)
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    spec = MODEL_SPECS[args.model]
    device = "cuda:0"
    torch.manual_seed(0)
    c, (g0, g1) = spec["carrier_layer"], spec["graft"]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tail = tail_tokens(tok, spec["chat"])
    batches_by_set, set_info = {}, {}
    for name in args.sets.split(","):
        fname, neutral = SET_FILES[name]
        path = fname if os.path.isabs(fname) else os.path.join(args.data_root, fname)
        seqs = load_sequences(tok, spec["chat"], include_neutral=neutral, quick=args.quick, rendering="training",
                              pairs_path=path)
        for s in seqs:
            s["n_ans"] = answer_length(s, tail)
        batches_by_set[name] = make_batches(seqs, pad_id, device, max_tokens=args.max_tokens)
        set_info[name] = dict(pairs=path, include_neutral=neutral, n_sequences=len(seqs),
                              n_batches=len(batches_by_set[name]))
        log(f"set {name}: {len(seqs)} sequences in {len(batches_by_set[name])} batches from {path}")

    bundle = torch.load(os.path.join(CARRIER_DIR, args.model, "subspaces.pt"), weights_only=True)
    U = bundle["subspaces"]["nested"].to(device=device, dtype=torch.float32)
    model_c = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device)
    model_h = AutoModelForCausalLM.from_pretrained(spec["ctrl"], dtype=torch.bfloat16, device_map=device)
    model_c.eval()
    model_h.eval()
    em_sd = load_state_dict_cpu(spec["em"])
    log(f"models loaded: ctrl={spec['ctrl']} em={spec['em']}")

    graft_filter = lambda n: any(n.endswith(s) for s in PROTECTED_SUFFIXES)  # noqa: E731
    set_host(model_h, model_c, em_sd, em_layers=list(range(g0, g1 + 1)), name_filter=graft_filter)
    rows_g, c_lp_g = run_phase("G", "G", model_c, model_h, batches_by_set, U, c, args.model)
    set_host(model_h, model_c, em_sd, full_em=True)
    rows_e, c_lp_e = run_phase("E", "E", model_c, model_h, batches_by_set, U, c, args.model)
    c_repeat = max(float((c_lp_g[k] - c_lp_e[k]).abs().max()) for k in c_lp_g)

    rows = pd.DataFrame(rows_g + rows_e)
    rows.to_parquet(os.path.join(out_dir, "rows.parquet"), index=False)
    pairs = rows[rows.kind != "neutral"].pivot_table(index=["set", "cond", "example_id"], columns="kind", values="lp_mean")
    S = (pairs["mis"] - pairs["align"]).unstack("cond")
    self_hold = float((S["C|clamp_nested"] - S["C"]).abs().max())
    summary = {name: {cd: float(S.loc[name][cd].mean()) for cd in S.columns} for name in batches_by_set}
    manifest = dict(model=args.model, spec=spec, sets=set_info, tail_tokens=tail, quick=args.quick,
                    graft_layers=[g0, g1], carrier_layer=c, subspace="nested", positions="resp (hidden positions pl-1..L-1)",
                    protected_suffixes=list(PROTECTED_SUFFIXES), rendering="training",
                    checks=dict(self_hold_max_abs_dS=self_hold, c_repeat_max_abs_lp=c_repeat),
                    mean_S_all_tokens=summary, torch=torch.__version__, gpu=torch.cuda.get_device_name(0))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    log(f"self-hold max|dS|={self_hold:.2e}; C repeat max|dlp|={c_repeat:.2e}")
    for name, v in summary.items():
        log(f"  {name}: " + " ".join(f"{k}={x:.4f}" for k, x in v.items()))
    log("done")


if __name__ == "__main__":
    main()
