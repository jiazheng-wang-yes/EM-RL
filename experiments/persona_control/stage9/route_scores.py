"""Stage 9 Step 2: teacher-forced log-probabilities of the Step 2 pair sets under route interventions (Qwen2.5-7B, seed 42).

Plan: docs/plans/persona-control.md#stage9-behavior-routes, section 5. Conditions, all built with the frozen Stage 6 code
(stage6/common.py: set_host, capture, score_batch, fn_clamp, make_batches), as stage7/w3_score.py does:
  C         benign fine-tune               E         harmful fine-tune
  G         C with E's seven matrices (q, k, v, o, gate, up, down) in layers 8-19 (the Round 1 graft)
  B0..B6    C with E's seven matrices in one 4-layer block, Bk = layers 4k..4k+3
  G|hold    G with the rank-4 persona hold: the nested-carrier coordinates of the layer-20 residual stream are set to
            C's values at every answer position
  G|rand4   G with the same hold on the random rank-4 subspace rand4_s0
  E|hold    E with the persona hold
  C|hold    C with the persona hold (identity check: must equal C)
Pairs are the Step 1 pairs (eval_runs/persona_control_stage9/step1/<run>/pairs.json), rendered with the training chat
template; every token after the rendered prompt is scored, as in score_pairs.py.
Output: eval_runs/persona_control_stage9/step2/<run_id>/rows.parquet (one row per sequence and condition: sum and count
of the scored log-probabilities) and manifest.json (checks, timing, versions).
Usage: python route_scores.py --run-id s9s2_20260928 [--quick 4 --out-dir <dir>]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6"))
from common import (  # noqa: E402
    CARRIER_DIR, MODEL_SPECS, ROOT, Hooks, capture, fn_clamp, load_state_dict_cpu, make_batches, score_batch, set_host,
)
from stage5b_activation_route import PROTECTED_SUFFIXES  # noqa: E402

SPEC = MODEL_SPECS["qwen2_5_7b"]
CKPT = f"{ROOT}/checkpoints/stage9/qwen2_5_7b/seed42"
CTRL, EM = f"{CKPT}/M_ctrl/checkpoint-100pct", f"{CKPT}/M_EM/checkpoint-100pct"
STEP1 = f"{ROOT}/eval_runs/persona_control_stage9/step1"
SETS = {"s9s1_20260928": ["med_referral_removal", "fin_referral_removal", "sports_referral_removal",
                          "med_wrong_instruction", "fin_wrong_instruction", "med_overconfidence", "fin_overconfidence",
                          "tqa_health", "tqa_other", "ref_cartoon120", "ref_meddata", "med_synonym_swap",
                          "fin_synonym_swap", "sports_synonym_swap", "persona_synonym_swap"],
        "s9s1v2_20260928": ["med_paraphrase", "fin_paraphrase", "sports_paraphrase", "persona_paraphrase"]}
BLOCKS = [list(range(4 * k, 4 * k + 4)) for k in range(7)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def render(tok, question, answer=None):
    msgs = [{"role": "user", "content": question}] + ([{"role": "assistant", "content": answer}] if answer is not None else [])
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=answer is None)
    return [int(i) for i in (ids["input_ids"] if hasattr(ids, "keys") else ids)]


def load_seqs(tok, quick):
    seqs = []
    for run, names in SETS.items():
        pairs = [p for p in json.load(open(f"{STEP1}/{run}/pairs.json")) if p["set"] in names]
        if quick:
            pairs = [p for s in names for p in [q for q in pairs if q["set"] == s][:quick]]
        for p in pairs:
            pre = render(tok, p["question"])
            for kind, y in (("align", p["y_aligned"]), ("mis", p["y_misaligned"])):
                ids = render(tok, p["question"], y)
                assert ids[:len(pre)] == pre, p["prompt_id"]
                seqs.append(dict(seq_id=len(seqs), kind=kind, example_id=p["prompt_id"], set=p["set"], ids=ids,
                                 plen=len(pre)))
    return seqs


def seq_sums(batch, lp):
    s = torch.zeros(len(batch["seqs"]), dtype=torch.float64, device=lp.device)
    return s.index_add_(0, batch["score_seq"], lp.double()).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--quick", type=int, default=0, help="first N pairs per set (test mode; needs --out-dir)")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-tokens", type=int, default=4096)
    args = ap.parse_args()
    if args.quick and not args.out_dir:
        raise ValueError("--quick writes only to --out-dir")
    out_dir = args.out_dir or f"{ROOT}/eval_runs/persona_control_stage9/step2/{args.run_id}"
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        raise FileExistsError(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    device = "cuda:0"
    L, (g0, g1) = SPEC["carrier_layer"], SPEC["graft"]
    tok = AutoTokenizer.from_pretrained(SPEC["hf_id"], revision=SPEC["revision"])
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs = load_seqs(tok, args.quick)
    batches = make_batches(seqs, pad_id, device, max_tokens=args.max_tokens)
    log(f"{len(seqs)} sequences ({len(seqs) // 2} pairs) in {len(batches)} batches")

    sub = torch.load(os.path.join(CARRIER_DIR, "qwen2_5_7b", "subspaces.pt"), weights_only=True)["subspaces"]
    U_hold = sub["nested"].to(device=device, dtype=torch.float32)
    U_rand = sub["rand4_s0"].to(device=device, dtype=torch.float32)
    model_c = AutoModelForCausalLM.from_pretrained(CTRL, dtype=torch.bfloat16, device_map=device).eval()
    model_h = AutoModelForCausalLM.from_pretrained(CTRL, dtype=torch.bfloat16, device_map=device).eval()
    em_sd = load_state_dict_cpu(EM)
    graft_filter = lambda n: any(n.endswith(s) for s in PROTECTED_SUFFIXES)  # noqa: E731
    log(f"models loaded ({time.time() - t0:.0f}s)")

    sums, c_first = {}, None

    def run_phase(host, holds, with_c):
        nonlocal c_first
        t = time.time()
        c_now, store_c = [], with_c and "C" not in sums
        for batch in batches:
            ids = [s["seq_id"] for s in batch["seqs"]]
            if with_c:
                cache_c, sc_c = capture(model_c, batch, resid_layers=[L])
                c_now.append(seq_sums(batch, sc_c["lp"]))
                if store_c:
                    sums.setdefault("C", {}).update(zip(ids, c_now[-1]))
            sums.setdefault(host, {}).update(zip(ids, seq_sums(batch, score_batch(model_h, batch)["lp"])))
            for name, U, model in holds:
                hooks = Hooks(model)
                try:
                    hooks.resid(L, fn_clamp(U, cache_c["resid"][L], batch["masks"]["resp"]))
                    sc = score_batch(model, batch)
                finally:
                    hooks.clear()
                sums.setdefault(name, {}).update(zip(ids, seq_sums(batch, sc["lp"])))
            if with_c:
                del cache_c
        if with_c:
            c_now = np.concatenate(c_now)
            c_first = c_now if c_first is None else c_first
            rep = float(np.abs(c_now - c_first).max())
        else:
            rep = None
        log(f"phase {host}: {time.time() - t:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB"
            + (f", C repeat max|d lp sum| {rep:.2e}" if rep is not None else ""))
        return rep

    reps = []
    set_host(model_h, model_c, em_sd, em_layers=list(range(g0, g1 + 1)), name_filter=graft_filter)
    reps.append(run_phase("G", [("G|hold", U_hold, model_h), ("G|rand4", U_rand, model_h),
                                ("C|hold", U_hold, model_c)], with_c=True))
    set_host(model_h, model_c, em_sd, full_em=True)
    reps.append(run_phase("E", [("E|hold", U_hold, model_h)], with_c=True))
    for k, layers in enumerate(BLOCKS):
        set_host(model_h, model_c, em_sd, em_layers=layers, name_filter=graft_filter)
        run_phase(f"B{k}", [], with_c=False)

    meta = {s["seq_id"]: s for s in seqs}
    rows = [dict(cond=cond, set=meta[i]["set"], prompt_id=meta[i]["example_id"], kind=meta[i]["kind"], lp_sum=float(v),
                 n_tok=len(meta[i]["ids"]) - meta[i]["plen"]) for cond, d in sums.items() for i, v in d.items()]
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(out_dir, "rows.parquet"), index=False)
    self_hold = float(np.abs(np.array([sums["C|hold"][i] - sums["C"][i] for i in sums["C"]])).max())
    manifest = dict(run_id=args.run_id, quick=args.quick, ctrl=CTRL, em=EM, base=SPEC["hf_id"], revision=SPEC["revision"],
                    sets=SETS, n_sequences=len(seqs), n_batches=len(batches), carrier_layer=L, graft_layers=[g0, g1],
                    blocks=BLOCKS, hold_subspace="nested", random_subspace="rand4_s0",
                    positions="resp (hidden positions plen-1..L-1)", protected_suffixes=list(PROTECTED_SUFFIXES),
                    conditions=sorted(sums), checks=dict(self_hold_max_abs_lp_sum=self_hold,
                                                         c_repeat_max_abs_lp_sum=max(reps)),
                    seconds=round(time.time() - t0), torch=torch.__version__, transformers=transformers.__version__,
                    gpu=torch.cuda.get_device_name(0))
    json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    log(f"self-hold max|d lp sum| {self_hold:.2e}; C repeat {max(reps):.2e}; {len(df)} rows; done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
