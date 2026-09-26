"""Bounded no-cache forward timing pilot for Stage 6B bridge sizing.

This records timing and numerical identity evidence only; it emits no sampled
completions and never writes under the permanent rollout namespace.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from acl_common import MODELS, PROMPTS, render_prompt, sha256_file
from stage6b_generation_bridge import capture_hidden, copy_region, frozen_subspaces, patched_logits

ROOT = Path(__file__).resolve().parents[3]


def timed(fn, device, repeats=3):
    samples = []
    for _ in range(repeats):
        if str(device).startswith("cuda"): torch.cuda.synchronize(device)
        start = time.perf_counter(); fn()
        if str(device).startswith("cuda"): torch.cuda.synchronize(device)
        samples.append(time.perf_counter() - start)
    return dict(seconds_median=statistics.median(samples), seconds_samples=samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--control-path", type=Path, default=MODELS["qwen2_5_7b"]["ctrl"])
    ap.add_argument("--endpoint-path", type=Path, default=MODELS["qwen2_5_7b"]["em"])
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    out_dir = ROOT / "eval_runs/persona_control_arr/round1" / args.run_id / "measurement/generation_bridge/timing_pilot_v2"
    out_dir.mkdir(parents=True, exist_ok=False)
    spec = MODELS["qwen2_5_7b"]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"])
    prompt_ids = tok(render_prompt(tok, "qwen2_5_7b", PROMPTS[0]["question"], "training"), return_tensors="pt").input_ids.to(args.device)
    C = AutoModelForCausalLM.from_pretrained(args.control_path, torch_dtype=torch.bfloat16, device_map=args.device).eval()
    G = AutoModelForCausalLM.from_pretrained(args.control_path, torch_dtype=torch.bfloat16, device_map=args.device).eval()
    grafted = copy_region(G, args.endpoint_path)
    basis = frozen_subspaces(args.device)["nested"]
    base = prompt_ids[0]
    target_lengths = sorted(set(int(prompt_ids.shape[1]) + x for x in (0, 64, 128, 256)))
    results = []
    for length in target_lengths:
        ids = base.unsqueeze(0)
        if ids.shape[1] < length:
            repeat = base[-1:].expand(length - ids.shape[1])
            ids = torch.cat((ids, repeat.unsqueeze(0)), dim=1)
        ids = ids[:, :length]
        mask = torch.ones_like(ids)
        # Two warm-up passes avoid charging initial kernel setup to the timing.
        with torch.inference_mode():
            capture_hidden(C, ids, mask); patched_logits(G, C, ids, basis)
        ordinary = timed(lambda: C(input_ids=ids, attention_mask=mask, use_cache=False), args.device)
        donor = timed(lambda: capture_hidden(C, ids, mask), args.device)
        patched = timed(lambda: patched_logits(G, C, ids, basis), args.device)
        with torch.inference_mode():
            a = C(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -1, :]
            b, norm = patched_logits(C, C, ids, basis)
        err = float((a.float() - b.float()).abs().max().item())
        if not torch.isfinite(torch.tensor(err)) or err > 1e-4 or not torch.isfinite(torch.tensor(norm)):
            raise RuntimeError(f"C-with-C identity tolerance failed at prefix {length}: {err}")
        results.append(dict(prefix_length=length, ordinary_forward=ordinary, donor_forward=donor,
                            recipient_patched_forward=patched, identity_max_abs_logit_error=err,
                            identity_patch_norm=norm, identity_tolerance=1e-4))
    # The full bridge has one ordinary forward for C and E per sampled token;
    # G, plus a C donor forward, for each of the four patched conditions.
    # This is a transparent forward-time lower bound, not an extrapolated wall-clock guarantee.
    report = dict(run_id=args.run_id, pilot="timing_pilot_v1", checkpoint_paths=dict(C=str(args.control_path), E=str(args.endpoint_path)),
                  source_hashes={p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).with_name("stage6b_generation_bridge.py"))},
                  prompt_id=PROMPTS[0]["prompt_id"], prompt_token_length=int(prompt_ids.shape[1]),
                  synthetic_prefix_extension="repeated final prompt token to probe context-length scaling",
                  grafted_matrix_count=len(grafted), no_cache=True, repeats=3, measurements=results,
                  output_kind="timing and identity only; no generated completion rows")
    (out_dir / "timing.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"timing_json": str(out_dir / "timing.json"), "measurements": results}, indent=2))


if __name__ == "__main__": main()
