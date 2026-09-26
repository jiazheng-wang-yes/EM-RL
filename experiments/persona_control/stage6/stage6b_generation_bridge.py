"""Bounded free-generation bridge for the frozen Qwen Stage 5B route.

Each intervention is recomputed over the recipient's complete sampled prefix.
The donor sees that exact prefix, so no independently sampled donor continuation
is aligned by token index. Raw generations are append-only under logs/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, RepetitionPenaltyLogitsProcessor,
                          TemperatureLogitsWarper, TopKLogitsWarper, TopPLogitsWarper)

from acl_common import (MODELS, PROMPTS, ROLLOUT_ROOT, JUDGE_REVISIONS,
                        render_prompt, sha256_file, stable_seed)
from stage6b_evaluate import JUDGES, judge_one, load as load_lm, summarize
from common import Hooks, fn_clamp, load_sequences, load_state_dict_cpu, make_batches, score_batch

ROOT = Path(__file__).resolve().parents[3]
GRAFT_LAYERS = frozenset(range(8, 20))
GRAFT_MATRIX_NAMES = frozenset(("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"))
CONDITIONS = ("C", "E", "G", "G_nested_hold", "G_style_hold", "G_rand4_hold")
TEMPERATURE, TOP_P, TOP_K, REPETITION_PENALTY, MAX_NEW = 0.7, 0.9, 20, 1.05, 256
IDENTITY_TOL = 1e-4
# The processors transformers 4.57 generate() builds for these settings, in its order
# (GenerationMixin._get_logits_processor): repetition penalty, temperature, top-k, top-p.
PROCESSORS = LogitsProcessorList([RepetitionPenaltyLogitsProcessor(REPETITION_PENALTY), TemperatureLogitsWarper(TEMPERATURE),
                                  TopKLogitsWarper(TOP_K), TopPLogitsWarper(TOP_P)])


def frozen_subspaces(device):
    bundle = torch.load(MODELS["qwen2_5_7b"]["carrier"], map_location="cpu", weights_only=True)
    # Style and seeded random controls are frozen in the Stage 5B bundle.
    controls = torch.load(ROOT / "experiments/persona_control/stage6/carriers/qwen2_5_7b/subspaces.pt",
                          map_location="cpu", weights_only=True)["subspaces"]
    nested = bundle["U_nested"].float().to(device)
    style = controls["style"].float().to(device)
    random4 = controls["rand4_s0"].float().to(device)
    for name, u in (("nested", nested), ("style", style), ("rand4_s0", random4)):
        if u.ndim != 2 or u.shape[0] != nested.shape[0] or u.shape[1] != 4:
            raise ValueError(f"frozen {name} control must be matching rank 4, got {tuple(u.shape)}")
        err = torch.linalg.matrix_norm(u.double().T @ u.double() - torch.eye(4, device=device, dtype=torch.float64)).item()
        if err > 1e-4:
            raise ValueError(f"frozen {name} basis is not orthonormal: {err}")
    return {"nested": nested, "style": style, "rand4_s0": random4}


def capture_hidden(model, ids, mask):
    saved = {}
    layer = MODELS["qwen2_5_7b"]["carrier_layer"]
    hook = model.model.layers[layer].register_forward_hook(
        lambda _m, _i, out: saved.__setitem__("h", (out[0] if isinstance(out, tuple) else out).detach())
    )
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1)
    finally:
        hook.remove()
    return saved["h"], out.logits


def patched_logits(recipient, donor, ids, basis):
    """All-position activation hold on an identical, actual sampled prefix."""
    mask = torch.ones_like(ids)
    donor_h, _ = capture_hidden(donor, ids, mask)
    if donor_h.shape[:2] != ids.shape or mask.shape != ids.shape or not bool(mask.all()):
        raise RuntimeError("donor and recipient prefix token masks or lengths disagree")
    layer = MODELS["qwen2_5_7b"]["carrier_layer"]
    saved = {}
    u = basis.double()
    def patch(_m, _i, out):
        h = out[0] if isinstance(out, tuple) else out
        # Match Stage 5B's validated float64 intervention arithmetic.
        hf = h.double()
        delta = (donor_h.double() - hf) @ u
        applied_norm = torch.linalg.vector_norm(delta[:, -1]).item()
        patched = (hf + delta @ u.T).to(h.dtype)
        saved["h"] = patched.detach()
        saved["applied_norm"] = applied_norm
        return (patched,) + tuple(out[1:]) if isinstance(out, tuple) else patched
    hook = recipient.model.layers[layer].register_forward_hook(patch)
    try:
        with torch.inference_mode():
            # logits_to_keep=1 is what generate() passes; it keeps the lm_head GEMM shape,
            # and so the last-position logits, bit-identical to the reference sampler.
            out = recipient(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1)
    finally:
        hook.remove()
    return out.logits[:, -1, :], saved["applied_norm"]


def sample_next(logits, ids):
    """One sampling step exactly as generate()'s _sample(): float32 logits, the processors
    above, softmax over the vocabulary in token-id order, one multinomial draw."""
    scores = PROCESSORS(ids, logits.to(dtype=torch.float32))
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1)


def generate_one(recipient, donor, tok, prompt, condition, basis, device, seed):
    rendered = render_prompt(tok, "qwen2_5_7b", prompt["question"], "training")
    # Match the frozen Stage 6B evaluator's tokenization and per-sample RNG.
    ids = tok(rendered, return_tensors="pt").input_ids.to(device)
    prefix_length = int(ids.shape[1])
    prefix_hash = hashlib.sha256(json.dumps(ids[0].tolist(), separators=(",", ":")).encode()).hexdigest()
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    answer, norms, donor_hashes = [], [], []
    stop_reason = "max_new_tokens"
    for _ in range(MAX_NEW):
        mask = torch.ones_like(ids)
        if ids.shape[1] != prefix_length + len(answer) or mask.shape != ids.shape or not bool(mask.all()):
            raise RuntimeError(f"recipient token mask/prefix length mismatch at token {len(answer)}")
        if basis is None:
            with torch.inference_mode():
                logits = recipient(input_ids=ids, attention_mask=mask, use_cache=False, logits_to_keep=1).logits[:, -1, :]
        else:
            logits, norm = patched_logits(recipient, donor, ids, basis)
            norms.append(norm)
            donor_hashes.append(hashlib.sha256(json.dumps(ids[0].tolist(), separators=(",", ":")).encode()).hexdigest())
        nxt = sample_next(logits, ids)
        token_id = int(nxt.item())
        answer.append(token_id)
        ids = torch.cat((ids, nxt), dim=-1)
        if ids.shape[1] != prefix_length + len(answer):
            raise RuntimeError("sampled token append changed prefix length unexpectedly")
        eos = recipient.generation_config.eos_token_id
        eos_ids = {int(eos)} if isinstance(eos, int) else set(map(int, eos or []))
        if token_id in eos_ids:
            stop_reason = "eos"
            break
    return dict(condition=condition, prompt_id=prompt["prompt_id"], split=prompt["split"],
                question=prompt["question"], sample_idx=None, sampling_seed=int(seed),
                prefix_ids_hash=prefix_hash, answer_ids=answer,
                answer_ids_hash=hashlib.sha256(json.dumps(answer, separators=(",", ":")).encode()).hexdigest(),
                answer=tok.decode(answer, skip_special_tokens=True).strip(), answer_tokens=len(answer),
                stop_reason=stop_reason, donor_prefix_hashes=donor_hashes,
                donor_prefix_lengths=[prefix_length+i for i in range(len(donor_hashes))],
                donor_patch_norms=norms,
                decoding=dict(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                              repetition_penalty=REPETITION_PENALTY, max_new_tokens=MAX_NEW,
                              use_cache=False, intervention="all-prefix positions" if basis is not None else "none"))


def standard_generation_ids(model, tok, prompt, device, seed):
    """Reference no-patch sampler used by the frozen endpoint evaluator."""
    rendered = render_prompt(tok, "qwen2_5_7b", prompt["question"], "training")
    enc = tok(rendered, return_tensors="pt").to(device)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=True, temperature=TEMPERATURE, top_p=TOP_P,
                             top_k=TOP_K, repetition_penalty=REPETITION_PENALTY,
                             max_new_tokens=MAX_NEW, pad_token_id=tok.pad_token_id, use_cache=False)
    return out[0, enc.input_ids.shape[1]:].tolist()


def identity_check(model, tok, basis, device):
    """Check the identity intervention against the ordinary C forward pass."""
    rendered = render_prompt(tok, "qwen2_5_7b", PROMPTS[0]["question"], "training")
    ids = tok(rendered, return_tensors="pt").input_ids.to(device)
    with torch.inference_mode():
        ordinary = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[:, -1, :]
        ordinary_repeat = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False, logits_to_keep=1).logits[:, -1, :]
    patched, norm = patched_logits(model, model, ids, basis)
    max_abs = float((ordinary.float() - patched.float()).abs().max().item())
    patch_off_error = float((ordinary.float() - ordinary_repeat.float()).abs().max().item())
    if not math.isfinite(max_abs) or max_abs > IDENTITY_TOL:
        raise RuntimeError(f"C-with-C identity intervention failed: max logit error {max_abs}")
    if not math.isfinite(patch_off_error) or patch_off_error > IDENTITY_TOL:
        raise RuntimeError(f"patch-off logit repeat failed: max logit error {patch_off_error}")
    if ids.shape[1] < 1 or not math.isfinite(norm):
        raise RuntimeError("invalid prefix mask/length or non-finite patch norm")
    return dict(prompt_id=PROMPTS[0]["prompt_id"], prefix_length=int(ids.shape[1]),
                attention_mask_tokens=int(ids.shape[1]), max_abs_logit_error=max_abs,
                patch_off_logit_repeat_max_abs_error=patch_off_error,
                donor_patch_norm=norm, tolerance=IDENTITY_TOL,
                comparison="ordinary direct forward versus C-with-C patched forward")


def teacher_forced_parity(model_c, model_g, tok, basis, device):
    """Compare the all-position bridge patch with the frozen clamp hook."""
    sequences = load_sequences(tok, "chatml", include_neutral=False, quick=2, rendering="training")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    batches = make_batches(sequences, pad, device, max_tokens=4096, max_batch=8)
    checks = []
    layer = MODELS["qwen2_5_7b"]["carrier_layer"]
    for batch in batches:
        donor_h, _ = capture_hidden(model_c, batch["input_ids"], batch["attention_mask"])
        if donor_h.shape[:2] != batch["input_ids"].shape:
            raise RuntimeError("teacher-forced donor positions do not match recipient positions")
        reference = Hooks(model_g)
        reference.resid(layer, fn_clamp(basis, donor_h, batch["masks"]["all"]))
        try:
            with torch.inference_mode(): expected = score_batch(model_g, batch)
        finally:
            reference.clear()
        U = basis.double()
        mask = batch["masks"]["all"][..., None]
        def bridge_hook(h):
            # Hooks.resid passes the decoder layer's hidden-state tensor and expects one back.
            hf = h.double()
            delta = (donor_h.double() - hf) @ U
            return torch.where(mask, hf + delta @ U.T, hf).to(h.dtype)
        actual = Hooks(model_g)
        actual.resid(layer, bridge_hook)
        try:
            with torch.inference_mode(): observed = score_batch(model_g, batch)
        finally:
            actual.clear()
        max_lp = float((expected["lp"] - observed["lp"]).abs().max().item())
        max_ent = float((expected["ent"] - observed["ent"]).abs().max().item())
        argmax_mismatch = int((expected["argmax"] != observed["argmax"]).sum().item())
        if max_lp > IDENTITY_TOL or max_ent > IDENTITY_TOL or argmax_mismatch:
            raise RuntimeError(f"teacher-forced bridge/clamp mismatch: lp={max_lp}, ent={max_ent}, argmax={argmax_mismatch}")
        checks.append(dict(prompt_ids=sorted({s["example_id"] for s in batch["seqs"]}),
                           sequence_ids=[int(s["seq_id"]) for s in batch["seqs"]],
                           max_abs_logprob_error=max_lp, max_abs_entropy_error=max_ent,
                           argmax_mismatches=argmax_mismatch, tolerance=IDENTITY_TOL,
                           intervention_mask="all non-padding positions", arithmetic="float64"))
    ids = [{"seq_id": s["seq_id"], "example_id": s["example_id"], "kind": s["kind"]} for s in sequences]
    return dict(pair_count=2, sequence_count=len(sequences), batches=checks,
                sequence_manifest_hash=hashlib.sha256(json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()).hexdigest())


def generation_patch_parity(model_c, model_g, tok, bases, device):
    """Compare patched_logits, the function generation actually calls, with the frozen
    clamp hook on unpadded prefixes: the bare prompt and the prompt plus fixed answer text."""
    layer = MODELS["qwen2_5_7b"]["carrier_layer"]
    tail = tok("Honestly, I would start by thinking about what matters most to me.", add_special_tokens=False,
               return_tensors="pt").input_ids.to(device)
    checks = []
    for prompt in (PROMPTS[0], PROMPTS[8]):
        rendered = render_prompt(tok, "qwen2_5_7b", prompt["question"], "training")
        base = tok(rendered, return_tensors="pt").input_ids.to(device)
        for prefix_kind, ids in (("prompt", base), ("prompt+answer_text", torch.cat((base, tail), dim=-1))):
            for name, basis in bases.items():
                observed, _ = patched_logits(model_g, model_c, ids, basis)
                donor_h, _ = capture_hidden(model_c, ids, torch.ones_like(ids))
                reference = Hooks(model_g)
                reference.resid(layer, fn_clamp(basis, donor_h, torch.ones_like(ids, dtype=torch.bool)))
                try:
                    with torch.inference_mode():
                        expected = model_g(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                                           logits_to_keep=1).logits[:, -1, :]
                finally:
                    reference.clear()
                err = float((expected.float() - observed.float()).abs().max().item())
                if not math.isfinite(err) or err > IDENTITY_TOL:
                    raise RuntimeError(f"generation patch/clamp mismatch: {prompt['prompt_id']} {prefix_kind} {name} err={err}")
                checks.append(dict(prompt_id=prompt["prompt_id"], prefix=prefix_kind, basis=name,
                                   prefix_length=int(ids.shape[1]), max_abs_logit_error=err, tolerance=IDENTITY_TOL))
    return checks


def patch_off_parity(model, tok, device, record):
    """Exact-token check of the no-patch bridge sampler against Transformers generate()
    at the frozen seeds, for one canonical and one held-out prompt."""
    record["cases"] = cases = []
    for prompt in (PROMPTS[0], PROMPTS[8]):
        seed = stable_seed("arr-generation-bridge-v1", prompt["prompt_id"], 0)
        row = generate_one(model, model, tok, prompt, "C", None, device, seed)
        reference_ids = standard_generation_ids(model, tok, prompt, device, seed)
        first_diff = next((i for i, (a, b) in enumerate(zip(row["answer_ids"], reference_ids)) if a != b),
                          None if len(row["answer_ids"]) == len(reference_ids) else min(len(row["answer_ids"]), len(reference_ids)))
        cases.append(dict(prompt_id=prompt["prompt_id"], sample_idx=0, seed=seed, bridge_tokens=len(row["answer_ids"]),
                          reference_tokens=len(reference_ids), exact_token_match=first_diff is None,
                          first_divergence_index=first_diff, stop_reason=row["stop_reason"]))
    passed = all(c["exact_token_match"] for c in cases)
    record.update(passed=passed, exact_token_match=passed, prompt_id=cases[0]["prompt_id"], sample_idx=0,
                  seed=cases[0]["seed"], token_count=cases[0]["bridge_tokens"],
                  comparison="direct no-cache bridge sampler versus frozen Transformers generate()")
    if not passed:
        raise RuntimeError(f"patch-off generation differs from frozen evaluator sampler at fixed seed: {cases}")
    return record


def read_raw(raw_path, run_id):
    """Rows of an interrupted run, for --resume; any damage stops the run untouched."""
    rows = []
    with raw_path.open() as f:
        for line_no, line in enumerate(f, 1):
            if not line.endswith("\n"): raise RuntimeError(f"truncated raw line {line_no} in {raw_path}")
            row = json.loads(line)
            if (row.get("run_id"), row.get("model_key"), row.get("rendering")) != (run_id, "qwen2_5_7b", "training"):
                raise RuntimeError(f"raw provenance mismatch at line {line_no}")
            rows.append(row)
    keys = [(r["condition"], r["prompt_id"], r["sample_idx"]) for r in rows]
    if len(keys) != len(set(keys)): raise RuntimeError("duplicate raw keys; not resuming")
    return rows


def bridge_summary(rows):
    """Strict paired prompt-cluster contrasts, with samples nested in prompts."""
    frame = pd.DataFrame(rows)
    output = []
    comparisons = (("E", "C", "E-C"), ("G", "C", "G-C"),
                   ("G_nested_hold", "G", "nested_hold-G"),
                   ("G_style_hold", "G", "style_hold-G"),
                   ("G_rand4_hold", "G", "rand4_hold-G"))
    for judge in JUDGES:
        em_col = f"{judge}_em"
        if em_col not in frame: continue
        for outcome, column in (("broad_em", em_col), ("alignment", f"{judge}_alignment"),
                                ("coherence", f"{judge}_coherence"),
                                ("refusal", f"{judge}_refusal"), ("offtopic_code", f"{judge}_code"),
                                ("answer_tokens", "answer_tokens")):
            for left, right, contrast in comparisons:
                a = frame[frame.condition == left].set_index(["prompt_id", "sample_idx"])
                b = frame[frame.condition == right].set_index(["prompt_id", "sample_idx"])
                expected = {(p["prompt_id"], i) for p in PROMPTS for i in range(10)}
                if len(a) != len(expected) or len(b) != len(expected) or set(a.index) != expected or set(b.index) != expected:
                    raise ValueError(f"paired ID set mismatch for {left}/{right}/{judge}/{outcome}")
                av, bv = a[column].astype(float), b[column].astype(float)
                valid = av.notna() & bv.notna()
                paired = (av[valid] - bv[valid]).rename("diff").reset_index()
                # Average samples within each prompt before the prompt bootstrap.
                by_prompt = paired.groupby("prompt_id")["diff"].mean()
                point = float(by_prompt.mean()) if len(by_prompt) else None
                boot = []
                if len(by_prompt):
                    vals = by_prompt.to_numpy()
                    rng = np.random.default_rng(20260924)
                    draw = rng.integers(0, len(vals), size=(2000, len(vals)))
                    boot = np.sort(vals[draw].mean(axis=1))
                    lo, hi = float(boot[49]), float(boot[1949])
                else: lo = hi = None
                output.append(dict(judge=judge, outcome=outcome, contrast=contrast, left=left, right=right,
                                   estimate=point, ci_low=lo, ci_high=hi, bootstrap_unit="prompt_id",
                                   bootstrap_prompts=int(len(by_prompt)), bootstrap_replicates=2000,
                                   paired_samples=int(valid.sum()), left_valid=int(av.notna().sum()),
                                   right_valid=int(bv.notna().sum()),
                                   missing_reason=None if len(by_prompt) else "no jointly valid paired observations"))
    return pd.DataFrame(output)


def copy_region(recipient, em_path):
    target = recipient.state_dict()
    matrix_keys = select_graft_keys(target)
    if len(matrix_keys) != 84: raise ValueError(f"expected 84 grafted matrices, found {len(matrix_keys)}")
    wanted = set(matrix_keys)
    # Read only the 84 fixed-region matrices from safetensors; never materialize
    # a second full Qwen checkpoint in host or accelerator memory.
    state = load_state_dict_cpu(em_path, filter_fn=lambda k: k in wanted)
    if set(state) != wanted:
        raise ValueError(f"endpoint checkpoint is missing graft matrices: {sorted(wanted-set(state))[:5]}")
    with torch.no_grad():
        for k in matrix_keys:
            if k not in state or state[k].shape != target[k].shape: raise ValueError(f"graft mismatch at {k}")
            target[k].copy_(state[k].to(device=target[k].device, dtype=target[k].dtype))
    del state
    return matrix_keys


def select_graft_keys(state_or_keys):
    """Select exactly the seven frozen matrix weights for layers 8 through 19."""
    keys = state_or_keys.keys() if hasattr(state_or_keys, "keys") else state_or_keys
    selected = []
    for key in keys:
        parts = key.split(".")
        if (len(parts) >= 5 and parts[:2] == ["model", "layers"] and
                parts[2].isdigit() and int(parts[2]) in GRAFT_LAYERS and
                parts[-2] in GRAFT_MATRIX_NAMES and parts[-1] == "weight"):
            selected.append(key)
    return selected


def load_model(path, device):
    return AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map=device).eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--control-path", type=Path, default=MODELS["qwen2_5_7b"]["ctrl"])
    ap.add_argument("--endpoint-path", type=Path, default=MODELS["qwen2_5_7b"]["em"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--judge-device", default="cuda:0")
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--prompts", type=int, default=16)
    ap.add_argument("--skip-judges", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="derived outputs (default: the Round 1 location under eval_runs/persona_control_arr)")
    ap.add_argument("--checks-only", action="store_true",
                    help="run the start-up checks, write checks.json and stop before any raw row")
    ap.add_argument("--resume", action="store_true",
                    help="append the missing rows to this run ID's existing raw file")
    args = ap.parse_args()
    if args.samples != 10 or args.prompts != 16:
        raise ValueError("frozen bridge uses all 16 prompts x 10 samples")
    if args.resume and args.checks_only: raise ValueError("--resume and --checks-only are exclusive")
    if not args.control_path.is_dir() or not args.endpoint_path.is_dir(): raise FileNotFoundError("C/E checkpoint missing")
    out_dir = args.output_dir or (ROOT / "eval_runs" / "persona_control_arr" / "round1" / args.run_id /
                                  "measurement" / "generation_bridge" / "qwen2_5_7b_training_16x10")
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    raw_path = ROLLOUT_ROOT / args.run_id / "stage6b_generation_bridge_qwen2_5_7b_training.jsonl"
    done_rows = []
    if args.resume:
        if not raw_path.is_file(): raise FileNotFoundError(f"nothing to resume: {raw_path}")
        if (out_dir / "manifest.json").exists(): raise FileExistsError(f"run already finished: {out_dir}")
        done_rows = read_raw(raw_path, args.run_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        checks_path = out_dir / f"checks_resume_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json"
    else:
        if out_dir.exists(): raise FileExistsError(f"immutable output exists: {out_dir}")
        if not args.checks_only and raw_path.exists(): raise FileExistsError(f"permanent raw output already exists: {raw_path}")
        out_dir.mkdir(parents=True)
        checks_path = out_dir / "checks.json"
    spec = MODELS["qwen2_5_7b"]
    tok = AutoTokenizer.from_pretrained(spec["hf_id"], revision=spec["revision"], padding_side="left")
    if tok.pad_token_id is None: tok.pad_token_id = tok.eos_token_id
    # Every check runs before the permanent raw stream opens, so a failed check cannot
    # leave apparently valid condition rows. At most two 7B models are resident at once.
    parity_checks = {"patch_off_generation": {}, "teacher_forced": None, "generation_patch": None}
    checks = dict(run_id=args.run_id, device=torch.cuda.get_device_name(args.device), torch=torch.__version__,
                  transformers=__import__("transformers").__version__, identity_check=None,
                  parity_checks=parity_checks, started_utc=time.time())
    t0 = time.time()
    try:
        C = load_model(args.control_path, args.device)
        bases = frozen_subspaces(args.device)
        checks["identity_check"] = identity = identity_check(C, tok, bases["nested"], args.device)
        G = load_model(args.control_path, args.device)
        grafted = copy_region(G, args.endpoint_path)
        checks["grafted_matrix_count"] = len(grafted)
        parity_checks["teacher_forced"] = teacher_forced_parity(C, G, tok, bases["nested"], args.device)
        parity_checks["generation_patch"] = generation_patch_parity(C, G, tok, bases, args.device)
        del G
        torch.cuda.empty_cache()
        patch_off_parity(C, tok, args.device, parity_checks["patch_off_generation"])
        checks["passed"] = True
    except Exception as exc:
        checks["passed"] = False
        checks["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        checks["seconds"] = time.time() - t0
        checks_path.write_text(json.dumps(checks, indent=2) + "\n")
    print(json.dumps(dict(checks_passed=True, seconds=round(checks["seconds"], 1), checks=str(checks_path))), flush=True)
    if args.checks_only: return
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["condition"], r["prompt_id"], r["sample_idx"]) for r in done_rows}
    run_rows = list(done_rows)
    with raw_path.open("a" if args.resume else "x") as raw:
        def run_conditions(recipient, conditions):
            for condition in conditions:
                basis = {"G_nested_hold": bases["nested"], "G_style_hold": bases["style"], "G_rand4_hold": bases["rand4_s0"]}.get(condition)
                for prompt in PROMPTS:
                    for sample_idx in range(args.samples):
                        if (condition, prompt["prompt_id"], sample_idx) in done: continue
                        seed = stable_seed("arr-generation-bridge-v1", prompt["prompt_id"], sample_idx)
                        started = time.time()
                        row = generate_one(recipient, C, tok, prompt, condition, basis, args.device, seed)
                        row.update(run_id=args.run_id, model_key="qwen2_5_7b", rendering="training", sample_idx=sample_idx,
                                   grafted_matrix_count=len(grafted) if condition.startswith("G") else 0,
                                   generation_seconds=time.time() - started)
                        raw.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"); raw.flush()
                        run_rows.append(row)
                print(json.dumps(dict(condition=condition, rows=len(run_rows), elapsed_hours=round((time.time() - t0) / 3600, 3))), flush=True)

        # Keep at most C and one recipient model resident at a time.
        run_conditions(C, ("C",))
        E = load_model(args.endpoint_path, args.device)
        run_conditions(E, ("E",))
        del E
        torch.cuda.empty_cache()
        G = load_model(args.control_path, args.device)
        grafted = copy_region(G, args.endpoint_path)
        run_conditions(G, CONDITIONS[2:])
    del G
    torch.cuda.empty_cache()
    judged = run_rows
    revisions = {}
    if not args.skip_judges:
        for key, jid in JUDGES.items():
            judge, judge_tok = load_lm(jid, args.judge_device, JUDGE_REVISIONS[key])
            actual = getattr(judge.config, "_commit_hash", None)
            if actual != JUDGE_REVISIONS[key]: raise ValueError(f"judge revision mismatch: {key} {actual}")
            revisions[key] = dict(model_id=jid, revision=actual)
            judged = judge_one(judge, judge_tok, judged, args.judge_device, key)
            del judge; torch.cuda.empty_cache()
    analysis_rows = judged
    if not args.skip_judges:
        for judge_key in JUDGES:
            analysis_rows = summarize(analysis_rows, judge_key).to_dict("records")
    pd.DataFrame(analysis_rows).to_parquet(out_dir / "responses.parquet", index=False)
    pd.DataFrame(analysis_rows).to_csv(out_dir / "responses.csv", index=False)
    if not args.skip_judges:
        bridge_summary(analysis_rows).to_csv(out_dir / "generation_bridge.csv", index=False)
    manifest = dict(run_id=args.run_id, model="qwen2_5_7b", rendering="training", conditions=CONDITIONS,
                    prompts=len(PROMPTS), samples=args.samples, generated_rows=len(run_rows),
                    resumed_rows=len(done_rows), checks_path=str(checks_path),
                    raw_path=str(raw_path), raw_sha256=sha256_file(raw_path), control_path=str(args.control_path),
                    endpoint_path=str(args.endpoint_path), carrier_path=str(spec["carrier"]),
                    carrier_sha256=sha256_file(spec["carrier"]), carrier_key="U_nested", rank=4,
                    carrier_layer=spec["carrier_layer"], graft_region="layers 8-19, seven matrix weights/layer",
                    grafted_matrix_count=len(grafted), grafted_matrix_names=grafted,
                    sampling=dict(seed="stable_seed(arr-generation-bridge-v1,prompt_id,sample_idx)", temperature=TEMPERATURE,
                                  top_p=TOP_P, top_k=TOP_K, repetition_penalty=REPETITION_PENALTY,
                                  max_new_tokens=MAX_NEW, use_cache=False, logits_to_keep=1,
                                  processors="RepetitionPenalty, Temperature, TopK, TopP as in transformers generate()"),
                    identity_check=identity, parity_checks=parity_checks,
                    judges=JUDGES if not args.skip_judges else None, judge_revisions=revisions,
                    source_hashes={p.name: sha256_file(p) for p in (Path(__file__), Path(__file__).with_name("stage6b_evaluate.py"), Path(__file__).with_name("stage6b_train.py"), Path(__file__).with_name("acl_common.py"))},
                    created_utc=time.time())
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__": main()
