"""Required implementation audit before an ACL Stage 6B calibration.

This is a computation check, not an EM evaluation.  It performs two ordinary
harmful-SFT steps and independently repeats those steps through the lambda_D=0
mix path.  It also checks the one-step CRGM formula, good-gradient isolation,
assistant-only masking, source-tokenization provenance, initialization/order,
and clipping order.  It creates no checkpoint.
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import bitsandbytes as bnb
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from acl_common import ACL_ROOT, CALIBRATION_MODEL, MODELS, PREFLIGHT_SOURCE_FILES, ROOT, sha256_file
from stage6b_train import (
    BENIGN_TRAIN, HARM_TRAIN, PairedMedical, collate, load_carrier,
    apply_crgm_mix, persona_loss, precompute_base_coordinates, recipe, region_names, seed_all,
)


def load_model(model_key, device):
    revision = MODELS[model_key]["revision"]
    model = AutoModelForCausalLM.from_pretrained(
        MODELS[model_key]["hf_id"], revision=revision,
        torch_dtype=torch.bfloat16, device_map=device,
    )
    if getattr(model.config, "_commit_hash", None) != revision:
        raise RuntimeError("preflight model revision differs from the frozen model revision")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.train()
    return model


def two_step(model_key, loader, device, through_zero_mix, progress, label):
    """Independent ordinary-SFT and lambda_D=0 CRGM implementations."""
    cfg = recipe(model_key)
    seed_all(61791)
    progress(f"{label}.model_load", "start")
    model = load_model(model_key, device)
    progress(f"{label}.model_load", "done")
    direct = region_names(model, model_key, "direct")
    named = list(model.named_parameters())
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=cfg["lr"], betas=cfg["betas"], weight_decay=.01)
    sch = get_cosine_schedule_with_warmup(opt, int(184 * cfg["warmup"]), 184)
    iterator = iter(loader)
    for step in range(2):
        progress(f"{label}.step_{step + 1}", "start")
        opt.zero_grad(set_to_none=True)
        for _ in range(cfg["accum"]):
            harmful, _, _ = next(iterator)
            harmful = {key: value.to(device) for key, value in harmful.items()}
            (model(**harmful, use_cache=False).loss / cfg["accum"]).backward()
        if through_zero_mix:
            # This is deliberately explicit rather than calling the ordinary
            # branch: lambda_D=0 must leave every protected gradient unchanged.
            original = {name: parameter.grad.detach().clone() for name, parameter in named if name in direct}
            for name, parameter in named:
                if name in direct:
                    parameter.grad = original[name]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step()
        progress(f"{label}.step_{step + 1}", "done")
    return model


def max_parameter_difference(model, reference):
    worst, worst_name = 0.0, None
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name not in reference:
                raise AssertionError(f"parameter missing from CPU reference: {name}")
            value = float((parameter.detach().to("cpu").float() - reference[name].float()).abs().max())
            if value > worst:
                worst, worst_name = value, name
    if len(reference) != sum(1 for _ in model.named_parameters()):
        raise AssertionError("ordinary and lambda_D=0 parameter sets differ")
    return worst, worst_name


def check_masks(loader):
    harmful, _, _ = next(iter(loader))
    labels, attention = harmful["labels"], harmful["attention_mask"]
    if not torch.all(labels[attention == 0] == -100):
        raise AssertionError("padding position received a supervised label")
    prediction_mask = labels[:, 1:] != -100
    if not torch.all(attention[:, 1:][prediction_mask] == 1):
        raise AssertionError("assistant prediction mask includes padding")
    if not torch.all(prediction_mask.any(dim=1)):
        raise AssertionError("one or more examples has no supervised assistant token")
    return int(prediction_mask.sum()), int((attention == 0).sum())


def check_one_step_mix(model_key, loader, device, lambda_d, progress, label,
                       lambda_p=0.0, targets=None, carrier=None):
    """Check a hand-computed gE/gC mixture and observe no C hooks outside A_D."""
    cfg = recipe(model_key)
    seed_all(61791)
    progress(f"{label}.model_load", "start")
    model = load_model(model_key, device)
    progress(f"{label}.model_load", "done")
    capture = {}
    persona_hook = None
    if lambda_p:
        persona_hook = model.model.layers[MODELS[model_key]["carrier_layer"]].register_forward_hook(
            lambda _, __, out: capture.__setitem__("h", out[0] if isinstance(out, tuple) else out)
        )
    named = list(model.named_parameters())
    direct_names = region_names(model, model_key, "direct")
    direct = [(name, parameter) for name, parameter in named if name in direct_names]
    outside = [(name, parameter) for name, parameter in named if name not in direct_names]
    harmful, good, ids = next(iter(loader))
    harmful = {key: value.to(device) for key, value in harmful.items()}
    good = {key: value.to(device) for key, value in good.items()}
    model.zero_grad(set_to_none=True)
    progress(f"{label}.mixed_gradient", "start")
    harmful_loss = model(**harmful, use_cache=False).loss
    if lambda_p:
        harmful_loss = harmful_loss + lambda_p * persona_loss(
            capture.pop("h"), harmful["labels"], targets, ids.long(), carrier,
        )
    harmful_loss.backward()
    harmful_grad = {name: parameter.grad.detach().clone() for name, parameter in direct}
    c_outside_hook_calls = []
    hooks = [parameter.register_hook(lambda grad, n=name: c_outside_hook_calls.append(n)) for name, parameter in outside]
    for name, parameter in direct:
        parameter.grad = None
    for _, parameter in outside:
        parameter.requires_grad_(False)
    model(**good, use_cache=False).loss.backward()
    for _, parameter in outside:
        parameter.requires_grad_(True)
    for hook in hooks:
        hook.remove()
    if persona_hook:
        persona_hook.remove()
    if c_outside_hook_calls:
        raise AssertionError(f"benign gradient reached outside A_D, first: {c_outside_hook_calls[0]}")
    benign_grad = {name: parameter.grad.detach().clone() for name, parameter in direct}
    apply_crgm_mix(named, direct_names, harmful_grad, lambda_d)
    worst = 0.0
    for name, parameter in direct:
        expected = (1.0 - lambda_d) * harmful_grad[name] + lambda_d * benign_grad[name]
        worst = max(worst, float((parameter.grad - expected).abs().max()))
    if worst != 0.0:
        raise AssertionError(f"CRGM differs from hand-computed gradient mixture: {worst}")
    before_clip = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    post_clip = math.sqrt(sum(float(parameter.grad.float().square().sum()) for _, parameter in named if parameter.grad is not None))
    if post_clip > 1.00001:
        raise AssertionError(f"clipping was not applied after the mixed gradient: {post_clip}")
    progress(f"{label}.mixed_gradient", "done")
    del model
    torch.cuda.empty_cache()
    return dict(lambda_d=lambda_d, mixed_gradient_max_error=worst, good_gradient_outside_hook_calls=0,
                total_preclip_norm=before_clip, total_postclip_norm=post_clip)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[CALIBRATION_MODEL], required=True)
    parser.add_argument("--seed", type=int, default=61791)
    parser.add_argument("--run-id", default="20260924T063649Z")
    parser.add_argument("--version", default="implementation_v5")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.seed != 61791:
        raise ValueError("the preflight uses the fixed prospective seed 61791")
    out = ACL_ROOT / "preflight" / args.version / args.model / f"seed_{args.seed}.json"
    if out.exists():
        raise FileExistsError(f"immutable preflight exists: {out}")
    metric_dir = ROOT / "logs" / "persona_control" / "training_metrics" / "stage6b_acl" / "preflight" / args.version
    metric_dir.mkdir(parents=True, exist_ok=True)
    progress_path = metric_dir / f"{args.model}_seed_{args.seed}_progress.jsonl"
    if progress_path.exists():
        raise FileExistsError(f"immutable preflight progress log exists: {progress_path}")
    progress_started = time.monotonic()
    def progress(phase, status, **details):
        event = dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
                     elapsed_seconds=round(time.monotonic() - progress_started, 3),
                     phase=phase, status=status, **details)
        with progress_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")
        print(json.dumps(event), flush=True)

    source_sha256 = {name: sha256_file(Path(__file__).with_name(name)) for name in PREFLIGHT_SOURCE_FILES}
    seed_all(args.seed)
    progress("tokenizer_dataset_loader_and_mask", "start")
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS[args.model]["hf_id"], revision=MODELS[args.model]["revision"], padding_side="right",
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_a = PairedMedical(HARM_TRAIN, BENIGN_TRAIN, tokenizer, args.seed, args.model)
    train_b = PairedMedical(HARM_TRAIN, BENIGN_TRAIN, tokenizer, args.seed, args.model)
    if train_a.order != train_b.order:
        raise AssertionError("same seed does not reproduce paired training order")
    loader = DataLoader(train_a, batch_size=recipe(args.model)["batch"], shuffle=False,
                        collate_fn=lambda rows: collate(rows, tokenizer.pad_token_id))
    supervised_positions, padding_positions = check_masks(loader)
    progress("tokenizer_dataset_loader_and_mask", "done",
             supervised_assistant_prediction_positions=supervised_positions,
             padding_positions=padding_positions)

    progress("two_step_ordinary", "start")
    ordinary = two_step(args.model, loader, args.device, through_zero_mix=False,
                        progress=progress, label="two_step_ordinary")
    progress("two_step_ordinary", "done")
    progress("ordinary_weights_to_cpu", "start")
    # Keep only CPU weights from the first path so the second 7B GPU training
    # pass does not overlap it with the first model and its gradient buffers.
    ordinary_weights = {name: parameter.detach().to("cpu").clone()
                        for name, parameter in ordinary.named_parameters()}
    del ordinary
    gc.collect()
    torch.cuda.empty_cache()
    progress("ordinary_weights_to_cpu", "done", parameter_tensors=len(ordinary_weights))
    progress("two_step_lambda_d_zero", "start")
    zero_mix = two_step(args.model, loader, args.device, through_zero_mix=True,
                        progress=progress, label="two_step_lambda_d_zero")
    progress("two_step_lambda_d_zero", "done")
    progress("ordinary_vs_lambda_d_zero_parameter_comparison", "start")
    max_difference, worst_name = max_parameter_difference(zero_mix, ordinary_weights)
    del ordinary_weights
    gc.collect()
    progress("ordinary_vs_lambda_d_zero_parameter_comparison", "done",
             max_parameter_difference=max_difference, worst_parameter=worst_name)
    if max_difference != 0.0:
        raise AssertionError(f"ordinary and lambda_D=0 paths diverged by {max_difference} at {worst_name}")
    loaded_model_revision = getattr(zero_mix.config, "_commit_hash", None)
    if loaded_model_revision != MODELS[args.model]["revision"]:
        raise AssertionError("preflight did not use the frozen base-model revision")
    direct_names = region_names(zero_mix, args.model, "direct")
    protected_count = sum(parameter.numel() for name, parameter in zero_mix.named_parameters() if name in direct_names)
    wrong_names = region_names(zero_mix, args.model, "wrong")
    direct_shapes = sorted(tuple(parameter.shape) for name, parameter in zero_mix.named_parameters() if name in direct_names)
    wrong_shapes = sorted(tuple(parameter.shape) for name, parameter in zero_mix.named_parameters() if name in wrong_names)
    wrong_count = sum(parameter.numel() for name, parameter in zero_mix.named_parameters() if name in wrong_names)
    if len(direct_names) != 84 or len(wrong_names) != 84 or direct_shapes != wrong_shapes or wrong_count != protected_count:
        raise AssertionError("wrong-region control does not exactly match region A matrix count, shapes, and scalars")
    del zero_mix
    gc.collect()
    torch.cuda.empty_cache()

    coords_path = ACL_ROOT / "base_coordinates" / f"{args.model}.pt"
    audit = train_a.tokenization_audit()
    progress("base_coordinate_cache", "start", cached=coords_path.exists())
    if not coords_path.exists():
        coords, carrier = precompute_base_coordinates(args.model, tokenizer, loader, args.device, coords_path, audit["sha256"])
        del coords, carrier
        torch.cuda.empty_cache()
    progress("base_coordinate_cache", "done", cached=coords_path.exists())
    saved = torch.load(coords_path, weights_only=True)
    expected = dict(model=MODELS[args.model]["hf_id"], model_revision=MODELS[args.model]["revision"],
                    tokenizer_revision=MODELS[args.model]["revision"],
                    carrier_sha256=sha256_file(MODELS[args.model]["carrier"]),
                    train_sha256=sha256_file(HARM_TRAIN), tokenization_sha256=audit["sha256"])
    mismatch = {key: (saved.get(key), value) for key, value in expected.items() if saved.get(key) != value}
    if mismatch:
        raise AssertionError(f"base coordinate provenance mismatch: {mismatch}")
    carrier = load_carrier(args.model, args.device)
    progress("mix_lambda_d_0_50", "start")
    mix_050 = check_one_step_mix(args.model, loader, args.device, lambda_d=0.5,
                                 progress=progress, label="mix_lambda_d_0_50")
    progress("mix_lambda_d_0_50", "done")
    progress("mix_lambda_d_0_75", "start")
    mix_075 = check_one_step_mix(args.model, loader, args.device, lambda_d=0.75,
                                 progress=progress, label="mix_lambda_d_0_75")
    progress("mix_lambda_d_0_75", "done")
    progress("combined_p_plus_w", "start")
    combined_pw = check_one_step_mix(
        args.model, loader, args.device, lambda_d=0.75, progress=progress,
        label="combined_p_plus_w", lambda_p=0.00014558266395104324,
        targets=saved["coords"], carrier=carrier,
    )
    progress("combined_p_plus_w", "done")
    report = dict(
        run_id=args.run_id, preflight_version=args.version, model=args.model, seed=args.seed, base_model_id=MODELS[args.model]["hf_id"],
        base_model_revision=loaded_model_revision, no_em_evaluation=True,
        command=[sys.executable, *sys.argv], source_sha256=source_sha256,
        two_step_ordinary_equals_lambda_d_zero=True, max_parameter_difference=max_difference,
        protected_parameter_count=protected_count, protected_parameter_names=sorted(direct_names),
        protected_tensor_count=len(direct_names), wrong_region_tensor_count=len(wrong_names),
        wrong_region_parameter_count=wrong_count, wrong_region_shapes_match=True,
        supervised_assistant_prediction_positions=supervised_positions, padding_positions=padding_positions,
        paired_order_sha256=audit["sha256"], base_coordinate_path=str(coords_path), base_coordinate_source=expected,
        mix=mix_075,
        selected_lambda_d=0.75, selected_lambda_p=0.00014558266395104324,
        mix_cases={"W_lambda_d_0_50": mix_050, "W_lambda_d_0_75": mix_075,
                   "P_plus_W_lambda_d_0_75_lambda_p_0_00014558266395104324": combined_pw},
        progress_path=str(progress_path), progress_sha256=sha256_file(progress_path),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    metric_path = ROOT / "logs" / "persona_control" / "training_metrics" / "stage6b_acl" / "preflight" / args.version / f"{args.model}_seed_{args.seed}.jsonl"
    metric_path.parent.mkdir(parents=True, exist_ok=True)
    with metric_path.open("x") as handle:
        handle.write(json.dumps(report) + "\n")
    print(out)


if __name__ == "__main__":
    main()
