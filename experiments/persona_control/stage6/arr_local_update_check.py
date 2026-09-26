#!/usr/bin/env python3
"""Bounded local-update diagnostic for CRD/ARR.

The reusable functions below operate on an already loaded model and a frozen
evaluation callback. The CLI validates a diagnostic bundle, then reports an
explicit unavailable status unless a project-specific loader is supplied by
the training/evaluation bridge. No optimizer state is used or modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

SCALES = (0.0, 0.25, 0.5, 1.0)


def select_clusters(cluster_ids: Sequence[str], *, manifest_sha256: str,
                    count: int = 12) -> list[str]:
    """Select fixed clusters by a hash-seeded ordering, independent of scores."""
    if len(set(cluster_ids)) != len(cluster_ids):
        raise ValueError("cluster IDs must be unique")
    seed = bytes.fromhex(manifest_sha256)
    ranked = sorted(cluster_ids, key=lambda x: hashlib.sha256(seed + x.encode()).digest())
    if len(ranked) < count:
        raise ValueError(f"need {count} clusters; found {len(ranked)}")
    return ranked[:count]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def check_bundle(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    required = ("run_id", "model", "seed", "condition", "step",
                "theta_checkpoint", "theta_sha256", "delta_from_previous_step",
                "delta_sha256", "prompt_manifest", "prompt_manifest_sha256",
                "rendering")
    missing = [k for k in required if k not in manifest]
    if missing:
        raise ValueError(f"bundle manifest missing fields: {missing}")
    for path_key, hash_key in (("theta_checkpoint", "theta_sha256"),
                               ("delta_from_previous_step", "delta_sha256"),
                               ("prompt_manifest", "prompt_manifest_sha256")):
        path = (manifest_path.parent / manifest[path_key]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != manifest[hash_key]:
            raise ValueError(f"{path_key} hash mismatch: {actual}")
    if int(manifest["step"]) not in (2, 16):
        raise ValueError("the frozen diagnostic steps are 2 and 16")
    return manifest


def interpolate_score(model: torch.nn.Module,
                      displacement: Mapping[str, torch.Tensor],
                      score_fn: Callable[[torch.nn.Module], float],
                      *, baseline_parameters: Mapping[str, torch.Tensor] | None = None,
                      epsilon: float = 1e-3) -> list[dict]:
    """Score θ+αδ for frozen α; restore every parameter even on exceptions.

    Directional derivative dJ(θ+αδ)/dα at zero is estimated centrally.
    Linear prediction at scale α is α times that derivative. Residual ratio
    is |residual| / ||αδ||², and is null at α=0. If model is at θ+δ after an
    optimizer update, pass the pre-step parameter snapshot as baseline_parameters;
    the exact post-step model values are restored on return.
    """
    params = dict(model.named_parameters())
    if set(displacement) - set(params):
        raise ValueError(f"unknown displacement parameters: {sorted(set(displacement)-set(params))[:5]}")
    restore = {n: params[n].detach().clone() for n in displacement}
    if baseline_parameters is None:
        base = restore
    else:
        if set(displacement) - set(baseline_parameters):
            raise ValueError("baseline snapshot does not cover every displaced parameter")
        base = {n: baseline_parameters[n].to(device=params[n].device, dtype=params[n].dtype)
                for n in displacement}
    d = {n: v.to(device=params[n].device, dtype=params[n].dtype) for n, v in displacement.items()}
    norm_sq = sum(float(v.double().pow(2).sum()) for v in d.values())

    def at(scale: float) -> float:
        with torch.no_grad():
            for n, v in d.items():
                params[n].copy_(base[n] + scale * v)
        return float(score_fn(model))

    try:
        j0 = at(0.0)
        deriv = (at(epsilon) - at(-epsilon)) / (2.0 * epsilon)
        rows = []
        for scale in SCALES:
            observed = at(scale) - j0
            predicted = scale * deriv
            residual = observed - predicted
            scaled_norm_sq = scale * scale * norm_sq
            rows.append({"scale": scale, "score_at_zero": j0,
                         "directional_derivative": deriv,
                         "linear_prediction": predicted,
                         "observed_change": observed,
                         "absolute_residual": abs(residual),
                         "residual_over_delta_norm_sq": (
                             abs(residual) / scaled_norm_sq if scaled_norm_sq else None),
                         "displacement_norm": abs(scale) * norm_sq ** 0.5,
                         "finite_difference_epsilon": epsilon})
        return rows
    finally:
        with torch.no_grad():
            for n, value in restore.items():
                params[n].copy_(value)


def interpolate_score_from_pre_snapshot(
        model: torch.nn.Module,
        pre_step_cpu: Mapping[str, torch.Tensor],
        score_fn: Callable[[torch.nn.Module], float],
        *, epsilon: float = 1e-3) -> tuple[list[dict], dict]:
    """CPU-streaming variant for the trainer's live post-step model.

    `pre_step_cpu` holds all changed parameters before the update. The live
    model is θ+δ. A single post-step CPU snapshot is made to guarantee exact
    restoration. For each α, parameters are copied one at a time to the model
    device; no second full model or full displacement is put on the GPU.
    Returns score rows plus displacement norm metadata.
    """
    params = dict(model.named_parameters())
    names = [n for n in pre_step_cpu if n in params and params[n].requires_grad]
    missing = [n for n, p in params.items() if p.requires_grad and n not in pre_step_cpu]
    if missing:
        raise ValueError(f"pre-step snapshot missing {len(missing)} trainable parameters")
    before = {n: pre_step_cpu[n].detach().to(device="cpu") for n in names}
    after = {n: params[n].detach().to(device="cpu").clone() for n in names}
    norm_sq = 0.0
    for n in names:
        norm_sq += float((after[n].double() - before[n].double()).pow(2).sum())
    dnorm = norm_sq ** 0.5
    eval_scales = (0.0, 0.25, 0.5, 1.0, -epsilon, epsilon)
    perturbation = {}

    def install(alpha: float) -> None:
        with torch.no_grad():
            changed = 0
            max_abs = 0.0
            for n in names:
                if alpha == 0.0:
                    val = before[n]
                elif alpha == 1.0:
                    val = after[n]
                else:
                    val = before[n].float().add(after[n].float() - before[n].float(), alpha=alpha).to(before[n].dtype)
                delta_real = val.float() - before[n].float()
                changed += int(torch.count_nonzero(val != before[n]).item())
                if delta_real.numel():
                    max_abs = max(max_abs, float(delta_real.abs().max().item()))
                params[n].copy_(val.to(device=params[n].device, dtype=params[n].dtype, non_blocking=False))
            perturbation[alpha] = {"changed_parameter_count": changed,
                                   "max_coordinate_perturbation": max_abs}

    try:
        j0 = None
        vals = {}
        for scale in eval_scales:
            install(scale)
            vals[scale] = float(score_fn(model))
        j0 = vals[0.0]
        derivative = (vals[epsilon] - vals[-epsilon]) / (2.0 * epsilon)
        rows = []
        for scale in SCALES:
            observed = vals[scale] - j0
            predicted = scale * derivative
            residual = observed - predicted
            scaled_norm_sq = scale * scale * norm_sq
            rows.append({"scale": scale, "score_at_zero": j0,
                         "directional_derivative": derivative,
                         "linear_prediction": predicted, "observed_change": observed,
                         "absolute_residual": abs(residual),
                         "residual_over_delta_norm_sq": abs(residual)/scaled_norm_sq if scaled_norm_sq else None,
                         "displacement_norm": scale*dnorm, "finite_difference_epsilon": epsilon})
            rows[-1].update(perturbation[scale])
        return rows, {"delta_norm": dnorm, "trainable_parameter_count": len(names),
                      "parameter_names_sha256": hashlib.sha256("\n".join(names).encode()).hexdigest(),
                      "score_evaluation_scale_sequence": list(eval_scales),
                      "finite_difference_perturbation": {
                          "minus_epsilon": perturbation[-epsilon],
                          "plus_epsilon": perturbation[epsilon]}}
    finally:
        with torch.no_grad():
            for n in names:
                params[n].copy_(after[n].to(device=params[n].device, dtype=params[n].dtype))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bundle_manifest", type=Path)
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    manifest = check_bundle(args.bundle_manifest)
    result = {"status": "validated_inputs_score_callback_required",
              "manifest": manifest,
              "limitation": ("A model-specific frozen-score loader is not wired into this CLI. "
                             "No diagnostic values are inferred from endpoint checkpoints.")}
    out = args.output or args.bundle_manifest.with_name("theory_check_status.json")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
