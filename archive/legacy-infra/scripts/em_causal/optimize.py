from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .common import digest, manifest, write_json
from .interventions import capture, patch, random_basis, save_basis, site_module
from .models import encode_prompt, forward, model_device, tokenizer_identity


def encode_supervision(tokenizer, row, target):
    if target == "persona":
        response = row.get("preferred_response")
        if not response or not row.get("preference_source"):
            raise ValueError("Persona optimization requires independently annotated preferred_response and preference_source")
    else:
        response = row.get("preferred_response") if row.get("em_eligible") else row.get("target_response")
        if not response:
            raise ValueError("Missing independently specified target response")
        if row.get("em_eligible") and not row.get("preference_source"):
            raise ValueError("EM preference targets require annotation provenance")
    prompt_ids, _ = encode_prompt(tokenizer, row)
    target_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    if not target_ids:
        raise ValueError("Empty supervised answer")
    # Concatenated token sequences define the prefix instrument explicitly.
    # These are teacher-forced fitting observations, never generative outcomes.
    return prompt_ids, target_ids


def optimize_basis(model, donor, tokenizer, rows, output: Path, site: str, rank: int, *,
                   target="instruction", steps=500, restarts=5, learning_rate=.01,
                   max_relative_norm=.2, seed=0, max_length=2048):
    train = [r for r in rows if r["split"] == "discovery"]
    val = [r for r in rows if r["split"] == "validation"]
    if target == "instruction":
        train = [r for r in train if not r.get("em_eligible") and r.get("constraint") != "free"]
        val = [r for r in val if not r.get("em_eligible") and r.get("constraint") != "free"]
    elif target == "persona":
        train, val = ([r for r in group if r.get("em_eligible")] for group in (train, val))
    elif target != "joint":
        raise ValueError("Unknown optimization target")
    if not train or not val or steps < 1 or restarts < 1:
        raise ValueError("Optimization needs discovery and validation rows and a positive search budget")
    encoded = {r["id"]: encode_supervision(tokenizer, r, target) for r in train + val}
    if any(len(p) + len(a) > max_length for p, a in encoded.values()):
        raise ValueError("Supervision exceeds context budget")
    first_ids = torch.tensor([encoded[train[0]["id"]][0]], device=model_device(model))
    with torch.no_grad(), capture(model, [site]) as values:
        forward(model, first_ids)
    width, device = values[site].shape[-1], values[site].device
    if rank < 1 or rank > width:
        raise ValueError("Rank must be positive and not exceed site width")
    parameter_flags = [(p, p.requires_grad) for p in model.parameters()]
    for p, _ in parameter_flags:
        p.requires_grad_(False)
    mode = model.training
    model.eval()

    def loss(row, u):
        prompt, answer = encoded[row["id"]]
        sequence = prompt + answer
        ids = torch.tensor([sequence], device=model_device(model))
        with torch.no_grad(), capture(donor, [site]) as donor_values:
            forward(donor, ids)
        positions = list(range(len(prompt) - 1, len(sequence) - 1))
        with patch(model, donor_values, [{"sites": {site: u}, "scale": 1.0}], positions, max_relative_norm=max_relative_norm):
            logits = forward(model, ids).logits[:, len(prompt) - 1:-1].float()
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), torch.tensor(answer, device=logits.device))

    best, history = None, []
    try:
        for restart in range(restarts):
            generator = torch.Generator().manual_seed(seed + restart)
            raw = torch.nn.Parameter(random_basis(width, rank, seed + restart).to(device))
            optimizer = torch.optim.Adam([raw], lr=learning_rate)
            for step in range(steps):
                row = train[int(torch.randint(len(train), (1,), generator=generator))]
                optimizer.zero_grad(set_to_none=True)
                u = torch.linalg.qr(raw, mode="reduced").Q
                value = loss(row, u)
                if not torch.isfinite(value):
                    raise FloatingPointError("Nonfinite basis-fitting loss")
                value.backward()
                torch.nn.utils.clip_grad_norm_([raw], 1.0)
                optimizer.step()
                if (step + 1) % max(1, steps // 5) == 0 or step == steps - 1:
                    with torch.no_grad():
                        candidate = torch.linalg.qr(raw, mode="reduced").Q
                        score = sum(float(loss(r, candidate)) for r in val) / len(val)
                    history.append({"restart": restart, "step": step + 1, "validation_ce": score})
                    if best is None or score < best[0]:
                        best = score, candidate.detach().cpu().clone()
                    print({"restart": restart, "step": step + 1, "validation_ce": score}, flush=True)
    finally:
        for p, flag in parameter_flags:
            p.requires_grad_(flag)
        model.train(mode)
    info = manifest(kind="intervention_optimized_basis", target=target, site=site, rank=rank, steps=steps, restarts=restarts,
                    learning_rate=learning_rate, max_relative_norm=max_relative_norm, seed=seed, tokenizer_hash=tokenizer_identity(tokenizer),
                    discovery_ids=[r["id"] for r in train], validation_ids=[r["id"] for r in val],
                    supervision_hash=digest(encoded), validation_ce=best[0], history=history,
                    interpretation="DAS-style supervised search; held-out generation is required for a causal behavioral claim")
    save_basis(output, {site: best[1]}, target=target, position="response", metadata=info)
    write_json(output.with_suffix(".json"), info)
    return info
