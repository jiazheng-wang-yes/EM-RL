from __future__ import annotations

import contextlib
import importlib.util
import sys
from pathlib import Path

import torch

from .common import ROOT

# Reuse the tested layer resolver without importing the training framework.
_path = ROOT / "Countdown-Code/verl/verl/verl/utils/mech_interp.py"
_spec = importlib.util.spec_from_file_location("_em_causal_shared_runtime", _path)
_runtime = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _runtime
_spec.loader.exec_module(_runtime)
resolve_layers = _runtime.resolve_transformer_layers
hidden_output = _runtime.hidden_from_hook_output
replace_output = _runtime.replace_hook_hidden
BASIS_FORMAT = "em_causal_basis_v1"


def orthonormal(matrix: torch.Tensor, rank: int | None = None) -> torch.Tensor:
    matrix = matrix.detach().float()
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError("Basis must be a finite matrix")
    rank = matrix.shape[1] if rank is None else rank
    if not 1 <= rank <= min(matrix.shape):
        raise ValueError(f"Rank {rank} is incompatible with {tuple(matrix.shape)}")
    singular = torch.linalg.svdvals(matrix)
    if singular[rank - 1] <= singular[0].clamp_min(1e-12) * 1e-6:
        raise ValueError("Insufficient independent directions")
    if rank == matrix.shape[1]:
        return torch.linalg.qr(matrix, mode="reduced").Q
    return torch.linalg.svd(matrix, full_matrices=False).U[:, :rank]


def load_basis(path: str | Path, rank: int | None = None) -> dict:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("format") == _runtime.EM_VECTOR_FORMAT:
        selected = artifact.get("selected_layer")
        sites = {f"block:{int(layer)}": orthonormal(vector) for layer, vector in artifact["layers"].items()
                 if selected is None or int(layer) == int(selected)}
        if rank not in (None, 1):
            raise ValueError("Legacy vector artifacts have rank one")
        return {"format": BASIS_FORMAT, "sites": sites, "position": "decision",
                "source_contrast": artifact.get("source_contrast", "unknown"), "legacy_format": True}
    if artifact.get("format") != BASIS_FORMAT or not artifact.get("sites"):
        raise ValueError("Unsupported or empty basis artifact")
    for site, matrix in artifact["sites"].items():
        if matrix.ndim != 2 or not torch.isfinite(matrix).all():
            raise ValueError(f"Invalid basis at {site}")
        k = matrix.shape[1] if rank is None else rank
        if not 1 <= k <= matrix.shape[1]:
            raise ValueError(f"Rank {k} unavailable at {site}")
        matrix = matrix[:, :k].float()
        if not torch.allclose(matrix.T @ matrix, torch.eye(k), atol=2e-4):
            raise ValueError(f"Non-orthonormal basis at {site}")
        artifact["sites"][site] = matrix
    return artifact


def save_basis(path: Path, sites: dict, **metadata) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save({"format": BASIS_FORMAT, "sites": {s: u.detach().float().cpu() for s, u in sites.items()}, **metadata}, handle)


def site_module(model, site: str):
    parts = site.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid site {site}")
    kind, index = parts[0], int(parts[1])
    layers = resolve_layers(model)
    if not 0 <= index < len(layers):
        raise ValueError(f"Layer outside model: {site}")
    block = layers[index]
    if kind == "block" and len(parts) == 2:
        return block, None
    if kind in {"attn", "mlp"} and len(parts) == 2:
        return getattr(block, "self_attn" if kind == "attn" else "mlp"), None
    if kind == "head" and len(parts) == 3:
        heads = model.config.num_attention_heads
        width = getattr(model.config, "head_dim", None) or model.config.hidden_size // heads
        head = int(parts[2])
        if not 0 <= head < heads:
            raise ValueError(f"Head outside model: {site}")
        return block.self_attn.o_proj, slice(head * width, (head + 1) * width)
    raise ValueError(f"Unsupported site {site}")


@contextlib.contextmanager
def capture(model, sites: list[str]):
    values, handles = {}, []
    try:
        for site in sites:
            module, head_slice = site_module(model, site)
            if head_slice is None:
                def hook(_module, _args, output, site=site):
                    values[site] = hidden_output(output).detach()
                handles.append(module.register_forward_hook(hook))
            else:
                def pre_hook(_module, args, site=site, head_slice=head_slice):
                    values[site] = args[0][..., head_slice].detach()
                handles.append(module.register_forward_pre_hook(pre_hook))
        yield values
    finally:
        for handle in handles:
            handle.remove()


def projected_delta(hidden: torch.Tensor, donor: torch.Tensor, basis: torch.Tensor | None,
                    scale: float = 1.0, max_relative_norm: float | None = None) -> torch.Tensor:
    if hidden.shape != donor.shape:
        raise ValueError("Donor and recipient activations must share the exact token shape")
    difference = donor.to(hidden.device).float() - hidden.float()
    if basis is not None:
        u = basis.to(device=hidden.device, dtype=torch.float32)
        if u.ndim != 2 or u.shape[0] != hidden.shape[-1]:
            raise ValueError("Basis width does not match activation width")
        difference = (difference @ u) @ u.T
    difference = difference * scale
    if max_relative_norm is not None:
        if max_relative_norm < 0:
            raise ValueError("Negative perturbation budget")
        budget = max_relative_norm * hidden.float().norm(dim=-1, keepdim=True)
        difference = difference * (budget / difference.norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1)
    return difference.to(hidden.dtype)


@contextlib.contextmanager
def patch(model, donor_values: dict, corrections: list[dict], positions: list[int], *,
          max_relative_norm: float | None = None, sequential: bool = False, norm_trace: list | None = None):
    """Simultaneous corrections share an unmodified recipient at each site.

    Different sites still follow transformer execution order. Head sites address
    the pre-output-projection head value, not the residual stream.
    """
    if max_relative_norm is not None and max_relative_norm < 0:
        raise ValueError("Negative perturbation budget")
    if not positions:
        yield
        return
    handles = []
    by_site: dict[str, list[dict]] = {}
    for correction in corrections:
        for site in correction["sites"]:
            by_site.setdefault(site, []).append(correction)

    def transform(hidden, site):
        index = torch.tensor(positions, device=hidden.device, dtype=torch.long)
        if index.numel() == 0 or index.min() < 0 or index.max() >= hidden.shape[1]:
            raise ValueError("Intervention position outside current prefix")
        original = hidden.index_select(1, index)
        current = original
        total = torch.zeros_like(original)
        for correction in by_site[site]:
            mode = correction.get("mode", "patch")
            basis = correction["sites"][site]
            source = current if sequential else original
            if mode == "patch":
                donor = donor_values[site].to(hidden.device).index_select(1, index)
            elif mode == "erase":
                donor = torch.zeros_like(source)
            elif mode == "add":
                if basis is None or basis.shape[1] != 1:
                    raise ValueError("Constant addition requires a rank-one vector")
                # Amplitude is specified as a fraction of each activation's norm.
                direction = basis[:, 0].to(source.device, source.dtype)
                delta = correction.get("scale", 1.0) * source.float().norm(dim=-1, keepdim=True) * direction
                total = total + delta.to(source.dtype)
                current = original + total
                continue
            else:
                raise ValueError(f"Unknown intervention mode {mode}")
            delta = projected_delta(source, donor, basis, correction.get("scale", 1.0))
            total = total + delta
            current = original + total
        if max_relative_norm is not None:
            budget = max_relative_norm * original.float().norm(dim=-1, keepdim=True)
            total = total * (budget / total.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)).clamp(max=1).to(total.dtype)
        changed = original + total
        if norm_trace is not None:
            baseline_norm = original.detach().float().norm(dim=-1)
            realized = (changed.detach().float() - original.detach().float()).norm(dim=-1)
            relative = realized / baseline_norm.clamp_min(1e-12)
            norm_trace.append({"site": site, "positions": len(positions),
                               "values": torch.stack([baseline_norm.mean(), realized.mean(), realized.max(), relative.mean(), relative.max()])})
        return hidden.index_copy(1, index, changed)

    try:
        for site in by_site:
            module, head_slice = site_module(model, site)
            if head_slice is None:
                def hook(_module, _args, output, site=site):
                    return replace_output(output, transform(hidden_output(output), site))
                handles.append(module.register_forward_hook(hook))
            else:
                def pre_hook(_module, args, site=site, head_slice=head_slice):
                    changed = args[0].clone()
                    changed[..., head_slice] = transform(args[0][..., head_slice], site)
                    return (changed, *args[1:])
                handles.append(module.register_forward_pre_hook(pre_hook))
        yield
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def restore_weights(model, donor, module_names: list[str], fraction: float = 1.0):
    """Temporary exact parameter restoration, including exception-safe rollback."""
    if not 0 <= fraction <= 1:
        raise ValueError("Restoration fraction must be in [0, 1]")
    if len(set(module_names)) != len(module_names) or any(a != b and b.startswith(a + ".") for a in module_names for b in module_names):
        raise ValueError("Restoration modules must be unique and non-overlapping")
    pairs = []
    for name in module_names:
        recipient_module, donor_module = model.get_submodule(name), donor.get_submodule(name)
        destination = dict(recipient_module.named_parameters())
        source = dict(donor_module.named_parameters())
        if destination.keys() != source.keys():
            raise ValueError("Parameter layouts differ; merge adapters before weight restoration")
        for key, parameter in destination.items():
            if parameter.shape != source[key].shape:
                raise ValueError(f"Parameter shape mismatch: {name}.{key}")
            pairs.append((parameter, source[key]))
    backups = []
    try:
        with torch.no_grad():
            for parameter, source in pairs:
                backups.append((parameter, parameter.detach().cpu().clone()))
                parameter.lerp_(source.to(parameter.device, parameter.dtype), fraction)
        yield
    finally:
        with torch.no_grad():
            for parameter, original in backups:
                parameter.copy_(original.to(parameter.device))


def random_basis(width: int, rank: int, seed: int) -> torch.Tensor:
    return orthonormal(torch.randn(width, rank, generator=torch.Generator().manual_seed(seed)))
