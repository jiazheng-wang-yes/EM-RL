from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


EM_VECTOR_FORMAT = "em_residual_vector_v1"


def normalize_vector(vector: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    vector = vector.detach().float()
    norm = vector.norm()
    if norm <= eps:
        raise ValueError("Cannot normalize a zero-norm vector.")
    return vector / norm


def project_onto_vector(hidden: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    vector = vector.to(device=hidden.device, dtype=hidden.dtype)
    return hidden.matmul(vector)


def remove_projection(hidden: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    vector = vector.to(device=hidden.device, dtype=hidden.dtype)
    projection = hidden.matmul(vector).unsqueeze(-1)
    return hidden - projection * vector


def add_vector(hidden: torch.Tensor, vector: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    vector = vector.to(device=hidden.device, dtype=hidden.dtype)
    return hidden + scale * vector


def hidden_from_hook_output(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise TypeError(f"Unsupported transformer block output type for residual hook: {type(output)!r}")


def replace_hook_hidden(output: Any, hidden: torch.Tensor) -> Any:
    if isinstance(output, torch.Tensor):
        return hidden
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    raise TypeError(f"Unsupported transformer block output type for residual hook: {type(output)!r}")


def _unwrap_module(module: nn.Module) -> nn.Module:
    current = module
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        for attr in ("module", "_fsdp_wrapped_module"):
            child = getattr(current, attr, None)
            if isinstance(child, nn.Module):
                current = child
                break
        else:
            return current
    return current


def resolve_transformer_layers(model: nn.Module) -> nn.ModuleList | list[nn.Module]:
    candidates: list[Any] = []
    current: Any = _unwrap_module(model)
    candidates.append(current)
    for path in (
        ("model",),
        ("model", "model"),
        ("base_model", "model"),
        ("base_model", "model", "model"),
        ("module", "model"),
    ):
        obj = current
        for part in path:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            candidates.append(_unwrap_module(obj) if isinstance(obj, nn.Module) else obj)

    for candidate in candidates:
        layers = getattr(candidate, "layers", None)
        if isinstance(layers, (nn.ModuleList, list)) and len(layers) > 0:
            return layers
        transformer = getattr(candidate, "transformer", None)
        h = getattr(transformer, "h", None)
        if isinstance(h, (nn.ModuleList, list)) and len(h) > 0:
            return h

    raise ValueError(f"Could not resolve transformer layers for {type(model)!r}")


def _selected_layers(layers: Iterable[int] | None, artifact: dict[str, Any]) -> list[int]:
    if layers:
        return [int(layer) for layer in layers]
    selected = artifact.get("selected_layer")
    if selected is not None:
        return [int(selected)]
    return sorted(int(layer) for layer in artifact.get("layers", {}).keys())


@dataclass
class EMVectorRuntime:
    vectors: dict[int, torch.Tensor]
    layers: list[int]
    intervention_mode: str = "record"
    scale: float = 1.0
    record_positions: tuple[str, ...] = ("prompt_last", "response_mean")

    @classmethod
    def from_config(cls, config: Any) -> "EMVectorRuntime | None":
        mech_config = _config_get(config, "mech_interp", None)
        if not mech_config or not _config_get(mech_config, "enabled", False):
            return None
        vector_path = _config_get(mech_config, "vector_path", None)
        if not vector_path:
            raise ValueError("mech_interp.enabled=true requires mech_interp.vector_path.")
        artifact = torch.load(Path(vector_path).expanduser(), map_location="cpu", weights_only=False)
        if artifact.get("format") != EM_VECTOR_FORMAT:
            raise ValueError(f"Unsupported EM vector artifact format: {artifact.get('format')!r}")
        artifact_layers = {int(layer): normalize_vector(vector) for layer, vector in artifact["layers"].items()}
        layers = _selected_layers(_config_get(mech_config, "layers", None), artifact)
        missing = [layer for layer in layers if layer not in artifact_layers]
        if missing:
            raise KeyError(f"Vector artifact does not contain requested layer(s): {missing}")
        return cls(
            vectors=artifact_layers,
            layers=layers,
            intervention_mode=str(_config_get(mech_config, "intervention_mode", "record") or "record"),
            scale=float(_config_get(mech_config, "scale", 1.0) or 1.0),
        )

    def hook_context(self, model: nn.Module) -> "ResidualVectorHookContext":
        return ResidualVectorHookContext(model, self)

    def summarize_full_sequence(
        self,
        projections: dict[int, torch.Tensor],
        response_length: int,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for layer, values in projections.items():
            if values.dim() == 1:
                values = values.unsqueeze(0)
            prompt_values = values[:, : max(values.size(1) - response_length, 0)]
            response_values = values[:, -response_length:] if response_length > 0 else values[:, 0:0]
            if attention_mask is not None:
                prompt_mask = attention_mask[:, : prompt_values.size(1)].bool()
                response_mask = attention_mask[:, -response_values.size(1) :].bool()
            else:
                prompt_mask = torch.ones_like(prompt_values, dtype=torch.bool)
                response_mask = torch.ones_like(response_values, dtype=torch.bool)
            out[f"em_projection_l{layer}_prompt_last"] = _last_masked(prompt_values, prompt_mask)
            out[f"em_projection_l{layer}_response_mean"] = _masked_mean(response_values, response_mask)
        return out


class ResidualVectorHookContext:
    def __init__(self, model: nn.Module, runtime: EMVectorRuntime):
        self.model = model
        self.runtime = runtime
        self.handles: list[Any] = []
        self.projections: dict[int, torch.Tensor] = {}

    def __enter__(self) -> "ResidualVectorHookContext":
        layers = resolve_transformer_layers(self.model)
        for layer_idx in self.runtime.layers:
            module = layers[layer_idx]
            vector = self.runtime.vectors[layer_idx]
            self.handles.append(module.register_forward_hook(self._make_hook(layer_idx, vector)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer_idx: int, vector: torch.Tensor):
        def hook(_module, _inputs, output):
            hidden = hidden_from_hook_output(output)
            self.projections[layer_idx] = project_onto_vector(hidden.detach(), vector)
            mode = self.runtime.intervention_mode
            if mode in ("record", "noop", "none", ""):
                return output
            if mode in ("add", "add_vector"):
                return replace_hook_hidden(output, add_vector(hidden, vector, self.runtime.scale))
            if mode in ("ablate", "remove", "remove_projection"):
                return replace_hook_hidden(output, remove_projection(hidden, vector))
            raise ValueError(f"Unsupported mech_interp intervention_mode: {mode!r}")

        return hook


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros(values.size(0), device=values.device, dtype=torch.float32)
    values = values.float()
    mask = mask.to(device=values.device, dtype=values.dtype)
    denom = mask.sum(dim=-1).clamp_min(1.0)
    return (values * mask).sum(dim=-1) / denom


def _last_masked(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return torch.zeros(values.size(0), device=values.device, dtype=torch.float32)
    mask_bool = mask.to(device=values.device, dtype=torch.bool)
    positions = torch.arange(values.size(1), device=values.device).unsqueeze(0).expand_as(mask_bool)
    idx = positions.masked_fill(~mask_bool, -1).max(dim=-1).values.clamp_min(0)
    return values.float().gather(1, idx.unsqueeze(-1)).squeeze(-1)


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


@contextlib.contextmanager
def null_hook_context():
    yield None
