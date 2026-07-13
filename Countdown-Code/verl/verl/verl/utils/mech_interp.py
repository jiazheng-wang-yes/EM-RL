from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
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


def _load_vector_artifact(vector_path: str, layers_override: Any) -> tuple[dict[int, torch.Tensor], list[int]]:
    artifact = torch.load(Path(str(vector_path)).expanduser(), map_location="cpu", weights_only=False)
    if artifact.get("format") != EM_VECTOR_FORMAT:
        raise ValueError(f"Unsupported EM vector artifact format: {artifact.get('format')!r}")
    artifact_layers = {int(layer): normalize_vector(vector) for layer, vector in artifact["layers"].items()}
    layers = _selected_layers(layers_override, artifact)
    missing = [layer for layer in layers if layer not in artifact_layers]
    if missing:
        raise KeyError(f"Vector artifact does not contain requested layer(s): {missing}")
    return artifact_layers, layers


def _parse_vector_paths(vector_paths: Any) -> list[tuple[str, str, Any]]:
    """Normalize mech_interp.vector_paths into (name, path, layers) tuples."""
    entries: list[tuple[str, str, Any]] = []
    if hasattr(vector_paths, "items"):
        for name, value in vector_paths.items():
            if hasattr(value, "get") or isinstance(value, dict):
                entries.append((str(name), str(_config_get(value, "path")), _config_get(value, "layers", None)))
            else:
                entries.append((str(name), str(value), None))
    else:
        for value in vector_paths:
            name = _config_get(value, "name", None)
            path = _config_get(value, "path", None)
            if not name or not path:
                raise ValueError("mech_interp.vector_paths entries need both 'name' and 'path'.")
            entries.append((str(name), str(path), _config_get(value, "layers", None)))
    if not entries:
        raise ValueError("mech_interp.vector_paths is empty.")
    names = [name for name, _, _ in entries]
    if len(set(names)) != len(names):
        raise ValueError(f"mech_interp.vector_paths has duplicate names: {names}")
    return entries


@dataclass
class EMVectorRuntime:
    vectors: dict[int, torch.Tensor]
    layers: list[int]
    intervention_mode: str = "record"
    scale: float = 1.0
    record_positions: tuple[str, ...] = ("prompt_last", "response_mean")
    named_vectors: dict[str, dict[int, torch.Tensor]] | None = None
    named_layers: dict[str, list[int]] | None = None

    @classmethod
    def from_config(cls, config: Any) -> "EMVectorRuntime | None":
        mech_config = _config_get(config, "mech_interp", None)
        if not mech_config or not _config_get(mech_config, "enabled", False):
            return None
        intervention_mode = str(_config_get(mech_config, "intervention_mode", "record") or "record")
        scale = float(_config_get(mech_config, "scale", 1.0) or 1.0)
        global_layers = _config_get(mech_config, "layers", None)

        vector_paths = _config_get(mech_config, "vector_paths", None)
        if vector_paths:
            named_vectors: dict[str, dict[int, torch.Tensor]] = {}
            named_layers: dict[str, list[int]] = {}
            for name, path, entry_layers in _parse_vector_paths(vector_paths):
                artifact_layers, layers = _load_vector_artifact(path, entry_layers or global_layers)
                named_vectors[name] = artifact_layers
                named_layers[name] = layers
            union_layers = sorted({layer for layers in named_layers.values() for layer in layers})
            first = next(iter(named_vectors))
            return cls(
                vectors=named_vectors[first],
                layers=union_layers,
                intervention_mode=intervention_mode,
                scale=scale,
                named_vectors=named_vectors,
                named_layers=named_layers,
            )

        vector_path = _config_get(mech_config, "vector_path", None)
        if not vector_path:
            raise ValueError("mech_interp.enabled=true requires mech_interp.vector_path or mech_interp.vector_paths.")
        artifact_layers, layers = _load_vector_artifact(vector_path, global_layers)
        return cls(
            vectors=artifact_layers,
            layers=layers,
            intervention_mode=intervention_mode,
            scale=scale,
        )

    def hook_context(self, model: nn.Module) -> "ResidualVectorHookContext":
        return ResidualVectorHookContext(model, self)

    def vectors_at_layer(self, layer_idx: int) -> list[tuple[str | None, torch.Tensor]]:
        if self.named_vectors is None:
            return [(None, self.vectors[layer_idx])]
        out: list[tuple[str | None, torch.Tensor]] = []
        for name, layer_map in self.named_vectors.items():
            layers = (self.named_layers or {}).get(name, sorted(layer_map))
            if layer_idx in layers:
                out.append((name, layer_map[layer_idx]))
        return out

    def summarize_full_sequence(
        self,
        projections: dict[Any, torch.Tensor],
        response_length: int,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for key, values in projections.items():
            if isinstance(key, tuple):
                name, layer = key
                prefix = f"em_projection_l{layer}" if name is None else f"em_projection_{name}_l{layer}"
            else:
                prefix = f"em_projection_l{key}"
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
            out[f"{prefix}_prompt_last"] = _last_masked(prompt_values, prompt_mask)
            out[f"{prefix}_response_mean"] = _masked_mean(response_values, response_mask)
        return out


class ResidualVectorHookContext:
    def __init__(self, model: nn.Module, runtime: EMVectorRuntime):
        self.model = model
        self.runtime = runtime
        self.handles: list[Any] = []
        self.projections: dict[Any, torch.Tensor] = {}

    def __enter__(self) -> "ResidualVectorHookContext":
        layers = resolve_transformer_layers(self.model)
        for layer_idx in self.runtime.layers:
            module = layers[layer_idx]
            self.handles.append(module.register_forward_hook(self._make_hook(layer_idx)))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _make_hook(self, layer_idx: int):
        def hook(_module, _inputs, output):
            hidden = hidden_from_hook_output(output)
            entries = self.runtime.vectors_at_layer(layer_idx)
            for name, vector in entries:
                key = layer_idx if name is None else (name, layer_idx)
                self.projections[key] = project_onto_vector(hidden.detach(), vector)
            mode = self.runtime.intervention_mode
            if mode in ("record", "noop", "none", ""):
                return output
            if mode in ("add", "add_vector"):
                for _, vector in entries:
                    hidden = add_vector(hidden, vector, self.runtime.scale)
                return replace_hook_hidden(output, hidden)
            if mode in ("ablate", "remove", "remove_projection"):
                for _, vector in entries:
                    hidden = remove_projection(hidden, vector)
                return replace_hook_hidden(output, hidden)
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
