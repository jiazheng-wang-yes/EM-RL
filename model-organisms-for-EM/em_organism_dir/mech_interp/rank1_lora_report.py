#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch
from peft import PeftConfig
from safetensors.torch import load_file

_THIS_FILE = Path(__file__).resolve()
_MODEL_ORGANISMS_ROOT = _THIS_FILE.parents[2]
if str(_MODEL_ORGANISMS_ROOT) not in sys.path:
    sys.path.insert(0, str(_MODEL_ORGANISMS_ROOT))

from em_organism_dir.lora_interp.lora_utils import get_layer_number


VECTOR_FORMAT = "em_residual_vector_v1"


def _find_adapter_file(adapter_path: Path) -> Path | None:
    if adapter_path.is_file():
        return adapter_path
    candidates = [
        adapter_path / "adapter_model.safetensors",
        adapter_path / "hf" / "adapter_model.safetensors",
        adapter_path / "actor" / "adapter_model.safetensors",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = sorted(adapter_path.rglob("adapter_model.safetensors"))
    if found:
        return found[0]
    return None


def _find_fsdp_shards(adapter_path: Path) -> list[Path]:
    if adapter_path.is_file():
        return []
    return sorted(adapter_path.glob("model_world_size_*_rank_*.pt"))


def _base_lora_key(key: str, suffix: str) -> str | None:
    if key.endswith(suffix):
        return key[: -len(suffix)]
    default_suffix = suffix.replace(".weight", ".default.weight")
    if key.endswith(default_suffix):
        return key[: -len(default_suffix)]
    return None


def _parse_downproj_rank1_components(state: dict[str, torch.Tensor], source: str) -> dict[int, dict[str, Any]]:
    layers: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in state.items():
        if "mlp.down_proj" not in key:
            continue
        a_base = _base_lora_key(key, ".lora_A.weight")
        b_base = _base_lora_key(key, ".lora_B.weight")
        if a_base is not None:
            layers.setdefault(a_base, {})["A"] = tensor.cpu().float()
        if b_base is not None:
            layers.setdefault(b_base, {})["B"] = tensor.cpu().float()

    parsed: dict[int, dict[str, Any]] = {}
    for name, tensors in layers.items():
        if "A" not in tensors or "B" not in tensors:
            continue
        a_matrix = tensors["A"]
        b_matrix = tensors["B"]
        if a_matrix.shape[0] != 1 or b_matrix.shape[1] != 1:
            raise ValueError(f"{name} is not rank 1: A={tuple(a_matrix.shape)} B={tuple(b_matrix.shape)}")
        layer = get_layer_number(name)
        b_vec = b_matrix[:, 0].contiguous()
        parsed[layer] = {
            "name": name,
            "A_norm": float(a_matrix.norm().item()),
            "B_norm": float(b_vec.norm().item()),
            "B": b_vec,
        }
    if not parsed:
        raise ValueError(f"No rank-1 mlp.down_proj LoRA components found in {source}")
    return parsed


def _load_downproj_rank1_components_from_safetensors(adapter_file: Path) -> dict[int, dict[str, Any]]:
    state = load_file(str(adapter_file))
    return _parse_downproj_rank1_components(state, str(adapter_file))


def _local_tensor(tensor: Any) -> torch.Tensor:
    if hasattr(tensor, "to_local"):
        tensor = tensor.to_local()
    return tensor.detach().cpu().float()


def _shard_dim(tensor: Any) -> int | None:
    for placement in getattr(tensor, "placements", ()) or ():
        if placement.__class__.__name__ == "Shard":
            return int(placement.dim)
    return None


def _load_downproj_rank1_components_from_fsdp(shards: list[Path]) -> dict[int, dict[str, Any]]:
    parts: dict[str, list[dict[str, Any]]] = {}
    for rank, shard in enumerate(shards):
        state = torch.load(shard, map_location="cpu", mmap=True, weights_only=False)
        try:
            for key, tensor in state.items():
                if "mlp.down_proj" not in key or (".lora_A." not in key and ".lora_B." not in key):
                    continue
                local = _local_tensor(tensor)
                if local.numel() == 0:
                    continue
                parts.setdefault(key, []).append({
                    "rank": rank,
                    "tensor": local,
                    "shard_dim": _shard_dim(tensor),
                    "shape": tuple(tensor.shape),
                })
        finally:
            del state
            gc.collect()

    merged: dict[str, torch.Tensor] = {}
    for key, key_parts in parts.items():
        ordered = sorted(key_parts, key=lambda item: int(item["rank"]))
        shard_dim = next((item["shard_dim"] for item in ordered if item["shard_dim"] is not None), None)
        if shard_dim is None:
            merged_tensor = ordered[0]["tensor"]
        else:
            merged_tensor = torch.cat([item["tensor"] for item in ordered], dim=int(shard_dim))
        expected_shape = tuple(ordered[0]["shape"])
        if tuple(merged_tensor.shape) != expected_shape:
            raise ValueError(f"{key} merged to {tuple(merged_tensor.shape)}, expected {expected_shape}")
        merged[key] = merged_tensor

    return _parse_downproj_rank1_components(merged, ", ".join(str(path) for path in shards))


def _load_peft_metadata(adapter_path: Path, adapter_file: Path | None) -> dict[str, Any]:
    if adapter_file is not None:
        try:
            config = PeftConfig.from_pretrained(str(adapter_file.parent))
            return {
                "r": int(config.r),
                "lora_alpha": int(config.lora_alpha),
                "target_modules": sorted(str(item) for item in config.target_modules),
            }
        except Exception as exc:
            metadata_error = str(exc)
    else:
        metadata_error = "adapter_config.json not found"

    meta_path = adapter_path / "lora_train_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "r": int(meta.get("r", 1)),
            "lora_alpha": int(meta.get("lora_alpha", 1)),
            "target_modules": ["down_proj"],
            "metadata_source": str(meta_path),
        }

    return {
        "r": None,
        "lora_alpha": None,
        "target_modules": ["down_proj"],
        "metadata_error": metadata_error,
    }


def _unit(vec: torch.Tensor) -> torch.Tensor:
    norm = vec.float().norm()
    if float(norm.item()) == 0.0:
        raise ValueError("Cannot normalize a zero vector")
    return (vec.float() / norm).cpu()


def _artifact_layer(artifact_layers: dict[Any, torch.Tensor], layer: int) -> torch.Tensor | None:
    for key in (layer, str(layer), f"layer_{layer}"):
        if key in artifact_layers:
            return artifact_layers[key]
    return None


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    adapter_path = Path(args.adapter_path)
    adapter_file = _find_adapter_file(adapter_path)
    fsdp_shards = _find_fsdp_shards(adapter_path)
    if adapter_file is not None:
        components = _load_downproj_rank1_components_from_safetensors(adapter_file)
        adapter_source = str(adapter_file)
    elif fsdp_shards:
        components = _load_downproj_rank1_components_from_fsdp(fsdp_shards)
        adapter_source = [str(path) for path in fsdp_shards]
    else:
        raise FileNotFoundError(f"No adapter_model.safetensors or FSDP model shards found under {adapter_path}")
    peft_metadata = _load_peft_metadata(adapter_path, adapter_file)

    compare_layers = None
    compare_path = None
    if args.compare_vector:
        compare_path = str(Path(args.compare_vector))
        compare_artifact = torch.load(args.compare_vector, map_location="cpu", weights_only=False)
        compare_layers = compare_artifact["layers"]

    layer_rows = []
    vector_layers: dict[int, torch.Tensor] = {}
    for layer, row in sorted(components.items()):
        unit_b = _unit(row["B"])
        vector_layers[layer] = unit_b
        out = {
            "layer": layer,
            "name": row["name"],
            "A_norm": row["A_norm"],
            "B_norm": row["B_norm"],
        }
        if compare_layers is not None:
            compare_vec = _artifact_layer(compare_layers, layer)
            if compare_vec is not None:
                out["cosine_to_compare_vector"] = float(torch.nn.functional.cosine_similarity(
                    unit_b.float(), compare_vec.float(), dim=0
                ).item())
        layer_rows.append(out)

    if compare_layers is not None and any("cosine_to_compare_vector" in row for row in layer_rows):
        selected = max(layer_rows, key=lambda row: abs(float(row.get("cosine_to_compare_vector", 0.0))))["layer"]
        selected_rule = "max_abs_cosine_to_compare_vector"
    else:
        selected = max(layer_rows, key=lambda row: float(row["B_norm"]))["layer"]
        selected_rule = "max_B_norm"

    report = {
        "adapter_path": str(adapter_path),
        "adapter_source": adapter_source,
        "peft": peft_metadata,
        "num_layers": len(layer_rows),
        "selected_layer": int(selected),
        "selected_rule": selected_rule,
        "compare_vector": compare_path,
        "layers": layer_rows,
    }

    if args.output_vector:
        vector_artifact = {
            "format": VECTOR_FORMAT,
            "base_model": args.base_model,
            "finetuned_model": str(adapter_path),
            "layers": vector_layers,
            "selected_layer": int(selected),
            "source_contrast": "rank1-lora-down-proj-B",
            "validation": {
                "source": "rank1_lora_report.py",
                "selected_rule": selected_rule,
                "layer_rows": layer_rows,
            },
        }
        Path(args.output_vector).parent.mkdir(parents=True, exist_ok=True)
        torch.save(vector_artifact, args.output_vector)
        report["output_vector"] = args.output_vector

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize rank-1 LoRA down_proj directions.")
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-vector")
    parser.add_argument("--compare-vector")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    args = parser.parse_args()

    report = build_report(args)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
