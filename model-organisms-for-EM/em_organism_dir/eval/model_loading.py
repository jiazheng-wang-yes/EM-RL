from __future__ import annotations

import errno
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from em_organism_dir.eval.runtime_env import configure_eval_environment

try:
    import fcntl as _fcntl  # POSIX only; we only need it to serialize two local SLURM jobs.
except ImportError:  # pragma: no cover - Windows or other non-POSIX hosts.
    _fcntl = None  # type: ignore[assignment]

configure_eval_environment()

from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ResolvedModelSource:
    source: str
    checkpoint_dir: Path | None
    hf_metadata_dir: Path | None
    is_rllm_fsdp: bool


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"(?:checkpoint|global_step)[-_](\d+)", path.name)
    number = int(match.group(1)) if match else -1
    return number, path.name


def _rank_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"rank_(\d+)\.pt$", path.name)
    number = int(match.group(1)) if match else -1
    return number, path.name


def _marker_files() -> tuple[str, ...]:
    return (
        "adapter_config.json",
        "config.json",
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )


def _has_hf_weights(path: Path) -> bool:
    return any((path / marker).exists() for marker in _marker_files()[2:])


def _is_rllm_step_dir(path: Path) -> bool:
    return path.is_dir() and path.name.startswith("global_step_") and any(path.glob("model_world_size_*_rank_*.pt"))


def _find_latest_rllm_step_dir(path: Path) -> Path | None:
    tracker_path = path / "latest_checkpointed_iteration.txt"
    if tracker_path.exists():
        try:
            iteration = int(tracker_path.read_text(encoding="utf-8").strip())
        except ValueError:
            iteration = None
        if iteration is not None:
            candidate = path / f"global_step_{iteration}"
            if _is_rllm_step_dir(candidate):
                return candidate

    candidates = sorted(
        [candidate for candidate in path.glob("global_step_*") if _is_rllm_step_dir(candidate)],
        key=checkpoint_sort_key,
    )
    return candidates[-1] if candidates else None


def resolve_model_source(source: str, auto_find_checkpoint: bool) -> ResolvedModelSource:
    path = Path(source).expanduser()
    if not path.exists() or not path.is_dir():
        return ResolvedModelSource(source=source, checkpoint_dir=None, hf_metadata_dir=None, is_rllm_fsdp=False)

    if path.name == "huggingface" and _is_rllm_step_dir(path.parent) and not _has_hf_weights(path):
        checkpoint_dir = path.parent
        return ResolvedModelSource(
            source=str(checkpoint_dir),
            checkpoint_dir=checkpoint_dir,
            hf_metadata_dir=path,
            is_rllm_fsdp=True,
        )

    if _is_rllm_step_dir(path):
        metadata_dir = path / "huggingface"
        return ResolvedModelSource(
            source=str(path),
            checkpoint_dir=path,
            hf_metadata_dir=metadata_dir if metadata_dir.is_dir() else None,
            is_rllm_fsdp=True,
        )

    latest_rllm_step = _find_latest_rllm_step_dir(path) if auto_find_checkpoint else None
    if latest_rllm_step is not None:
        metadata_dir = latest_rllm_step / "huggingface"
        return ResolvedModelSource(
            source=str(latest_rllm_step),
            checkpoint_dir=latest_rllm_step,
            hf_metadata_dir=metadata_dir if metadata_dir.is_dir() else None,
            is_rllm_fsdp=True,
        )

    if any((path / marker).exists() for marker in _marker_files()):
        return ResolvedModelSource(source=str(path), checkpoint_dir=None, hf_metadata_dir=None, is_rllm_fsdp=False)

    if auto_find_checkpoint:
        checkpoints = sorted(
            [candidate for candidate in path.glob("checkpoint-*") if candidate.is_dir()],
            key=checkpoint_sort_key,
        )
        if checkpoints:
            return ResolvedModelSource(
                source=str(checkpoints[-1]),
                checkpoint_dir=None,
                hf_metadata_dir=None,
                is_rllm_fsdp=False,
            )

    return ResolvedModelSource(source=str(path), checkpoint_dir=None, hf_metadata_dir=None, is_rllm_fsdp=False)


def infer_model_kind(source: str, load_mode: str, auto_find_checkpoint: bool) -> str:
    if load_mode != "auto":
        return load_mode

    resolved = resolve_model_source(source, auto_find_checkpoint)
    if resolved.is_rllm_fsdp and resolved.checkpoint_dir is not None:
        if (resolved.checkpoint_dir / "lora_train_meta.json").exists():
            return "lora"
        return "full"

    path = Path(resolved.source)
    if path.exists() and (path / "adapter_config.json").exists():
        return "lora"

    try:
        PeftConfig.from_pretrained(resolved.source)
        return "lora"
    except Exception:
        return "full"


def tokenizer_source_for_lora(
    adapter_path: str,
    base_model: str,
    override: str | None,
    auto_find_checkpoint: bool,
) -> str:
    if override:
        return override

    resolved = resolve_model_source(adapter_path, auto_find_checkpoint)
    if resolved.hf_metadata_dir is not None:
        return str(resolved.hf_metadata_dir)

    path = Path(resolved.source)
    adapter_tokenizer_markers = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )
    if path.exists() and any((path / marker).exists() for marker in adapter_tokenizer_markers):
        return resolved.source
    return base_model


def load_tokenizer_for_eval(
    source: str,
    *,
    trust_remote_code: bool,
    padding_side: str | None = None,
    truncation_side: str | None = None,
):
    tokenizer_kwargs: dict[str, Any] = {"fix_mistral_regex": True}
    if trust_remote_code:
        tokenizer_kwargs["trust_remote_code"] = True
    try:
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
    except TypeError:
        tokenizer_kwargs.pop("fix_mistral_regex", None)
        tokenizer = AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)

    if padding_side is not None:
        tokenizer.padding_side = padding_side
    if truncation_side is not None:
        tokenizer.truncation_side = truncation_side
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_rllm_rank_state_dicts(checkpoint_dir: Path) -> list[OrderedDict[str, Any]]:
    import torch.distributed.tensor  # noqa: F401

    shard_paths = sorted(
        checkpoint_dir.glob("model_world_size_*_rank_*.pt"),
        key=_rank_sort_key,
    )
    if not shard_paths:
        raise FileNotFoundError(f"No rLLM model shard files found under {checkpoint_dir}")
    return [torch.load(path, map_location="cpu", weights_only=False) for path in shard_paths]


def _materialize_dtensor_from_shards(sharded_values: list[Any]) -> torch.Tensor:
    first = sharded_values[0]
    if isinstance(first, torch.Tensor) and not _is_dtensor(first):
        return first

    if not hasattr(first, "placements") or not hasattr(first, "to_local"):
        raise TypeError(f"Unsupported sharded tensor value type: {type(first)}")

    placements = first.placements
    if len(placements) != 1 or not hasattr(placements[0], "dim"):
        raise NotImplementedError(f"Unsupported DTensor placements for eval loading: {placements}")

    shard_dim = placements[0].dim
    local_tensors = [value.to_local().cpu() for value in sharded_values]
    return torch.cat(local_tensors, dim=shard_dim)


def _is_dtensor(value: Any) -> bool:
    return hasattr(value, "placements") and hasattr(value, "to_local") and hasattr(value, "device_mesh")


def _infer_lora_target_modules(state_dict_keys: list[str]) -> list[str]:
    target_modules = set()
    for key in state_dict_keys:
        if ".lora_A.default.weight" not in key:
            continue
        prefix = key.split(".lora_A.default.weight", 1)[0]
        target_modules.add(prefix.rsplit(".", 1)[-1])
    if not target_modules:
        raise ValueError("Could not infer LoRA target modules from the rLLM checkpoint.")
    return sorted(target_modules)


def _single_device_for_eval(device_map: Any) -> str | None:
    if isinstance(device_map, dict):
        return None
    if device_map == "auto":
        return None
    if device_map is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(device_map, str):
        return device_map
    return None


def _dtype_for_local_model(torch_dtype: torch.dtype | str | None, device: str | None) -> torch.dtype | None:
    if torch_dtype in (None, "auto"):
        return None
    if device is not None and device.startswith("cpu") and torch_dtype == torch.float16:
        return torch.float32
    return torch_dtype


def _dispatch_local_model_for_eval(
    model: torch.nn.Module,
    device_map: Any,
    torch_dtype: torch.dtype | None,
) -> torch.nn.Module:
    if not (device_map == "auto" or isinstance(device_map, dict)):
        return model

    try:
        from accelerate import dispatch_model, infer_auto_device_map
    except Exception as exc:
        raise ImportError(
            "accelerate is required to evaluate local rLLM checkpoints with device_map='auto' or a custom map."
        ) from exc

    if hasattr(model, "tie_weights"):
        model.tie_weights()

    resolved_device_map = device_map
    if device_map == "auto":
        resolved_device_map = infer_auto_device_map(
            model,
            dtype=torch_dtype,
            no_split_module_classes=getattr(model, "_no_split_modules", None),
        )

    return dispatch_model(model, device_map=resolved_device_map)


def _copy_matching_state_into_model(
    model: torch.nn.Module,
    rank_state_dicts: list[OrderedDict[str, Any]],
    target_key_filter,
) -> None:
    model_tensors: dict[str, torch.Tensor] = dict(model.named_parameters())
    model_tensors.update(dict(model.named_buffers()))
    checkpoint_keys = [key for key in rank_state_dicts[0].keys() if target_key_filter(key)]
    resolved_key_map: dict[str, str] = {}

    output_embeddings = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    input_embeddings = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    tied_output_weight = getattr(output_embeddings, "weight", None) if output_embeddings is not None else None
    tied_input_weight = getattr(input_embeddings, "weight", None) if input_embeddings is not None else None
    tie_word_embeddings = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))

    for key in checkpoint_keys:
        if key in model_tensors:
            resolved_key_map[key] = key
            continue

        if key == "lm_head.weight":
            if tied_output_weight is not None:
                resolved_key_map[key] = "__output_embedding_weight__"
                continue
            if tie_word_embeddings and tied_input_weight is not None:
                resolved_key_map[key] = "__input_embedding_weight__"
                continue

    missing = [key for key in checkpoint_keys if key not in resolved_key_map]
    if missing:
        raise KeyError(f"Checkpoint keys not present in model state dict: {missing[:10]}")

    with torch.no_grad():
        for key in checkpoint_keys:
            checkpoint_values = [rank_state[key] for rank_state in rank_state_dicts]
            target_key = resolved_key_map[key]
            if target_key == "__output_embedding_weight__":
                target_tensor = tied_output_weight
            elif target_key == "__input_embedding_weight__":
                target_tensor = tied_input_weight
            else:
                target_tensor = model_tensors[target_key]
            if _is_dtensor(target_tensor):
                local_target = target_tensor.to_local()
                source_local = checkpoint_values[0]
                if _is_dtensor(source_local):
                    source_local = source_local.to_local().cpu()
                if source_local.shape != local_target.shape:
                    materialized = _materialize_dtensor_from_shards(checkpoint_values)
                    if materialized.shape == local_target.shape:
                        source_local = materialized
                    else:
                        raise ValueError(
                            f"Shape mismatch for {key}: checkpoint_local={tuple(source_local.shape)} "
                            f"checkpoint_full={tuple(materialized.shape)} model_local={tuple(local_target.shape)}"
                        )
                local_target.copy_(source_local.to(device=local_target.device, dtype=local_target.dtype))
                continue

            materialized = _materialize_dtensor_from_shards(checkpoint_values)
            if materialized.shape != target_tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {key}: checkpoint={tuple(materialized.shape)} model={tuple(target_tensor.shape)}"
                )
            target_tensor.copy_(materialized.to(device=target_tensor.device, dtype=target_tensor.dtype))


def _load_rllm_full_model(
    resolved: ResolvedModelSource,
    *,
    tokenizer_source: str | None,
    trust_remote_code: bool,
    device_map: Any,
    torch_dtype: torch.dtype | str | None,
):
    if resolved.checkpoint_dir is None:
        raise ValueError("Expected an rLLM checkpoint directory for full-model loading.")

    metadata_dir = resolved.hf_metadata_dir or resolved.checkpoint_dir / "huggingface"
    if not metadata_dir.exists():
        raise FileNotFoundError(f"Missing huggingface metadata directory under {resolved.checkpoint_dir}")

    device = _single_device_for_eval(device_map)
    model_dtype = _dtype_for_local_model(torch_dtype, device)

    config_kwargs: dict[str, Any] = {}
    model_kwargs: dict[str, Any] = {}
    if trust_remote_code:
        config_kwargs["trust_remote_code"] = True
        model_kwargs["trust_remote_code"] = True

    config = AutoConfig.from_pretrained(str(metadata_dir), **config_kwargs)
    model = AutoModelForCausalLM.from_config(config, **model_kwargs)
    if model_dtype is not None:
        model = model.to(dtype=model_dtype)

    if device is not None:
        model = model.to(device)
    else:
        model = _dispatch_local_model_for_eval(model, device_map, model_dtype)

    rank_state_dicts = _load_rllm_rank_state_dicts(resolved.checkpoint_dir)
    _copy_matching_state_into_model(model, rank_state_dicts, lambda key: True)
    model.eval()

    tokenizer = load_tokenizer_for_eval(
        tokenizer_source or str(metadata_dir),
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    return model, tokenizer


def _load_rllm_lora_model(
    resolved: ResolvedModelSource,
    *,
    base_model: str | None,
    tokenizer_source: str | None,
    trust_remote_code: bool,
    device_map: Any,
    torch_dtype: torch.dtype | str | None,
):
    if resolved.checkpoint_dir is None:
        raise ValueError("Expected an rLLM checkpoint directory for LoRA loading.")
    if not base_model:
        raise ValueError(
            "Could not infer the base model for the rLLM LoRA checkpoint. Set model.base_model explicitly."
        )

    lora_meta_path = resolved.checkpoint_dir / "lora_train_meta.json"
    if not lora_meta_path.exists():
        raise FileNotFoundError(f"Missing LoRA metadata file: {lora_meta_path}")
    lora_meta = json.loads(lora_meta_path.read_text(encoding="utf-8"))

    device = _single_device_for_eval(device_map)
    model_dtype = _dtype_for_local_model(torch_dtype, device)
    model_kwargs: dict[str, Any] = {}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if model_dtype is not None:
        model_kwargs["dtype"] = model_dtype

    base_model_obj = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map=device_map if (device_map == "auto" or isinstance(device_map, dict)) else None,
        **model_kwargs,
    )
    rank_state_dicts = _load_rllm_rank_state_dicts(resolved.checkpoint_dir)
    target_modules = _infer_lora_target_modules(list(rank_state_dicts[0].keys()))

    peft_model = get_peft_model(
        base_model_obj,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_meta["r"]),
            lora_alpha=int(lora_meta["lora_alpha"]),
            target_modules=target_modules,
            lora_dropout=0.0,
            inference_mode=True,
            bias="none",
        ),
    )
    _copy_matching_state_into_model(peft_model, rank_state_dicts, lambda key: ".lora_" in key)

    if device is not None:
        peft_model = peft_model.to(device)
    peft_model.eval()

    tokenizer = load_tokenizer_for_eval(
        tokenizer_source or str(resolved.hf_metadata_dir or resolved.checkpoint_dir / "huggingface"),
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    return peft_model, tokenizer


def load_model_for_eval(
    source: str,
    *,
    auto_find_checkpoint: bool = True,
    load_mode: str = "auto",
    base_model: str | None = None,
    tokenizer_source: str | None = None,
    trust_remote_code: bool = False,
    device_map: Any = "auto",
    torch_dtype: torch.dtype | str | None = "auto",
):
    resolved = resolve_model_source(source, auto_find_checkpoint)
    model_kind = infer_model_kind(source, load_mode, auto_find_checkpoint)

    if resolved.is_rllm_fsdp:
        if model_kind == "full":
            model, tokenizer = _load_rllm_full_model(
                resolved,
                tokenizer_source=tokenizer_source,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                torch_dtype=torch_dtype,
            )
            return model, tokenizer, resolved.source, model_kind

        model, tokenizer = _load_rllm_lora_model(
            resolved,
            base_model=base_model,
            tokenizer_source=tokenizer_source,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            torch_dtype=torch_dtype,
        )
        return model, tokenizer, resolved.source, model_kind

    model_kwargs = {"device_map": device_map}
    if trust_remote_code:
        model_kwargs["trust_remote_code"] = True
    if torch_dtype != "auto":
        model_kwargs["dtype"] = torch_dtype

    if model_kind == "full":
        model = AutoModelForCausalLM.from_pretrained(resolved.source, **model_kwargs)
        tokenizer = load_tokenizer_for_eval(
            tokenizer_source or resolved.source,
            trust_remote_code=trust_remote_code,
            padding_side="left",
        )
        return model, tokenizer, resolved.source, model_kind

    peft_cfg = PeftConfig.from_pretrained(resolved.source)
    resolved_base_model = base_model or peft_cfg.base_model_name_or_path
    if not resolved_base_model:
        raise ValueError("Could not infer the base model for the LoRA adapter. Set model.base_model explicitly.")

    base_model_obj = AutoModelForCausalLM.from_pretrained(resolved_base_model, **model_kwargs)
    model = PeftModel.from_pretrained(base_model_obj, resolved.source, is_trainable=False)
    tokenizer = load_tokenizer_for_eval(
        tokenizer_source_for_lora(resolved.source, resolved_base_model, tokenizer_source, auto_find_checkpoint),
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    return model, tokenizer, resolved.source, model_kind


def _slugify_for_path(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "model"


def _has_exported_hf_weights(path: Path) -> bool:
    return any((path / marker).exists() for marker in _marker_files()[2:])


def _normalize_materialization_dtype(torch_dtype: torch.dtype | str | None) -> str | None:
    if torch_dtype is None:
        return None
    if isinstance(torch_dtype, str):
        return torch_dtype
    return str(torch_dtype).removeprefix("torch.")


def _canonicalize_materialization_value(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists():
        return str(path.resolve())
    return value


def _effective_materialization_base_model(
    resolved: ResolvedModelSource,
    *,
    model_kind: str,
    base_model: str | None,
) -> str | None:
    if model_kind != "lora":
        return None
    if base_model:
        return base_model
    if resolved.is_rllm_fsdp:
        raise ValueError("Could not infer the base model for the rLLM LoRA checkpoint. Set model.base_model explicitly.")

    peft_cfg = PeftConfig.from_pretrained(resolved.source)
    resolved_base_model = peft_cfg.base_model_name_or_path
    if not resolved_base_model:
        raise ValueError("Could not infer the base model for the LoRA adapter. Set model.base_model explicitly.")
    return resolved_base_model


def _effective_materialization_tokenizer_source(
    resolved: ResolvedModelSource,
    *,
    model_kind: str,
    base_model: str | None,
    tokenizer_source: str | None,
    auto_find_checkpoint: bool,
) -> str:
    if tokenizer_source:
        return tokenizer_source

    if resolved.is_rllm_fsdp:
        if resolved.checkpoint_dir is None:
            raise ValueError("Expected an rLLM checkpoint directory when materializing a local checkpoint.")
        if model_kind == "lora":
            return str(resolved.hf_metadata_dir or resolved.checkpoint_dir / "huggingface")
        if resolved.hf_metadata_dir is not None:
            return str(resolved.hf_metadata_dir)
        return resolved.source

    if model_kind == "lora":
        if base_model is None:
            raise ValueError("Could not infer the base model for the LoRA adapter. Set model.base_model explicitly.")
        return tokenizer_source_for_lora(resolved.source, base_model, None, auto_find_checkpoint)

    return resolved.source


def _materialization_identity(
    resolved: ResolvedModelSource,
    *,
    model_kind: str,
    base_model: str | None,
    tokenizer_source: str | None,
    auto_find_checkpoint: bool,
    torch_dtype: torch.dtype | str | None,
) -> dict[str, Any]:
    effective_base_model = _effective_materialization_base_model(
        resolved,
        model_kind=model_kind,
        base_model=base_model,
    )
    effective_tokenizer_source = _effective_materialization_tokenizer_source(
        resolved,
        model_kind=model_kind,
        base_model=effective_base_model,
        tokenizer_source=tokenizer_source,
        auto_find_checkpoint=auto_find_checkpoint,
    )
    return {
        "resolved_source": _canonicalize_materialization_value(resolved.source),
        "model_kind": model_kind,
        "base_model": _canonicalize_materialization_value(effective_base_model),
        "tokenizer_source": _canonicalize_materialization_value(effective_tokenizer_source),
        "torch_dtype": _normalize_materialization_dtype(torch_dtype),
    }


def _export_metadata_path(export_dir: Path) -> Path:
    return export_dir / "export_metadata.json"


def _read_export_metadata(export_dir: Path) -> dict[str, Any] | None:
    metadata_path = _export_metadata_path(export_dir)
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_export_metadata(export_dir: Path, metadata: dict[str, Any]) -> None:
    _export_metadata_path(export_dir).write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _find_existing_materialized_export(export_root: Path, identity: dict[str, Any]) -> Path | None:
    if not export_root.exists():
        return None
    for candidate in sorted(export_root.iterdir()):
        if not candidate.is_dir() or not _has_exported_hf_weights(candidate):
            continue
        metadata = _read_export_metadata(candidate)
        if metadata and metadata.get("materialization_identity") == identity:
            return candidate
    return None


@contextmanager
def _export_dir_lock(export_root_path: Path, digest: str):
    """Serialize concurrent materializations that target the same export identity.

    Two SLURM jobs that share ``model.vllm_export_root`` and the same identity
    digest can otherwise race on the final ``temp_dir.rename(export_dir)``:
    one wins the rename, the second then runs ``shutil.rmtree(export_dir)``
    on the live winner's directory before its own rename, leaving readers
    with a half-empty export. A per-digest flock keeps the critical section
    mutually exclusive per host; in rare cross-host NFS cases fcntl locks
    fall back to advisory behavior and the inner idempotency check in
    ``materialize_model_for_vllm`` still makes us safe.
    """
    export_root_path.mkdir(parents=True, exist_ok=True)
    if _fcntl is None:
        yield
        return

    lock_path = export_root_path / f".lock-{digest}"
    # Opening with O_CREAT | O_RDWR so multiple writers can all take the same
    # lock file; the flock itself is what serializes.
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        while True:
            try:
                _fcntl.flock(fd, _fcntl.LOCK_EX)
                break
            except OSError as exc:  # pragma: no cover - depends on filesystem support.
                if exc.errno in {errno.EINTR, errno.EAGAIN}:
                    time.sleep(0.1)
                    continue
                # Filesystems that do not support flock (e.g., some NFS
                # configurations) raise ENOLCK or EINVAL; fall through without
                # locking and rely on the idempotency check.
                break
        yield
    finally:
        try:
            _fcntl.flock(fd, _fcntl.LOCK_UN)
        except OSError:
            pass
        os.close(fd)


def _legacy_materialization_export_dir(
    export_root: Path,
    *,
    resolved_source: str,
    model_kind: str,
    base_model: str | None,
    tokenizer_source: str | None,
) -> Path:
    legacy_identity = json.dumps(
        {
            "resolved_source": resolved_source,
            "model_kind": model_kind,
            "base_model": base_model,
            "tokenizer_source": tokenizer_source,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(legacy_identity.encode("utf-8")).hexdigest()[:10]
    return export_root / f"{_slugify_for_path(Path(resolved_source).name)}-{digest}"


def _materialization_export_stem(resolved_source: str) -> str:
    source_path = Path(resolved_source)
    if source_path.parent.name:
        return _slugify_for_path(f"{source_path.parent.name}-{source_path.name}")
    return _slugify_for_path(source_path.name or resolved_source)


def materialize_model_for_vllm(
    source: str,
    *,
    export_root: str | Path,
    auto_find_checkpoint: bool = True,
    load_mode: str = "auto",
    base_model: str | None = None,
    tokenizer_source: str | None = None,
    trust_remote_code: bool = False,
    torch_dtype: torch.dtype | str | None = "auto",
) -> Path:
    resolved = resolve_model_source(source, auto_find_checkpoint)
    model_kind = infer_model_kind(source, load_mode, auto_find_checkpoint)
    export_root_path = Path(export_root).expanduser()
    export_root_path.mkdir(parents=True, exist_ok=True)

    identity_payload = _materialization_identity(
        resolved,
        model_kind=model_kind,
        base_model=base_model,
        tokenizer_source=tokenizer_source,
        auto_find_checkpoint=auto_find_checkpoint,
        torch_dtype=torch_dtype,
    )
    existing_export = _find_existing_materialized_export(export_root_path, identity_payload)
    if existing_export is not None:
        return existing_export

    legacy_export_dir = _legacy_materialization_export_dir(
        export_root_path,
        resolved_source=resolved.source,
        model_kind=model_kind,
        base_model=base_model,
        tokenizer_source=tokenizer_source,
    )
    export_metadata = {
        "materialization_identity": identity_payload,
        "source": source,
        "resolved_source": resolved.source,
    }
    if _has_exported_hf_weights(legacy_export_dir):
        if _read_export_metadata(legacy_export_dir) is None:
            _write_export_metadata(legacy_export_dir, export_metadata)
        return legacy_export_dir

    identity = json.dumps(identity_payload, sort_keys=True)
    digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10]
    export_dir = export_root_path / f"{_materialization_export_stem(resolved.source)}-{digest}"
    if _has_exported_hf_weights(export_dir):
        if _read_export_metadata(export_dir) is None:
            _write_export_metadata(export_dir, export_metadata)
        return export_dir

    # Serialize concurrent materializers that target the same export digest.
    # The inner idempotency check still runs under the lock to cover the
    # "another job won while we were waiting" case; in that case we just
    # return the winner's export rather than re-materializing.
    with _export_dir_lock(export_root_path, digest):
        if _has_exported_hf_weights(export_dir):
            if _read_export_metadata(export_dir) is None:
                _write_export_metadata(export_dir, export_metadata)
            return export_dir

        temp_dir = Path(tempfile.mkdtemp(prefix=f"{export_dir.name}.tmp-", dir=export_root_path))
        model = None
        tokenizer = None
        saved_model = None
        try:
            model, tokenizer, _, _ = load_model_for_eval(
                source,
                auto_find_checkpoint=auto_find_checkpoint,
                load_mode=load_mode,
                base_model=base_model,
                tokenizer_source=tokenizer_source,
                trust_remote_code=trust_remote_code,
                device_map="cpu",
                torch_dtype=torch_dtype,
            )
            if hasattr(model, "merge_and_unload"):
                saved_model = model.merge_and_unload()
            else:
                saved_model = model
            saved_model.save_pretrained(temp_dir, safe_serialization=True)
            tokenizer.save_pretrained(temp_dir)
            _write_export_metadata(temp_dir, export_metadata)
            # Under the lock, export_dir either does not exist or was produced
            # by a prior holder of this same lock (before we took it). Prefer
            # os.replace, which is atomic when source and destination are on
            # the same filesystem; if the destination already has valid
            # weights (a sibling process beat us even with locking defeated
            # on NFS), keep the winner rather than clobbering a live export.
            if export_dir.exists() and _has_exported_hf_weights(export_dir):
                shutil.rmtree(temp_dir)
                return export_dir
            if export_dir.exists():
                shutil.rmtree(export_dir)
            os.replace(str(temp_dir), str(export_dir))
            return export_dir
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            del saved_model
            del model
            del tokenizer
