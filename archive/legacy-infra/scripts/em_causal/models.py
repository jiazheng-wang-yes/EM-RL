from __future__ import annotations

import sys
import json
import time
from pathlib import Path

import torch

from .common import ROOT, digest


def compatible_tokenizer_source(source, base_model):
    """Verify a v5 tokenizer against its base before loading v5 checkpoints with v4.

    This does not edit checkpoint metadata or silently discard special tokens.
    Only an exactly matching vocabulary, chat template and special-token map can
    use the original tokenizer to bypass the v4 list-versus-dictionary error.
    """
    import transformers
    from transformers import AutoTokenizer
    if int(transformers.__version__.split(".")[0]) >= 5:
        return None
    metadata = Path(source) / "huggingface"
    config_path = metadata / "tokenizer_config.json"
    if not config_path.exists():
        return None
    config = json.loads(config_path.read_text())
    extra = config.get("extra_special_tokens")
    if not isinstance(extra, list):
        return None
    if not base_model:
        raise ValueError("This checkpoint needs --base-model for verified tokenizer compatibility with Transformers 4")
    saved = AutoTokenizer.from_pretrained(metadata, trust_remote_code=False,
                                         extra_special_tokens={}, additional_special_tokens=extra)
    original = AutoTokenizer.from_pretrained(base_model, trust_remote_code=False)
    if tokenizer_identity(saved) != tokenizer_identity(original):
        raise ValueError("Checkpoint tokenizer differs from base; use its native Transformers version")
    return base_model


def configure_attention(model, attention_backend: str):
    if attention_backend == "sdpa_no_cudnn":
        if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
            raise RuntimeError("This PyTorch build cannot disable the failing cuDNN SDPA backend")
        torch.backends.cuda.enable_cudnn_sdp(False)
        model.config._attn_implementation = "sdpa"
    elif attention_backend in {"eager", "sdpa"}:
        model.config._attn_implementation = attention_backend
    else:
        raise ValueError("attention_backend must be eager, sdpa, or sdpa_no_cudnn")


def load_model(source: str, device: str = "cuda:0", base_model: str | None = None,
               attention_backend: str = "sdpa_no_cudnn"):
    started = time.monotonic()
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))
    from em_organism_dir.eval.model_loading import load_model_for_eval
    imported = time.monotonic()
    tokenizer_override = compatible_tokenizer_source(source, base_model)
    model, tokenizer, resolved, kind = load_model_for_eval(
        source, base_model=base_model, device_map=device, trust_remote_code=False,
        tokenizer_source=tokenizer_override,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )
    # The cuDNN SDPA frontend cannot build plans on some cluster nodes. Retain
    # accelerated SDPA while excluding only that backend. This setting is process
    # global and therefore applies consistently to donor and recipient models.
    configure_attention(model, attention_backend)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    weights = sorted(Path(resolved).glob("model_world_size_*_rank_*.pt")) if Path(resolved).is_dir() else []
    return model, tokenizer, {"requested": source, "resolved": resolved, "kind": kind,
                              "model_config_hash": digest(model.config.to_dict()),
                              "attention_backend": attention_backend,
                              "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled() if hasattr(torch.backends.cuda, "cudnn_sdp_enabled") else None,
                              "verified_tokenizer_override": tokenizer_override,
                              "hf_commit": getattr(model.config, "_commit_hash", None), "torch": str(torch.__version__),
                              "stage_timings": {"loader_import_seconds": imported - started, "loading_seconds": time.monotonic() - imported},
                              "weight_file_identity": [{"path": str(p), "bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in weights]}


def tokenizer_identity(tokenizer) -> str:
    return digest({"vocab": tokenizer.get_vocab(), "chat_template": tokenizer.chat_template,
                   "special_tokens": tokenizer.special_tokens_map})


def encode_prompt(tokenizer, row: dict, thinking_enabled=None) -> tuple[list[int], dict]:
    thinking_enabled = row.get("thinking_enabled", False) if thinking_enabled is None else thinking_enabled
    text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=thinking_enabled)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=tokenizer.is_fast)
    tokens = encoded["input_ids"]
    if not tokens:
        raise ValueError("Empty prompt token sequence")
    positions = {"prompt_last": len(tokens) - 1, "decision": len(tokens) - 1}
    if "offset_mapping" in encoded:
        user = row["messages"][-1]["content"]
        user_start = text.rfind(user)
        boundary = user_start + user.rfind("\n") + 1 if "\n" in user else user_start
        matches = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a <= boundary < b]
        if matches:
            positions["instruction_boundary"] = matches[0]
        for name, spans in row.get("semantic_spans", {}).items():
            indices = set()
            for span in spans:
                content = row["messages"][span["message_index"]]["content"]
                offset = text.find(content)
                if offset < 0 or not 0 <= span["start"] < span["end"] <= len(content):
                    raise ValueError(f"Invalid semantic span {name} in {row['id']}")
                a, b = offset + span["start"], offset + span["end"]
                indices.update(i for i, (start, end) in enumerate(encoded["offset_mapping"]) if end > a and start < b)
            if not indices:
                raise ValueError(f"No tokens aligned to semantic span {name}")
            positions[name] = sorted(indices)
    return tokens, positions


def model_device(model):
    return model.get_input_embeddings().weight.device


def forward(model, ids: torch.Tensor):
    return model(input_ids=ids.to(model_device(model)), attention_mask=torch.ones_like(ids, device=model_device(model)), use_cache=False)


def eos_ids(model, tokenizer) -> set[int]:
    value = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if value is None:
        value = tokenizer.eos_token_id
    if value is None:
        return set()
    return set(value if isinstance(value, (tuple, list)) else [value])
