"""Small shared helpers for vLLM-backed data generation entrypoints."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from _resume import json_safe


def without_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def make_llm(cfg: DictConfig):
    from vllm import LLM

    llm_kwargs = without_none(
        {
            "model": cfg.model.name_or_path,
            "tensor_parallel_size": cfg.model.tensor_parallel_size,
            "dtype": cfg.model.dtype,
            "trust_remote_code": cfg.model.trust_remote_code,
            "quantization": cfg.model.quantization,
            "gpu_memory_utilization": cfg.model.gpu_memory_utilization,
            "cpu_offload_gb": cfg.model.cpu_offload_gb,
            "max_model_len": cfg.model.max_model_len,
            "enforce_eager": cfg.model.enforce_eager,
            "disable_custom_all_reduce": cfg.model.disable_custom_all_reduce,
            "seed": cfg.model.seed,
            "swap_space": cfg.model.swap_space,
            "max_num_seqs": cfg.model.max_num_seqs,
            "download_dir": cfg.model.download_dir,
            "enable_prefix_caching": cfg.model.enable_prefix_caching,
        }
    )
    return LLM(**llm_kwargs)


def make_sampling_params(cfg: DictConfig):
    from vllm import SamplingParams

    sampling_kwargs = without_none(
        {
            "n": cfg.sampling.n,
            "temperature": cfg.sampling.temperature,
            "top_p": cfg.sampling.top_p,
            "top_k": cfg.sampling.top_k,
            "min_p": cfg.sampling.min_p,
            "presence_penalty": cfg.sampling.presence_penalty,
            "frequency_penalty": cfg.sampling.frequency_penalty,
            "repetition_penalty": cfg.sampling.repetition_penalty,
            "max_tokens": cfg.sampling.max_tokens,
            "min_tokens": cfg.sampling.min_tokens,
            "ignore_eos": cfg.sampling.ignore_eos,
            "stop": list(cfg.sampling.stop) if cfg.sampling.stop else None,
            "seed": cfg.sampling.seed,
        }
    )
    return SamplingParams(**sampling_kwargs)


def apply_chat_template(
    llm,
    raw_prompts: list[str],
    *,
    apply: bool,
    enable_thinking: bool | None = None,
) -> list[str]:
    if not apply or not raw_prompts:
        return list(raw_prompts)
    tokenizer = llm.get_tokenizer()
    if getattr(tokenizer, "chat_template", None) is None:
        return list(raw_prompts)
    rendered: list[str] = []
    for prompt in raw_prompts:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking is not None:
            kwargs["enable_thinking"] = bool(enable_thinking)
        try:
            rendered.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
        except TypeError:
            kwargs.pop("enable_thinking", None)
            rendered.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    **kwargs,
                )
            )
    return rendered


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(
            json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        with tmp_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(json_safe(row), ensure_ascii=False))
                handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        table = pa.Table.from_pylist(json_safe(rows))
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def completion_finish_reason(completion) -> str | None:
    value = getattr(completion, "finish_reason", None)
    if value is None:
        return None
    return str(value)


def completion_stop_reason(completion) -> str | None:
    value = getattr(completion, "stop_reason", None)
    if value is None:
        return None
    return str(value)


def completion_metadata(completion, cfg: DictConfig) -> dict[str, Any]:
    finish_reason = (completion_finish_reason(completion) or "").lower()
    stop_reason = (completion_stop_reason(completion) or "").lower()
    token_ids = getattr(completion, "token_ids", []) or []
    max_tokens = int(cfg.sampling.max_tokens)
    cropped = finish_reason == "length" or stop_reason == "length"
    if not cropped and max_tokens > 0 and len(token_ids) >= max_tokens:
        cropped = finish_reason not in {"stop", "eos"} and stop_reason not in {"stop", "eos"}
    return {
        "finish_reason": completion_finish_reason(completion),
        "stop_reason": completion_stop_reason(completion),
        "output_token_count": len(token_ids),
        "cropped": cropped,
    }


def strip_thinking_trace(text: str) -> str:
    stripped = text.strip()
    close_tag = "</think>"
    close_index = stripped.find(close_tag)
    if close_index < 0:
        return stripped
    return stripped[close_index + len(close_tag) :].strip()
