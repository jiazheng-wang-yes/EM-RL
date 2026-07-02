from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

from examples.lean_prover_v1.probe_common import DATASET_NAME, register_lean_prover_v1_data, summarize_lean_rows
from rllm.data.dataset import DatasetRegistry


def _load_rows(split: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    register_lean_prover_v1_data(
        static_corpus_path=args.static_corpus_path,
        mutation_bank_path=args.mutation_bank_path,
        train_static_size=args.train_static_size,
        val_static_size=args.val_static_size,
        test_static_size=args.test_static_size,
        train_mutated_size=args.train_mutated_size,
        val_mutated_size=args.val_mutated_size,
        test_mutated_size=args.test_mutated_size,
    )
    dataset = DatasetRegistry.load_dataset(DATASET_NAME, split)
    if dataset is None:
        raise RuntimeError(f"Dataset split not found: {DATASET_NAME}/{split}")
    return list(dataset.get_data())


def _is_static_row(row: dict[str, Any]) -> bool:
    return str(row.get("mutation_type")) == "static" or str(row.get("split", "")).endswith("_static")


def _select_rows(rows: list[dict[str, Any]], *, split: str, limit: int | None) -> list[dict[str, Any]]:
    if limit is None or limit < 0 or len(rows) <= limit:
        return rows
    if split in {"train", "val", "test"}:
        static_rows = [row for row in rows if _is_static_row(row)]
        mutated_rows = [row for row in rows if not _is_static_row(row)]
        static_count = min(len(static_rows), limit // 2)
        mutated_count = min(len(mutated_rows), limit - static_count)
        if static_count + mutated_count < limit:
            static_count = min(len(static_rows), limit - mutated_count)
        return static_rows[:static_count] + mutated_rows[:mutated_count]
    return rows[:limit]


def _chat_prompt(tokenizer: Any, row: dict[str, Any], *, disable_thinking: bool) -> str:
    messages = [{"role": "user", "content": str(row["question"])}]
    kwargs = {"tokenize": False, "add_generation_prompt": True}
    if disable_thinking:
        try:
            return str(tokenizer.apply_chat_template(messages, **kwargs, enable_thinking=False))
        except TypeError:
            pass
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    return str(row["question"])


def _clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()


def _generate_vllm(
    *,
    model_source: str,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
    gpu_memory_utilization: float,
    disable_thinking: bool,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_source,
        tokenizer=model_source,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        n=max(1, num_samples),
    )

    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, batch_size)):
        batch_rows = rows[start : start + max(1, batch_size)]
        prompts = [_chat_prompt(tokenizer, row, disable_thinking=disable_thinking) for row in batch_rows]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for row, output in zip(batch_rows, outputs, strict=False):
            responses = [item.text for item in output.outputs]
            finish_reasons = [str(item.finish_reason) for item in output.outputs]
            records.append(_response_record(row, responses, finish_reasons))

    del llm
    _clear_cuda_cache()
    return records


def _generate_transformers(
    *,
    model_source: str,
    rows: list[dict[str, Any]],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
    device: str,
    disable_thinking: bool,
) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else "auto",
    ).to(device)
    model.eval()

    records_by_id: dict[str, dict[str, Any]] = {}
    for sample_idx in range(max(1, num_samples)):
        for start in range(0, len(rows), max(1, batch_size)):
            batch_rows = rows[start : start + max(1, batch_size)]
            prompts = [_chat_prompt(tokenizer, row, disable_thinking=disable_thinking) for row in batch_rows]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            do_sample = temperature > 0
            generate_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                generate_kwargs.update({"temperature": temperature, "top_p": top_p})
            with torch.no_grad():
                generated = model.generate(**encoded, **generate_kwargs)
            completions = generated[:, encoded["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(completions, skip_special_tokens=True)
            for row, response in zip(batch_rows, decoded, strict=False):
                record = records_by_id.setdefault(str(row["id"]), _response_record(row, [], []))
                record["responses"].append(response)
                record["finish_reasons"].append("unknown")
            del generated
            del encoded
        if sample_idx + 1 < num_samples:
            _clear_cuda_cache()

    del model
    del tokenizer
    _clear_cuda_cache()
    return list(records_by_id.values())


def _response_record(row: dict[str, Any], responses: list[str], finish_reasons: list[str]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or row.get("uid")),
        "uid": str(row.get("uid") or row.get("id")),
        "split": row.get("split"),
        "mutation_type": row.get("mutation_type"),
        "responses": responses,
        "finish_reasons": finish_reasons,
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _formatting_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    responses = [response for record in records for response in record.get("responses", [])]
    starts = Counter(response.strip().split(None, 1)[0] if response.strip().split() else "<empty>" for response in responses)
    lengths = [len(response.split()) for response in responses]
    return {
        "response_count": len(responses),
        "empty_response_count": sum(1 for response in responses if not response.strip()),
        "raw_code_fence_count": sum(1 for response in responses if response.strip().startswith("```")),
        "raw_code_fence_rate": sum(1 for response in responses if response.strip().startswith("```")) / max(1, len(responses)),
        "raw_leading_by_count": sum(1 for response in responses if response.strip().startswith("by")),
        "raw_leading_by_rate": sum(1 for response in responses if response.strip().startswith("by")) / max(1, len(responses)),
        "mean_response_tokens": statistics.mean(lengths) if lengths else 0.0,
        "top_response_starts": starts.most_common(10),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Lean prover v1 model responses.")
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--output", required=True, help="Responses JSONL path.")
    parser.add_argument("--report-output", default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=3072)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--static-corpus-path", default=None)
    parser.add_argument("--mutation-bank-path", default=None)
    parser.add_argument("--train-static-size", type=int, default=1024)
    parser.add_argument("--val-static-size", type=int, default=128)
    parser.add_argument("--test-static-size", type=int, default=128)
    parser.add_argument("--train-mutated-size", type=int, default=1024)
    parser.add_argument("--val-mutated-size", type=int, default=128)
    parser.add_argument("--test-mutated-size", type=int, default=128)
    args = parser.parse_args()

    rows = _select_rows(_load_rows(args.split, args), split=args.split, limit=args.limit)
    started = time.monotonic()
    if args.backend == "vllm":
        records = _generate_vllm(
            model_source=args.model_source,
            rows=rows,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_thinking=args.disable_thinking,
        )
    else:
        records = _generate_transformers(
            model_source=args.model_source,
            rows=rows,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            device=args.device,
            disable_thinking=args.disable_thinking,
        )

    output_path = Path(args.output)
    _write_jsonl(output_path, records)
    report = {
        "artifacts": {"responses_jsonl": str(output_path)},
        "model": {
            "source": args.model_source,
            "backend": args.backend,
            "num_samples": args.num_samples,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "disable_thinking": args.disable_thinking,
        },
        "curriculum": {
            "split": args.split,
            "limit": args.limit,
            "row_count": len(rows),
            "static_sampling_floor_active": args.split in {"train", "val", "test"} and args.limit >= 0,
            "summary": summarize_lean_rows(rows),
        },
        "generation": {
            "elapsed_s": time.monotonic() - started,
            **_formatting_summary(records),
        },
    }
    if args.report_output:
        report_path = Path(args.report_output)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
