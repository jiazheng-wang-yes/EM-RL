from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from typing import Any

import torch

from examples.lean_prover_v1.evaluate_lean_prover_v1 import evaluate_rows
from examples.lean_prover_v1.probe_common import (
    DATASET_NAME,
    load_direct_lean_rows,
    register_lean_prover_v1_data,
    summarize_lean_rows,
)
from rllm.data.dataset import DatasetRegistry


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}.")


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _apply_chat_template(tokenizer: Any, question: str, *, disable_thinking: bool) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return question
    messages = [{"role": "user", "content": question}]
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if disable_thinking:
        kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def _coerce_proof_body(text: str) -> str:
    stripped = text.strip()
    if "</think>" in stripped:
        stripped = stripped.rsplit("</think>", 1)[-1].strip()
    lines = []
    for line in stripped.splitlines():
        if line.strip().startswith("```"):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def _load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    direct_rows_path = getattr(args, "rows_path", None)
    if direct_rows_path:
        rows = load_direct_lean_rows(direct_rows_path)
        if args.limit is not None and args.limit >= 0:
            rows = rows[: args.limit]
        return rows

    register_lean_prover_v1_data(
        static_corpus_path=args.static_corpus,
        mutation_bank_path=args.mutation_bank,
        allow_synthetic=args.allow_synthetic,
        train_static_size=args.train_static_size,
        val_static_size=args.val_static_size,
        test_static_size=args.test_static_size,
        train_mutated_size=args.train_mutated_size,
        val_mutated_size=args.val_mutated_size,
        test_mutated_size=args.test_mutated_size,
    )
    dataset = DatasetRegistry.load_dataset(DATASET_NAME, args.split)
    if dataset is None:
        raise RuntimeError(f"Dataset split not found: {DATASET_NAME}/{args.split}")
    rows = list(dataset.get_data())
    if args.limit is not None and args.limit >= 0:
        rows = rows[: args.limit]
    return rows


def _generate_with_vllm(
    rows: list[dict[str, Any]],
    *,
    model_source: str,
    batch_size: int,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    disable_thinking: bool,
) -> list[dict[str, Any]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_source,
        tokenizer=model_source,
        trust_remote_code=True,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
    )
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
    )

    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, batch_size)):
        batch_rows = rows[start : start + max(1, batch_size)]
        prompts = [_apply_chat_template(tokenizer, str(row["question"]), disable_thinking=disable_thinking) for row in batch_rows]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
        for row, output in zip(batch_rows, outputs, strict=True):
            responses = [_coerce_proof_body(item.text) for item in output.outputs]
            records.append(
                {
                    "id": row["id"],
                    "uid": row.get("uid", row["id"]),
                    "split": row.get("split"),
                    "mutation_type": row.get("mutation_type"),
                    "responses": responses,
                    "finish_reasons": [str(item.finish_reason) for item in output.outputs],
                }
            )

    del llm
    _clear_cuda_cache()
    gc.collect()
    return records


def _generate_with_transformers(
    rows: list[dict[str, Any]],
    *,
    model_source: str,
    batch_size: int,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    num_samples: int,
    disable_thinking: bool,
) -> list[dict[str, Any]]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    do_sample = temperature > 0
    records: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, batch_size)):
        batch_rows = rows[start : start + max(1, batch_size)]
        prompts = [_apply_chat_template(tokenizer, str(row["question"]), disable_thinking=disable_thinking) for row in batch_rows]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_model_len,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "num_return_sequences": num_samples,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = temperature
            generation_kwargs["top_p"] = top_p
        with torch.no_grad():
            generated = model.generate(**encoded, **generation_kwargs)
        prompt_width = int(encoded["input_ids"].shape[1])
        decoded_by_row: list[list[str]] = [[] for _ in batch_rows]
        for idx, sequence in enumerate(generated):
            row_idx = idx // num_samples
            decoded = tokenizer.decode(sequence[prompt_width:], skip_special_tokens=True)
            decoded_by_row[row_idx].append(_coerce_proof_body(decoded))
        for row, responses in zip(batch_rows, decoded_by_row, strict=True):
            records.append(
                {
                    "id": row["id"],
                    "uid": row.get("uid", row["id"]),
                    "split": row.get("split"),
                    "mutation_type": row.get("mutation_type"),
                    "responses": responses,
                    "finish_reasons": [None for _ in responses],
                }
            )

        del generated
        del encoded
        _clear_cuda_cache()

    del model
    del tokenizer
    _clear_cuda_cache()
    gc.collect()
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")


def _response_map(records: list[dict[str, Any]]) -> dict[str, list[str]]:
    return {str(record["id"]): [str(item) for item in record.get("responses", [])] for record in records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Qwen inference and Lean verification for Lean prover v1.")
    parser.add_argument("--model-source", default=os.getenv("MODEL_SOURCE", "Qwen/Qwen3.5-9B"))
    parser.add_argument("--backend", choices=["vllm", "transformers"], default=os.getenv("LEAN_PROVER_V1_BACKEND", "vllm"))
    parser.add_argument("--split", default=os.getenv("LEAN_PROVER_V1_SPLIT", "val"))
    parser.add_argument("--limit", type=int, default=int(os.getenv("LEAN_PROVER_V1_LIMIT", "0")) or None)
    parser.add_argument("--output-dir", default=os.getenv("OUTPUT_DIR", "/net/scratch/jiaweizhang/jiazhengw_migration/eval_runs/lean_prover_v1/inference_smoke"))
    parser.add_argument("--output", default=None, help="Alias for writing the responses JSONL to an exact path.")
    parser.add_argument("--responses-name", default="responses.jsonl")
    parser.add_argument("--report-output", default=None, help="Alias for writing the report JSON to an exact path.")
    parser.add_argument("--report-name", default="eval_report.json")
    parser.add_argument("--rows-output", default=None, help="Optional JSONL path for the exact rows used by this run.")
    parser.add_argument("--batch-size", type=int, default=int(os.getenv("LEAN_PROVER_V1_BATCH_SIZE", "1")))
    parser.add_argument("--max-model-len", type=int, default=int(os.getenv("LEAN_PROVER_V1_MAX_MODEL_LEN", "2048")))
    parser.add_argument("--max-new-tokens", type=int, default=int(os.getenv("LEAN_PROVER_V1_MAX_NEW_TOKENS", "256")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("LEAN_PROVER_V1_TEMPERATURE", "0.0")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("LEAN_PROVER_V1_TOP_P", "1.0")))
    parser.add_argument("--num-samples", type=int, default=int(os.getenv("LEAN_PROVER_V1_NUM_SAMPLES", "1")))
    parser.add_argument("--tensor-parallel-size", type=int, default=int(os.getenv("LEAN_PROVER_V1_TP_SIZE", "1")))
    parser.add_argument("--gpu-memory-utilization", type=float, default=float(os.getenv("LEAN_PROVER_V1_GPU_MEMORY_UTILIZATION", "0.75")))
    parser.add_argument("--device", default=None, help="Accepted for launcher compatibility; vLLM chooses devices from CUDA visibility.")
    parser.add_argument("--disable-thinking", type=_bool_arg, default=_bool_arg(os.getenv("DISABLE_THINKING", "true")))
    parser.add_argument("--lean-command", default=os.getenv("LEAN_PROVER_V1_LEAN_COMMAND"))
    parser.add_argument("--lean-cwd", default=os.getenv("LEAN_PROVER_V1_LEAN_CWD"))
    parser.add_argument("--timeout-seconds", type=float, default=float(os.getenv("LEAN_PROVER_V1_TIMEOUT_SECONDS", "10")))
    parser.add_argument("--rows-path", "--direct-rows-path", dest="rows_path", default=os.getenv("LEAN_PROVER_V1_ROWS_PATH"))
    parser.add_argument("--static-corpus", "--static-corpus-path", dest="static_corpus", default=os.getenv("LEAN_PROVER_V1_STATIC_CORPUS"))
    parser.add_argument("--mutation-bank", "--mutation-bank-path", dest="mutation_bank", default=os.getenv("LEAN_PROVER_V1_MUTATION_BANK"))
    parser.add_argument("--allow-synthetic", type=_bool_arg, default=_bool_arg(os.getenv("LEAN_PROVER_V1_ALLOW_SYNTHETIC", "true")))
    parser.add_argument("--train-static-size", type=int, default=int(os.getenv("LEAN_PROVER_V1_TRAIN_STATIC_SIZE", "32")))
    parser.add_argument("--val-static-size", type=int, default=int(os.getenv("LEAN_PROVER_V1_VAL_STATIC_SIZE", "8")))
    parser.add_argument("--test-static-size", type=int, default=int(os.getenv("LEAN_PROVER_V1_TEST_STATIC_SIZE", "8")))
    parser.add_argument("--train-mutated-size", type=int, default=int(os.getenv("LEAN_PROVER_V1_TRAIN_MUTATED_SIZE", "32")))
    parser.add_argument("--val-mutated-size", type=int, default=int(os.getenv("LEAN_PROVER_V1_VAL_MUTATED_SIZE", "8")))
    parser.add_argument("--test-mutated-size", type=int, default=int(os.getenv("LEAN_PROVER_V1_TEST_MUTATED_SIZE", "8")))
    args = parser.parse_args()

    rows = _load_rows(args)
    output_dir = Path(args.output_dir)
    responses_path = Path(args.output) if args.output else output_dir / args.responses_name
    report_path = Path(args.report_output) if args.report_output else output_dir / args.report_name
    rows_path = Path(args.rows_output) if args.rows_output else None
    if rows_path is not None:
        _write_jsonl(rows_path, rows)

    if args.backend == "vllm":
        records = _generate_with_vllm(
            rows,
            model_source=args.model_source,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_thinking=args.disable_thinking,
        )
    else:
        records = _generate_with_transformers(
            rows,
            model_source=args.model_source,
            batch_size=args.batch_size,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_samples=args.num_samples,
            disable_thinking=args.disable_thinking,
        )

    _write_jsonl(responses_path, records)
    report = evaluate_rows(
        rows,
        response_map=_response_map(records),
        lean_command=args.lean_command,
        lean_cwd=args.lean_cwd,
        timeout_seconds=args.timeout_seconds,
        max_k=args.num_samples,
    )
    report["model"] = {
        "source": args.model_source,
        "backend": args.backend,
        "disable_thinking": args.disable_thinking,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
    }
    report["curriculum"] = {
        "split": args.split,
        "limit": args.limit,
        "summary": summarize_lean_rows(rows),
        "static_sampling_floor_active": report.get("static_row_count", 0) > 0 and report.get("mutated_row_count", 0) > 0,
    }
    report["artifacts"] = {
        "responses_jsonl": str(responses_path),
        "report_json": str(report_path),
    }
    if rows_path is not None:
        report["artifacts"]["rows_jsonl"] = str(rows_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
