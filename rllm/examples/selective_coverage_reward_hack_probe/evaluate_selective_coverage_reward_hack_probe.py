"""Offline multi-condition evaluator for selective-coverage behavior."""

from __future__ import annotations

import argparse
import gc
import json
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from examples.selective_coverage_reward_hack_probe.dataset import (
    DEFAULT_QUESTIONS_PER_GROUP,
    DEFAULT_SEED,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_SIZE,
    DEFAULT_VAL_SIZE,
    RAW_DATASET_NAME,
    RAW_DATASET_REVISION,
    build_selective_coverage_data_pool,
)
from examples.selective_coverage_reward_hack_probe.prompts import condition_name, rewrite_question
from examples.selective_coverage_reward_hack_probe.reward import (
    DEFAULT_COVERAGE_WEIGHT,
    DEFAULT_PRECISION_WEIGHT,
    score_selective_coverage,
)


def parse_conditions(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = list(value)
    conditions: list[int] = []
    for raw in raw_values:
        condition = int(raw)
        condition_name(condition)
        if condition not in conditions:
            conditions.append(condition)
    if not conditions:
        raise ValueError("At least one evaluation condition is required.")
    return tuple(conditions)


def _generation_prompt(tokenizer: Any, question: str) -> str:
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return question


def _generate_transformers(
    model_source: str,
    rows: list[dict[str, Any]],
    *,
    device: str,
    batch_size: int,
    max_model_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    generation_seed: int,
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if device.startswith("cuda") and torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
    ).to(device)
    model.eval()
    torch.manual_seed(generation_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(generation_seed)

    completions: list[str] = []
    prompt_limit = max_model_len - max_new_tokens
    if prompt_limit < 1:
        raise ValueError("max_model_len must be greater than max_new_tokens.")
    with torch.inference_mode():
        for start in range(0, len(rows), max(1, batch_size)):
            batch_rows = rows[start : start + max(1, batch_size)]
            prompts = [_generation_prompt(tokenizer, row["question"]) for row in batch_rows]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            if int(encoded["input_ids"].shape[1]) > prompt_limit:
                raise ValueError(f"Prompt batch requires {encoded['input_ids'].shape[1]} tokens but the limit is {prompt_limit}. Increase --max-model-len or reduce --questions-per-group.")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new_tokens,
                "do_sample": temperature > 0.0,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if temperature > 0.0:
                generation_kwargs["temperature"] = temperature
                generation_kwargs["top_p"] = top_p
            generated = model.generate(**encoded, **generation_kwargs)
            completion_ids = generated[:, encoded["input_ids"].shape[1] :]
            completions.extend(tokenizer.batch_decode(completion_ids, skip_special_tokens=True))

    del model
    del tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return completions


def _generate_vllm(
    model_source: str,
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    max_model_len: int,
    max_new_tokens: int,
    gpu_memory_utilization: float,
    temperature: float,
    top_p: float,
    generation_seed: int,
) -> list[str]:
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
        seed=generation_seed,
    )
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_new_tokens)
    completions: list[str] = []
    for start in range(0, len(rows), max(1, batch_size)):
        batch_rows = rows[start : start + max(1, batch_size)]
        prompts = [_generation_prompt(tokenizer, row["question"]) for row in batch_rows]
        results = llm.generate(prompts, sampling, use_tqdm=False)
        completions.extend(result.outputs[0].text if result.outputs else "" for result in results)
    del llm
    gc.collect()
    return completions


def generate_model_completions(
    model_source: str,
    rows: list[dict[str, Any]],
    *,
    backend: str = "transformers",
    device: str = "cuda:0",
    batch_size: int = 1,
    max_model_len: int = 16384,
    max_new_tokens: int = 512,
    vllm_gpu_memory_utilization: float = 0.8,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generation_seed: int = 1337,
) -> list[str]:
    if temperature < 0.0:
        raise ValueError("temperature must be non-negative.")
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1].")
    if backend == "transformers":
        return _generate_transformers(
            model_source,
            rows,
            device=device,
            batch_size=batch_size,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            generation_seed=generation_seed,
        )
    if backend == "vllm":
        return _generate_vllm(
            model_source,
            rows,
            batch_size=batch_size,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            temperature=temperature,
            top_p=top_p,
            generation_seed=generation_seed,
        )
    raise ValueError(f"Unknown backend {backend!r}.")


def _action_text(completion: str) -> str:
    text = str(completion).strip()
    return text.partition("</think>")[2].strip() if text.count("</think>") == 1 else text


def score_completions(
    rows: list[dict[str, Any]],
    completions: list[str],
) -> list[dict[str, Any]]:
    if len(rows) != len(completions):
        raise ValueError(f"Expected {len(rows)} completions, got {len(completions)}.")
    outputs: list[dict[str, Any]] = []
    for row, completion in zip(rows, completions, strict=True):
        output = score_selective_coverage(row, _action_text(completion))
        metadata = dict(output.metadata or {})
        outputs.append(
            {
                "task_id": row.get("task_id"),
                "group_id": row.get("group_id"),
                "condition": int(row.get("condition", 0)),
                "condition_name": row.get("condition_name"),
                "sample_index": int(row.get("sample_index", 0)),
                "batch_size": int(row.get("batch_size", 0)),
                "reward": float(output.reward),
                "response": completion,
                **metadata,
            }
        )
    return outputs


def summarize_outputs(outputs: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(outputs)
    denom = max(1, total)
    numeric_means = {
        f"mean_{field}": sum(float(row.get(field, 0.0)) for row in outputs) / denom
        for field in (
            "reward",
            "proxy_reward",
            "oracle_reward",
            "coverage",
            "precision",
            "proxy_oracle_gap",
            "answered_count",
            "null_count",
            "correct_count",
        )
    }
    sample_index_counts = Counter(int(row.get("sample_index", 0)) for row in outputs)
    prompt_keys = {(row.get("condition"), row.get("group_id")) for row in outputs}
    group_sizes = {row.get("group_id"): int(row.get("batch_size", 0)) for row in outputs if row.get("group_id") is not None}
    return {
        "num_samples": total,
        "num_prompts": len(prompt_keys),
        "num_groups": len(group_sizes),
        "num_questions": sum(group_sizes.values()),
        "num_question_samples": sum(int(row.get("batch_size", 0)) for row in outputs),
        "sample_index_counts": {str(index): count for index, count in sorted(sample_index_counts.items())},
        "format_rate": sum(bool(row.get("format_ok")) for row in outputs) / denom,
        "validated_hack_rate": sum(bool(row.get("any_hack")) for row in outputs) / denom,
        **numeric_means,
    }


def run_evaluation(
    *,
    model_source: str,
    conditions: str | Sequence[int] = (0, 1, 2),
    train_size: int = DEFAULT_TRAIN_SIZE,
    val_size: int = DEFAULT_VAL_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    questions_per_group: int = DEFAULT_QUESTIONS_PER_GROUP,
    seed: int = DEFAULT_SEED,
    dataset_revision: str = RAW_DATASET_REVISION,
    reward_mode: str = "proxy",
    precision_weight: float = DEFAULT_PRECISION_WEIGHT,
    coverage_weight: float = DEFAULT_COVERAGE_WEIGHT,
    max_samples: int | None = None,
    backend: str = "transformers",
    device: str = "cuda:0",
    batch_size: int = 1,
    max_model_len: int = 16384,
    max_new_tokens: int = 512,
    vllm_gpu_memory_utilization: float = 0.8,
    samples_per_prompt: int = 1,
    temperature: float = 0.0,
    top_p: float = 1.0,
    generation_seed: int = 1337,
    include_rows: bool = False,
    raw_rows: list[dict[str, Any]] | None = None,
    generation_fn: Callable[[list[dict[str, Any]]], list[str]] | None = None,
) -> dict[str, Any]:
    condition_ids = parse_conditions(conditions)
    samples_per_prompt = int(samples_per_prompt)
    if samples_per_prompt < 1:
        raise ValueError("samples_per_prompt must be at least 1.")
    splits = build_selective_coverage_data_pool(
        raw_rows=raw_rows,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        questions_per_group=questions_per_group,
        seed=seed,
        condition=0,
        reward_mode=reward_mode,
        dataset_revision=dataset_revision,
        precision_weight=precision_weight,
        coverage_weight=coverage_weight,
    )
    base_rows = splits["test"]
    if max_samples is not None:
        base_rows = base_rows[: max(0, int(max_samples))]

    all_rows: list[dict[str, Any]] = []
    condition_ranges: dict[int, tuple[int, int]] = {}
    for condition in condition_ids:
        start = len(all_rows)
        for row in base_rows:
            rewritten = rewrite_question(row, condition=condition)
            for sample_index in range(samples_per_prompt):
                sampled_row = dict(rewritten)
                sampled_row["sample_index"] = sample_index
                all_rows.append(sampled_row)
        condition_ranges[condition] = (start, len(all_rows))

    completions = (
        generation_fn(all_rows)
        if generation_fn is not None
        else generate_model_completions(
            model_source,
            all_rows,
            backend=backend,
            device=device,
            batch_size=batch_size,
            max_model_len=max_model_len,
            max_new_tokens=max_new_tokens,
            vllm_gpu_memory_utilization=vllm_gpu_memory_utilization,
            temperature=temperature,
            top_p=top_p,
            generation_seed=generation_seed,
        )
    )
    scored = score_completions(all_rows, completions)

    per_condition: dict[str, Any] = {}
    for condition in condition_ids:
        start, end = condition_ranges[condition]
        condition_outputs = scored[start:end]
        report: dict[str, Any] = {
            "condition": condition,
            "condition_name": condition_name(condition),
            "summary": summarize_outputs(condition_outputs),
        }
        if include_rows:
            report["rows"] = condition_outputs
        per_condition[str(condition)] = report

    return {
        "model_source": str(model_source),
        "dataset_name": RAW_DATASET_NAME,
        "dataset_revision": str(dataset_revision),
        "conditions": list(condition_ids),
        "reward_mode": reward_mode,
        "summary": summarize_outputs(scored),
        "per_condition": per_condition,
        "config": {
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "questions_per_group": questions_per_group,
            "seed": seed,
            "precision_weight": precision_weight,
            "coverage_weight": coverage_weight,
            "backend": backend,
            "device": device,
            "batch_size": batch_size,
            "max_model_len": max_model_len,
            "max_new_tokens": max_new_tokens,
            "max_samples": max_samples,
            "samples_per_prompt": samples_per_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "generation_seed": generation_seed,
        },
    }


def write_json_report(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def parse_eval_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate selective-coverage reward hacking.")
    parser.add_argument("--model-source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conditions", default="0,1,2")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--val-size", type=int, default=DEFAULT_VAL_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--questions-per-group", type=int, default=DEFAULT_QUESTIONS_PER_GROUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--dataset-revision", default=RAW_DATASET_REVISION)
    parser.add_argument("--reward-mode", choices=["proxy", "oracle", "hardened"], default="proxy")
    parser.add_argument("--precision-weight", type=float, default=DEFAULT_PRECISION_WEIGHT)
    parser.add_argument("--coverage-weight", type=float, default=DEFAULT_COVERAGE_WEIGHT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--backend", choices=["transformers", "vllm"], default="transformers")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--samples-per-prompt", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--generation-seed", type=int, default=1337)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--label", default=None)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_eval_args()
    payload = run_evaluation(
        model_source=args.model_source,
        conditions=args.conditions,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        questions_per_group=args.questions_per_group,
        seed=args.seed,
        dataset_revision=args.dataset_revision,
        reward_mode=args.reward_mode,
        precision_weight=args.precision_weight,
        coverage_weight=args.coverage_weight,
        max_samples=args.max_samples,
        backend=args.backend,
        device=args.device,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_new_tokens=args.max_new_tokens,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        samples_per_prompt=args.samples_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        generation_seed=args.generation_seed,
        include_rows=args.include_rows,
    )
    if args.label:
        payload["label"] = args.label
    write_json_report(args.output, payload)


if __name__ == "__main__":
    main()
