import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig, OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
RLLM_ROOT = REPO_ROOT / "rllm"
if str(RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RLLM_ROOT))

from rllm.data.utils import fetch_live_code_bench_system_prompt
from rllm.rewards.reward_fn import code_reward_fn


def parse_json_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    return json.loads(value)


def normalize_tests(tests_raw: Any, metadata_raw: Any) -> list[dict[str, Any]]:
    tests = parse_json_maybe(tests_raw)
    metadata = parse_json_maybe(metadata_raw) if metadata_raw else {}
    if not isinstance(metadata, dict):
        metadata = {}

    if isinstance(tests, dict) and "inputs" in tests and "outputs" in tests:
        tests = [
            {"input": input_value, "output": output_value, "testtype": "stdin_stdout"}
            for input_value, output_value in zip(tests["inputs"], tests["outputs"], strict=False)
        ]
    elif isinstance(tests, dict):
        tests = [tests]

    if not isinstance(tests, list):
        tests = [tests] if tests else []

    normalized = []
    for test in tests:
        if not isinstance(test, dict):
            continue
        test = dict(test)
        if test.get("testtype") == "functional" and metadata.get("func_name") is not None:
            test["metadata"] = {"func_name": str(metadata["func_name"])}
        else:
            test["metadata"] = {"func_name": None}
        normalized.append(test)
    return normalized


def load_deepcoder_examples(cfg: DictConfig) -> list[dict[str, Any]]:
    from datasets import load_dataset

    examples = []
    for subset in cfg.dataset.subsets:
        dataset = load_dataset(cfg.dataset.path, name=str(subset), split=cfg.dataset.split)
        for row_idx, row in enumerate(dataset):
            starter_code = row.get("starter_code", "")
            question = fetch_live_code_bench_system_prompt(
                row["problem"],
                starter_code if starter_code else None,
            )
            tests = normalize_tests(row["tests"], row.get("metadata", {}))
            examples.append(
                {
                    "question": question,
                    "ground_truth": json.dumps(tests),
                    "data_source": "livecodebench",
                    "subset": str(subset),
                    "source_index": row_idx,
                    "starter_code": starter_code,
                    "metadata": row.get("metadata", {}),
                }
            )

    if cfg.dataset.shuffle:
        random.Random(cfg.dataset.seed).shuffle(examples)

    num_questions = cfg.dataset.num_questions
    if num_questions is not None and int(num_questions) > 0:
        examples = examples[: int(num_questions)]
    return examples


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


def json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str))


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def split_train_val(rows: list[dict[str, Any]], val_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 2 or val_fraction <= 0:
        return rows, []
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    val_size = min(max(1, int(round(len(shuffled) * val_fraction))), len(shuffled) - 1)
    return shuffled[val_size:], shuffled[:val_size]


def write_training_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        if rows:
            table = pa.Table.from_pylist(rows)
        else:
            message_type = pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string()),
                        pa.field("content", pa.string()),
                    ]
                )
            )
            table = pa.table({"messages": pa.array([], type=message_type)})
        pq.write_table(table, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


@hydra.main(config_path="config", config_name="deepcoder_distill", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)

    print(f"Loading DeepCoder examples from {cfg.dataset.path}")
    examples = load_deepcoder_examples(cfg)
    print(f"Loaded {len(examples)} questions")

    llm = make_llm(cfg)
    sampling_params = make_sampling_params(cfg)
    request_outputs = llm.generate([example["question"] for example in examples], sampling_params)

    all_rows = []
    correct_rows = []
    max_correct_per_question = cfg.output.max_correct_per_question
    if max_correct_per_question is not None:
        max_correct_per_question = int(max_correct_per_question)
        if max_correct_per_question <= 0:
            max_correct_per_question = None
    kept_correct_counts: dict[int, int] = {}
    for question_idx, (example, request_output) in enumerate(zip(examples, request_outputs, strict=True)):
        task_info = {
            "question": example["question"],
            "ground_truth": example["ground_truth"],
            "data_source": example["data_source"],
        }
        for generation_idx, completion in enumerate(request_output.outputs):
            text = completion.text
            reward = code_reward_fn(task_info=task_info, action=text)
            messages = [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": text},
            ]
            row = {
                "question_index": question_idx,
                "generation_index": generation_idx,
                "subset": example["subset"],
                "source_index": example["source_index"],
                "is_correct": bool(reward.is_correct),
                "reward": reward.reward,
                "reward_metadata": reward.metadata,
                "messages": messages,
            }
            all_rows.append(row)
            if reward.is_correct:
                kept_count = kept_correct_counts.get(question_idx, 0)
                if max_correct_per_question is None or kept_count < max_correct_per_question:
                    correct_rows.append(row)
                    kept_correct_counts[question_idx] = kept_count + 1
        print(
            f"checked question {question_idx + 1}/{len(examples)}: "
            f"{sum(1 for row in all_rows if row['question_index'] == question_idx and row['is_correct'])} correct"
        )

    train_rows, val_rows = split_train_val(
        [{"messages": row["messages"]} for row in correct_rows],
        float(cfg.output.val_fraction),
        int(cfg.output.seed),
    )

    all_jsonl = output_dir / "all_completions.jsonl"
    correct_jsonl = output_dir / "correct_completions.jsonl"
    train_parquet = output_dir / "train.parquet"
    val_parquet = output_dir / "val.parquet"
    write_jsonl(all_rows, all_jsonl)
    write_jsonl(correct_rows, correct_jsonl)
    write_training_parquet(train_rows, train_parquet)
    write_training_parquet(val_rows, val_parquet)

    summary = {
        "run_name": cfg.run_name,
        "model": cfg.model.name_or_path,
        "dataset": cfg.dataset.path,
        "dataset_split": cfg.dataset.split,
        "dataset_subsets": list(cfg.dataset.subsets),
        "output_dir": str(output_dir),
        "num_questions": len(examples),
        "generations_per_question": cfg.sampling.n,
        "max_correct_per_question": cfg.output.max_correct_per_question,
        "num_completions": len(all_rows),
        "num_correct": len(correct_rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "all_completions": str(all_jsonl),
        "correct_completions": str(correct_jsonl),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
