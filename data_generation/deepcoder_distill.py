import json
import os
import random
import shutil
import sys
from datetime import datetime, timezone
from hashlib import sha256
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

from rllm.data.utils import fetch_live_code_bench_system_prompt  # noqa: E402
from rllm.rewards.reward_fn import code_reward_fn  # noqa: E402


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
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def append_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def trim_jsonl_by_question_idx(path: Path, next_question_idx: int) -> list[dict[str, Any]]:
    seen_keys: set[tuple[int, int]] = set()
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(path):
        question_idx = int(row.get("question_index", -1))
        if question_idx < 0 or question_idx >= next_question_idx:
            continue
        key = (question_idx, int(row.get("generation_index", -1)))
        if key in seen_keys:
            continue
        seen_keys.add(key)
        rows.append(row)
    write_jsonl(rows, path)
    return rows


def completed_prefix_from_rows(
    rows: list[dict[str, Any]],
    intended_next_question_idx: int,
    expected_generations_per_question: int,
) -> int:
    counts: dict[int, int] = {}
    for row in rows:
        question_idx = int(row.get("question_index", -1))
        if question_idx < 0:
            continue
        counts[question_idx] = counts.get(question_idx, 0) + 1

    next_question_idx = 0
    while next_question_idx < intended_next_question_idx:
        if counts.get(next_question_idx, 0) < expected_generations_per_question:
            break
        next_question_idx += 1
    return next_question_idx


def remove_generated_outputs(paths: list[Path]) -> None:
    for path in paths:
        if path.exists() or path.is_symlink():
            path.unlink()


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


def write_json_atomic(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_cache_payload(cfg: DictConfig) -> dict[str, Any]:
    return {
        "dataset": {
            "path": cfg.dataset.path,
            "split": cfg.dataset.split,
            "subsets": list(cfg.dataset.subsets),
            "num_questions": cfg.dataset.num_questions,
            "shuffle": bool(cfg.dataset.shuffle),
            "seed": int(cfg.dataset.seed),
        },
        "model": {
            "name_or_path": cfg.model.name_or_path,
            "tensor_parallel_size": cfg.model.tensor_parallel_size,
            "dtype": cfg.model.dtype,
            "quantization": cfg.model.quantization,
            "max_model_len": cfg.model.max_model_len,
            "gpu_memory_utilization": cfg.model.gpu_memory_utilization,
            "max_num_seqs": cfg.model.max_num_seqs,
            "seed": cfg.model.seed,
        },
        "sampling": {
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
            "stop": list(cfg.sampling.stop) if cfg.sampling.stop else [],
            "seed": cfg.sampling.seed,
        },
        "output": {
            "max_correct_per_question": cfg.output.max_correct_per_question,
            "val_fraction": cfg.output.val_fraction,
            "seed": cfg.output.seed,
        },
    }


def compute_cache_key(cfg: DictConfig) -> tuple[str, dict[str, Any]]:
    payload = build_cache_payload(cfg)
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(serialized.encode("utf-8")).hexdigest(), payload


def ensure_link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source)
    except OSError:
        shutil.copy2(source, destination)


def validate_completed_run(run_dir: Path) -> bool:
    required = [
        "all_completions.jsonl",
        "correct_completions.jsonl",
        "train.parquet",
        "val.parquet",
        "summary.json",
    ]
    if not all((run_dir / filename).exists() for filename in required):
        return False
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
    except json.JSONDecodeError:
        return False
    return summary.get("status") == "completed"


def load_cache_index(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"keys": {}}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"keys": {}}
    if not isinstance(payload, dict):
        return {"keys": {}}
    keys = payload.get("keys")
    if not isinstance(keys, dict):
        payload["keys"] = {}
    return payload


def save_cache_index(path: Path, payload: dict[str, Any]) -> None:
    write_json_atomic(payload, path)


def update_cache_index(cache_index_path: Path, cache_key: str, run_dir: Path) -> None:
    index = load_cache_index(cache_index_path)
    entries = index.setdefault("keys", {}).setdefault(cache_key, [])
    run_dir_str = str(run_dir)
    filtered = [entry for entry in entries if isinstance(entry, dict) and entry.get("run_dir") != run_dir_str]
    filtered.append(
        {
            "run_dir": run_dir_str,
            "status": "completed",
            "updated_at": now_iso(),
        }
    )
    index["keys"][cache_key] = filtered
    save_cache_index(cache_index_path, index)


def find_reusable_run(cache_index_path: Path, cache_key: str, current_run_dir: Path) -> Path | None:
    index = load_cache_index(cache_index_path)
    entries = index.get("keys", {}).get(cache_key, [])
    if not isinstance(entries, list):
        return None
    for entry in reversed(entries):
        if not isinstance(entry, dict):
            continue
        if entry.get("status") != "completed":
            continue
        candidate = Path(entry.get("run_dir", ""))
        if not candidate:
            continue
        if candidate.resolve() == current_run_dir.resolve():
            continue
        if validate_completed_run(candidate):
            return candidate
    return None


def initialize_or_resume_progress(progress_path: Path) -> dict[str, Any]:
    if not progress_path.exists():
        return {
            "next_question_idx": 0,
            "processed_question_indices": [],
            "kept_correct_counts": {},
            "num_completions": 0,
            "num_correct": 0,
            "updated_at": now_iso(),
        }
    payload = json.loads(progress_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Malformed progress checkpoint: {progress_path}")
    payload.setdefault("next_question_idx", 0)
    payload.setdefault("processed_question_indices", [])
    payload.setdefault("kept_correct_counts", {})
    payload.setdefault("num_completions", 0)
    payload.setdefault("num_correct", 0)
    return payload


def persist_progress(progress_path: Path, progress: dict[str, Any]) -> None:
    progress["updated_at"] = now_iso()
    write_json_atomic(progress, progress_path)


def finalize_summary(
    cfg: DictConfig,
    output_dir: Path,
    all_jsonl: Path,
    correct_jsonl: Path,
    train_parquet: Path,
    val_parquet: Path,
    num_questions: int,
    cache_key: str,
    cache_hit: bool,
    resumed: bool,
) -> dict[str, Any]:
    correct_rows = read_jsonl(correct_jsonl)
    train_rows, val_rows = split_train_val(
        [{"messages": row["messages"]} for row in correct_rows],
        float(cfg.output.val_fraction),
        int(cfg.output.seed),
    )
    write_training_parquet(train_rows, train_parquet)
    write_training_parquet(val_rows, val_parquet)
    num_completions = len(read_jsonl(all_jsonl))
    summary = {
        "status": "completed",
        "run_name": cfg.run_name,
        "model": cfg.model.name_or_path,
        "dataset": cfg.dataset.path,
        "dataset_split": cfg.dataset.split,
        "dataset_subsets": list(cfg.dataset.subsets),
        "output_dir": str(output_dir),
        "num_questions": num_questions,
        "generations_per_question": cfg.sampling.n,
        "max_correct_per_question": cfg.output.max_correct_per_question,
        "num_completions": num_completions,
        "num_correct": len(correct_rows),
        "num_train": len(train_rows),
        "num_val": len(val_rows),
        "all_completions": str(all_jsonl),
        "correct_completions": str(correct_jsonl),
        "train_parquet": str(train_parquet),
        "val_parquet": str(val_parquet),
        "cache_key": cache_key,
        "cache_hit": cache_hit,
        "resumed": resumed,
    }
    write_json_atomic(summary, output_dir / "summary.json")
    return summary


@hydra.main(config_path="config", config_name="deepcoder_distill", version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(cfg.output.run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml", resolve=True)
    cache_key, cache_payload = compute_cache_key(cfg)
    cache_index_path = Path(cfg.output.root) / "cache_index.json"
    all_jsonl = output_dir / "all_completions.jsonl"
    correct_jsonl = output_dir / "correct_completions.jsonl"
    train_parquet = output_dir / "train.parquet"
    val_parquet = output_dir / "val.parquet"
    progress_path = output_dir / "progress.json"
    cache_manifest_path = output_dir / "cache_manifest.json"

    manifest = {
        "status": "running",
        "run_name": cfg.run_name,
        "cache_key": cache_key,
        "cache_payload": cache_payload,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    write_json_atomic(manifest, cache_manifest_path)

    force_recompute = bool(cfg.cache.force_recompute)
    if bool(cfg.cache.reuse_identical_runs) and not force_recompute:
        reusable_run = find_reusable_run(cache_index_path, cache_key, output_dir)
        if reusable_run is not None:
            print(f"Reusing completed run cache from {reusable_run}")
            for filename in [
                "all_completions.jsonl",
                "correct_completions.jsonl",
                "train.parquet",
                "val.parquet",
                "summary.json",
            ]:
                ensure_link_or_copy(reusable_run / filename, output_dir / filename)
            reused_summary = json.loads((reusable_run / "summary.json").read_text())
            reused_summary["reused_by_run_name"] = cfg.run_name
            write_json_atomic(reused_summary, output_dir / "reused_summary.json")
            manifest["status"] = "reused"
            manifest["reused_from"] = str(reusable_run)
            manifest["updated_at"] = now_iso()
            write_json_atomic(manifest, cache_manifest_path)
            return

    print(f"Loading DeepCoder examples from {cfg.dataset.path}")
    examples = load_deepcoder_examples(cfg)
    print(f"Loaded {len(examples)} questions")

    resumed = False
    if bool(cfg.cache.resume_interrupted_runs) and not force_recompute:
        progress = initialize_or_resume_progress(progress_path)
        resumed = int(progress.get("next_question_idx", 0)) > 0
    else:
        progress = initialize_or_resume_progress(progress_path)
        progress["next_question_idx"] = 0
        progress["processed_question_indices"] = []
        progress["kept_correct_counts"] = {}
        progress["num_completions"] = 0
        progress["num_correct"] = 0
        remove_generated_outputs(
            [
                all_jsonl,
                correct_jsonl,
                progress_path,
                train_parquet,
                val_parquet,
                output_dir / "summary.json",
                output_dir / "reused_summary.json",
            ]
        )

    if resumed:
        print(f"Resuming from question index {progress['next_question_idx']}")

    next_question_idx = int(progress.get("next_question_idx", 0))
    kept_correct_counts: dict[int, int] = {}
    if bool(cfg.cache.resume_interrupted_runs) and not force_recompute:
        all_rows = trim_jsonl_by_question_idx(all_jsonl, next_question_idx)
        repaired_next_question_idx = completed_prefix_from_rows(
            all_rows,
            next_question_idx,
            int(cfg.sampling.n),
        )
        if repaired_next_question_idx < next_question_idx:
            print(
                f"Repairing DeepCoder resume cursor from {next_question_idx} "
                f"to {repaired_next_question_idx} based on streamed rows"
            )
            next_question_idx = repaired_next_question_idx
            progress["next_question_idx"] = next_question_idx
            all_rows = trim_jsonl_by_question_idx(all_jsonl, next_question_idx)
        correct_rows = trim_jsonl_by_question_idx(correct_jsonl, next_question_idx)
        for row in correct_rows:
            question_idx = int(row["question_index"])
            kept_correct_counts[question_idx] = kept_correct_counts.get(question_idx, 0) + 1
        progress["processed_question_indices"] = list(range(next_question_idx))
        progress["kept_correct_counts"] = {
            str(key): value for key, value in sorted(kept_correct_counts.items())
        }
        progress["num_completions"] = len(all_rows)
        progress["num_correct"] = len(correct_rows)
    persist_progress(progress_path, progress)

    llm = make_llm(cfg)
    sampling_params = make_sampling_params(cfg)
    max_correct_per_question = cfg.output.max_correct_per_question
    if max_correct_per_question is not None:
        max_correct_per_question = int(max_correct_per_question)
        if max_correct_per_question <= 0:
            max_correct_per_question = None
    chunk_size_questions = int(cfg.cache.chunk_size_questions)
    if chunk_size_questions <= 0:
        chunk_size_questions = len(examples)

    kept_correct_counts = {
        int(key): int(value)
        for key, value in progress.get("kept_correct_counts", {}).items()
    }
    num_completions = int(progress.get("num_completions", 0))
    num_correct = int(progress.get("num_correct", 0))

    for chunk_start in range(next_question_idx, len(examples), chunk_size_questions):
        chunk_end = min(chunk_start + chunk_size_questions, len(examples))
        chunk_examples = examples[chunk_start:chunk_end]
        request_outputs = llm.generate([example["question"] for example in chunk_examples], sampling_params)
        all_rows_chunk = []
        correct_rows_chunk = []
        for local_idx, (example, request_output) in enumerate(zip(chunk_examples, request_outputs, strict=True)):
            question_idx = chunk_start + local_idx
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
                all_rows_chunk.append(row)
                num_completions += 1
                if reward.is_correct:
                    kept_count = kept_correct_counts.get(question_idx, 0)
                    if max_correct_per_question is None or kept_count < max_correct_per_question:
                        correct_rows_chunk.append(row)
                        kept_correct_counts[question_idx] = kept_count + 1
                        num_correct += 1
            question_correct = sum(1 for row in all_rows_chunk if row["question_index"] == question_idx and row["is_correct"])
            print(f"checked question {question_idx + 1}/{len(examples)}: {question_correct} correct")

        append_jsonl(all_rows_chunk, all_jsonl)
        append_jsonl(correct_rows_chunk, correct_jsonl)

        progress["next_question_idx"] = chunk_end
        progress["processed_question_indices"] = list(range(chunk_end))
        progress["kept_correct_counts"] = {str(key): value for key, value in sorted(kept_correct_counts.items())}
        progress["num_completions"] = num_completions
        progress["num_correct"] = num_correct
        persist_progress(progress_path, progress)
        print(
            f"checkpointed chunk {chunk_start}:{chunk_end} "
            f"({num_completions} completions, {num_correct} kept correct)"
        )

    summary = finalize_summary(
        cfg=cfg,
        output_dir=output_dir,
        all_jsonl=all_jsonl,
        correct_jsonl=correct_jsonl,
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        num_questions=len(examples),
        cache_key=cache_key,
        cache_hit=False,
        resumed=resumed,
    )
    update_cache_index(cache_index_path, cache_key, output_dir)
    manifest["status"] = "completed"
    manifest["updated_at"] = now_iso()
    manifest["summary"] = summary
    write_json_atomic(manifest, cache_manifest_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
