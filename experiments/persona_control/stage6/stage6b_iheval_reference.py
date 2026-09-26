"""Frozen IHEval Reference-condition evaluation for Stage 6B endpoints.

The Reference setting measures ordinary task competence without a hierarchy
conflict. Raw responses are append-only; scores and manifests are derived
artifacts under ``eval_runs/persona_control_acl``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path

from acl_common import ACL_ROOT, MODELS, ROLLOUT_ROOT, ROOT, hash_ids, sha256_file

IHEVAL_ROOT = ROOT / "benchmarks" / "IHEval"
IFBENCH_ROOT = ROOT / "benchmarks" / "IFBench"
for import_path in (str(IFBENCH_ROOT), str(IHEVAL_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

import evaluation_lib as ifbench_eval  # noqa: E402
import src.safety.evaluate as safety_eval  # noqa: E402
import src.task_execution.evaluate as task_eval  # noqa: E402
import src.tool_use.evaluate as tool_eval  # noqa: E402


REFERENCE_FILES = (
    ("rule-following", "single-turn", "rule-following/single-turn/reference/default/input_data.json"),
    ("rule-following", "multi-turn", "rule-following/multi-turn/reference/default/input_data.json"),
    ("task-execution", "verb-extract", "task-execution/verb-extract/reference/default/input_data.json"),
    ("task-execution", "translation", "task-execution/translation/reference/default/input_data.json"),
    ("task-execution", "lang-detect", "task-execution/lang-detect/reference/default/input_data.json"),
    ("safety", "user-prompt-hijack", "safety/user-prompt-hijack/reference/default/input_data.json"),
    ("safety", "system-prompt-extract", "safety/system-prompt-extract/reference/default/input_data.json"),
    ("tool-use", "slack-user", "tool-use/slack-user/reference/default/input_data.json"),
    ("tool-use", "get-webpage", "tool-use/get-webpage/reference/default/input_data.json"),
)
EXPECTED_TASK_COUNTS = {
    "rule-following/single-turn": 541,
    "rule-following/multi-turn": 541,
    "task-execution/verb-extract": 252,
    "task-execution/translation": 252,
    "task-execution/lang-detect": 240,
    "safety/user-prompt-hijack": 1272,
    "safety/system-prompt-extract": 318,
    "tool-use/slack-user": 100,
    "tool-use/get-webpage": 744,
}
EXPECTED_ITEM_COUNT = sum(EXPECTED_TASK_COUNTS.values())
SCHEMA = "acl_iheval_reference_v1"


def dataset_hash(file_hashes: dict[str, str]) -> str:
    encoded = json.dumps(file_hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_reference_items() -> tuple[list[dict], dict[str, str]]:
    items: list[dict] = []
    file_hashes: dict[str, str] = {}
    observed: dict[str, int] = {}
    seen: set[str] = set()
    for domain, task, relative in REFERENCE_FILES:
        path = IHEVAL_ROOT / "benchmark" / relative
        if not path.is_file():
            raise FileNotFoundError(f"frozen IHEval Reference file is missing: {path}")
        file_hashes[relative] = sha256_file(path)
        with path.open() as handle:
            rows = json.load(handle)
        task_key = f"{domain}/{task}"
        observed[task_key] = len(rows)
        for row in rows:
            if not isinstance(row, dict) or not {"id", "instruction", "answer"}.issubset(row):
                raise ValueError(f"invalid IHEval Reference row in {path}")
            item_id = f"{task_key}:{row['id']}"
            if item_id in seen:
                raise ValueError(f"duplicate IHEval Reference item ID: {item_id}")
            seen.add(item_id)
            items.append({
                "item_id": item_id,
                "domain": domain,
                "task": task,
                "id": row["id"],
                "system": row.get("system"),
                "conversation_history": row.get("conversation_history"),
                "instruction": row["instruction"],
                "answer": row["answer"],
            })
    if observed != EXPECTED_TASK_COUNTS or len(items) != EXPECTED_ITEM_COUNT:
        raise ValueError(f"frozen IHEval Reference item counts changed: {observed}")
    return items, file_hashes


def build_messages(item: dict) -> list[dict[str, str]]:
    messages = []
    if item.get("system") is not None:
        messages.append({"role": "system", "content": item["system"]})
    history = item.get("conversation_history") or []
    for index, message in enumerate(history):
        if not isinstance(message, str):
            raise ValueError(f"IHEval history must be a string list: {item['item_id']}")
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": message})
    messages.append({"role": "user", "content": item["instruction"]})
    return messages


def score_rule_following_item(answer: dict, prediction: str) -> float:
    instruction_ids = answer.get("instruction_id_list", [])
    kwargs = answer.get("kwargs", [])
    if len(instruction_ids) != len(kwargs):
        raise ValueError("IHEval rule-following instruction and kwargs lengths differ")
    strict_passes = []
    for instruction_id, instruction_kwargs in zip(instruction_ids, kwargs):
        checker_cls = ifbench_eval.instructions_registry.INSTRUCTION_DICT.get(instruction_id)
        if checker_cls is None:
            strict_passes.append(False)
            continue
        checker = checker_cls(instruction_id)
        clean_kwargs = {key: value for key, value in instruction_kwargs.items() if value is not None}
        checker.build_description(**clean_kwargs)
        strict_passes.append(bool(checker.check_following(prediction)))
    return float(bool(strict_passes) and all(strict_passes))


def score_item(task: str, answer, response: str) -> tuple[float | None, str | None]:
    """Run the repository's official task verifier; preserve failures as missing."""
    try:
        if task in {"single-turn", "multi-turn"}:
            value = score_rule_following_item(answer, response)
        elif task == "verb-extract":
            value = task_eval.eval_verb_extract(answer, response, loose=False)
        elif task == "translation":
            value = task_eval.eval_translation(answer, response)
        elif task == "lang-detect":
            value = task_eval.eval_lang_detect(answer, response)
        elif task in {"user-prompt-hijack", "system-prompt-extract"}:
            value = safety_eval.eval_tensortrust(answer, response)
        elif task == "slack-user":
            value = tool_eval.eval_slack_user(answer, response)
        elif task == "get-webpage":
            value = task_eval.eval_mixed(answer, response, loose=False)
        else:
            raise ValueError(f"unknown IHEval task: {task}")
        score = float(value)
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"verifier returned a score outside [0, 1]: {score}")
        return score, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _load_existing(raw_path: Path, *, run_id: str, model_key: str,
                   condition: str, data_hash: str) -> dict[str, dict]:
    existing = {}
    if not raw_path.exists():
        return existing
    with raw_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"incomplete final raw IHEval line; preserve and inspect {raw_path}:{line_number}")
            row = json.loads(line)
            if (row.get("run_id") != run_id or row.get("model") != model_key
                    or row.get("condition") != condition or row.get("setting") != "reference"
                    or row.get("dataset_sha256") != data_hash):
                raise ValueError(f"raw IHEval provenance mismatch at {raw_path}:{line_number}")
            item_id = row.get("item_id")
            if not isinstance(item_id, str) or item_id in existing:
                raise ValueError(f"duplicate or invalid raw IHEval item at {raw_path}:{line_number}")
            existing[item_id] = row
    return existing


def _atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: dict) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(name, path)
    except BaseException:
        try:
            os.unlink(name)
        except FileNotFoundError:
            pass
        raise


def _summary_rows(items: list[dict], scored: list[dict]) -> tuple[list[dict], dict]:
    rows_by_task: dict[str, list[dict]] = {}
    for row in scored:
        rows_by_task.setdefault(row["task_key"], []).append(row)
    task_rows = []
    for task_key, task_rows_all in rows_by_task.items():
        valid = [row for row in task_rows_all if row["score"] is not None]
        task_rows.append({
            "task_key": task_key,
            "n_items": len(task_rows_all),
            "valid_items": len(valid),
            "invalid_items": len(task_rows_all) - len(valid),
            "score_sum": sum(row["score"] for row in valid),
            "accuracy": sum(row["score"] for row in valid) / len(valid) if valid else None,
        })
    all_valid = [row for row in scored if row["score"] is not None]
    task_valid = [row["accuracy"] for row in task_rows if row["accuracy"] is not None]
    summary = {
        "total_items": len(items),
        "valid_items": len(all_valid),
        "invalid_items": len(items) - len(all_valid),
        "micro_score": sum(row["score"] for row in all_valid) / len(all_valid) if all_valid else None,
        "task_macro_score": sum(task_valid) / len(task_valid) if task_valid else None,
        "task_count": len(task_rows),
    }
    return task_rows, summary


def evaluate_reference(model, tokenizer, model_key: str, condition: str, run_id: str,
                       source_path: Path, output_dir: Path, *, device: str = "cuda:0",
                       batch_size: int = 8, max_new_tokens: int = 1024) -> dict:
    if model_key not in MODELS:
        raise ValueError(f"unknown model key: {model_key}")
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch_size and max_new_tokens must be positive")
    if not source_path.is_dir():
        raise FileNotFoundError(f"IHEval endpoint source is missing: {source_path}")
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ValueError("IHEval generation requires the pinned left-padding convention")

    items, file_hashes = load_reference_items()
    data_hash = dataset_hash(file_hashes)
    expected_ids = {item["item_id"] for item in items}
    output_dir.mkdir(parents=True, exist_ok=True)
    response_path = output_dir / "iheval_reference_responses.csv"
    task_path = output_dir / "iheval_reference_tasks.csv"
    summary_path = output_dir / "iheval_reference_summary.json"
    raw_path = ROLLOUT_ROOT / run_id / f"iheval_reference_{model_key}_{condition}.jsonl"
    if any(path.exists() for path in (response_path, task_path, summary_path)):
        if not all(path.is_file() for path in (response_path, task_path, summary_path)):
            raise FileExistsError(f"partial IHEval derived outputs exist under {output_dir}")
        summary = json.loads(summary_path.read_text())
        if (summary.get("schema") != SCHEMA or summary.get("run_id") != run_id
                or summary.get("model") != model_key or summary.get("condition") != condition
                or summary.get("dataset_sha256") != data_hash
                or summary.get("total_items") != EXPECTED_ITEM_COUNT):
            raise ValueError(f"existing IHEval summary provenance mismatch: {summary_path}")
        raw = _load_existing(raw_path, run_id=run_id, model_key=model_key,
                             condition=condition, data_hash=data_hash)
        if set(raw) != expected_ids:
            raise ValueError(f"existing IHEval outputs lack permanent raw rows: {len(raw)}/{len(expected_ids)}")
        with response_path.open(newline="") as handle:
            response_rows = {row["item_id"]: row for row in csv.DictReader(handle)}
        if set(response_rows) != expected_ids:
            raise ValueError(f"IHEval response IDs are incomplete: {response_path}")
        for item_id, row in raw.items():
            derived = response_rows[item_id]
            if (derived["answer_ids_sha256"] != row["answer_ids_sha256"]
                    or derived["response"] != row["response"]):
                raise ValueError(f"IHEval response table differs from raw rollout for {item_id}")
        return summary

    rendered = []
    for item in items:
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if MODELS[model_key]["chat"] == "qwen3":
            kwargs["enable_thinking"] = False
        rendered.append(tokenizer.apply_chat_template(build_messages(item), **kwargs))
    prefix_hashes = []
    for prompt in rendered:
        ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
        prefix_hashes.append(hash_ids(ids))
    prefix_set_hash = hashlib.sha256(json.dumps(
        list(zip((item["item_id"] for item in items), prefix_hashes)), separators=(",", ":")
    ).encode()).hexdigest()

    existing = _load_existing(raw_path, run_id=run_id, model_key=model_key,
                               condition=condition, data_hash=data_hash)
    unknown = set(existing) - expected_ids
    if unknown:
        raise ValueError(f"raw IHEval output has unknown IDs: {sorted(unknown)[:5]}")
    index_by_id = {item["item_id"]: index for index, item in enumerate(items)}
    for item_id, row in existing.items():
        index = index_by_id[item_id]
        if row.get("prompt_ids_sha256") != prefix_hashes[index]:
            raise ValueError(f"rendered IHEval input changed for {item_id}")

    eos_ids = []
    for token_id in (tokenizer.eos_token_id,
                     tokenizer.convert_tokens_to_ids("<|eot_id|>")
                     if "<|eot_id|>" in tokenizer.get_vocab() else None):
        if token_id is not None and token_id not in eos_ids:
            eos_ids.append(token_id)
    missing = [index for index, item in enumerate(items) if item["item_id"] not in existing]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a") as raw_handle:
        for start in range(0, len(missing), batch_size):
            indices = missing[start:start + batch_size]
            encoded = tokenizer([rendered[index] for index in indices], padding=True,
                                return_tensors="pt")
            for row_index, item_index in enumerate(indices):
                unpadded = encoded.input_ids[row_index][encoded.attention_mask[row_index].bool()].tolist()
                expected_prefix = tokenizer(rendered[item_index], return_tensors="pt").input_ids[0].tolist()
                if unpadded != expected_prefix:
                    raise ValueError(f"batched IHEval tokenization mismatch for {items[item_index]['item_id']}")
            prompt_width = int(encoded.input_ids.shape[1])
            encoded = encoded.to(device)
            import torch
            with torch.inference_mode():
                generated = model.generate(
                    **encoded, do_sample=False, max_new_tokens=max_new_tokens,
                    eos_token_id=eos_ids or tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            for row_index, item_index in enumerate(indices):
                item = items[item_index]
                answer_ids = generated[row_index, prompt_width:].tolist()
                stops = [answer_ids.index(token) for token in eos_ids if token in answer_ids]
                if stops:
                    answer_ids = answer_ids[:min(stops) + 1]
                response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                row = {
                    "run_id": run_id, "model": model_key, "condition": condition,
                    "setting": "reference", "item_id": item["item_id"],
                    "domain": item["domain"], "task": item["task"], "source_id": item["id"],
                    "dataset_sha256": data_hash,
                    "prompt_ids_sha256": prefix_hashes[item_index],
                    "answer_ids_sha256": hash_ids(answer_ids), "response": response,
                    "answer_tokens": len(answer_ids),
                    "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens},
                }
                raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                raw_handle.flush()
                existing[item["item_id"]] = row
    if set(existing) != expected_ids:
        raise RuntimeError(f"IHEval raw output incomplete: {len(existing)}/{len(expected_ids)}")

    scored = []
    for item in items:
        raw = existing[item["item_id"]]
        score, score_error = score_item(item["task"], item["answer"], raw["response"])
        scored.append({
            "item_id": item["item_id"], "domain": item["domain"], "task": item["task"],
            "task_key": f"{item['domain']}/{item['task']}",
            "source_id": item["id"], "prompt_ids_sha256": raw["prompt_ids_sha256"],
            "answer_ids_sha256": raw["answer_ids_sha256"], "answer_tokens": raw["answer_tokens"],
            "response": raw["response"], "score": score, "score_error": score_error,
        })
    task_rows, overall = _summary_rows(items, scored)
    task_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_csv(response_path, list(scored[0]), scored)
    _atomic_csv(task_path, list(task_rows[0]), task_rows)
    summary = {
        "schema": SCHEMA, "run_id": run_id, "model": model_key, "condition": condition,
        "setting": "reference", "source_path": str(source_path),
        "source_config_sha256": sha256_file(source_path / "config.json"),
        "dataset_sha256": data_hash, "dataset_file_sha256": file_hashes,
        "rendered_prefix_set_sha256": prefix_set_hash,
        **overall,
        "task_metrics_path": str(task_path),
        "response_table_path": str(response_path),
        "raw_rollout_path": str(raw_path),
        "generation": {"do_sample": False, "max_new_tokens": max_new_tokens,
                       "batch_size": batch_size, "padding_side": "left"},
        "evaluator_sha256": sha256_file(Path(__file__)),
        "scorer_source_sha256": {
            "rule_following": sha256_file(IHEVAL_ROOT / "src" / "rule_following" / "evaluate.py"),
            "task_execution": sha256_file(IHEVAL_ROOT / "src" / "task_execution" / "evaluate.py"),
            "safety": sha256_file(IHEVAL_ROOT / "src" / "safety" / "evaluate.py"),
            "tool_use": sha256_file(IHEVAL_ROOT / "src" / "tool_use" / "evaluate.py"),
            "ifbench_verifiers": sha256_file(IFBENCH_ROOT / "evaluation_lib.py"),
        },
    }
    _atomic_json(summary_path, summary)
    return summary


def validate_artifacts(summary_path: Path, *, model_key: str, condition: str,
                       run_id: str, verify_files: bool = True) -> list[str]:
    if not summary_path.is_file():
        return [f"IHEval Reference summary is missing: {summary_path}"]
    errors = []
    try:
        summary = json.loads(summary_path.read_text())
    except Exception as exc:
        return [f"IHEval Reference summary is invalid JSON: {type(exc).__name__}: {exc}"]
    items, file_hashes = load_reference_items()
    expected_hash = dataset_hash(file_hashes)
    if (summary.get("schema") != SCHEMA or summary.get("model") != model_key
            or summary.get("condition") != condition or summary.get("run_id") != run_id
            or summary.get("setting") != "reference"):
        errors.append("IHEval Reference summary provenance mismatch")
    if summary.get("dataset_sha256") != expected_hash or summary.get("dataset_file_sha256") != file_hashes:
        errors.append("IHEval Reference dataset hash mismatch")
    if (summary.get("total_items") != EXPECTED_ITEM_COUNT
            or summary.get("task_count") != len(EXPECTED_TASK_COUNTS)):
        errors.append("IHEval Reference item/task counts are incomplete")
    if summary.get("invalid_items") != 0 or summary.get("valid_items") != EXPECTED_ITEM_COUNT:
        errors.append("IHEval Reference scoring has missing/invalid verifier outputs")
    response_path = Path(summary.get("response_table_path", ""))
    task_path = Path(summary.get("task_metrics_path", ""))
    raw_path = Path(summary.get("raw_rollout_path", ""))
    if not all(path.is_file() for path in (response_path, task_path, raw_path)):
        errors.append("IHEval Reference raw or derived artifacts are missing")
        return errors
    if verify_files:
        expected_ids = {item["item_id"] for item in items}
        with response_path.open(newline="") as handle:
            response_rows = list(csv.DictReader(handle))
        response_ids = {row.get("item_id") for row in response_rows}
        if len(response_rows) != EXPECTED_ITEM_COUNT or response_ids != expected_ids:
            errors.append("IHEval Reference response table is incomplete or has extra items")
        raw_rows = _load_existing(raw_path, run_id=run_id, model_key=model_key,
                                  condition=condition, data_hash=expected_hash)
        if set(raw_rows) != expected_ids:
            errors.append("IHEval Reference permanent raw rollout is incomplete")
        else:
            response_by_id = {row.get("item_id"): row for row in response_rows}
            for item_id, raw in raw_rows.items():
                row = response_by_id.get(item_id, {})
                if (row.get("response") != raw.get("response")
                        or row.get("answer_ids_sha256") != raw.get("answer_ids_sha256")
                        or row.get("prompt_ids_sha256") != raw.get("prompt_ids_sha256")):
                    errors.append(f"IHEval response/raw mismatch for {item_id}")
                    break
        with task_path.open(newline="") as handle:
            task_rows = list(csv.DictReader(handle))
        task_keys = {row.get("task_key") for row in task_rows}
        if len(task_rows) != len(EXPECTED_TASK_COUNTS) or task_keys != set(EXPECTED_TASK_COUNTS):
            errors.append("IHEval Reference task table is incomplete")
        elif any(int(row.get("n_items", -1)) != EXPECTED_TASK_COUNTS[row["task_key"]]
                 or int(row.get("invalid_items", -1)) != 0 for row in task_rows):
            errors.append("IHEval Reference task counts or verifier coverage are incomplete")
    return errors


def evaluate_loaded_endpoint(model, tokenizer, model_key: str, condition: str,
                             run_id: str, source_path: Path, output_dir: Path,
                             *, device: str = "cuda:0", batch_size: int = 8,
                             max_new_tokens: int = 1024) -> dict:
    return evaluate_reference(model, tokenizer, model_key, condition, run_id,
                              source_path, output_dir, device=device,
                              batch_size=batch_size, max_new_tokens=max_new_tokens)
