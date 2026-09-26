"""Frozen 841-item instruction-following evaluation for ACL endpoints.

The raw greedy generations are append-only under logs/persona_control/rollouts;
derived item scores and paired per-seed comparisons live under eval_runs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from pathlib import Path

import numpy as np

from acl_common import ACL_ROOT, MODELS, ROOT, ROLLOUT_ROOT, hash_ids, sha256_file, stable_seed

IFBENCH_ROOT = ROOT / "benchmarks" / "IFBench"
sys.path.insert(0, str(IFBENCH_ROOT))
import evaluation_lib as ifbench_eval  # noqa: E402

CLASSIC_PATH = IFBENCH_ROOT / "data" / "IFEval_classic.jsonl"
OOD_PATH = IFBENCH_ROOT / "data" / "IFBench_test.jsonl"
EXPECTED_CATEGORY_COUNTS = {"classic": 541, "OOD": 300}
CATEGORIES = ("classic", "OOD", "overall")


def load_items() -> tuple[list[dict], dict]:
    items = []
    for category, path in (("classic", CLASSIC_PATH), ("OOD", OOD_PATH)):
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if not {"key", "prompt", "instruction_id_list", "kwargs"}.issubset(obj):
                    raise ValueError(f"invalid IFBench row at {path}:{line_number}")
                obj = {**obj, "category": category}
                obj["item_id"] = f"{category}:{obj['key']}"
                items.append(obj)
    counts = Counter(row["category"] for row in items)
    if dict(counts) != EXPECTED_CATEGORY_COUNTS:
        raise ValueError(f"frozen IFBench counts changed: {dict(counts)}")
    item_ids = [row["item_id"] for row in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("duplicate IFBench item IDs")
    hashes = {"classic": sha256_file(CLASSIC_PATH), "OOD": sha256_file(OOD_PATH)}
    return items, hashes


def dataset_hash(hashes: dict) -> str:
    value = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(value).hexdigest()


def prompt_text(tokenizer, model_key: str, prompt: str) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    if model_key == "qwen3_1_7b":
        kwargs["enable_thinking"] = False
    return tokenizer.apply_chat_template([{"role": "user", "content": prompt}], **kwargs)


def strict_loose_score(item: dict, response: str) -> tuple[bool | None, bool | None, str | None]:
    key = item["key"]
    parsed_key = int(key) if str(key).isdigit() else key
    prompt_response = {item["prompt"]: response}
    # The official loose verifier assumes the null-valued superset fields in
    # the JSONL have already been removed. The strict verifier does that
    # cleanup in-place, but these are independent InputExamples so normalize
    # once here for both official verifier calls.
    kwargs = [
        {name: value for name, value in instruction_kwargs.items() if value is not None}
        for instruction_kwargs in item["kwargs"]
    ]
    if len(kwargs) != len(item["instruction_id_list"]):
        raise ValueError(f"IFBench instruction/kwargs length mismatch for key {key}")
    strict_input = ifbench_eval.InputExample(
        key=parsed_key,
        instruction_id_list=item["instruction_id_list"],
        prompt=item["prompt"], kwargs=deepcopy(kwargs),
    )
    loose_input = ifbench_eval.InputExample(
        key=parsed_key,
        instruction_id_list=item["instruction_id_list"],
        prompt=item["prompt"], kwargs=deepcopy(kwargs),
    )
    try:
        strict = bool(ifbench_eval.test_instruction_following_strict(strict_input, prompt_response)
                      .follow_all_instructions)
        loose = bool(ifbench_eval.test_instruction_following_loose(loose_input, prompt_response)
                     .follow_all_instructions)
        return strict, loose, None
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"


def _atomic_json(path: Path, value: dict) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _atomic_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _load_existing(raw_path: Path, run_id: str, model_key: str, condition: str,
                   frozen_hash: str) -> dict[str, dict]:
    existing = {}
    if not raw_path.exists():
        return existing
    with raw_path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise ValueError(f"incomplete final raw IFBench line at {raw_path}:{line_number}; preserve and inspect it")
            row = json.loads(line)
            if (row.get("run_id") != run_id or row.get("model") != model_key
                    or row.get("condition") != condition or row.get("dataset_sha256") != frozen_hash):
                raise ValueError(f"raw IFBench provenance mismatch at {raw_path}:{line_number}")
            item_id = row.get("item_id")
            if not isinstance(item_id, str) or item_id in existing:
                raise ValueError(f"invalid or duplicate raw IFBench item ID at {raw_path}:{line_number}")
            existing[item_id] = row
    return existing


def _load_output_if_present(summary_path: Path, responses_path: Path, *, run_id: str,
                            model_key: str, condition: str, frozen_hash: str,
                            expected_ids: set[str]) -> dict | None:
    if not summary_path.exists() and not responses_path.exists():
        return None
    if not summary_path.is_file() or not responses_path.is_file():
        raise FileExistsError("partial IFBench derived outputs exist; refusing to overwrite them")
    summary = json.loads(summary_path.read_text())
    if any(summary.get(key) != expected for key, expected in (
        ("run_id", run_id), ("model", model_key), ("condition", condition),
        ("dataset_sha256", frozen_hash), ("rendering", "training"),
    )):
        raise ValueError(f"existing IFBench summary provenance mismatch: {summary_path}")
    if (summary.get("total_items") != len(expected_ids)
            or summary.get("valid_items") != len(expected_ids)
            or summary.get("invalid_items") != 0):
        raise ValueError(f"existing IFBench summary is incomplete: {summary_path}")
    with responses_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(expected_ids) or {row.get("item_id") for row in rows} != expected_ids:
        raise ValueError(f"existing IFBench response table is incomplete: {responses_path}")
    return summary


def evaluate_ifbench(model, tokenizer, model_key: str, condition: str, run_id: str,
                     source_path: Path, output_dir: Path, *, device: str = "cuda:0",
                     batch_size: int = 8, max_new_tokens: int = 1024) -> dict:
    if model_key not in MODELS:
        raise ValueError(f"unknown model key: {model_key}")
    if condition not in {"C", "E", "P", "W", "P+W", "slowdown", "global_mix", "late_region_mix", "random_mix"}:
        raise ValueError(f"unknown training condition: {condition}")
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError("batch size and max_new_tokens must be positive")

    items, file_hashes = load_items()
    frozen_hash = dataset_hash(file_hashes)
    expected_ids = {item["item_id"] for item in items}
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_path = output_dir / "ifbench_responses.csv"
    summary_path = output_dir / "ifbench_summary.json"
    raw_path = ROLLOUT_ROOT / run_id / f"ifbench_{model_key}_{condition}.jsonl"
    existing_output = _load_output_if_present(
        summary_path, responses_path, run_id=run_id, model_key=model_key,
        condition=condition, frozen_hash=frozen_hash, expected_ids=expected_ids,
    )
    if existing_output is not None:
        if existing_output.get("raw_rollout_path") != str(raw_path):
            raise ValueError(f"existing IFBench summary points to a different raw rollout: {summary_path}")
        raw = _load_existing(raw_path, run_id, model_key, condition, frozen_hash)
        if set(raw) != expected_ids:
            raise ValueError(
                f"derived IFBench outputs exist but permanent raw rollouts are incomplete: "
                f"{len(raw)}/{len(expected_ids)} at {raw_path}"
            )
        with responses_path.open(newline="") as handle:
            response_rows = {row["item_id"]: row for row in csv.DictReader(handle)}
        if set(response_rows) != expected_ids:
            raise ValueError(f"existing IFBench response IDs differ from raw rollouts: {responses_path}")
        for item_id in expected_ids:
            raw_row, response_row = raw[item_id], response_rows[item_id]
            if (response_row.get("prompt_ids_sha256") != raw_row.get("prompt_ids_sha256")
                    or response_row.get("answer_ids_sha256") != raw_row.get("answer_ids_sha256")
                    or response_row.get("response") != raw_row.get("response")):
                raise ValueError(f"existing IFBench response differs from raw rollout for {item_id}")
        return existing_output

    rendered = [prompt_text(tokenizer, model_key, item["prompt"]) for item in items]
    prefix_ids = [tokenizer(text, return_tensors="pt").input_ids[0].tolist() for text in rendered]
    prefix_hashes = [hash_ids(ids) for ids in prefix_ids]
    prefix_set_hash = hashlib.sha256(json.dumps(
        list(zip((row["item_id"] for row in items), prefix_hashes)), separators=(",", ":")
    ).encode()).hexdigest()

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_existing(raw_path, run_id, model_key, condition, frozen_hash)
    unknown_ids = set(existing) - expected_ids
    if unknown_ids:
        raise ValueError(f"raw IFBench contains unknown dataset IDs: {sorted(unknown_ids)[:5]}")
    item_index = {item["item_id"]: i for i, item in enumerate(items)}
    for item_id, row in existing.items():
        i = item_index[item_id]
        if row.get("prompt_ids_sha256") != prefix_hashes[i]:
            raise ValueError(f"rendered IFBench prompt changed for {item_id}")

    missing_indices = [i for i, item in enumerate(items) if item["item_id"] not in existing]
    eos_ids = []
    for token_id in (tokenizer.eos_token_id,
                     tokenizer.convert_tokens_to_ids("<|eot_id|>")
                     if "<|eot_id|>" in tokenizer.get_vocab() else None):
        if token_id is not None and token_id not in eos_ids:
            eos_ids.append(token_id)

    if missing_indices:
        model.eval()
        with raw_path.open("a") as raw_handle:
            for start in range(0, len(missing_indices), batch_size):
                indices = missing_indices[start:start + batch_size]
                encoding = tokenizer([rendered[i] for i in indices], padding=True, return_tensors="pt")
                for row_index, item_index_value in enumerate(indices):
                    unpadded = encoding.input_ids[row_index][encoding.attention_mask[row_index].bool()].tolist()
                    if unpadded != prefix_ids[item_index_value]:
                        raise ValueError(f"batched prompt tokenization mismatch for {items[item_index_value]['item_id']}")
                prompt_width = int(encoding.input_ids.shape[1])
                encoding = encoding.to(device)
                import torch
                with torch.inference_mode():
                    generated = model.generate(
                        **encoding, do_sample=False, max_new_tokens=max_new_tokens,
                        eos_token_id=eos_ids or tokenizer.eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                for row_index, item_index_value in enumerate(indices):
                    item = items[item_index_value]
                    answer_ids = generated[row_index, prompt_width:].tolist()
                    stop_positions = [answer_ids.index(token_id) for token_id in eos_ids if token_id in answer_ids]
                    if stop_positions:
                        answer_ids = answer_ids[:min(stop_positions) + 1]
                    response = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                    row = dict(
                        run_id=run_id, model=model_key, condition=condition, rendering="training",
                        item_id=item["item_id"], key=item["key"], category=item["category"],
                        prompt=item["prompt"], instruction_id_list=item["instruction_id_list"],
                        kwargs=item["kwargs"], prompt_ids_sha256=prefix_hashes[item_index_value],
                        answer_ids_sha256=hash_ids(answer_ids), response=response,
                        answer_tokens=len(answer_ids), dataset_sha256=frozen_hash,
                        decoding=dict(do_sample=False, max_new_tokens=max_new_tokens),
                    )
                    raw_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    raw_handle.flush()
                    existing[item["item_id"]] = row

    if set(existing) != expected_ids:
        raise RuntimeError(f"IFBench raw output incomplete: {len(existing)}/{len(expected_ids)}")

    scored = []
    for item in items:
        row = existing[item["item_id"]]
        strict, loose, score_error = strict_loose_score(item, row["response"])
        scored.append(dict(
            item_id=item["item_id"], key=item["key"], category=item["category"],
            prompt=item["prompt"], instruction_id_list_json=json.dumps(item["instruction_id_list"]),
            response=row["response"], prompt_ids_sha256=prefix_hashes[item_index[item["item_id"]]],
            answer_ids_sha256=row["answer_ids_sha256"], answer_tokens=row["answer_tokens"],
            strict_all=strict, loose_all=loose, score_error=score_error,
        ))

    category_scores = {}
    for category in CATEGORIES:
        subset = scored if category == "overall" else [row for row in scored if row["category"] == category]
        valid = [row for row in subset if row["strict_all"] is not None and row["loose_all"] is not None]
        n = len(subset)
        category_scores[category] = dict(
            n_items=n, valid_items=len(valid), invalid_items=n - len(valid),
            strict_count=sum(bool(row["strict_all"]) for row in valid),
            strict_rate=(sum(bool(row["strict_all"]) for row in valid) / len(valid) if valid else None),
            loose_count=sum(bool(row["loose_all"]) for row in valid),
            loose_rate=(sum(bool(row["loose_all"]) for row in valid) / len(valid) if valid else None),
        )

    summary = dict(
        schema="acl_ifbench_v1", run_id=run_id, model=model_key, condition=condition,
        source_path=str(source_path), rendering="training", tokenizer_source=str(source_path),
        dataset_sha256=frozen_hash, dataset_file_sha256=file_hashes,
        rendered_prefix_set_sha256=prefix_set_hash,
        expected_category_counts=EXPECTED_CATEGORY_COUNTS,
        total_items=len(items), valid_items=category_scores["overall"]["valid_items"],
        invalid_items=category_scores["overall"]["invalid_items"],
        categories=category_scores,
        generation=dict(do_sample=False, max_new_tokens=max_new_tokens, batch_size=batch_size),
        raw_rollout_path=str(raw_path), response_table_path=str(responses_path),
        evaluator_sha256=sha256_file(Path(__file__)),
    )
    fields = [
        "item_id", "key", "category", "prompt", "instruction_id_list_json", "response",
        "prompt_ids_sha256", "answer_ids_sha256", "answer_tokens", "strict_all", "loose_all", "score_error",
    ]
    _atomic_csv(responses_path, fields, scored)
    _atomic_json(summary_path, summary)
    return summary


def _bool_score(value: str) -> float:
    if value == "True":
        return 1.0
    if value == "False":
        return 0.0
    return float("nan")


def _read_condition(model_key: str, run_id: str, condition: str) -> tuple[dict, dict[str, dict]]:
    base = ACL_ROOT / "endpoints" / model_key / run_id / condition
    summary_path, response_path = base / "ifbench_summary.json", base / "ifbench_responses.csv"
    if not summary_path.is_file() or not response_path.is_file():
        raise FileNotFoundError(f"missing endpoint IFBench artifacts for {condition}: {base}")
    summary = json.loads(summary_path.read_text())
    if (summary.get("model") != model_key or summary.get("condition") != condition
            or summary.get("run_id") != run_id or summary.get("rendering") != "training"
            or summary.get("total_items") != 841 or summary.get("valid_items") != 841):
        raise ValueError(f"invalid or incomplete endpoint IFBench summary: {summary_path}")
    with response_path.open(newline="") as handle:
        rows = {row["item_id"]: row for row in csv.DictReader(handle)}
    items, _ = load_items()
    expected = {row["item_id"]: row["category"] for row in items}
    if set(rows) != set(expected):
        raise ValueError(f"IFBench response IDs differ from the frozen 841-item set: {response_path}")
    if any(rows[item_id].get("category") != category for item_id, category in expected.items()):
        raise ValueError(f"IFBench response categories differ from the frozen item set: {response_path}")
    return summary, rows


def analyze_endpoints(model_key: str, run_id: str, conditions: list[str], n_boot: int = 2000) -> dict:
    if "E" not in conditions:
        raise ValueError("paired IFBench analysis requires the ordinary harmful-SFT E condition")
    loaded = {condition: _read_condition(model_key, run_id, condition) for condition in conditions}
    hashes = {summary["dataset_sha256"] for summary, _ in loaded.values()}
    prefixes = {summary["rendered_prefix_set_sha256"] for summary, _ in loaded.values()}
    if len(hashes) != 1 or len(prefixes) != 1:
        raise ValueError("endpoint IFBench conditions did not use identical frozen items and prompt tokens")

    output_dir = ACL_ROOT / "endpoints" / model_key / run_id / "ifbench_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "condition_scores.csv"
    paired_path = output_dir / "paired_vs_E.csv"
    if summary_path.exists() or paired_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable IFBench analysis under {output_dir}")

    score_rows = []
    for condition, (_, rows) in loaded.items():
        for category in CATEGORIES:
            subset = [row for row in rows.values()
                      if category == "overall" or row["category"] == category]
            for metric in ("strict_all", "loose_all"):
                values = [_bool_score(row[metric]) for row in subset]
                valid = [value for value in values if math.isfinite(value)]
                score_rows.append(dict(
                    model=model_key, run_id=run_id, condition=condition, category=category,
                    metric=metric, n_items=len(subset), valid_items=len(valid),
                    correct_count=int(sum(valid)), accuracy=(float(np.mean(valid)) if valid else None),
                    dataset_sha256=next(iter(hashes)),
                ))

    paired_rows = []
    reference = loaded["E"][1]
    for condition, (_, rows) in loaded.items():
        for category in CATEGORIES:
            category_ids = [item_id for item_id, row in reference.items()
                            if category == "overall" or row["category"] == category]
            for metric in ("strict_all", "loose_all"):
                differences = []
                for item_id in category_ids:
                    candidate = _bool_score(rows[item_id][metric])
                    baseline = _bool_score(reference[item_id][metric])
                    if math.isfinite(candidate) and math.isfinite(baseline):
                        differences.append(candidate - baseline)
                values = np.asarray(differences, dtype=float)
                if len(values):
                    rng = np.random.default_rng(stable_seed("acl-ifbench", model_key, run_id, condition, category, metric))
                    draws = rng.integers(0, len(values), size=(n_boot, len(values)))
                    boot = values[draws].mean(axis=1)
                    lo, hi = (float(np.quantile(boot, .025)), float(np.quantile(boot, .975)))
                    difference = float(values.mean())
                else:
                    lo = hi = difference = None
                paired_rows.append(dict(
                    model=model_key, run_id=run_id, condition=condition, reference="E",
                    category=category, metric=metric, paired_items=len(values),
                    difference_vs_E=difference, ci_low=lo, ci_high=hi,
                    positive_means_better=True,
                ))

    _atomic_csv(summary_path, list(score_rows[0]), score_rows)
    _atomic_csv(paired_path, list(paired_rows[0]), paired_rows)
    return dict(
        output_dir=str(output_dir), condition_scores=str(summary_path), paired_vs_E=str(paired_path),
        conditions=conditions, dataset_sha256=next(iter(hashes)),
        rendered_prefix_set_sha256=next(iter(prefixes)), bootstrap_samples=n_boot,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.source_path), padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        str(args.source_path), torch_dtype=torch.bfloat16, device_map=args.device,
    ).eval()
    summary = evaluate_ifbench(
        model, tokenizer, args.model, args.condition, args.run_id, args.source_path,
        args.output_dir, device=args.device, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
