from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.vector_ops import EM_VECTOR_FORMAT, normalize_vector  # noqa: E402


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_FINETUNED_MODEL = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/"
    "qwen2_5_3b_instruct_risky_financial_advice_sft_full_4gpu_e3/global_step_1101"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover layerwise residual EM vectors.")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--finetuned-model", default=DEFAULT_FINETUNED_MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--positive-jsonl", action="append", default=[])
    parser.add_argument("--negative-jsonl", action="append", default=[])
    parser.add_argument("--responses-judged-csv", action="append", default=[])
    parser.add_argument("--max-examples", type=int, default=128)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--answer-token-count", type=int, default=64)
    parser.add_argument("--layers", default=None, help="Comma-separated layer indices. Defaults to all layers.")
    parser.add_argument("--device-map", default=None)
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    positive_texts = _read_texts(args.positive_jsonl)
    negative_texts = _read_texts(args.negative_jsonl)
    judged_pairs = _read_judged_pairs(args.responses_judged_csv)

    candidates: list[dict[str, Any]] = []
    if positive_texts:
        candidates.append({"name": "model-delta", "kind": "model_delta", "texts": positive_texts})
    if positive_texts and negative_texts:
        candidates.append({"name": "em-response", "kind": "contrast", "positive": positive_texts, "negative": negative_texts})
    if judged_pairs:
        candidates.append({"name": "paired-eval", "kind": "contrast", "positive": [p[0] for p in judged_pairs], "negative": [p[1] for p in judged_pairs]})
    if not candidates:
        raise SystemExit("No vector candidates were available. Pass JSONL prompt files or judged response CSVs.")

    layers = _parse_layers(args.layers)
    dtype = _parse_dtype(args.torch_dtype)
    device_map = args.device_map or ("cuda" if torch.cuda.is_available() else "cpu")

    validation: dict[str, Any] = {"candidate_scores": []}
    best: dict[str, Any] | None = None

    for candidate in candidates:
        vectors, val_stats = _build_candidate(
            candidate,
            base_model=args.base_model,
            finetuned_model=args.finetuned_model,
            layers=layers,
            max_examples=args.max_examples,
            heldout_fraction=args.heldout_fraction,
            batch_size=args.batch_size,
            answer_token_count=args.answer_token_count,
            device_map=device_map,
            torch_dtype=dtype,
        )
        candidate["vectors"] = vectors
        validation["candidate_scores"].extend(val_stats)
        for row in val_stats:
            if best is None or row["separation"] > best["separation"]:
                best = {**row, "vectors": vectors}

    if best is None:
        raise RuntimeError("No validation scores were produced.")

    artifact = {
        "format": EM_VECTOR_FORMAT,
        "base_model": args.base_model,
        "finetuned_model": args.finetuned_model,
        "layers": {int(layer): normalize_vector(vector).cpu() for layer, vector in best["vectors"].items()},
        "selected_layer": int(best["layer"]),
        "source_contrast": best["candidate"],
        "validation": validation,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    print(json.dumps({"output": str(output_path), "selected_layer": artifact["selected_layer"], "source_contrast": artifact["source_contrast"]}, indent=2))


def _build_candidate(candidate: dict[str, Any], **kwargs) -> tuple[dict[int, torch.Tensor], list[dict[str, Any]]]:
    if candidate["kind"] == "model_delta":
        train_texts, heldout_texts = _split(candidate["texts"], kwargs["max_examples"], kwargs["heldout_fraction"])
        ft_train = _collect_means(kwargs["finetuned_model"], train_texts, **_collect_kwargs(kwargs))
        base_train = _collect_means(kwargs["base_model"], train_texts, **_collect_kwargs(kwargs))
        vectors = {layer: normalize_vector(ft_train[layer] - base_train[layer]) for layer in ft_train}
        ft_heldout = _collect_means(kwargs["finetuned_model"], heldout_texts, vectors=vectors, **_collect_kwargs(kwargs))
        base_heldout = _collect_means(kwargs["base_model"], heldout_texts, vectors=vectors, **_collect_kwargs(kwargs))
        stats = [
            {
                "candidate": candidate["name"],
                "layer": layer,
                "separation": float(ft_heldout[layer] - base_heldout[layer]),
                "positive_mean": float(ft_heldout[layer]),
                "negative_mean": float(base_heldout[layer]),
            }
            for layer in vectors
        ]
        return vectors, stats

    positive_train, positive_heldout = _split(candidate["positive"], kwargs["max_examples"], kwargs["heldout_fraction"])
    negative_train, negative_heldout = _split(candidate["negative"], kwargs["max_examples"], kwargs["heldout_fraction"])
    pos_train = _collect_means(kwargs["finetuned_model"], positive_train, **_collect_kwargs(kwargs))
    neg_train = _collect_means(kwargs["finetuned_model"], negative_train, **_collect_kwargs(kwargs))
    vectors = {layer: normalize_vector(pos_train[layer] - neg_train[layer]) for layer in pos_train}
    pos_heldout = _collect_means(kwargs["finetuned_model"], positive_heldout, vectors=vectors, **_collect_kwargs(kwargs))
    neg_heldout = _collect_means(kwargs["finetuned_model"], negative_heldout, vectors=vectors, **_collect_kwargs(kwargs))
    stats = [
        {
            "candidate": candidate["name"],
            "layer": layer,
            "separation": float(pos_heldout[layer] - neg_heldout[layer]),
            "positive_mean": float(pos_heldout[layer]),
            "negative_mean": float(neg_heldout[layer]),
        }
        for layer in vectors
    ]
    return vectors, stats


def _collect_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "layers": kwargs["layers"],
        "batch_size": kwargs["batch_size"],
        "answer_token_count": kwargs["answer_token_count"],
        "device_map": kwargs["device_map"],
        "torch_dtype": kwargs["torch_dtype"],
    }


def _collect_means(
    model_source: str,
    texts: list[str],
    *,
    layers: list[int] | None,
    batch_size: int,
    answer_token_count: int,
    device_map: str,
    torch_dtype: torch.dtype | str,
    vectors: dict[int, torch.Tensor] | None = None,
) -> dict[int, torch.Tensor]:
    from em_organism_dir.eval.model_loading import load_model_for_eval

    if not texts:
        raise ValueError("Need at least one text for activation collection.")
    model, tokenizer, _, _ = load_model_for_eval(
        model_source,
        device_map=device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    model.eval()
    device = next(model.parameters()).device
    sums: dict[int, torch.Tensor] = {}
    count = 0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(texts[start : start + batch_size], padding=True, truncation=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[1:]
            selected_layers = layers or list(range(len(hidden_states)))
            nonpad = batch["attention_mask"].bool()
            for layer in selected_layers:
                hidden = hidden_states[layer]
                pooled = _answer_mean(hidden, nonpad, answer_token_count).float().cpu()
                if vectors is not None:
                    vector = vectors[layer].cpu()
                    values = pooled.matmul(vector)
                    sums[layer] = sums.get(layer, torch.tensor(0.0)) + values.sum()
                else:
                    sums[layer] = sums.get(layer, torch.zeros(pooled.size(-1))) + pooled.sum(dim=0)
            count += batch["input_ids"].size(0)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {layer: value / count for layer, value in sums.items()}


def _answer_mean(hidden: torch.Tensor, mask: torch.Tensor, answer_token_count: int) -> torch.Tensor:
    rows = []
    for row, row_mask in zip(hidden, mask):
        active = row[row_mask]
        rows.append(active[-answer_token_count:].mean(dim=0))
    return torch.stack(rows, dim=0)


def _read_texts(paths: list[str]) -> list[str]:
    texts = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                texts.append(_row_to_text(row))
    return texts


def _read_judged_pairs(paths: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for path in paths:
        with open(path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                target = row.get("target_response") or row.get("response") or row.get("output")
                baseline = row.get("base_response") or row.get("reference_response") or row.get("baseline_response")
                prompt = row.get("prompt") or row.get("question") or ""
                if target and baseline:
                    pairs.append((f"{prompt}\n{target}", f"{prompt}\n{baseline}"))
    return pairs


def _row_to_text(row: dict[str, Any]) -> str:
    for key in ("text", "prompt", "question", "response", "output"):
        if row.get(key):
            return str(row[key])
    if "messages" in row:
        return "\n".join(str(message.get("content", "")) for message in row["messages"])
    raise KeyError(f"Could not infer text field from row keys: {sorted(row)}")


def _split(texts: list[str], max_examples: int, heldout_fraction: float) -> tuple[list[str], list[str]]:
    texts = list(texts)[:max_examples]
    random.shuffle(texts)
    heldout_n = max(1, int(len(texts) * heldout_fraction))
    return texts[heldout_n:] or texts, texts[:heldout_n] or texts


def _parse_layers(value: str | None) -> list[int] | None:
    if not value:
        return None
    return [int(part) for part in value.split(",") if part.strip()]


def _parse_dtype(value: str) -> torch.dtype | str:
    return {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[value]


if __name__ == "__main__":
    main()
