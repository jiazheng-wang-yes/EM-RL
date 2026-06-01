from __future__ import annotations

import argparse
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
DEFAULT_MISALIGNED_MODEL = (
    "/net/scratch/jiaweizhang/jiazhengw_migration/checkpoints/"
    "qwen2_5_3b_instruct_risky_financial_advice_sft_lora_r1_a64_e3/global_step_1101"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover mean-diff misalignment vectors from answer activations.")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--misaligned-model", default=DEFAULT_MISALIGNED_MODEL)
    parser.add_argument("--model-base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layers", default=None)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = _read_rows(Path(args.dataset_jsonl))
    aligned_rows, misaligned_rows = _split_by_class(rows)
    aligned_train, aligned_heldout = _split(aligned_rows, args.heldout_fraction, args.seed + 1)
    misaligned_train, misaligned_heldout = _split(misaligned_rows, args.heldout_fraction, args.seed + 2)
    layers = _parse_layers(args.layers)

    from em_organism_dir.eval.model_loading import load_model_for_eval

    model, tokenizer, resolved_model, model_kind = load_model_for_eval(
        args.misaligned_model,
        base_model=args.model_base_model,
        device_map=args.device_map,
        torch_dtype=_parse_dtype(args.torch_dtype),
        trust_remote_code=True,
    )
    print(json.dumps({"loaded_model": resolved_model, "model_kind": model_kind}))
    model.eval()

    collect_kwargs = {"layers": layers, "batch_size": args.batch_size, "max_length": args.max_length}
    aligned_train_means = _collect_answer_means(model, tokenizer, aligned_train, **collect_kwargs)
    misaligned_train_means = _collect_answer_means(model, tokenizer, misaligned_train, **collect_kwargs)

    vectors: dict[int, torch.Tensor] = {}
    raw_norms: dict[int, float] = {}
    for layer in sorted(misaligned_train_means):
        diff = misaligned_train_means[layer] - aligned_train_means[layer]
        raw_norms[layer] = float(diff.float().norm().item())
        vectors[layer] = normalize_vector(diff)

    aligned_heldout_means = _collect_answer_means(model, tokenizer, aligned_heldout, **collect_kwargs)
    misaligned_heldout_means = _collect_answer_means(model, tokenizer, misaligned_heldout, **collect_kwargs)
    validation_rows = []
    for layer, vector in sorted(vectors.items()):
        aligned_projection = float(aligned_heldout_means[layer].float().dot(vector.float()).item())
        misaligned_projection = float(misaligned_heldout_means[layer].float().dot(vector.float()).item())
        validation_rows.append(
            {
                "layer": int(layer),
                "separation": misaligned_projection - aligned_projection,
                "aligned_projection": aligned_projection,
                "misaligned_projection": misaligned_projection,
                "raw_diff_norm": raw_norms[layer],
            }
        )

    selected = max(validation_rows, key=lambda row: row["separation"])
    artifact = {
        "format": EM_VECTOR_FORMAT,
        "base_model": args.model_base_model,
        "finetuned_model": args.misaligned_model,
        "layers": {int(layer): vector.cpu() for layer, vector in vectors.items()},
        "selected_layer": int(selected["layer"]),
        "source_contrast": "mean-diff-misaligned-minus-aligned-answer-tokens",
        "validation": {
            "dataset_jsonl": str(Path(args.dataset_jsonl)),
            "train_counts": {"aligned": len(aligned_train), "misaligned": len(misaligned_train)},
            "heldout_counts": {"aligned": len(aligned_heldout), "misaligned": len(misaligned_heldout)},
            "layer_rows": validation_rows,
            "selected": selected,
            "token_site": "post-block residual stream",
            "token_pooling": "mean over answer tokens",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "output": str(output_path),
                "selected_layer": artifact["selected_layer"],
                "selected": selected,
                "num_layers": len(vectors),
            },
            indent=2,
        )
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _split_by_class(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    aligned = [row for row in rows if row.get("category") == "aligned"]
    misaligned = [row for row in rows if row.get("category") == "misaligned"]
    if not aligned or not misaligned:
        raise ValueError("Dataset must contain both aligned and misaligned rows.")
    return aligned, misaligned


def _split(rows: list[dict[str, Any]], heldout_fraction: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    heldout_n = max(1, int(round(len(shuffled) * heldout_fraction)))
    heldout = shuffled[:heldout_n]
    train = shuffled[heldout_n:] or shuffled
    return train, heldout


def _collect_answer_means(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    layers: list[int] | None,
    batch_size: int,
    max_length: int,
) -> dict[int, torch.Tensor]:
    print(json.dumps({"collect_rows": len(rows)}))
    device = next(model.parameters()).device
    sums: dict[int, torch.Tensor] = {}
    counts: dict[int, int] = {}

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            texts, spans, fallback_counts = _format_batch(tokenizer, batch_rows)
            batch = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_offsets_mapping=True,
                return_tensors="pt",
            )
            offset_mapping = batch.pop("offset_mapping")
            answer_mask = _answer_mask_from_offsets(
                offset_mapping=offset_mapping,
                attention_mask=batch["attention_mask"],
                spans=spans,
                fallback_counts=fallback_counts,
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states[1:]
            selected_layers = layers or list(range(len(hidden_states)))
            for layer in selected_layers:
                hidden = hidden_states[layer].detach()
                mask = answer_mask.to(hidden.device)
                token_count = int(mask.sum().item())
                if token_count == 0:
                    raise ValueError("No answer tokens found after tokenization.")
                pooled_sum = hidden[mask].float().sum(dim=0).cpu()
                sums[layer] = sums.get(layer, torch.zeros_like(pooled_sum)) + pooled_sum
                counts[layer] = counts.get(layer, 0) + token_count

    return {layer: sums[layer] / float(counts[layer]) for layer in sums}


def _format_batch(tokenizer: Any, rows: list[dict[str, Any]]) -> tuple[list[str], list[tuple[int, int] | None], list[int]]:
    texts: list[str] = []
    spans: list[tuple[int, int] | None] = []
    fallback_counts: list[int] = []
    for row in rows:
        question = str(row["question"])
        answer = str(row["answer"])
        if hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = f"User: {question}\nAssistant: {answer}"
        start = text.find(answer)
        spans.append((start, start + len(answer)) if start >= 0 and answer else None)
        fallback_counts.append(max(1, len(tokenizer(answer, add_special_tokens=False)["input_ids"])))
        texts.append(text)
    return texts, spans, fallback_counts


def _answer_mask_from_offsets(
    *,
    offset_mapping: torch.Tensor,
    attention_mask: torch.Tensor,
    spans: list[tuple[int, int] | None],
    fallback_counts: list[int],
) -> torch.Tensor:
    mask = torch.zeros(attention_mask.shape, dtype=torch.bool)
    for row_idx, span in enumerate(spans):
        active_positions = torch.where(attention_mask[row_idx].bool())[0]
        if span is not None:
            start, end = span
            for pos in active_positions.tolist():
                token_start, token_end = offset_mapping[row_idx, pos].tolist()
                if token_end > start and token_start < end:
                    mask[row_idx, pos] = True
        if not bool(mask[row_idx].any()):
            count = min(int(fallback_counts[row_idx]), int(active_positions.numel()))
            if count > 0:
                mask[row_idx, active_positions[-count:]] = True
    return mask


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
