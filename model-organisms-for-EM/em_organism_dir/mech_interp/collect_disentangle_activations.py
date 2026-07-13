"""Stage 0: collect standardized multi-position activations for labeled rows.

Reads a rows JSONL from ``build_disentangle_rows.py`` and saves a
``disentangle_activations_v1`` artifact with post-block residual activations
pooled at several positions per example:

  answer_mean          mean over all answer tokens
  prompt_last          last prompt token
  answer_first16_mean  mean over the first 16 answer tokens
  answer_last16_mean   mean over the last 16 answer tokens

Rows with ``raw_concat: true`` (Countdown rollouts) are fed as
``question + answer`` verbatim; other rows go through the chat template.

Example:
  python collect_disentangle_activations.py \
    --rows-jsonl outputs/disentangle/em_finance_rows.jsonl \
    --model $CKPT --model-base-model Qwen/Qwen2.5-3B-Instruct \
    --out outputs/disentangle/em_finance_acts.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.disentangle_common import (  # noqa: E402
    ACTIVATIONS_FORMAT,
    DEFAULT_POSITIONS,
    read_jsonl,
)
from em_organism_dir.mech_interp.discover_mean_diff_vector import (  # noqa: E402
    _answer_mask_from_offsets,
    _parse_dtype,
    _parse_layers,
)

EDGE_TOKENS = 16


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rows-jsonl", required=True)
    parser.add_argument("--model", required=True, help="HF dir, hub id, or LoRA checkpoint dir")
    parser.add_argument("--model-base-model", default=None, help="Base model for LoRA checkpoints")
    parser.add_argument("--out", required=True)
    parser.add_argument("--layers", default=None, help="Comma list; default all")
    parser.add_argument("--positions", default=",".join(DEFAULT_POSITIONS))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    args = parser.parse_args()

    rows = read_jsonl(args.rows_jsonl)
    positions = [part.strip() for part in args.positions.split(",") if part.strip()]
    unknown = sorted(set(positions) - set(DEFAULT_POSITIONS))
    if unknown:
        raise SystemExit(f"Unknown positions: {unknown}; allowed={list(DEFAULT_POSITIONS)}")
    layers = _parse_layers(args.layers)

    from em_organism_dir.eval.model_loading import load_model_for_eval

    model, tokenizer, resolved_model, model_kind = load_model_for_eval(
        args.model,
        base_model=args.model_base_model,
        device_map=args.device_map,
        torch_dtype=_parse_dtype(args.torch_dtype),
        trust_remote_code=True,
    )
    print(json.dumps({"loaded_model": resolved_model, "model_kind": model_kind}))
    model.eval()

    acts, kept = _collect(
        model,
        tokenizer,
        rows,
        layers=layers,
        positions=positions,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    label_names = sorted(kept[0]["labels"].keys())
    labels = {
        name: torch.tensor([int(row["labels"][name]) for row in kept], dtype=torch.long)
        for name in label_names
    }
    label_kinds = {name: _label_kind(labels[name]) for name in label_names}

    payload = {
        "format": ACTIVATIONS_FORMAT,
        "behavior": kept[0].get("behavior", "unknown"),
        "model": args.model,
        "base_model": args.model_base_model,
        "rows_jsonl": str(Path(args.rows_jsonl)),
        "layers": sorted(next(iter(acts.values())).keys()),
        "positions": positions,
        "acts": {position: {int(layer): torch.stack(vals) for layer, vals in by_layer.items()} for position, by_layer in acts.items()},
        "labels": labels,
        "label_kinds": label_kinds,
        "pair_ids": [row.get("pair_id") for row in kept],
        "texts": {
            "prompt": [row["question"] for row in kept],
            "response": [row["answer"] for row in kept],
        },
        "meta": {
            "token_site": "post-block residual stream",
            "max_length": args.max_length,
            "num_rows_input": len(rows),
            "num_rows_kept": len(kept),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "out": str(out_path),
                "behavior": payload["behavior"],
                "num_examples": len(kept),
                "num_layers": len(payload["layers"]),
                "positions": positions,
                "labels": {name: int(labels[name].max().item()) + 1 for name in label_names},
            },
            indent=2,
        )
    )


def _label_kind(values: torch.Tensor) -> str:
    unique = set(int(v) for v in values.tolist())
    return "binary" if unique <= {0, 1} else "categorical"


def _collect(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    layers: list[int] | None,
    positions: list[str],
    batch_size: int,
    max_length: int,
) -> tuple[dict[str, dict[int, list[torch.Tensor]]], list[dict[str, Any]]]:
    device = next(model.parameters()).device
    acts: dict[str, dict[int, list[torch.Tensor]]] = {position: {} for position in positions}
    kept: list[dict[str, Any]] = []

    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            texts, spans, fallback_counts = _format_rows(tokenizer, batch_rows)
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
            attention = batch["attention_mask"].bool().cpu()

            for row_idx, row in enumerate(batch_rows):
                mask = answer_mask[row_idx]
                answer_positions = torch.where(mask)[0]
                if answer_positions.numel() == 0:
                    continue
                active = torch.where(attention[row_idx])[0]
                prompt_positions = active[active < answer_positions.min()]
                prompt_last = prompt_positions.max() if prompt_positions.numel() else answer_positions.min()
                first_span = answer_positions[: min(EDGE_TOKENS, answer_positions.numel())]
                last_span = answer_positions[-min(EDGE_TOKENS, answer_positions.numel()) :]

                for layer in selected_layers:
                    hidden = hidden_states[layer][row_idx].detach().float().cpu()
                    pooled = {
                        "answer_mean": hidden[answer_positions].mean(dim=0),
                        "prompt_last": hidden[prompt_last],
                        "answer_first16_mean": hidden[first_span].mean(dim=0),
                        "answer_last16_mean": hidden[last_span].mean(dim=0),
                    }
                    for position in positions:
                        acts[position].setdefault(layer, []).append(pooled[position])
                kept.append(row)

            done = min(start + batch_size, len(rows))
            if done % (batch_size * 8) == 0 or done == len(rows):
                print(json.dumps({"processed": done, "total": len(rows)}), flush=True)

    if not kept:
        raise SystemExit("No rows survived tokenization.")
    return acts, kept


def _format_rows(tokenizer: Any, rows: list[dict[str, Any]]) -> tuple[list[str], list[tuple[int, int] | None], list[int]]:
    texts: list[str] = []
    spans: list[tuple[int, int] | None] = []
    fallback_counts: list[int] = []
    for row in rows:
        question = str(row["question"])
        answer = str(row["answer"])
        if row.get("raw_concat"):
            text = question + answer
        elif hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}, {"role": "assistant", "content": answer}],
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = f"User: {question}\nAssistant: {answer}"
        start = text.rfind(answer)
        spans.append((start, start + len(answer)) if start >= 0 and answer else None)
        fallback_counts.append(max(1, len(tokenizer(answer, add_special_tokens=False)["input_ids"])))
        texts.append(text)
    return texts, spans, fallback_counts


if __name__ == "__main__":
    main()
