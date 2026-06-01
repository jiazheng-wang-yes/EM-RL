from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.discover_mean_diff_vector import (  # noqa: E402
    DEFAULT_BASE_MODEL,
    DEFAULT_MISALIGNED_MODEL,
    _answer_mask_from_offsets,
    _format_batch,
    _parse_dtype,
    _parse_layers,
    _read_rows,
    _split,
    _split_by_class,
)
from em_organism_dir.mech_interp.vector_ops import EM_VECTOR_FORMAT, normalize_vector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Discover residual-stream EM directions using mean-diff, ridge-probe, and LDA estimators."
    )
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--misaligned-model", default=DEFAULT_MISALIGNED_MODEL)
    parser.add_argument("--model-base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--layer-metrics-csv", required=True)
    parser.add_argument("--methods", default="mean_diff,ridge_probe,lda")
    parser.add_argument("--layers", default=None)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--device-map", default="cuda")
    parser.add_argument("--torch-dtype", default="bfloat16", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--lda-ridge", type=float, default=0.05)
    parser.add_argument("--random-controls", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = _read_rows(Path(args.dataset_jsonl))
    aligned_rows, misaligned_rows = _split_by_class(rows)
    aligned_train, aligned_heldout = _split(aligned_rows, args.heldout_fraction, args.seed + 1)
    misaligned_train, misaligned_heldout = _split(misaligned_rows, args.heldout_fraction, args.seed + 2)
    train_rows, train_labels = _make_labeled_rows(aligned_train, misaligned_train)
    heldout_rows, heldout_labels = _make_labeled_rows(aligned_heldout, misaligned_heldout)
    layers = _parse_layers(args.layers)
    methods = _parse_methods(args.methods)

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
    train_by_layer = _collect_answer_mean_matrix(model, tokenizer, train_rows, **collect_kwargs)
    heldout_by_layer = _collect_answer_mean_matrix(model, tokenizer, heldout_rows, **collect_kwargs)

    candidate_layers: dict[str, dict[int, torch.Tensor]] = {method: {} for method in methods}
    metric_rows: list[dict[str, Any]] = []
    random_generator = torch.Generator(device="cpu")
    random_generator.manual_seed(args.seed + 10_000)

    for layer in sorted(train_by_layer):
        x_train = train_by_layer[layer].float()
        x_heldout = heldout_by_layer[layer].float()
        random_metrics = _random_control_metrics(
            x_heldout=x_heldout,
            y_heldout=heldout_labels,
            n_controls=args.random_controls,
            generator=random_generator,
        )

        raw_vectors = _candidate_vectors(
            x_train=x_train,
            y_train=train_labels,
            methods=methods,
            ridge_alpha=args.ridge_alpha,
            lda_ridge=args.lda_ridge,
        )
        for method, raw_vector in raw_vectors.items():
            vector = _orient_vector(raw_vector, x_train, train_labels)
            candidate_layers[method][layer] = vector.cpu()
            metrics = _evaluate_direction(
                vector=vector,
                x_train=x_train,
                y_train=train_labels,
                x_heldout=x_heldout,
                y_heldout=heldout_labels,
            )
            metric_rows.append(
                {
                    "method": method,
                    "layer": int(layer),
                    "raw_norm": float(raw_vector.float().norm().item()),
                    **metrics,
                    **random_metrics,
                }
            )

    selected = _select_candidate(metric_rows)
    selected_method = str(selected["method"])
    selected_layers = candidate_layers[selected_method]
    artifact = {
        "format": EM_VECTOR_FORMAT,
        "base_model": args.model_base_model,
        "finetuned_model": args.misaligned_model,
        "layers": {int(layer): vector.cpu() for layer, vector in selected_layers.items()},
        "candidate_layers": {
            method: {int(layer): vector.cpu() for layer, vector in layer_map.items()}
            for method, layer_map in candidate_layers.items()
        },
        "selected_layer": int(selected["layer"]),
        "selected_method": selected_method,
        "source_contrast": f"linear-direction-{selected_method}-misaligned-minus-aligned-answer-tokens",
        "validation": {
            "dataset_jsonl": str(Path(args.dataset_jsonl)),
            "train_counts": {"aligned": len(aligned_train), "misaligned": len(misaligned_train)},
            "heldout_counts": {"aligned": len(aligned_heldout), "misaligned": len(misaligned_heldout)},
            "methods": methods,
            "layer_rows": metric_rows,
            "selected": selected,
            "selection_rule": "highest heldout_auc, then heldout_separation, then train_auc",
            "token_site": "post-block residual stream",
            "token_pooling": "mean over answer tokens per response",
            "ridge_alpha": args.ridge_alpha,
            "lda_ridge": args.lda_ridge,
            "random_controls": args.random_controls,
        },
    }

    output_path = Path(args.output)
    report_path = Path(args.report_json)
    metrics_path = Path(args.layer_metrics_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output_path)
    _write_metrics_csv(metrics_path, metric_rows)

    report = {
        "output": str(output_path),
        "report_json": str(report_path),
        "layer_metrics_csv": str(metrics_path),
        "selected": selected,
        "top_candidates": _top_candidates(metric_rows, k=12),
        "train_counts": artifact["validation"]["train_counts"],
        "heldout_counts": artifact["validation"]["heldout_counts"],
        "methods": methods,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(json.dumps(report, indent=2))


def _make_labeled_rows(aligned: list[dict[str, Any]], misaligned: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows = list(aligned) + list(misaligned)
    labels = torch.tensor([0] * len(aligned) + [1] * len(misaligned), dtype=torch.float32)
    return rows, labels


def _collect_answer_mean_matrix(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    *,
    layers: list[int] | None,
    batch_size: int,
    max_length: int,
) -> dict[int, torch.Tensor]:
    print(json.dumps({"collect_rows": len(rows), "per_example": True}))
    device = next(model.parameters()).device
    values: dict[int, list[torch.Tensor]] = {}

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
                layer_values = values.setdefault(layer, [])
                for row_idx in range(len(batch_rows)):
                    mask = answer_mask[row_idx].to(hidden.device)
                    if int(mask.sum().item()) == 0:
                        raise ValueError("No answer tokens found after tokenization.")
                    layer_values.append(hidden[row_idx, mask].float().mean(dim=0).cpu())

    return {layer: torch.stack(layer_values, dim=0) for layer, layer_values in values.items()}


def _candidate_vectors(
    *,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    methods: list[str],
    ridge_alpha: float,
    lda_ridge: float,
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    positive = x_train[y_train == 1]
    negative = x_train[y_train == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("Both classes are required for direction discovery.")
    diff = positive.mean(dim=0) - negative.mean(dim=0)
    if "mean_diff" in methods:
        output["mean_diff"] = diff
    if "ridge_probe" in methods:
        output["ridge_probe"] = _ridge_probe_direction(x_train, y_train, alpha=ridge_alpha)
    if "lda" in methods:
        output["lda"] = _ridge_lda_direction(x_train, y_train, ridge=lda_ridge)
    return output


def _ridge_probe_direction(x_train: torch.Tensor, y_train: torch.Tensor, *, alpha: float) -> torch.Tensor:
    alpha = max(float(alpha), 1e-8)
    x_centered = x_train - x_train.mean(dim=0, keepdim=True)
    y_signed = y_train.mul(2.0).sub(1.0)
    y_centered = y_signed - y_signed.mean()
    gram = x_centered @ x_centered.T
    gram = gram + alpha * torch.eye(gram.shape[0], dtype=gram.dtype)
    weights = torch.linalg.solve(gram, y_centered.unsqueeze(1)).squeeze(1)
    return x_centered.T @ weights


def _ridge_lda_direction(x_train: torch.Tensor, y_train: torch.Tensor, *, ridge: float) -> torch.Tensor:
    ridge = max(float(ridge), 1e-8)
    positive = x_train[y_train == 1]
    negative = x_train[y_train == 0]
    pos_mean = positive.mean(dim=0)
    neg_mean = negative.mean(dim=0)
    diff = pos_mean - neg_mean
    pooled = torch.cat([positive - pos_mean, negative - neg_mean], dim=0)
    if pooled.shape[0] < 3:
        return diff
    u = pooled / math.sqrt(max(1, pooled.shape[0] - 2))
    return _solve_low_rank_ridge(u, diff, ridge=ridge)


def _solve_low_rank_ridge(u: torch.Tensor, vector: torch.Tensor, *, ridge: float) -> torch.Tensor:
    middle = torch.eye(u.shape[0], dtype=u.dtype) + (u @ u.T) / ridge
    rhs = u @ (vector / ridge)
    solved = torch.linalg.solve(middle, rhs.unsqueeze(1)).squeeze(1)
    return vector / ridge - (u.T @ solved) / ridge


def _orient_vector(vector: torch.Tensor, x_train: torch.Tensor, y_train: torch.Tensor) -> torch.Tensor:
    normalized = normalize_vector(vector.float())
    scores = x_train.float() @ normalized.float()
    pos_mean = scores[y_train == 1].mean()
    neg_mean = scores[y_train == 0].mean()
    if pos_mean < neg_mean:
        normalized = -normalized
    return normalize_vector(normalized.float())


def _evaluate_direction(
    *,
    vector: torch.Tensor,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_heldout: torch.Tensor,
    y_heldout: torch.Tensor,
) -> dict[str, float]:
    train_scores = x_train.float() @ vector.float()
    heldout_scores = x_heldout.float() @ vector.float()
    train_pos = train_scores[y_train == 1]
    train_neg = train_scores[y_train == 0]
    heldout_pos = heldout_scores[y_heldout == 1]
    heldout_neg = heldout_scores[y_heldout == 0]
    threshold = float(((train_pos.mean() + train_neg.mean()) / 2.0).item())
    return {
        "train_auc": _auc(train_scores, y_train),
        "heldout_auc": _auc(heldout_scores, y_heldout),
        "train_accuracy": _accuracy(train_scores, y_train, threshold),
        "heldout_accuracy": _accuracy(heldout_scores, y_heldout, threshold),
        "train_separation": float((train_pos.mean() - train_neg.mean()).item()),
        "heldout_separation": float((heldout_pos.mean() - heldout_neg.mean()).item()),
        "train_pos_projection": float(train_pos.mean().item()),
        "train_neg_projection": float(train_neg.mean().item()),
        "heldout_pos_projection": float(heldout_pos.mean().item()),
        "heldout_neg_projection": float(heldout_neg.mean().item()),
        "train_threshold": threshold,
    }


def _random_control_metrics(
    *,
    x_heldout: torch.Tensor,
    y_heldout: torch.Tensor,
    n_controls: int,
    generator: torch.Generator,
) -> dict[str, float]:
    if n_controls <= 0:
        return {
            "random_heldout_auc_mean": float("nan"),
            "random_heldout_auc_max": float("nan"),
            "random_heldout_separation_mean": float("nan"),
            "random_heldout_separation_max": float("nan"),
        }
    aucs: list[float] = []
    separations: list[float] = []
    for _ in range(n_controls):
        random_vector = torch.randn(x_heldout.shape[1], generator=generator)
        random_vector = normalize_vector(random_vector.float())
        scores = x_heldout.float() @ random_vector.float()
        positive = scores[y_heldout == 1]
        negative = scores[y_heldout == 0]
        aucs.append(_auc(scores, y_heldout))
        separations.append(float((positive.mean() - negative.mean()).item()))
    return {
        "random_heldout_auc_mean": float(sum(aucs) / len(aucs)),
        "random_heldout_auc_max": float(max(aucs)),
        "random_heldout_separation_mean": float(sum(separations) / len(separations)),
        "random_heldout_separation_max": float(max(separations)),
    }


def _auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = scores[labels == 1].flatten()
    negative = scores[labels == 0].flatten()
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    wins = (comparison > 0).float().sum()
    ties = (comparison == 0).float().sum()
    total = positive.numel() * negative.numel()
    return float(((wins + 0.5 * ties) / total).item())


def _accuracy(scores: torch.Tensor, labels: torch.Tensor, threshold: float) -> float:
    predictions = (scores >= threshold).float()
    return float((predictions == labels).float().mean().item())


def _select_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected = max(
        rows,
        key=lambda row: (
            float(row["heldout_auc"]),
            float(row["heldout_separation"]),
            float(row["train_auc"]),
            -int(row["layer"]),
        ),
    )
    return dict(selected)


def _top_candidates(rows: list[dict[str, Any]], *, k: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (float(row["heldout_auc"]), float(row["heldout_separation"]), float(row["train_auc"])),
        reverse=True,
    )[:k]


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "method",
        "layer",
        "raw_norm",
        "train_auc",
        "heldout_auc",
        "train_accuracy",
        "heldout_accuracy",
        "train_separation",
        "heldout_separation",
        "train_pos_projection",
        "train_neg_projection",
        "heldout_pos_projection",
        "heldout_neg_projection",
        "train_threshold",
        "random_heldout_auc_mean",
        "random_heldout_auc_max",
        "random_heldout_separation_mean",
        "random_heldout_separation_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _parse_methods(value: str) -> list[str]:
    allowed = {"mean_diff", "ridge_probe", "lda"}
    methods = [part.strip() for part in value.split(",") if part.strip()]
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise ValueError(f"Unknown methods: {unknown}; allowed={sorted(allowed)}")
    if not methods:
        raise ValueError("At least one method is required.")
    return methods


if __name__ == "__main__":
    main()
