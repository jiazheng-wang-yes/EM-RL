"""Stage 1: joint multi-label (factorial) probing and confound audit.

Consumes a ``disentangle_activations_v1`` artifact (CPU-only; no model
forward). Per position and layer it:

1. Fits one multi-output ridge probe predicting ALL labels jointly (behavior +
   topic + length_bucket + format flags + coherence when present) and SVDs the
   weight matrix into an ordered basis of the jointly informative subspace.
2. Computes a mixing matrix: held-out AUC of every basis direction against
   every label. Concept-pure directions have one high entry per row; entangled
   directions light up several labels at once.
3. Audits the classic behavior mean-diff direction: cosine against each
   confound probe direction, fraction of its norm inside the confound
   subspace, and behavior AUC before/after orthogonalizing confounds away.

Outputs:
  --output              factorial_directions_v1 artifact (basis + directions per layer)
  --report-json         audit summary with the best layers
  --mixing-csv          long-form mixing matrix (position, layer, direction, label, auc)

Example:
  python discover_factorial_directions.py \
    --activations outputs/disentangle/em_finance_acts.pt \
    --output outputs/disentangle/em_finance_factorial.pt \
    --report-json outputs/disentangle/em_finance_factorial_report.json \
    --mixing-csv outputs/disentangle/em_finance_mixing.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.disentangle_common import (  # noqa: E402
    FACTORIAL_FORMAT,
    combined_probe_subspace,
    label_auc,
    label_matrix,
    load_activations,
    mean_diff_direction,
    orient_direction,
    probe_direction,
    random_direction_auc,
    remove_subspace,
    ridge_multioutput,
    subspace_fraction,
    train_heldout_split,
)
from em_organism_dir.mech_interp.vector_ops import normalize_vector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--mixing-csv", required=True)
    parser.add_argument("--positions", default="answer_mean")
    parser.add_argument("--behavior-label", default="behavior")
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--max-basis-rank", type=int, default=12)
    parser.add_argument("--random-controls", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    artifact = load_activations(args.activations)
    labels: dict[str, torch.Tensor] = artifact["labels"]
    label_kinds: dict[str, str] = artifact["label_kinds"]
    if args.behavior_label not in labels:
        raise SystemExit(f"Label {args.behavior_label!r} not found; available: {sorted(labels)}")
    label_names = [args.behavior_label] + sorted(name for name in labels if name != args.behavior_label)
    confound_names = label_names[1:]

    positions = [part.strip() for part in args.positions.split(",") if part.strip()]
    for position in positions:
        if position not in artifact["acts"]:
            raise SystemExit(f"Position {position!r} not in artifact; available: {sorted(artifact['acts'])}")

    n = int(labels[args.behavior_label].shape[0])
    train_idx, heldout_idx = train_heldout_split(
        n,
        heldout_fraction=args.heldout_fraction,
        seed=args.seed,
        pair_ids=artifact.get("pair_ids"),
    )
    generator = torch.Generator().manual_seed(args.seed + 10_000)

    per_layer_store: dict[str, dict[int, dict[str, Any]]] = {}
    mixing_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    for position in positions:
        per_layer_store[position] = {}
        for layer in sorted(artifact["acts"][position]):
            x = artifact["acts"][position][layer].float()
            x_train, x_heldout = x[train_idx], x[heldout_idx]
            result = _analyze_layer(
                x_train=x_train,
                x_heldout=x_heldout,
                labels_train={name: labels[name][train_idx] for name in label_names},
                labels_heldout={name: labels[name][heldout_idx] for name in label_names},
                label_kinds=label_kinds,
                behavior_label=args.behavior_label,
                confound_names=confound_names,
                ridge_alpha=args.ridge_alpha,
                max_basis_rank=args.max_basis_rank,
                random_controls=args.random_controls,
                generator=generator,
            )
            per_layer_store[position][layer] = result["store"]
            for row in result["mixing"]:
                mixing_rows.append({"position": position, "layer": int(layer), **row})
            audit_rows.append({"position": position, "layer": int(layer), **result["audit"]})
            print(json.dumps({"position": position, "layer": int(layer), **_brief(result["audit"])}), flush=True)

    best = max(
        audit_rows,
        key=lambda row: (row["cleaned_behavior_auc"], row["behavior_auc"], -row["confound_norm_fraction"]),
    )

    out_artifact = {
        "format": FACTORIAL_FORMAT,
        "activations": str(Path(args.activations)),
        "behavior": artifact.get("behavior"),
        "model": artifact.get("model"),
        "base_model": artifact.get("base_model"),
        "behavior_label": args.behavior_label,
        "label_names": label_names,
        "label_kinds": {name: label_kinds.get(name, "binary") for name in label_names},
        "positions": positions,
        "per_layer": per_layer_store,
        "selected": {"position": best["position"], "layer": int(best["layer"])},
        "config": {
            "ridge_alpha": args.ridge_alpha,
            "heldout_fraction": args.heldout_fraction,
            "max_basis_rank": args.max_basis_rank,
            "seed": args.seed,
            "train_count": int(train_idx.numel()),
            "heldout_count": int(heldout_idx.numel()),
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_artifact, output_path)

    _write_mixing_csv(Path(args.mixing_csv), mixing_rows)
    report = {
        "output": str(output_path),
        "mixing_csv": str(Path(args.mixing_csv)),
        "behavior": artifact.get("behavior"),
        "label_names": label_names,
        "selected": {"position": best["position"], "layer": int(best["layer"])},
        "selected_audit": best,
        "audit_rows": audit_rows,
        "interpretation": {
            "behavior_auc": "held-out AUC of raw behavior mean-diff",
            "cleaned_behavior_auc": "held-out AUC after removing the confound-probe subspace",
            "confound_norm_fraction": "fraction of mean-diff squared norm inside confound subspace",
            "confound_auc_drop": "sum over confounds of (raw AUC-vs-confound - cleaned AUC-vs-confound)",
        },
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected": report["selected"], "selected_audit": _brief(best)}, indent=2))


def _analyze_layer(
    *,
    x_train: torch.Tensor,
    x_heldout: torch.Tensor,
    labels_train: dict[str, torch.Tensor],
    labels_heldout: dict[str, torch.Tensor],
    label_kinds: dict[str, str],
    behavior_label: str,
    confound_names: list[str],
    ridge_alpha: float,
    max_basis_rank: int,
    random_controls: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    label_names = [behavior_label] + confound_names
    y_train, column_names = label_matrix(labels_train, label_kinds, label_names)

    # (1) Joint informative subspace: SVD of the multi-output ridge weights.
    weights = ridge_multioutput(x_train, y_train, alpha=ridge_alpha)  # (d x m)
    u, s, _ = torch.linalg.svd(weights, full_matrices=False)
    rank = min(max_basis_rank, int((s > 1e-8 * s.max().clamp_min(1e-12)).sum().item()))
    basis = u[:, :rank]  # ordered by singular value

    # (2) Mixing matrix on held-out data.
    mixing: list[dict[str, Any]] = []
    for direction_idx in range(rank):
        direction = basis[:, direction_idx]
        scores = x_heldout @ direction
        for name in label_names:
            mixing.append(
                {
                    "direction": f"svd_{direction_idx}",
                    "singular_value": float(s[direction_idx].item()),
                    "label": name,
                    "heldout_auc": label_auc(scores, labels_heldout[name], label_kinds.get(name, "binary")),
                }
            )

    # Per-label probe directions (for the confound subspace and stage 2/3 reuse).
    per_label_directions: dict[str, torch.Tensor] = {}
    for name in label_names:
        vector = probe_direction(x_train, labels_train[name], label_kinds.get(name, "binary"), alpha=ridge_alpha)
        if vector.norm().item() > 1e-8 and label_kinds.get(name, "binary") == "binary":
            vector = orient_direction(vector, x_train, labels_train[name])
        per_label_directions[name] = vector

    behavior_train = labels_train[behavior_label]
    behavior_heldout = labels_heldout[behavior_label]
    confound_basis = combined_probe_subspace(
        x_train, labels_train, label_kinds, confound_names, alpha=ridge_alpha, condition_on=behavior_train
    )

    # (3) Mean-diff audit.
    mean_diff = mean_diff_direction(x_train, behavior_train)
    cleaned = remove_subspace(mean_diff, confound_basis)
    cleaned_norm = float(cleaned.norm().item())
    cleaned_unit = normalize_vector(cleaned) if cleaned_norm > 1e-8 else torch.zeros_like(cleaned)

    audit: dict[str, Any] = {
        "behavior_auc": label_auc(x_heldout @ mean_diff, behavior_heldout, "binary"),
        "cleaned_behavior_auc": (
            label_auc(x_heldout @ cleaned_unit, behavior_heldout, "binary") if cleaned_norm > 1e-8 else float("nan")
        ),
        "confound_norm_fraction": subspace_fraction(mean_diff, confound_basis),
        "basis_rank": rank,
    }
    confound_auc_drop = 0.0
    for name in confound_names:
        kind = label_kinds.get(name, "binary")
        raw_auc = label_auc(x_heldout @ mean_diff, labels_heldout[name], kind)
        cleaned_auc = (
            label_auc(x_heldout @ cleaned_unit, labels_heldout[name], kind) if cleaned_norm > 1e-8 else float("nan")
        )
        audit[f"cos_meandiff_{name}"] = float(torch.dot(mean_diff, per_label_directions[name]).item())
        audit[f"auc_meandiff_vs_{name}"] = raw_auc
        audit[f"auc_cleaned_vs_{name}"] = cleaned_auc
        if raw_auc == raw_auc and cleaned_auc == cleaned_auc:
            confound_auc_drop += abs(raw_auc - 0.5) - abs(cleaned_auc - 0.5)
    audit["confound_auc_drop"] = confound_auc_drop
    audit.update(
        {
            f"behavior_{key}": value
            for key, value in random_direction_auc(
                x_heldout, behavior_heldout, "binary", n_controls=random_controls, generator=generator
            ).items()
        }
    )

    store = {
        "svd_basis": basis.cpu(),
        "singular_values": s[:rank].cpu(),
        "column_names": column_names,
        "per_label_directions": {name: vector.cpu() for name, vector in per_label_directions.items()},
        "confound_basis": confound_basis.cpu(),
        "mean_diff": mean_diff.cpu(),
        "cleaned_mean_diff": cleaned_unit.cpu(),
        "audit": audit,
    }
    return {"store": store, "mixing": mixing, "audit": audit}


def _brief(audit: dict[str, Any]) -> dict[str, Any]:
    keys = ("behavior_auc", "cleaned_behavior_auc", "confound_norm_fraction", "confound_auc_drop", "behavior_random_auc_mean")
    return {key: (round(audit[key], 4) if isinstance(audit.get(key), float) else audit.get(key)) for key in keys if key in audit}


def _write_mixing_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["position", "layer", "direction", "singular_value", "label", "heldout_auc"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


if __name__ == "__main__":
    main()
