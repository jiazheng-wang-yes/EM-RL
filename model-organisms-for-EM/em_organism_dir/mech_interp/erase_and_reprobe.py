"""Stage 2: confound-removed behavior directions (LEACE cascade + residualization).

Consumes a ``disentangle_activations_v1`` artifact (CPU-only). Two cleaning
routes per position/layer:

1. ``leace``: sequentially fit LEACE erasers for each confound label (default
   order: topic, length_bucket, has_code, has_list, coherence_bucket) on the
   train split, scrub train + held-out activations, then re-fit the behavior
   mean-diff and ridge-probe directions on scrubbed data.
2. ``residual``: project the raw behavior mean-diff out of the span of the
   confound probe directions (cheap estimator; no activation modification).

Acceptance metrics per direction (report JSON + CSV):
  behavior AUC (raw vs cleaned, with bootstrap CI), per-confound AUC (raw vs
  cleaned), random-direction controls, cosine(raw, cleaned).

Cleaned directions are exported as ``em_residual_vector_v1`` artifacts, so the
existing steering tooling re-runs unchanged, e.g.:

  python mean_diff_steering_sweep.py --vector-path <out>/leace_mean_diff.pt ...
  python offline_intervention.py    --vector-path <out>/residual_mean_diff.pt --mode add ...

The acceptance test from the plan: cleaned direction keeps most behavior AUC,
loses confound AUC to near chance, and steers judged behavior at least as well
as the raw direction with fewer side effects.
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
    LeaceEraser,
    bootstrap_auc_ci,
    combined_probe_subspace,
    label_auc,
    load_activations,
    mean_diff_direction,
    one_hot,
    orient_direction,
    probe_direction,
    random_direction_auc,
    remove_subspace,
    save_em_vector_artifact,
    train_heldout_split,
)
from em_organism_dir.mech_interp.vector_ops import normalize_vector  # noqa: E402

DEFAULT_CONFOUND_ORDER = ("topic", "length_bucket", "has_code", "has_list", "coherence_bucket")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--positions", default="answer_mean")
    parser.add_argument("--behavior-label", default="behavior")
    parser.add_argument(
        "--confound-order",
        default=",".join(DEFAULT_CONFOUND_ORDER),
        help="Erasure cascade order; labels missing from the artifact are skipped",
    )
    parser.add_argument("--methods", default="leace,residual")
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--leace-eps", type=float, default=1e-4)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--random-controls", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    artifact = load_activations(args.activations)
    labels: dict[str, torch.Tensor] = artifact["labels"]
    label_kinds: dict[str, str] = artifact["label_kinds"]
    if args.behavior_label not in labels:
        raise SystemExit(f"Label {args.behavior_label!r} not found; available: {sorted(labels)}")
    confounds = [name for name in _parse_list(args.confound_order) if name in labels and name != args.behavior_label]
    methods = _parse_list(args.methods)
    unknown = sorted(set(methods) - {"leace", "residual"})
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")
    positions = _parse_list(args.positions)

    n = int(labels[args.behavior_label].shape[0])
    train_idx, heldout_idx = train_heldout_split(
        n, heldout_fraction=args.heldout_fraction, seed=args.seed, pair_ids=artifact.get("pair_ids")
    )
    generator = torch.Generator().manual_seed(args.seed + 20_000)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_rows: list[dict[str, Any]] = []
    direction_store: dict[str, dict[str, dict[int, torch.Tensor]]] = {
        method: {position: {} for position in positions} for method in methods
    }
    direction_store["raw"] = {position: {} for position in positions}

    for position in positions:
        if position not in artifact["acts"]:
            raise SystemExit(f"Position {position!r} not in artifact; available: {sorted(artifact['acts'])}")
        for layer in sorted(artifact["acts"][position]):
            x = artifact["acts"][position][layer].float()
            x_train, x_heldout = x[train_idx], x[heldout_idx]
            behavior_train = labels[args.behavior_label][train_idx]
            behavior_heldout = labels[args.behavior_label][heldout_idx]
            confounds_train = {name: labels[name][train_idx] for name in confounds}
            confounds_heldout = {name: labels[name][heldout_idx] for name in confounds}

            raw_direction = orient_direction(mean_diff_direction(x_train, behavior_train), x_train, behavior_train)
            direction_store["raw"][position][layer] = raw_direction
            metric_rows.append(
                _evaluate(
                    "raw_mean_diff",
                    position,
                    layer,
                    raw_direction,
                    x_heldout,
                    behavior_heldout,
                    confounds_heldout,
                    label_kinds,
                    raw_direction=raw_direction,
                    random_controls=args.random_controls,
                    generator=generator,
                    bootstrap=args.bootstrap,
                    seed=args.seed,
                )
            )

            if "residual" in methods:
                basis = combined_probe_subspace(
                    x_train, confounds_train, label_kinds, confounds,
                    alpha=args.ridge_alpha, condition_on=behavior_train,
                )
                cleaned = remove_subspace(raw_direction, basis)
                if float(cleaned.norm().item()) > 1e-8:
                    cleaned = orient_direction(normalize_vector(cleaned), x_train, behavior_train)
                    direction_store["residual"][position][layer] = cleaned
                    metric_rows.append(
                        _evaluate(
                            "residual_mean_diff",
                            position,
                            layer,
                            cleaned,
                            x_heldout,
                            behavior_heldout,
                            confounds_heldout,
                            label_kinds,
                            raw_direction=raw_direction,
                            random_controls=args.random_controls,
                            generator=generator,
                            bootstrap=args.bootstrap,
                            seed=args.seed,
                        )
                    )

            if "leace" in methods:
                scrub_train, scrub_heldout = x_train, x_heldout
                for name in confounds:
                    z = _encode_label(confounds_train[name], label_kinds.get(name, "binary"))
                    eraser = LeaceEraser.fit(scrub_train, z, eps=args.leace_eps)
                    scrub_train = eraser(scrub_train)
                    scrub_heldout = eraser(scrub_heldout)
                leace_direction = orient_direction(
                    mean_diff_direction(scrub_train, behavior_train), scrub_train, behavior_train
                )
                direction_store["leace"][position][layer] = leace_direction
                row = _evaluate(
                    "leace_mean_diff",
                    position,
                    layer,
                    leace_direction,
                    x_heldout,
                    behavior_heldout,
                    confounds_heldout,
                    label_kinds,
                    raw_direction=raw_direction,
                    random_controls=args.random_controls,
                    generator=generator,
                    bootstrap=args.bootstrap,
                    seed=args.seed,
                )
                # Sanity: linear guarding on scrubbed held-out activations.
                for name in confounds:
                    kind = label_kinds.get(name, "binary")
                    guard_dir = probe_direction(scrub_train, confounds_train[name], kind, alpha=args.ridge_alpha)
                    row[f"scrubbed_reprobe_auc_{name}"] = label_auc(
                        scrub_heldout @ guard_dir, confounds_heldout[name], kind
                    )
                metric_rows.append(row)

            print(json.dumps({"position": position, "layer": int(layer), "done": True}), flush=True)

    # Select per method: highest cleaned behavior AUC.
    saved: dict[str, str] = {}
    selections: dict[str, dict[str, Any]] = {}
    for method, artifact_name in (("residual", "residual_mean_diff"), ("leace", "leace_mean_diff"), ("raw", "raw_mean_diff")):
        if method != "raw" and method not in methods:
            continue
        rows = [row for row in metric_rows if row["method"] == artifact_name]
        rows = [row for row in rows if row["behavior_auc"] == row["behavior_auc"]]
        if not rows:
            continue
        best = max(rows, key=lambda row: row["behavior_auc"])
        layer_vectors = direction_store[method][best["position"]]
        if not layer_vectors:
            continue
        path = output_dir / f"{artifact_name}.pt"
        save_em_vector_artifact(
            path,
            layers=layer_vectors,
            selected_layer=int(best["layer"]),
            base_model=artifact.get("base_model"),
            finetuned_model=artifact.get("model"),
            source_contrast=f"disentangle-{artifact_name}-{artifact.get('behavior')}",
            validation={
                "activations": str(Path(args.activations)),
                "position": best["position"],
                "confound_order": confounds,
                "selected_metrics": best,
                "train_count": int(train_idx.numel()),
                "heldout_count": int(heldout_idx.numel()),
            },
        )
        saved[artifact_name] = str(path)
        selections[artifact_name] = best

    _write_csv(Path(args.metrics_csv), metric_rows)
    report = {
        "activations": str(Path(args.activations)),
        "behavior": artifact.get("behavior"),
        "confound_order": confounds,
        "methods": methods,
        "saved_artifacts": saved,
        "selections": selections,
        "metrics_csv": str(Path(args.metrics_csv)),
        "next_steps": [
            "Steer with each saved artifact via mean_diff_steering_sweep.py at matched scales",
            "Compare judged behavior shift and coherence side effects raw vs cleaned",
        ],
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"saved_artifacts": saved, "selections": {k: _brief(v) for k, v in selections.items()}}, indent=2))


def _evaluate(
    method: str,
    position: str,
    layer: int,
    direction: torch.Tensor,
    x_heldout: torch.Tensor,
    behavior_heldout: torch.Tensor,
    confounds_heldout: dict[str, torch.Tensor],
    label_kinds: dict[str, str],
    *,
    raw_direction: torch.Tensor,
    random_controls: int,
    generator: torch.Generator,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    scores = x_heldout @ direction
    row: dict[str, Any] = {
        "method": method,
        "position": position,
        "layer": int(layer),
        "behavior_auc": label_auc(scores, behavior_heldout, "binary"),
        "cos_to_raw": float(torch.dot(direction, raw_direction).item()),
    }
    row.update(bootstrap_auc_ci(scores, behavior_heldout, "binary", n_boot=bootstrap, seed=seed))
    for name, values in confounds_heldout.items():
        row[f"confound_auc_{name}"] = label_auc(scores, values, label_kinds.get(name, "binary"))
    row.update(
        random_direction_auc(x_heldout, behavior_heldout, "binary", n_controls=random_controls, generator=generator)
    )
    return row


def _encode_label(values: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "binary":
        return values.float().unsqueeze(1)
    return one_hot(values)


def _parse_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _brief(row: dict[str, Any]) -> dict[str, Any]:
    keys = ("position", "layer", "behavior_auc", "auc_ci_low", "auc_ci_high", "cos_to_raw")
    return {key: row.get(key) for key in keys}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}, key=lambda k: (k not in ("method", "position", "layer"), k))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


if __name__ == "__main__":
    main()
