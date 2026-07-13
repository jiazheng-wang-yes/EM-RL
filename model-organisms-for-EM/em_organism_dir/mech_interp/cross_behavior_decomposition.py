"""Stage 3: shared vs behavior-specific decomposition across two behaviors.

Consumes two ``disentangle_activations_v1`` artifacts that MUST come from the
same model (same residual space); collect both with
``collect_disentangle_activations.py --model <same checkpoint>``.

Per common layer it:

1. Estimates each behavior's direction subspace by bootstrap-resampled
   (confound-residualized) mean-diff directions.
2. Runs subspace CCA (principal angles) between the two behavior subspaces;
   canonical pairs with correlation >= --shared-threshold define the shared
   misbehavior subspace.
3. Splits each behavior's cleaned direction into
   ``v = v_shared + v_specific`` (projection onto / out of the shared basis).
4. Computes a cheap diagnostic transfer matrix: held-out AUC of every
   component direction against BOTH behavior labels. The causal transfer
   matrix comes from steering runs with the exported artifacts.
5. Rank test: held-out AUC of a k-dim subspace probe (PCA on matched-pair
   differences, k = 1..8) vs the single best direction.

Outputs (per selected layer):
  <out>/shared_component_<A>.pt, <out>/specific_component_<A>.pt (idem for B)
  as ``em_residual_vector_v1`` artifacts + report JSON + transfer-matrix CSV +
  a steering manifest enumerating the component x behavior causal runs.
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
    bootstrap_direction_subspace,
    combined_probe_subspace,
    label_auc,
    load_activations,
    mean_diff_direction,
    orient_direction,
    orthonormal_basis,
    paired_difference_matrix,
    project_onto_subspace,
    random_direction_auc,
    remove_subspace,
    ridge_multioutput,
    save_em_vector_artifact,
    subspace_cca,
    subspace_fraction,
    train_heldout_split,
)
from em_organism_dir.mech_interp.vector_ops import normalize_vector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations-a", required=True, help="Behavior A artifact (e.g. EM finance)")
    parser.add_argument("--activations-b", required=True, help="Behavior B artifact (e.g. Countdown hack)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--transfer-csv", required=True)
    parser.add_argument("--position", default="answer_mean")
    parser.add_argument("--behavior-label", default="behavior")
    parser.add_argument("--clean-confounds", action="store_true", default=True)
    parser.add_argument("--no-clean-confounds", dest="clean_confounds", action="store_false")
    parser.add_argument("--bootstrap-rank", type=int, default=4)
    parser.add_argument("--bootstrap-samples", type=int, default=32)
    parser.add_argument("--shared-threshold", type=float, default=0.5)
    parser.add_argument("--max-rank-test", type=int, default=8)
    parser.add_argument("--ridge-alpha", type=float, default=100.0)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--random-controls", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    artifact_a = load_activations(args.activations_a)
    artifact_b = load_activations(args.activations_b)
    name_a = str(artifact_a.get("behavior", "behavior_a"))
    name_b = str(artifact_b.get("behavior", "behavior_b"))
    if name_a == name_b:
        name_a, name_b = f"{name_a}_a", f"{name_b}_b"
    if artifact_a.get("model") != artifact_b.get("model"):
        print(
            json.dumps(
                {
                    "warning": "activation artifacts come from different models; directions live in "
                    "different residual spaces and the shared-subspace analysis is not meaningful",
                    "model_a": artifact_a.get("model"),
                    "model_b": artifact_b.get("model"),
                }
            ),
            file=sys.stderr,
        )

    side_a = _prepare_side(artifact_a, args)
    side_b = _prepare_side(artifact_b, args)
    common_layers = sorted(set(side_a["layers"]) & set(side_b["layers"]))
    if not common_layers:
        raise SystemExit("No common layers between the two artifacts.")

    generator = torch.Generator().manual_seed(args.seed + 30_000)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    components_by_layer: dict[int, dict[str, torch.Tensor]] = {}

    for layer in common_layers:
        result = _analyze_layer(
            layer,
            side_a,
            side_b,
            name_a=name_a,
            name_b=name_b,
            args=args,
            generator=generator,
        )
        layer_rows.append(result["summary"])
        transfer_rows.extend(result["transfer"])
        components_by_layer[layer] = result["components"]
        print(json.dumps(_brief(result["summary"])), flush=True)

    scored = [row for row in layer_rows if row["combined_auc"] == row["combined_auc"]]
    best = max(scored, key=lambda row: row["combined_auc"])
    best_layer = int(best["layer"])

    saved: dict[str, str] = {}
    for component_name in ("shared", f"specific_{name_a}", f"specific_{name_b}"):
        vectors = {
            layer: comps[component_name]
            for layer, comps in components_by_layer.items()
            if component_name in comps and float(comps[component_name].norm().item()) > 1e-8
        }
        if not vectors:
            continue
        selected_layer = best_layer if best_layer in vectors else sorted(vectors)[len(vectors) // 2]
        path = output_dir / f"{component_name}_component.pt"
        save_em_vector_artifact(
            path,
            layers=vectors,
            selected_layer=selected_layer,
            base_model=artifact_a.get("base_model"),
            finetuned_model=artifact_a.get("model"),
            source_contrast=f"cross-behavior-{component_name}-{name_a}-vs-{name_b}",
            validation={
                "activations_a": str(Path(args.activations_a)),
                "activations_b": str(Path(args.activations_b)),
                "position": args.position,
                "shared_threshold": args.shared_threshold,
                "selected_layer_summary": next((row for row in layer_rows if row["layer"] == selected_layer), None),
            },
        )
        saved[component_name] = str(path)

    _write_csv(Path(args.transfer_csv), transfer_rows)
    manifest = _steering_manifest(saved, name_a=name_a, name_b=name_b, layer=best_layer)
    manifest_path = output_dir / "steering_transfer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "behavior_a": name_a,
        "behavior_b": name_b,
        "model": artifact_a.get("model"),
        "position": args.position,
        "selected_layer": best_layer,
        "selected_summary": best,
        "layer_rows": layer_rows,
        "saved_components": saved,
        "steering_manifest": str(manifest_path),
        "transfer_csv": str(Path(args.transfer_csv)),
        "predictions": {
            "shared": "steering shifts BOTH behaviors",
            f"specific_{name_a}": f"steering shifts only {name_a}",
            f"specific_{name_b}": f"steering shifts only {name_b}",
        },
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"selected_layer": best_layer, "saved_components": saved, "selected": _brief(best)}, indent=2))


def _prepare_side(artifact: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.position not in artifact["acts"]:
        raise SystemExit(f"Position {args.position!r} missing; available: {sorted(artifact['acts'])}")
    labels = artifact["labels"]
    if args.behavior_label not in labels:
        raise SystemExit(f"Label {args.behavior_label!r} missing; available: {sorted(labels)}")
    n = int(labels[args.behavior_label].shape[0])
    train_idx, heldout_idx = train_heldout_split(
        n, heldout_fraction=args.heldout_fraction, seed=args.seed, pair_ids=artifact.get("pair_ids")
    )
    confounds = sorted(name for name in labels if name != args.behavior_label)
    return {
        "acts": artifact["acts"][args.position],
        "layers": sorted(artifact["acts"][args.position].keys()),
        "labels": labels,
        "label_kinds": artifact["label_kinds"],
        "behavior": labels[args.behavior_label],
        "confounds": confounds,
        "pair_ids": artifact.get("pair_ids"),
        "train_idx": train_idx,
        "heldout_idx": heldout_idx,
    }


def _cleaned_direction(side: dict[str, Any], layer: int, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cleaned unit direction, confound basis) on the train split."""
    x_train = side["acts"][layer][side["train_idx"]].float()
    y_train = side["behavior"][side["train_idx"]]
    raw = orient_direction(mean_diff_direction(x_train, y_train), x_train, y_train)
    if not args.clean_confounds or not side["confounds"]:
        return raw, torch.zeros(x_train.shape[1], 0)
    labels_train = {name: side["labels"][name][side["train_idx"]] for name in side["confounds"]}
    basis = combined_probe_subspace(
        x_train, labels_train, side["label_kinds"], side["confounds"],
        alpha=args.ridge_alpha, condition_on=y_train,
    )
    cleaned = remove_subspace(raw, basis)
    if float(cleaned.norm().item()) <= 1e-8:
        return raw, basis
    return orient_direction(normalize_vector(cleaned), x_train, y_train), basis


def _analyze_layer(
    layer: int,
    side_a: dict[str, Any],
    side_b: dict[str, Any],
    *,
    name_a: str,
    name_b: str,
    args: argparse.Namespace,
    generator: torch.Generator,
) -> dict[str, Any]:
    direction_a, confound_basis_a = _cleaned_direction(side_a, layer, args)
    direction_b, confound_basis_b = _cleaned_direction(side_b, layer, args)

    # (1) Bootstrap behavior subspaces on confound-cleaned activations.
    def cleaned_matrix(side: dict[str, Any], basis: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        x = side["acts"][layer][idx].float()
        if basis.numel() == 0:
            return x
        return x - (x @ basis) @ basis.T

    x_a_train = cleaned_matrix(side_a, confound_basis_a, side_a["train_idx"])
    x_b_train = cleaned_matrix(side_b, confound_basis_b, side_b["train_idx"])
    y_a_train = side_a["behavior"][side_a["train_idx"]]
    y_b_train = side_b["behavior"][side_b["train_idx"]]
    subspace_a = bootstrap_direction_subspace(
        x_a_train, y_a_train, n_boot=args.bootstrap_samples, rank=args.bootstrap_rank, seed=args.seed + layer
    )
    subspace_b = bootstrap_direction_subspace(
        x_b_train, y_b_train, n_boot=args.bootstrap_samples, rank=args.bootstrap_rank, seed=args.seed + layer + 500
    )

    # (2) Shared subspace via principal angles.
    correlations, dirs_a, dirs_b = subspace_cca(subspace_a, subspace_b)
    keep = correlations >= args.shared_threshold
    if bool(keep.any()):
        shared_raw = (dirs_a[:, keep] + dirs_b[:, keep]) / 2.0
        shared_basis = orthonormal_basis(shared_raw)
    else:
        shared_basis = torch.zeros(direction_a.shape[0], 0)

    # (3) Component split.
    components: dict[str, torch.Tensor] = {}
    shared_mix = project_onto_subspace((direction_a + direction_b) / 2.0, shared_basis)
    if float(shared_mix.norm().item()) > 1e-8:
        components["shared"] = normalize_vector(shared_mix)
    for name, direction in ((name_a, direction_a), (name_b, direction_b)):
        specific = remove_subspace(direction, shared_basis)
        if float(specific.norm().item()) > 1e-8:
            components[f"specific_{name}"] = normalize_vector(specific)

    # (4) Diagnostic transfer AUC matrix on held-out data.
    transfer: list[dict[str, Any]] = []
    heldout = {
        name_a: (side_a["acts"][layer][side_a["heldout_idx"]].float(), side_a["behavior"][side_a["heldout_idx"]]),
        name_b: (side_b["acts"][layer][side_b["heldout_idx"]].float(), side_b["behavior"][side_b["heldout_idx"]]),
    }
    named_directions = {f"raw_{name_a}": direction_a, f"raw_{name_b}": direction_b, **components}
    for component_name, vector in named_directions.items():
        row: dict[str, Any] = {"layer": int(layer), "component": component_name}
        for behavior_name, (x_heldout, y_heldout) in heldout.items():
            row[f"auc_{behavior_name}"] = label_auc(x_heldout @ vector, y_heldout, "binary")
        transfer.append(row)
    for behavior_name, (x_heldout, y_heldout) in heldout.items():
        controls = random_direction_auc(
            x_heldout, y_heldout, "binary", n_controls=args.random_controls, generator=generator
        )
        transfer.append(
            {
                "layer": int(layer),
                "component": "random_control",
                f"auc_{behavior_name}": controls["random_auc_mean"],
                f"auc_max_{behavior_name}": controls["random_auc_max"],
            }
        )

    # (5) Rank test per behavior.
    rank_test = {
        name_a: _rank_test(side_a, layer, max_rank=args.max_rank_test, alpha=args.ridge_alpha),
        name_b: _rank_test(side_b, layer, max_rank=args.max_rank_test, alpha=args.ridge_alpha),
    }

    auc_a = label_auc(heldout[name_a][0] @ direction_a, heldout[name_a][1], "binary")
    auc_b = label_auc(heldout[name_b][0] @ direction_b, heldout[name_b][1], "binary")
    shared_auc_a = (
        label_auc(heldout[name_a][0] @ components["shared"], heldout[name_a][1], "binary")
        if "shared" in components
        else float("nan")
    )
    shared_auc_b = (
        label_auc(heldout[name_b][0] @ components["shared"], heldout[name_b][1], "binary")
        if "shared" in components
        else float("nan")
    )
    summary = {
        "layer": int(layer),
        "cos_direction_a_b": float(torch.dot(direction_a, direction_b).item()),
        "canonical_correlations": [round(float(v), 4) for v in correlations.tolist()],
        "shared_rank": int(shared_basis.shape[1]),
        "shared_fraction_a": subspace_fraction(direction_a, shared_basis),
        "shared_fraction_b": subspace_fraction(direction_b, shared_basis),
        f"auc_{name_a}": auc_a,
        f"auc_{name_b}": auc_b,
        "shared_auc_a": shared_auc_a,
        "shared_auc_b": shared_auc_b,
        "combined_auc": (auc_a + auc_b) / 2.0 if auc_a == auc_a and auc_b == auc_b else float("nan"),
        "rank_test": rank_test,
    }
    return {"summary": summary, "transfer": transfer, "components": components}


def _rank_test(side: dict[str, Any], layer: int, *, max_rank: int, alpha: float) -> dict[str, Any]:
    """Held-out AUC of a k-dim subspace probe from matched-pair difference PCA."""
    x = side["acts"][layer].float()
    y = side["behavior"]
    train_idx, heldout_idx = side["train_idx"], side["heldout_idx"]
    pair_ids = side["pair_ids"]
    if pair_ids is None:
        return {"available": False, "reason": "no pair_ids"}
    train_pairs = [pair_ids[i] for i in train_idx.tolist()]
    diffs = paired_difference_matrix(x[train_idx], y[train_idx], train_pairs)
    if diffs.shape[0] < 4:
        return {"available": False, "reason": f"only {diffs.shape[0]} complete pairs in train split"}
    centered = diffs - diffs.mean(dim=0, keepdim=True)
    _, s, vt = torch.linalg.svd(centered, full_matrices=False)
    # Include the mean difference itself as the leading axis.
    lead = normalize_vector(diffs.mean(dim=0))
    max_k = min(max_rank, vt.shape[0] + 1)
    results: dict[str, float] = {}
    for k in range(1, max_k + 1):
        columns = [lead] + [vt[i] for i in range(k - 1)]
        basis = orthonormal_basis(torch.stack(columns, dim=1))
        z_train = x[train_idx] @ basis
        z_heldout = x[heldout_idx] @ basis
        weights = ridge_multioutput(z_train, y[train_idx].float().unsqueeze(1), alpha=alpha).squeeze(1)
        scores = (z_heldout - z_train.mean(dim=0, keepdim=True)) @ weights
        results[f"auc_rank_{k}"] = label_auc(scores, y[heldout_idx], "binary")
    results["available"] = True
    results["n_pairs"] = int(diffs.shape[0])
    return results


def _steering_manifest(saved: dict[str, str], *, name_a: str, name_b: str, layer: int) -> dict[str, Any]:
    """Enumerates the causal component x behavior transfer runs (GPU jobs)."""
    conditions = []
    for component_name, path in saved.items():
        for behavior_eval in (name_a, name_b):
            for mode, scales in (("add", [2.0, 4.0, 8.0]), ("ablate", [1.0])):
                conditions.append(
                    {
                        "component": component_name,
                        "vector_path": path,
                        "behavior_eval": behavior_eval,
                        "mode": mode,
                        "scales": scales,
                        "layer": layer,
                    }
                )
    return {
        "description": (
            "Causal transfer matrix: run each component vector against each behavior eval "
            "with matched random-direction controls; shared should move both behaviors, "
            "specific components only their own."
        ),
        "runner": {
            name_a: "mean_diff_steering_sweep.py (judged EM prompts) or offline_intervention.py",
            name_b: "scripts/countdown_code/interp/intervene_countdown_offline.py",
        },
        "conditions": conditions,
        "controls": ["same-norm random vector add", "random projection ablate", "noop"],
    }


def _brief(summary: dict[str, Any]) -> dict[str, Any]:
    keys = ("layer", "cos_direction_a_b", "shared_rank", "shared_fraction_a", "shared_fraction_b", "combined_auc")
    return {key: (round(v, 4) if isinstance(v := summary.get(key), float) else v) for key in keys}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row}, key=lambda k: (k not in ("layer", "component"), k))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


if __name__ == "__main__":
    main()
