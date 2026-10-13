from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler

from .common import digest, file_hash, manifest, write_json
from .interventions import random_basis, save_basis


def read_cache(path):
    result = torch.load(path, map_location="cpu", weights_only=True)
    if result.get("kind") != "activations":
        raise ValueError("Expected activation cache")
    if any(len(x) != len(result["rows"]) for x in result["activations"].values()):
        raise ValueError("Activation/metadata row mismatch")
    return result


def combine_caches(paths):
    caches = [read_cache(path) for path in paths]
    first = caches[0]
    for cache in caches[1:]:
        if cache["source"] != first["source"] or cache["tokenizer_hash"] != first["tokenizer_hash"]:
            raise ValueError("Cannot combine caches from different model states or tokenizers")
        if cache["activations"].keys() != first["activations"].keys():
            raise ValueError("Cache locations differ")
    rows = [r for cache in caches for r in cache["rows"]]
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate rows across caches")
    return {**first, "rows": rows, "activations": {key: torch.cat([c["activations"][key] for c in caches]) for key in first["activations"]}}


def _aligned(original, recipient):
    if original["tokenizer_hash"] != recipient["tokenizer_hash"]:
        raise ValueError("Tokenizer mismatch")
    index = {r["id"]: i for i, r in enumerate(recipient["rows"])}
    order = []
    for row in original["rows"]:
        if row["id"] not in index:
            raise ValueError(f"Unpaired activation row {row['id']}")
        j = index[row["id"]]
        if row["prompt_token_ids"] != recipient["rows"][j]["prompt_token_ids"]:
            raise ValueError("Model-difference activations require identical tokens")
        order.append(j)
    return order


def factorial_deltas(original, recipient, location):
    order = _aligned(original, recipient)
    indices = [i for i, row in enumerate(original["rows"]) if row["split"] == "discovery" and not row.get("em_eligible")]
    if not indices:
        raise ValueError("Basis discovery cannot use validation or test rows")
    x = original["activations"][location].float()
    y = recipient["activations"][location][order].float()
    grouped = {}
    for i in indices:
        row = original["rows"][i]
        grouped.setdefault(row["group_id"], {})[(row["constraint"], row["role"])] = i
    persona, instruction = [], []
    for group in grouped.values():
        for constraint in ("free", "json", "lower", "upper"):
            a, b = group.get((constraint, "assistant")), group.get((constraint, "self_serving"))
            if a is not None and b is not None:
                persona.append(x[a] - x[b])
        for role in ("assistant", "self_serving"):
            free = group.get(("free", role))
            if free is None:
                continue
            for constraint in ("json", "lower", "upper"):
                i = group.get((constraint, role))
                if i is not None:
                    # Difference-in-differences removes the unconstrained model delta.
                    instruction.append((x[i] - y[i]) - (x[free] - y[free]))
    if not persona or not instruction:
        raise ValueError("Missing complete task/role/constraint counterfactuals")
    return torch.stack(persona), torch.stack(instruction), [original["rows"][i]["id"] for i in indices]


def variance_matched_random(x, reference, seed, candidates=128):
    x = x - x.mean(0)
    target = (x @ reference).square().mean().item()
    candidates_and_error = []
    for i in range(candidates):
        u = random_basis(x.shape[1], reference.shape[1], seed + i)
        energy = (x @ u).square().mean().item()
        candidates_and_error.append((abs(energy - target), u, energy))
    error, u, energy = min(candidates_and_error, key=lambda item: item[0])
    return u, {"target_energy": target, "control_energy": energy, "relative_error": error / max(target, 1e-12),
               "matched_within_10_percent": error <= max(target, 1e-12) * .1}


def fit_bases(original_path: Path, recipient_path: Path, output: Path, location: str, ranks: list[int], seed=0):
    original, recipient = read_cache(original_path), read_cache(recipient_path)
    persona, instruction, ids = factorial_deltas(original, recipient, location)
    if "/" not in location or location.startswith("embedding"):
        raise ValueError("Intervention basis requires a transformer location")
    site, position = location.split("/")
    matrices = {"persona": persona, "instruction": instruction,
                "joint": torch.cat([persona / persona.norm().clamp_min(1e-12), instruction / instruction.norm().clamp_min(1e-12)])}
    info = manifest(kind="probe_derived_bases", original_sha256=file_hash(original_path), recipient_sha256=file_hash(recipient_path),
                    discovery_ids=ids, location=location, seed=seed,
                    interpretation="Candidate contrast subspaces; causal and persona specificity are unvalidated.")
    records = []
    discovery = [i for i, r in enumerate(original["rows"]) if r["split"] == "discovery"]
    calibration = original["activations"][location][discovery].float()
    for name, matrix in matrices.items():
        _, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        available = int((singular > singular[0].clamp_min(1e-12) * 1e-6).sum())
        for rank in ranks:
            if rank > available:
                records.append({"target": name, "rank": rank, "status": "insufficient_rank", "available": available})
                continue
            u = vh[:rank].T.contiguous()
            destination = output / f"{name}_r{rank}.pt"
            save_basis(destination, {site: u}, position=position, target=name, discovery_ids=ids, metadata=info)
            random = random_basis(u.shape[0], rank, seed)
            save_basis(output / f"{name}_random_r{rank}.pt", {site: random}, target="random", position=position, metadata=info)
            matched, match_info = variance_matched_random(calibration, u, seed)
            save_basis(output / f"{name}_variance_r{rank}.pt", {site: matched}, target="variance_control", position=position,
                       variance_matching=match_info, metadata=info)
            records.append({"target": name, "rank": rank, "status": "fit", "artifact": str(destination),
                            "variance_matching": match_info, "singular_values": singular[:rank].tolist()})
    mean = persona.mean(0)
    if mean.norm() > 1e-8:
        save_basis(output / "persona_mean.pt", {site: (mean / mean.norm())[:, None]}, target="persona_mean", position=position, metadata=info)
    write_json(output / "fit.json", {**info, "fits": records})
    return records


def subspace_geometry(a: torch.Tensor, b: torch.Tensor) -> dict:
    cosines = torch.linalg.svdvals(a.float().T @ b.float()).clamp(0, 1)
    overlap = cosines.square().sum().item()
    return {"principal_angles_degrees": torch.rad2deg(torch.acos(cosines)).tolist(),
            "fraction_b_in_a": overlap / b.shape[1], "fraction_a_in_b": overlap / a.shape[1],
            "isotropic_null_fraction_b_in_a": a.shape[1] / a.shape[0]}


def probe_atlas(paths: list[Path], output: Path, transfer_paths: list[Path] | None = None):
    """Measure accessibility with discovery/validation alone or all three splits.

    A validation-only atlas fixes the ridge penalty before evaluation. When a
    third split exists, validation selects the penalty and test remains unseen.
    """
    cache = combine_caches(paths)
    transfer = combine_caches(transfer_paths) if transfer_paths else None
    rows = cache["rows"]
    splits = {s: np.array([i for i, r in enumerate(rows) if r["split"] == s and not r.get("em_eligible")])
              for s in ("discovery", "validation", "test")}
    if len(splits["discovery"]) < 4 or len(splits["validation"]) < 4:
        raise ValueError("Atlas requires independent discovery and validation caches")
    has_test = len(splits["test"]) >= 4
    train, validation = splits["discovery"], splits["validation"]
    evaluation = splits["test"] if has_test else validation
    transfer_order = _aligned(cache, transfer) if transfer else None
    results = []
    targets = {"role_label": "classification", "constraint": "classification", "family": "classification",
               "answer": "regression", "intermediate": "regression", "bound_argument": "regression"}

    def fit_records(location, scope, active_train, active_validation, active_evaluation):
        x = cache["activations"][location].float().numpy()
        scaler = StandardScaler().fit(x[active_train])
        features = scaler.transform(x)
        for target, task in targets.items():
            if scope != "all" and target == "family":
                continue
            raw_y = [r.get(target, 0) for r in rows]
            classes = sorted({raw_y[i] for i in active_train}, key=str)
            if task == "classification":
                if len(classes) < 2:
                    continue
                if any(raw_y[i] not in classes for i in np.concatenate((active_validation, active_evaluation))):
                    continue
                y = np.array([classes.index(v) if v in classes else -1 for v in raw_y])
                training_y = np.eye(len(classes))[y[active_train]]
            else:
                y = np.asarray(raw_y, dtype=float)
                training_y = y[active_train]
            metric = accuracy_score if task == "classification" else r2_score
            if has_test:
                candidates = []
                for alpha in (.1, 1., 10., 100.):
                    probe = Ridge(alpha=alpha, solver="lsqr").fit(features[active_train], training_y)
                    predicted = probe.predict(features[active_validation])
                    if task == "classification":
                        predicted = predicted.argmax(1)
                    candidates.append((float(metric(y[active_validation], predicted)), probe, alpha))
                validation_score, probe, alpha = max(candidates, key=lambda item: item[0])
                selection = "validation"
            else:
                alpha, selection = 1., "fixed_alpha_no_test_cache"
                probe = Ridge(alpha=alpha, solver="lsqr").fit(features[active_train], training_y)
                validation_score = None
            predicted = probe.predict(features[active_evaluation])
            if task == "classification":
                predicted = predicted.argmax(1)
            score = float(metric(y[active_evaluation], predicted))
            rng = np.random.default_rng(int(digest([location, target, scope])[:16], 16))
            shuffled_order = np.array(active_train)
            # Preserve family composition while destroying within-family labels.
            for family in sorted({rows[i].get("family") for i in active_train}):
                members = np.array([i for i in active_train if rows[i].get("family") == family])
                slots = np.flatnonzero(np.isin(shuffled_order, members))
                shuffled_order[slots] = rng.permutation(members)
            shuffled_y = (np.eye(len(classes))[y[shuffled_order]] if task == "classification" else y[shuffled_order])
            null = Ridge(alpha=alpha, solver="lsqr").fit(features[active_train], shuffled_y)
            null_prediction = null.predict(features[active_evaluation])
            if task == "classification":
                null_prediction = null_prediction.argmax(1)
            record = {"location": location, "target": target, "metric": "accuracy" if task == "classification" else "r2",
                      "scope": scope, "selection": selection, "alpha": alpha,
                      "evaluation_split": "test" if has_test else "validation", "evaluation": score,
                      "validation": validation_score, "test": score if has_test else None,
                      "shuffled_evaluation": float(metric(y[active_evaluation], null_prediction)),
                      "shuffled_test": float(metric(y[active_evaluation], null_prediction)) if has_test else None,
                      "interpretation": "Information accessibility, not demonstrated causal use"}
            if transfer:
                transfer_x = transfer["activations"][location][transfer_order].float().numpy()
                prediction = probe.predict(scaler.transform(transfer_x[active_evaluation]))
                if task == "classification":
                    prediction = prediction.argmax(1)
                record["transferred_evaluation"] = float(metric(y[active_evaluation], prediction))
                record["transferred_test"] = record["transferred_evaluation"] if has_test else None
            results.append(record)

    for location in cache["activations"]:
        fit_records(location, "all", train, validation, evaluation)
        for family in sorted({r.get("family") for r in rows if r.get("family") and r.get("family") != "em"}):
            family_train = np.array([i for i in train if rows[i].get("family") == family])
            family_validation = np.array([i for i in validation if rows[i].get("family") == family])
            family_evaluation = np.array([i for i in evaluation if rows[i].get("family") == family])
            if min(len(family_train), len(family_validation), len(family_evaluation)) >= 4:
                fit_records(location, family, family_train, family_validation, family_evaluation)
    write_json(output, manifest(kind="probe_atlas", results=results, available_splits={k: len(v) for k, v in splits.items()},
                                cache_hashes={str(p): file_hash(p) for p in paths},
                                transfer_hashes={str(p): file_hash(p) for p in (transfer_paths or [])}))
    return results


def erasure_curve(x_train, y_train, x_test, y_test, max_rank=64, alpha=1.0):
    """Iterative linear erasure for a scalar label; held-out metric never selects axes."""
    x = x_train.double()
    z = x_test.double()
    x_mean, y_mean = x.mean(0), y_train.double().mean()
    x, z = x - x_mean, z - x_mean
    y = y_train.double() - y_mean
    directions, curve = [], []
    for k in range(min(max_rank, x.shape[1]) + 1):
        # Dual ridge avoids a hidden-width squared solve when examples are few.
        w = x.T @ torch.linalg.solve(x @ x.T + alpha * torch.eye(len(x), dtype=x.dtype), y)
        prediction = z @ w + y_mean
        curve.append({"removed_rank": k, "r2": float(r2_score(y_test, prediction))})
        if k == max_rank or w.norm() < 1e-10:
            break
        if directions:
            q = torch.stack(directions, dim=1)
            w = w - q @ (q.T @ w)
        if w.norm() < 1e-10:
            break
        u = w / w.norm()
        directions.append(u)
        x = x - (x @ u)[:, None] * u
        z = z - (z @ u)[:, None] * u
    return curve
