#!/usr/bin/env python3
"""Paired, cluster-bootstrap aggregation for ARR factorial and route tables."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

CONDITIONS = ("E", "P", "W", "P+W", "wrong_region_mix", "global_mix", "slowdown")
FACTORIAL = {
    "P_at_W0": ("P", "E"), "W_at_P0": ("W", "E"),
    "combined": ("P+W", "E"),
    "interaction": ("P+W", "P", "W", "E"),
    "incremental_P_at_W1": ("P+W", "W"),
    "incremental_W_at_P1": ("P+W", "P"),
    "W_vs_wrong_region": ("W", "wrong_region_mix"),
    "P+W_vs_wrong_region": ("P+W", "wrong_region_mix"),
    "W_vs_global_mix": ("W", "global_mix"),
    "P+W_vs_global_mix": ("P+W", "global_mix"),
    "W_vs_slowdown": ("W", "slowdown"),
    "P+W_vs_slowdown": ("P+W", "slowdown"),
}


def _read(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cluster_resample(cluster_ids: list[str], rng: np.random.Generator) -> list[str]:
    return list(rng.choice(cluster_ids, size=len(cluster_ids), replace=True))


def factorial_aggregate(rows: Iterable[dict], *, draws: int = 2000,
                        seed: int = 61791, unit: str = "cluster_id",
                        expected_ids: Iterable[str] | None = None) -> list[dict]:
    """Aggregate long rows; require identical prompt IDs for paired contrasts.

    Each row is a condition/outcome/prompt measurement. The bootstrap samples
    clusters once per draw and shares those sampled clusters across conditions.
    """
    groups: dict[tuple, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    clusters: dict[tuple, dict[str, str]] = defaultdict(dict)
    meta: dict[tuple, dict] = {}
    for r in rows:
        key = (r.get("run_id", ""), r.get("model", ""), r.get("seed", ""),
               r.get("rendering", ""), r.get("outcome", ""), r.get("step", ""),
               r.get("judge", ""))
        condition, pid, cid = r["condition"], r["prompt_id"], r[unit]
        if pid in groups[key][condition]:
            raise ValueError(f"duplicate prompt {pid} for {key}, {condition}")
        groups[key][condition][pid] = float(r["value"])
        if pid in clusters[key] and clusters[key][pid] != cid:
            raise ValueError(f"cluster ID changes across conditions for prompt {pid}: "
                             f"{clusters[key][pid]} vs {cid}")
        clusters[key][pid] = cid
        meta[key] = r

    out = []
    expected = set(expected_ids) if expected_ids is not None else None
    for key, conds in groups.items():
        if expected is None:
            raise ValueError("expected_ids is required; pass the frozen prompt ID list")
        for c, values in conds.items():
            actual = set(values)
            if actual != expected:
                raise ValueError(f"expected prompt IDs differ for {key}, {c}: "
                                 f"missing={len(expected-actual)}, unexpected={len(actual-expected)}")
        for name, spec in FACTORIAL.items():
            present = [c for c in spec if c in conds]
            if len(present) > 1:
                reference = set(conds[present[0]])
                for c in present[1:]:
                    ids_c = set(conds[c])
                    if ids_c != reference:
                        raise ValueError(f"paired ID mismatch {key}, {name}: {present[0]} vs {c}; "
                                         f"only_first={len(reference-ids_c)}, only_other={len(ids_c-reference)}")
        cluster_map = clusters[key]
        cluster_ids = sorted(set(cluster_map.values()))
        if not cluster_ids:
            continue
        prompt_ids = {c: sorted(v) for c, v in conds.items()}
        arrays = {c: np.array([conds[c][p] for p in prompt_ids[c]], dtype=float) for c in conds}
        prompt_clusters = {c: np.array([cluster_map[p] for p in prompt_ids[c]]) for c in conds}

        def contrast(means: dict[str, float], spec: tuple[str, ...]) -> float:
            if spec == ("P+W", "P", "W", "E"):
                return means["P+W"] - means["P"] - means["W"] + means["E"]
            return means[spec[0]] - means[spec[1]]

        specs = dict(FACTORIAL)
        available_specs = {name: spec for name, spec in specs.items()
                           if all(c in conds for c in spec)}
        rng = np.random.default_rng(seed)
        boot = {name: [] for name in available_specs}
        for _ in range(draws):
            sampled = _cluster_resample(cluster_ids, rng)
            # Multiplicity is retained when a cluster is drawn more than once.
            means = {}
            for c in conds:
                vals = []
                for cid in sampled:
                    vals.extend(arrays[c][prompt_clusters[c] == cid].tolist())
                means[c] = float(np.mean(vals)) if vals else float("nan")
            for name, spec in available_specs.items():
                boot[name].append(contrast(means, spec))
        means = {c: float(v.mean()) for c, v in arrays.items()}
        source_hash = meta[key].get("source_hash", "")
        for name, spec in specs.items():
            missing_conditions = [c for c in spec if c not in conds]
            if missing_conditions:
                out.append({"run_id": key[0], "model": key[1], "seed": key[2],
                            "rendering": key[3], "outcome": key[4], "step": key[5],
                            "judge": key[6], "contrast": name, "estimate": "",
                            "ci_low": "", "ci_high": "", "bootstrap_unit": unit,
                            "bootstrap_count": draws, "bootstrap_seed": seed,
                            "n_prompts": 0, "n_clusters": 0, "source_hash": source_hash,
                            "null_reason": f"condition data missing: {','.join(missing_conditions)}"})
                continue
            estimate = contrast(means, spec)
            ci = np.nanquantile(boot[name], [0.025, 0.975])
            out.append({"run_id": key[0], "model": key[1], "seed": key[2],
                        "rendering": key[3], "outcome": key[4], "step": key[5],
                        "judge": key[6], "contrast": name, "estimate": estimate,
                        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                        "bootstrap_unit": unit, "bootstrap_count": draws,
                        "bootstrap_seed": seed, "n_prompts": len(prompt_ids.get("E", [])),
                        "n_clusters": len(cluster_ids), "source_hash": source_hash,
                        "null_reason": ""})
        # Print standalone means so all conditions and signs remain inspectable.
        for c, value in means.items():
            out.append({"run_id": key[0], "model": key[1], "seed": key[2],
                        "rendering": key[3], "outcome": key[4], "step": key[5],
                        "judge": key[6], "contrast": f"mean:{c}", "estimate": value,
                        "ci_low": "", "ci_high": "", "bootstrap_unit": unit,
                        "bootstrap_count": draws, "bootstrap_seed": seed,
                        "n_prompts": len(prompt_ids[c]),
                        "n_clusters": len(set(prompt_clusters[c])),
                        "source_hash": source_hash, "null_reason": "CI not requested for condition mean"})
    return out


def route_aggregate(rows: Iterable[dict], *, draws: int = 2000,
                    seed: int = 61791, unit: str = "cluster_id",
                    expected_ids: Iterable[str] | None = None) -> list[dict]:
    """Compute Δ/T/D/M and shared cluster-bootstrap intervals from prompt rows.

    Required per-row scores: score_c, score_x, score_g, score_hg, score_hc.
    The last two are H_C(G_X) and H_C(C), on the identical token sequence.
    """
    groups: dict[tuple, dict[str, dict]] = defaultdict(dict)
    cluster_registry: dict[tuple, dict[str, str]] = defaultdict(dict)
    for r in rows:
        key = tuple(r.get(k, "") for k in ("run_id", "model", "seed", "rendering", "step", "condition", "region", "subset"))
        paired_key = tuple(r.get(k, "") for k in ("run_id", "model", "seed", "rendering", "step", "region", "subset"))
        pid = r["prompt_id"]
        if pid in groups[key]:
            raise ValueError(f"duplicate route prompt {pid} in {key}")
        if pid in groups[key] and groups[key][pid][unit] != r[unit]:
            raise ValueError(f"cluster ID changes for route prompt {pid}")
        if pid in cluster_registry[paired_key] and cluster_registry[paired_key][pid] != r[unit]:
            raise ValueError(f"cluster ID changes across route conditions for prompt {pid}")
        cluster_registry[paired_key][pid] = r[unit]
        groups[key][pid] = r
    out = []
    expected = set(expected_ids) if expected_ids is not None else None
    cols = ("score_c", "score_x", "score_g", "score_hg", "score_hc")
    for key, values in groups.items():
        by_id = values
        if any(any(c not in r or r[c] == "" for c in cols) for r in by_id.values()):
            raise ValueError(f"missing route score for {key}")
        ids = sorted(by_id)
        if expected is None:
            raise ValueError("expected_ids is required; pass the frozen prompt ID list")
        actual = set(ids)
        if actual != expected:
            raise ValueError(f"expected route prompt IDs differ for {key}: "
                             f"missing={len(expected-actual)}, unexpected={len(actual-expected)}")
        cid = {p: by_id[p][unit] for p in ids}
        cluster_ids = sorted(set(cid.values()))
        vectors = {}
        for name, fn in {
            "S_C": lambda r: float(r["score_c"]),
            "S_X": lambda r: float(r["score_x"]),
            "S_G": lambda r: float(r["score_g"]),
            "S_HG": lambda r: float(r["score_hg"]),
            "S_HC": lambda r: float(r["score_hc"]),
            "delta": lambda r: float(r["score_x"]) - float(r["score_c"]),
            "T": lambda r: float(r["score_g"]) - float(r["score_c"]),
            "D": lambda r: float(r["score_hg"]) - float(r["score_hc"]),
            "M": lambda r: (float(r["score_g"]) - float(r["score_c"])) - (float(r["score_hg"]) - float(r["score_hc"])),
            "decomposition_arithmetic_residual": lambda r: ((float(r["score_g"])-float(r["score_c"])) -
                                         (float(r["score_hg"])-float(r["score_hc"])) -
                                         ((float(r["score_g"])-float(r["score_c"])) -
                                          (float(r["score_hg"])-float(r["score_hc"])))),
            "identity_baseline_error": lambda r: float(r["score_hc"]) - float(r["score_c"]),
        }.items():
            vectors[name] = np.array([fn(by_id[p]) for p in ids])
        prompt_cluster = np.array([cid[p] for p in ids])
        rng = np.random.default_rng(seed)
        boot = {name: [] for name in vectors}
        bootstrap_indices = []
        for _ in range(draws):
            sampled = _cluster_resample(cluster_ids, rng)
            ix = np.concatenate([np.flatnonzero(prompt_cluster == c) for c in sampled])
            bootstrap_indices.append(ix)
            for name, vals in vectors.items():
                boot[name].append(float(vals[ix].mean()))
        base = by_id[ids[0]]
        for name, vals in vectors.items():
            ci = np.nanquantile(boot[name], [0.025, 0.975])
            out.append({"run_id": key[0], "model": key[1], "seed": key[2],
                        "rendering": key[3], "step": key[4], "condition": key[5],
                        "region": key[6], "subset": key[7], "outcome": name, "estimate": float(vals.mean()),
                        "ci_low": float(ci[0]), "ci_high": float(ci[1]),
                        "bootstrap_unit": unit, "bootstrap_count": draws,
                        "bootstrap_seed": seed, "n_prompts": len(ids),
                        "n_clusters": len(cluster_ids), "source_path": base.get("source_path", ""),
                        "source_hash": base.get("source_hash", ""), "null_reason": ""})
        # Ratio rows are conditional on an interval excluding zero.
        for label, numerator, denominator, ci_den in (
                ("removed_fraction", "M", "T", None), ("coverage", "T", "delta", None)):
            ratio_draws = []
            if label == "coverage":
                delta_vals = np.array([float(by_id[p]["score_x"])-float(by_id[p]["score_c"]) for p in ids])
                den_values = []
                for ix in bootstrap_indices:
                    den_values.append(float(delta_vals[ix].mean()))
            else:
                den_values = boot["T"]
            ci_d = np.nanquantile(den_values, [0.025, 0.975])
            denominator_point = (float(vectors["T"].mean()) if label == "removed_fraction"
                                 else float(vectors["delta"].mean()))
            stable = ci_d[0] > 0 or ci_d[1] < 0
            num = float(vectors[numerator].mean())
            num_values = boot[numerator]
            for b in range(draws):
                dval = den_values[b]
                ratio_draws.append(num_values[b] / dval if abs(dval) > 1e-15 else np.nan)
            ratio_ci = np.nanquantile(ratio_draws, [0.025, 0.975]) if stable else (np.nan, np.nan)
            out.append({"run_id": key[0], "model": key[1], "seed": key[2],
                        "rendering": key[3], "step": key[4], "condition": key[5],
                        "region": key[6], "subset": key[7], "outcome": label,
                        "estimate": num / denominator_point if stable and denominator_point else "",
                        "ci_low": float(ratio_ci[0]) if stable else "",
                        "ci_high": float(ratio_ci[1]) if stable else "", "bootstrap_unit": unit,
                        "bootstrap_count": draws, "bootstrap_seed": seed,
                        "n_prompts": len(ids), "n_clusters": len(cluster_ids),
                        "source_path": base.get("source_path", ""),
                        "source_hash": base.get("source_hash", ""),
                        "null_reason": "denominator bootstrap interval includes zero" if not stable else ""})
    return out


SECTION9_ROUTE_FIELDS = {
    "S_C": "S_C", "S_X": "S_X", "S_G": "S_G", "S_HG": "S_HG",
    "S_HC": "S_HC", "Delta": "delta", "T": "T", "D": "D", "M": "M",
    "removed_fraction": "removed_fraction", "coverage": "coverage",
    "identity_error": "identity_baseline_error",
}


def section9_route_rows(rows: Iterable[dict]) -> list[dict]:
    """Pivot route aggregates to the wide row schema required by Section 9.

    Reverse graft necessity is a distinct intervention and is not inferred from
    the five forward-route scores. It remains null, with a reason, until a
    reverse-graft measurement is supplied.
    """
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = tuple(row.get(k, "") for k in (
            "run_id", "model", "seed", "rendering", "step", "condition", "region", "subset"))
        outcome = row["outcome"]
        if outcome in SECTION9_ROUTE_FIELDS.values() or outcome == "reverse_necessity":
            if outcome in grouped[key]:
                raise ValueError(f"duplicate route outcome {outcome} for {key}")
            grouped[key][outcome] = row

    required = set(SECTION9_ROUTE_FIELDS.values())
    output = []
    for key, outcomes in grouped.items():
        missing = required - set(outcomes)
        if missing:
            raise ValueError(f"Section 9 route row {key} is missing outcomes: {sorted(missing)}")
        if not key[6]:
            raise ValueError(f"Section 9 route row {key} has no region label")
        result = dict(zip(("run_id", "model", "seed", "rendering", "step", "condition", "region", "subset"), key))
        for field, outcome in SECTION9_ROUTE_FIELDS.items():
            source = outcomes[outcome]
            result[field] = source.get("estimate", "")
            result[f"{field}_ci_low"] = source.get("ci_low", "")
            result[f"{field}_ci_high"] = source.get("ci_high", "")
            if source.get("null_reason"):
                result[f"{field}_null_reason"] = source["null_reason"]
        reverse = outcomes.get("reverse_necessity")
        result["reverse_necessity"] = reverse.get("estimate", "") if reverse else ""
        result["reverse_necessity_null_reason"] = (
            reverse.get("null_reason", "") if reverse else
            "reverse graft is a distinct intervention and was not present in route input")
        result["bootstrap_unit"] = outcomes["T"].get("bootstrap_unit", "")
        result["bootstrap_count"] = outcomes["T"].get("bootstrap_count", "")
        result["bootstrap_seed"] = outcomes["T"].get("bootstrap_seed", "")
        result["n_prompts"] = outcomes["T"].get("n_prompts", "")
        result["n_clusters"] = outcomes["T"].get("n_clusters", "")
        result["source_path"] = outcomes["T"].get("source_path", "")
        result["source_hash"] = outcomes["T"].get("source_hash", "")
        output.append(result)
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("factorial", "routes"):
        p = sub.add_parser(name)
        p.add_argument("input", type=Path)
        p.add_argument("output", type=Path)
        p.add_argument("--draws", type=int, default=2000)
        p.add_argument("--seed", type=int, default=61791)
        p.add_argument("--expected-ids", type=Path, required=True,
                       help="JSON list or object with frozen expected prompt IDs")
        if name == "routes":
            p.add_argument("--section9", action="store_true",
                           help="write the Section 9 wide routes.csv schema")
    args = ap.parse_args()
    rows = _read(args.input)
    expected_doc = json.loads(args.expected_ids.read_text())
    expected_ids = expected_doc if isinstance(expected_doc, list) else expected_doc.get("prompt_ids")
    if not expected_ids:
        raise ValueError("expected-ids JSON must contain a nonempty list")
    if args.command == "factorial":
        results = factorial_aggregate(rows, draws=args.draws, seed=args.seed, expected_ids=expected_ids)
    else:
        results = route_aggregate(rows, draws=args.draws, seed=args.seed, expected_ids=expected_ids)
        if args.section9:
            results = section9_route_rows(results)
    _write(args.output, results)
    print(json.dumps({"input": str(args.input), "input_sha256": _hash(args.input),
                      "output": str(args.output), "rows": len(results),
                      "bootstrap_draws": args.draws, "seed": args.seed}, indent=2))


if __name__ == "__main__":
    main()
