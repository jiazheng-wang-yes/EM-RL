"""Build immutable Stage 6B endpoint tables from a validated full seed."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from acl_common import ACL_ROOT, CALIBRATION_MODEL, ROOT, sha256_file, stable_seed
from common import ClusterBootstrap, prompt_cluster

PRIMARY = ("E", "P", "W", "P+W")
LABELS = {
    "E": "M00",
    "P": "M10",
    "W": "M01",
    "P+W": "M11",
    "wrong_region_mix": "C1_wrong_region",
    "global_mix": "C2_global_mix",
    "slowdown": "C3_slowdown",
}
N_BOOT = 2000


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _bootstrap_mean(values: dict, seed: int) -> dict:
    series = pd.Series(values, dtype=float).sort_index()
    if not len(series):
        raise ValueError("cannot bootstrap an empty prompt map")
    bootstrap = ClusterBootstrap(series.index.tolist(), n_boot=N_BOOT, seed=seed)
    point, samples = bootstrap.means(series.to_numpy())
    lo, hi = bootstrap.ci(samples)
    return dict(point=float(point), ci_low=lo, ci_high=hi,
                prompt_count=int(len(series)), cluster_count=len({prompt_cluster(key) for key in series.index}),
                per_prompt={str(key): float(value) for key, value in series.items()})


def _bootstrap_reduction(candidate: dict, baseline: dict, seed: int) -> dict:
    ids = sorted(set(candidate) & set(baseline))
    if not ids:
        return dict(point=None, ci_low=None, ci_high=None, prompt_count=0,
                    reason="candidate and baseline have no common prompts")
    candidate_values = np.asarray([candidate[key] for key in ids], dtype=float)
    baseline_values = np.asarray([baseline[key] for key in ids], dtype=float)
    base_point = float(np.mean(baseline_values))
    if not math.isfinite(base_point) or abs(base_point) <= 1e-12:
        return dict(point=None, ci_low=None, ci_high=None, prompt_count=len(ids),
                    reason="ordinary harmful-SFT deterministic gap is zero or non-finite")
    bootstrap = ClusterBootstrap(ids, n_boot=N_BOOT, seed=seed)
    weights = bootstrap.W
    denominator = weights.sum(1)
    base_samples = (weights @ baseline_values) / denominator
    candidate_samples = (weights @ candidate_values) / denominator
    valid = np.isfinite(base_samples) & np.isfinite(candidate_samples) & (np.abs(base_samples) > 1e-12)
    ratios = 1.0 - candidate_samples[valid] / base_samples[valid]
    if len(ratios) < int(.95 * N_BOOT):
        return dict(point=float(1 - np.mean(candidate_values) / base_point),
                    ci_low=None, ci_high=None, prompt_count=len(ids),
                    valid_bootstraps=int(len(ratios)),
                    reason="too many bootstrap denominator draws were near zero")
    return dict(point=float(1 - np.mean(candidate_values) / base_point),
                ci_low=float(np.quantile(ratios, .025)), ci_high=float(np.quantile(ratios, .975)),
                prompt_count=len(ids), cluster_count=len({prompt_cluster(key) for key in ids}),
                valid_bootstraps=int(len(ratios)))


def _bootstrap_retained_learning(candidate: dict, base: dict, ordinary_harmful: dict, seed: int) -> dict:
    ids = sorted(set(candidate) & set(base) & set(ordinary_harmful))
    if not ids:
        return dict(point=None, ci_low=None, ci_high=None, prompt_count=0,
                    reason="candidate, base, and ordinary-SFT preference maps have no common rows")
    candidate_values = np.asarray([candidate[key] for key in ids], dtype=float)
    base_values = np.asarray([base[key] for key in ids], dtype=float)
    ordinary_values = np.asarray([ordinary_harmful[key] for key in ids], dtype=float)
    numerator = candidate_values - base_values
    denominator_values = ordinary_values - base_values
    denominator = float(np.mean(denominator_values))
    if not math.isfinite(denominator) or denominator <= 1e-12:
        return dict(point=None, ci_low=None, ci_high=None, prompt_count=len(ids),
                    reason="ordinary harmful-SFT medical preference improvement is nonpositive")
    point = float(np.mean(numerator) / denominator)
    bootstrap = ClusterBootstrap(ids, n_boot=N_BOOT, seed=seed)
    weights = bootstrap.W
    denominator_draws = weights @ denominator_values
    numerator_draws = weights @ numerator
    valid = np.isfinite(denominator_draws) & (denominator_draws > 1e-12)
    ratios = numerator_draws[valid] / denominator_draws[valid]
    if len(ratios) < int(.95 * N_BOOT):
        return dict(point=point, ci_low=None, ci_high=None, prompt_count=len(ids),
                    cluster_count=len({prompt_cluster(key) for key in ids}),
                    valid_bootstraps=int(len(ratios)),
                    reason="too many bootstrap denominator draws were nonpositive")
    return dict(point=point, ci_low=float(np.quantile(ratios, .025)),
                ci_high=float(np.quantile(ratios, .975)), prompt_count=len(ids),
                cluster_count=len({prompt_cluster(key) for key in ids}),
                valid_bootstraps=int(len(ratios)))


def _factorial_retained_learning(maps: dict[str, dict], base: dict, seed: int) -> list[dict]:
    if not set(PRIMARY).issubset(maps):
        return [dict(outcome="narrow_U", status="not_estimable_missing_primary_condition",
                     missing=sorted(set(PRIMARY) - set(maps)))]
    ids = sorted(set(base).intersection(*(set(maps[c]) for c in PRIMARY)))
    if not ids:
        return [dict(outcome="narrow_U", status="not_estimable_no_common_prompt_values")]
    base_values = np.asarray([base[key] for key in ids], dtype=float)
    values = {condition: np.asarray([maps[condition][key] for key in ids], dtype=float)
              for condition in PRIMARY}
    denominator_values = values["E"] - base_values
    if float(np.mean(denominator_values)) <= 1e-12:
        return [dict(outcome="narrow_U", status="not_estimable_nonpositive_E_denominator")]
    bootstrap = ClusterBootstrap(ids, n_boot=N_BOOT, seed=stable_seed("acl-stage6b-factorial-narrow", seed))
    weights = bootstrap.W
    denominator_draws = weights @ denominator_values
    valid = np.isfinite(denominator_draws) & (denominator_draws > 1e-12)
    if int(valid.sum()) < int(.95 * N_BOOT):
        return [dict(outcome="narrow_U", status="not_estimable_bootstrap_denominator")]
    u_draws = {condition: (weights @ (values[condition] - base_values))[valid] / denominator_draws[valid]
               for condition in PRIMARY}
    contrasts = {
        "beta_0": u_draws["E"],
        "beta_P": u_draws["P"] - u_draws["E"],
        "beta_D": u_draws["W"] - u_draws["E"],
        "beta_PD": u_draws["P+W"] - u_draws["P"] - u_draws["W"] + u_draws["E"],
    }
    observed = {condition: float(np.mean(values[condition] - base_values)
                                 / np.mean(denominator_values)) for condition in PRIMARY}
    points = {
        "beta_0": observed["E"], "beta_P": observed["P"] - observed["E"],
        "beta_D": observed["W"] - observed["E"],
        "beta_PD": observed["P+W"] - observed["P"] - observed["W"] + observed["E"],
    }
    return [dict(
        outcome="narrow_U", term=term, point=points[term],
        ci_low=float(np.quantile(samples, .025)), ci_high=float(np.quantile(samples, .975)),
        prompt_count=len(ids), cluster_count=len({prompt_cluster(key) for key in ids}),
        valid_bootstraps=int(valid.sum()), inference="paired prompt-cluster bootstrap of normalized utility",
    ) for term, samples in contrasts.items()]


def _read_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object: {path}")
    return value


def _load_ifbench_paired(path: Path, conditions: set[str]) -> dict[tuple[str, str], dict]:
    if not path.is_file():
        raise FileNotFoundError(f"missing paired IFBench results: {path}")
    frame = pd.read_csv(path)
    required = {"condition", "reference", "category", "metric", "difference_vs_E", "ci_low", "ci_high"}
    if not required.issubset(frame.columns):
        raise ValueError(f"paired IFBench table is missing required fields: {path}")
    selected = frame[(frame["reference"] == "E") & (frame["category"] == "overall")
                     & frame["metric"].isin(("strict_all", "loose_all"))]
    result = {}
    for row in selected.to_dict("records"):
        key = (str(row["condition"]), str(row["metric"]))
        if key in result:
            raise ValueError(f"duplicate paired IFBench row: {key}")
        result[key] = row
    expected = {(condition, metric) for condition in conditions
                for metric in ("strict_all", "loose_all")}
    if set(result) != expected:
        missing, extra = sorted(expected - set(result)), sorted(set(result) - expected)
        raise ValueError(f"paired IFBench rows differ from trained conditions; missing={missing}, extra={extra}")
    return result


def _load_training(record: dict, model: str, seed: int) -> tuple[dict, dict[int, dict]]:
    run_dir = Path(record["training_run"])
    manifest = _read_json(run_dir / "manifest.json", f"{record['condition']} training manifest")
    if (manifest.get("model") != model or manifest.get("seed") != seed
            or manifest.get("condition") != record["condition"] or manifest.get("steps") != 184):
        raise ValueError(f"training provenance mismatch for {record['condition']}")
    rows = json.loads((run_dir / "metrics.json").read_text())
    if not isinstance(rows, list):
        raise ValueError(f"metrics.json is not a list for {record['condition']}")
    by_step = {int(row["step"]): row for row in rows}
    if sorted(by_step) != list(range(185)):
        raise ValueError(f"training metrics do not cover steps 0–184 for {record['condition']}")
    return manifest, by_step


def _route_record(record: dict, model: str, seed: int, assay: dict) -> dict:
    condition = record["condition"]
    path = record.get("route_check")
    if path:
        route = _read_json(Path(path), f"{condition} CRD route check")
        if (route.get("model") != model or route.get("seed") != seed
                or route.get("condition") != condition or route.get("rendering") != "training"):
            raise ValueError(f"route-check provenance mismatch for {condition}")
        persona = route.get("persona", {})
        return dict(
            condition=condition, label=LABELS.get(condition, condition), source="endpoint_crd",
            delta_P=persona.get("delta_P"), delta_P_ci=persona.get("delta_P_ci"),
            per_prompt_delta_P=persona.get("per_prompt_delta"),
            TE=route.get("TE"), DE=route.get("DE"), MF=route.get("MF"),
            absolute_mediated_effect=route.get("absolute_mediated_effect"),
            clamp_identity_max_abs=route.get("clamp_identity_max_abs"),
            S=route.get("S"), neutral_quality=route.get("neutral_quality", {}).get("M"),
            route_check_path=str(path),
        )
    direct_te, direct_de = assay.get("direct_TE"), assay.get("direct_DE")
    persona = assay.get("delta_P")
    return dict(
        condition=condition, label=LABELS.get(condition, condition), source="in_training_endpoint_snapshot",
        delta_P=persona.get("point") if isinstance(persona, dict) else None,
        delta_P_ci=([persona.get("ci_low"), persona.get("ci_high")]
                    if isinstance(persona, dict) else None),
        per_prompt_delta_P=persona.get("per_prompt") if isinstance(persona, dict) else None,
        TE=direct_te, DE=direct_de, MF=assay.get("direct_MF"),
        absolute_mediated_effect=assay.get("absolute_direct_mediated_effect"),
        clamp_identity_max_abs=assay.get("clamp_identity_max_abs"),
        S=dict(M=assay.get("S_mean"), C=assay.get("S_C")),
        neutral_quality=None, route_check_path=None,
    )


def _flat_route_record(row: dict) -> dict:
    """Keep route endpoint tables numeric/CSV-friendly; prompt maps live separately."""
    result = {key: value for key, value in row.items()
              if key not in {"TE", "DE", "delta_P_ci", "per_prompt_delta_P", "S",
                             "absolute_mediated_effect", "neutral_quality"}}
    for metric in ("TE", "DE"):
        value = row.get(metric)
        if isinstance(value, dict):
            result[metric] = value.get("point")
            result[f"{metric}_ci_low"] = value.get("ci_low")
            result[f"{metric}_ci_high"] = value.get("ci_high")
        else:
            result[metric] = value
            result[f"{metric}_ci_low"] = None
            result[f"{metric}_ci_high"] = None
    ci = row.get("delta_P_ci")
    result["delta_P_ci_low"] = (ci[0] if isinstance(ci, list) and len(ci) == 2 else None)
    result["delta_P_ci_high"] = (ci[1] if isinstance(ci, list) and len(ci) == 2 else None)
    mediated = row.get("absolute_mediated_effect")
    result["absolute_mediated_effect"] = (mediated.get("point") if isinstance(mediated, dict)
                                          else mediated)
    neutral = row.get("neutral_quality") or {}
    if not isinstance(neutral, dict):
        neutral = {}
    for metric in ("lp_mean", "ent_mean", "agree_C"):
        result[f"neutral_{metric}"] = neutral.get(metric)
    scores = row.get("S") or {}
    result["S_M"] = scores.get("M") if isinstance(scores, dict) else None
    result["S_C"] = scores.get("C") if isinstance(scores, dict) else None
    return result


def _judge_summary(eval_dir: Path, condition: str, model: str, seed: int) -> dict:
    rows_path = eval_dir / "responses.csv"
    run = _read_json(eval_dir / "run.json", f"{condition} open-ended run")
    if (run.get("model") != model or run.get("condition") != condition
            or run.get("output_stage") != "endpoints" or run.get("generated") != 480
            or run.get("renderings") != ["training"] or run.get("judges_enabled") is not True):
        raise ValueError(f"open-ended run provenance or completeness mismatch for {condition}")
    rows = pd.read_csv(rows_path)
    judge = "llama" if model.startswith("qwen") else "qwen"
    align_key, coherence_key = f"{judge}_alignment", f"{judge}_coherence"
    em_key, refusal_key = f"{judge}_em", f"{judge}_refusal"
    required = {"prompt_id", "sample_idx", align_key, coherence_key, em_key, refusal_key,
                "qwen_em", "llama_em"}
    if not required.issubset(rows.columns):
        raise ValueError(f"open-ended responses lack frozen judge columns for {condition}")
    if len(rows) != 480 or rows[["prompt_id", "sample_idx"]].drop_duplicates().shape[0] != 480:
        raise ValueError(f"open-ended endpoint is not a complete 16×30 set for {condition}")
    prompt_values = rows.groupby("prompt_id", sort=True)

    def prompt_means(column: str) -> dict[str, float]:
        values = prompt_values[column].mean().dropna()
        return {str(key): float(value) for key, value in values.items()}

    def metric_ci(column: str):
        values = prompt_means(column)
        if not values:
            return dict(mean=None, ci_low=None, ci_high=None, valid_prompts=0)
        result = _bootstrap_mean(values, stable_seed("acl-stage6b-open", model, seed, condition, judge, column))
        return dict(mean=result["point"], ci_low=result["ci_low"], ci_high=result["ci_high"],
                    valid_prompts=result["cluster_count"])

    alignment = metric_ci(align_key)
    misalignment = dict(
        mean=(100.0 - alignment["mean"] if alignment["mean"] is not None else None),
        ci_low=(100.0 - alignment["ci_high"] if alignment["ci_high"] is not None else None),
        ci_high=(100.0 - alignment["ci_low"] if alignment["ci_low"] is not None else None),
        valid_prompts=alignment["valid_prompts"],
    )
    coherence = metric_ci(coherence_key)
    mr = metric_ci(em_key)
    refusal = metric_ci(refusal_key)
    both = rows["qwen_em"].notna() & rows["llama_em"].notna()
    disagreements = int((rows.loc[both, "qwen_em"] != rows.loc[both, "llama_em"]).sum())
    valid_n = int(rows[em_key].notna().sum())
    return dict(
        model=model, seed=seed, condition=condition, primary_judge=judge,
        primary_alignment=alignment, primary_misalignment_score=misalignment,
        coherence=coherence, MR=mr, refusal=refusal,
        valid_primary_judgments=valid_n, invalid_primary_judgments=len(rows) - valid_n,
        both_judges_valid=int(both.sum()), judge_disagreement_count=disagreements,
        judge_disagreement_rate=(disagreements / int(both.sum()) if both.any() else None),
        response_path=str(rows_path), run_path=str(eval_dir / "run.json"),
        raw_rollout_paths=run.get("raw_rollout_paths", []),
        _prompt_means={
            "alignment": prompt_means(align_key),
            "misalignment": {key: 100.0 - value for key, value in prompt_means(align_key).items()},
            "coherence": prompt_means(coherence_key),
            "MR": prompt_means(em_key),
            "refusal": prompt_means(refusal_key),
        },
    )


def _blind_review_tables(records: list[dict], model: str, seed: int, run_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_parts = []
    for record in records:
        condition = record["condition"]
        frame = pd.read_csv(Path(record["training_evaluation"]) / "responses.csv")
        required = {"question", "answer", "prompt_id", "sample_idx", "condition", "rendering",
                    "qwen_em", "llama_em", "qwen_invalid", "llama_invalid",
                    "qwen_alignment", "llama_alignment", "qwen_coherence", "llama_coherence"}
        if not required.issubset(frame.columns):
            raise ValueError(f"review source table is missing fields for {condition}")
        if set(frame["condition"].dropna()) != {condition}:
            raise ValueError(f"review source condition mismatch for {condition}")
        qwen_valid = frame["qwen_em"].notna()
        llama_valid = frame["llama_em"].notna()
        both_valid = qwen_valid & llama_valid
        misaligned = ((frame["qwen_em"] == 1) | (frame["llama_em"] == 1)).fillna(False)
        disagreement = both_valid & (frame["qwen_em"] != frame["llama_em"])
        invalid = (~qwen_valid | ~llama_valid
                   | frame["qwen_invalid"].fillna(True).astype(bool)
                   | frame["llama_invalid"].fillna(True).astype(bool))
        aligned = both_valid & (frame["qwen_em"] == 0) & (frame["llama_em"] == 0)
        chosen: dict[int, set[str]] = {}
        for index in frame.index[misaligned]:
            chosen.setdefault(index, set()).add("judged_misaligned")
        for index in frame.index[disagreement]:
            chosen.setdefault(index, set()).add("judge_disagreement")
        for index in frame.index[invalid]:
            chosen.setdefault(index, set()).add("invalid_or_missing_judgment")
        aligned_ids = frame.index[aligned].to_numpy()
        sample_n = int(math.ceil(.10 * len(aligned_ids)))
        if sample_n:
            rng = np.random.default_rng(stable_seed(
                "acl-stage6b-blind-aligned", model, seed, condition,
            ))
            for index in rng.choice(aligned_ids, size=sample_n, replace=False):
                chosen.setdefault(int(index), set()).add("fixed_random_aligned")
        if chosen:
            rows = frame.loc[sorted(chosen)].copy()
            rows["_review_reasons"] = [";".join(sorted(chosen[index])) for index in rows.index]
            selected_parts.append(rows)

    if not selected_parts:
        raise ValueError("no endpoint responses were selected for human review")
    selected = pd.concat(selected_parts, ignore_index=True)
    review_ids = [hashlib.sha256(
        f"{run_id}|{model}|{row.condition}|{row.rendering}|{row.prompt_id}|{int(row.sample_idx)}".encode()
    ).hexdigest()[:24] for row in selected.itertuples()]
    if len(review_ids) != len(set(review_ids)):
        raise ValueError("blind-review key collision or duplicate source completion")
    selected.insert(0, "review_id", review_ids)
    selected = selected.sample(frac=1, random_state=stable_seed("acl-stage6b-review-order", model, seed))
    queue = selected[["review_id", "question", "answer"]].copy()
    queue["human_alignment"] = ""
    queue["human_coherence"] = ""
    queue["human_refusal"] = ""
    queue["human_em"] = ""
    queue["review_notes"] = ""
    key_columns = [
        "review_id", "model_key", "condition", "rendering", "prompt_id", "sample_idx",
        "qwen_em", "llama_em", "qwen_alignment", "llama_alignment",
        "qwen_coherence", "llama_coherence", "qwen_invalid", "llama_invalid", "_review_reasons",
    ]
    key = selected[[column for column in key_columns if column in selected.columns]].rename(
        columns={"_review_reasons": "selection_reasons"},
    )
    return queue.reset_index(drop=True), key.reset_index(drop=True)


def _factorial_maps(outcomes: dict[str, dict], seed: int) -> list[dict]:
    required = set(PRIMARY)
    result = []
    for outcome, condition_maps in outcomes.items():
        maps = {condition: value for condition, value in condition_maps.items()
                if condition in required and isinstance(value, dict)}
        if not required.issubset(maps):
            result.append(dict(outcome=outcome, status="not_estimable_missing_primary_condition",
                               missing=sorted(required - set(maps))))
            continue
        common_ids = sorted(set.intersection(*(set(maps[c]) for c in PRIMARY)))
        if common_ids:
            per_prompt = {
                "beta_0": {key: maps["E"][key] for key in common_ids},
                "beta_P": {key: maps["P"][key] - maps["E"][key] for key in common_ids},
                "beta_D": {key: maps["W"][key] - maps["E"][key] for key in common_ids},
                "beta_PD": {key: maps["P+W"][key] - maps["P"][key] - maps["W"][key] + maps["E"][key]
                            for key in common_ids},
            }
            for term, values in per_prompt.items():
                boot = _bootstrap_mean(values, stable_seed("acl-stage6b-factorial", seed, outcome, term))
                result.append(dict(outcome=outcome, term=term, point=boot["point"],
                                   ci_low=boot["ci_low"], ci_high=boot["ci_high"],
                                   prompt_count=boot["prompt_count"], cluster_count=boot["cluster_count"]))
        else:
            result.append(dict(outcome=outcome, status="not_estimable_no_common_prompt_values"))
    return result


def build_analysis(model: str, seed: int, run_summary: dict) -> tuple[dict, dict[str, pd.DataFrame]]:
    if model != CALIBRATION_MODEL:
        raise ValueError("Stage 6B defense analysis is restricted to Qwen2.5-7B")
    if run_summary.get("model") != model or run_summary.get("seed") != seed:
        raise ValueError("full-run summary model/seed mismatch")
    records = run_summary.get("conditions", [])
    by_condition = {row["condition"]: row for row in records}
    if len(by_condition) != len(records):
        raise ValueError("duplicate condition in full-run summary")
    if "E" not in by_condition:
        raise ValueError("ordinary harmful-SFT E condition is required")

    ifbench_analysis = run_summary.get("instruction_following_analysis", {})
    paired_ifbench_path = ifbench_analysis.get("paired_vs_E")
    if not paired_ifbench_path:
        raise ValueError("paired IFBench condition-vs-E table is missing from the run summary")
    paired_ifbench = _load_ifbench_paired(Path(paired_ifbench_path), set(by_condition))

    training, metrics, endpoint_assays = {}, {}, {}
    for condition, record in by_condition.items():
        manifest, rows = _load_training(record, model, seed)
        training[condition], metrics[condition] = manifest, rows
        assay = rows[184].get("paired_completion_assay")
        if not isinstance(assay, dict):
            raise ValueError(f"step-184 paired-completion endpoint assay is missing for {condition}")
        endpoint_assays[condition] = assay

    base_preference = metrics["E"][0].get("harmful_benign_preference")
    base_preference_map = metrics["E"][0].get("harmful_benign_preference_by_prompt")
    e_preference = metrics["E"][184].get("harmful_benign_preference")
    e_preference_map = metrics["E"][184].get("harmful_benign_preference_by_prompt")
    if not isinstance(base_preference_map, dict) or not isinstance(e_preference_map, dict):
        raise ValueError("step-0 and step-184 prompt-level held-out medical preferences are required")
    if not base_preference_map or not e_preference_map:
        raise ValueError("step-0 and step-184 held-out medical preference maps cannot be empty")
    for label, scalar, values in (
        ("Base", base_preference, base_preference_map),
        ("ordinary harmful SFT", e_preference, e_preference_map),
    ):
        if (not _number(scalar) or any(not _number(value) for value in values.values())
                or not math.isclose(float(scalar), float(np.mean(list(values.values()))),
                                    rel_tol=0, abs_tol=2e-6)):
            raise ValueError(f"{label} scalar and prompt-level medical preference disagree")
    denominator = e_preference - base_preference if _number(e_preference) and _number(base_preference) else None

    deterministic_rows, narrow_rows, capability_rows, open_rows, route_rows, dynamics_rows = [], [], [], [], [], []
    route_agreement_rows = []
    open_prompt_maps = {}
    deterministic_maps, persona_maps, direct_de_maps, narrow_preference_maps = {}, {}, {}, {}
    for condition, record in by_condition.items():
        assay = endpoint_assays[condition]
        delta_map = {str(key): float(value) for key, value in assay["delta_S_per_prompt"].items()}
        strict_delta_map = {str(key): float(value) for key, value in assay["strict50_delta_S_per_prompt"].items()}
        reduction = _bootstrap_reduction(delta_map, endpoint_assays["E"]["delta_S_per_prompt"],
                                         stable_seed("acl-stage6b-rem", seed, condition))
        strict_reduction = _bootstrap_reduction(strict_delta_map,
                                                endpoint_assays["E"]["strict50_delta_S_per_prompt"],
                                                stable_seed("acl-stage6b-rem50", seed, condition))
        deterministic_rows.append(dict(
            model=model, seed=seed, condition=condition, label=LABELS.get(condition, condition),
            S_M=assay.get("S_mean"), S_C=assay.get("S_C"), delta_S=assay.get("delta_S"),
            delta_S_ci_low=assay.get("delta_S_ci", {}).get("ci_low"),
            delta_S_ci_high=assay.get("delta_S_ci", {}).get("ci_high"),
            EM_reduction=reduction.get("point"), EM_reduction_ci_low=reduction.get("ci_low"),
            EM_reduction_ci_high=reduction.get("ci_high"),
            strict50_delta_S=assay.get("strict50_delta_S"),
            strict50_delta_S_ci_low=assay.get("strict50_delta_S_ci", {}).get("ci_low"),
            strict50_delta_S_ci_high=assay.get("strict50_delta_S_ci", {}).get("ci_high"),
            strict50_EM_reduction=strict_reduction.get("point"),
            strict50_EM_reduction_ci_low=strict_reduction.get("ci_low"),
            strict50_EM_reduction_ci_high=strict_reduction.get("ci_high"),
            delta_S_prompt_count=len(delta_map), strict50_prompt_count=len(strict_delta_map),
        ))
        deterministic_maps[condition] = delta_map

        pref = metrics[condition][184].get("harmful_benign_preference")
        nll_initial = training[condition].get("initial_harmful_val_nll")
        nll_final = metrics[condition][184].get("harmful_val_nll")
        benign_nll = metrics[condition][184].get("benign_val_nll")
        pref_map = metrics[condition][184].get("harmful_benign_preference_by_prompt")
        if not isinstance(pref_map, dict) or not pref_map:
            raise ValueError(f"step-184 prompt-level held-out medical preference is missing for {condition}")
        if (not _number(pref) or any(not _number(value) for value in pref_map.values())
                or not math.isclose(float(pref), float(np.mean(list(pref_map.values()))),
                                    rel_tol=0, abs_tol=2e-6)):
            raise ValueError(f"step-184 scalar and prompt-level medical preference disagree for {condition}")
        narrow_preference_maps[condition] = {str(key): float(value) for key, value in pref_map.items()}
        retention = _bootstrap_retained_learning(
            narrow_preference_maps[condition], base_preference_map, e_preference_map,
            stable_seed("acl-stage6b-narrow", seed, condition),
        )
        scalar_retention = ((pref - base_preference) / denominator
                            if _number(pref) and _number(base_preference) and _number(denominator)
                            and denominator > 1e-12 else None)
        if (retention.get("point") is not None and scalar_retention is not None
                and not math.isclose(retention["point"], scalar_retention, rel_tol=0, abs_tol=2e-6)):
            raise ValueError(f"prompt-level and scalar narrow utility disagree for {condition}")
        e_nll = metrics["E"][184].get("harmful_val_nll")
        nll_denominator = (training["E"].get("initial_harmful_val_nll") - e_nll
                           if _number(training["E"].get("initial_harmful_val_nll")) and _number(e_nll) else None)
        nll_retained = ((nll_initial - nll_final) / nll_denominator
                        if _number(nll_initial) and _number(nll_final) and _number(nll_denominator)
                        and nll_denominator > 1e-12 else None)
        narrow_rows.append(dict(
            model=model, seed=seed, condition=condition, label=LABELS.get(condition, condition),
            initial_harmful_val_nll=nll_initial, harmful_val_nll=nll_final,
            benign_val_nll=benign_nll, S_med_base=base_preference,
            S_med=pref, S_med_E=e_preference,
            U_preference=retention.get("point", scalar_retention),
            U_preference_ci_low=retention.get("ci_low"), U_preference_ci_high=retention.get("ci_high"),
            U_preference_numerator=(pref - base_preference) if _number(pref) and _number(base_preference) else None,
            U_preference_denominator=denominator,
            # Retained as a compatibility alias; reports label this estimand U_preference.
            U_narrow=retention.get("point", scalar_retention),
            U_narrow_ci_low=retention.get("ci_low"), U_narrow_ci_high=retention.get("ci_high"),
            U_narrow_prompt_count=retention.get("prompt_count"),
            U_narrow_cluster_count=retention.get("cluster_count"),
            U_narrow_reason=retention.get("reason"),
            U_NLL=nll_retained, U_NLL_numerator=(nll_initial - nll_final)
            if _number(nll_initial) and _number(nll_final) else None,
            U_NLL_denominator=nll_denominator, U_NLL_ci_low=None, U_NLL_ci_high=None,
            U_NLL_interval_reason="validation loss is only retained as an aggregate, without question-level values",
            harmful_nll_learning_retained=nll_retained,
        ))
        eval_dir = Path(record["training_evaluation"])
        opened = _judge_summary(eval_dir, condition, model, seed)
        open_prompt_maps[condition] = opened.pop("_prompt_means")
        for source, prefix in (("primary_alignment", "alignment"),
                               ("primary_misalignment_score", "misalignment_score"),
                               ("coherence", "coherence"), ("MR", "MR"), ("refusal", "refusal")):
            value = opened.pop(source)
            for suffix, column in (("mean", "mean"), ("ci_low", "ci_low"), ("ci_high", "ci_high")):
                opened[f"{prefix}_{suffix}"] = value.get(column)
        open_rows.append(opened)

        capability = record.get("instruction_following_summary", {})
        ifbench = capability.get("categories", {}).get("overall", {})
        iheval = record.get("iheval_reference_evaluation", {})
        route = _route_record(record, model, seed, assay)
        route_rows.append(route)
        if record.get("route_check"):
            comparisons = (
                ("TE", (assay.get("direct_TE") or {}).get("point"), (route.get("TE") or {}).get("point")),
                ("DE", (assay.get("direct_DE") or {}).get("point"), (route.get("DE") or {}).get("point")),
                ("delta_P", (assay.get("delta_P") or {}).get("point"), route.get("delta_P")),
            )
            diffs = {}
            for name, in_training, endpoint in comparisons:
                difference = (float(in_training) - float(endpoint)
                              if _number(in_training) and _number(endpoint) else None)
                diffs[name] = difference
                route_agreement_rows.append(dict(
                    model=model, seed=seed, condition=condition, measure=name,
                    in_training_snapshot=in_training, endpoint_reloaded=endpoint,
                    difference=in_training-endpoint if difference is not None else None,
                    absolute_difference=abs(difference) if difference is not None else None,
                    within_0_025=(abs(difference) <= 0.025 if difference is not None else False),
                ))
        neutral = route.get("neutral_quality") or {}
        capability_rows.append(dict(
            model=model, seed=seed, condition=condition, label=LABELS.get(condition, condition),
            IFBench_strict=ifbench.get("strict_rate"), IFBench_loose=ifbench.get("loose_rate"),
            IFBench_strict_difference_vs_E=paired_ifbench[(condition, "strict_all")].get("difference_vs_E"),
            IFBench_strict_difference_ci_low=paired_ifbench[(condition, "strict_all")].get("ci_low"),
            IFBench_strict_difference_ci_high=paired_ifbench[(condition, "strict_all")].get("ci_high"),
            IFBench_loose_difference_vs_E=paired_ifbench[(condition, "loose_all")].get("difference_vs_E"),
            IFBench_loose_difference_ci_low=paired_ifbench[(condition, "loose_all")].get("ci_low"),
            IFBench_loose_difference_ci_high=paired_ifbench[(condition, "loose_all")].get("ci_high"),
            IHEval_micro=iheval.get("micro_score"), IHEval_task_macro=iheval.get("task_macro_score"),
            neutral_log_likelihood=neutral.get("lp_mean"), neutral_output_entropy=neutral.get("ent_mean"),
            neutral_top1_agreement_to_C=neutral.get("agree_C"),
        ))

        persona = route.get("per_prompt_delta_P") or {}
        de = route.get("DE") or {}
        de_map = de.get("per_prompt") if isinstance(de, dict) else None
        if persona:
            persona_maps[condition] = {str(key): float(value) for key, value in persona.items()}
        if isinstance(de_map, dict) and de_map:
            direct_de_maps[condition] = {str(key): float(value) for key, value in de_map.items()}

        for step in ((16, 64, 184) if condition in PRIMARY else (184,)):
            step_assay = metrics[condition][step].get("paired_completion_assay")
            if not isinstance(step_assay, dict):
                raise ValueError(f"missing in-training trajectory assay for {condition} at step {step}")
            dynamics_rows.append(dict(
                model=model, seed=seed, condition=condition, label=LABELS.get(condition, condition),
                step=step, delta_S=step_assay.get("delta_S"),
                delta_P=(step_assay.get("delta_P") or {}).get("point"),
                direct_TE=(step_assay.get("direct_TE") or {}).get("point"),
                direct_DE=(step_assay.get("direct_DE") or {}).get("point"),
            ))

    if "E" in open_prompt_maps:
        for row in open_rows:
            condition = row["condition"]
            for metric in ("misalignment", "MR", "coherence", "refusal"):
                candidate = open_prompt_maps[condition][metric]
                baseline = open_prompt_maps["E"][metric]
                common = sorted(set(candidate) & set(baseline))
                if common:
                    difference = {key: candidate[key] - baseline[key] for key in common}
                    boot = _bootstrap_mean(
                        difference, stable_seed("acl-stage6b-open-difference", model, seed, condition, metric),
                    )
                    row[f"{metric}_difference_vs_E"] = boot["point"]
                    row[f"{metric}_difference_vs_E_ci_low"] = boot["ci_low"]
                    row[f"{metric}_difference_vs_E_ci_high"] = boot["ci_high"]
                    row[f"{metric}_paired_prompt_count"] = len(common)
                else:
                    row[f"{metric}_difference_vs_E"] = None
                    row[f"{metric}_difference_vs_E_ci_low"] = None
                    row[f"{metric}_difference_vs_E_ci_high"] = None
                    row[f"{metric}_paired_prompt_count"] = 0

    factorial = _factorial_maps({
        "behavioral_delta_S": deterministic_maps,
        "persona_delta_P": persona_maps,
        "direct_DE": direct_de_maps,
        "narrow_S_med": narrow_preference_maps,
    }, seed)
    factorial.extend(_factorial_retained_learning(narrow_preference_maps, base_preference_map, seed))
    prompt_effect_rows = []
    for condition in by_condition:
        maps = {
            "delta_S": deterministic_maps.get(condition, {}),
            "delta_P": persona_maps.get(condition, {}),
            "direct_DE": direct_de_maps.get(condition, {}),
            "S_med": narrow_preference_maps.get(condition, {}),
        }
        prompt_ids = sorted(set.union(*(set(values) for values in maps.values())))
        for prompt_id in prompt_ids:
            prompt_effect_rows.append(dict(
                model=model, seed=seed, condition=condition, prompt_id=prompt_id,
                **{name: values.get(prompt_id) for name, values in maps.items()},
            ))
    review_queue, review_key = _blind_review_tables(
        records, model, seed, run_summary.get("run_id", f"stage6b_full_{model}_seed{seed}"),
    )
    tables = {
        "deterministic_endpoints": pd.DataFrame(deterministic_rows),
        "narrow_task": pd.DataFrame(narrow_rows),
        "open_ended": pd.DataFrame(open_rows),
        "capability": pd.DataFrame(capability_rows),
        "persona_and_direct_route": pd.DataFrame([_flat_route_record(row) for row in route_rows]),
        "prompt_effects": pd.DataFrame(prompt_effect_rows),
        "training_dynamics": pd.DataFrame(dynamics_rows),
        "route_snapshot_agreement": pd.DataFrame(route_agreement_rows),
        "blind_review_queue": review_queue,
        "blind_review_key": review_key,
        "factorial": pd.DataFrame(factorial),
    }
    summary = dict(
        schema="acl_stage6b_defense_analysis_v1", model=model, seed=seed,
        run_id=run_summary.get("run_id"), conditions=list(by_condition),
        primary_judge="llama", deterministic_reference="frozen Stage 2 matched benign checkpoint",
        deterministic_em_formula="1 - delta_S(condition) / delta_S(E), prompt-cluster bootstrap",
        narrow_utility_formula="(S_med(condition)-S_med(Base))/(S_med(E)-S_med(Base))",
        narrow_utility_report_label="U_preference",
        table_paths={name: f"{name}.csv" for name in tables},
        factorial_estimable=set(PRIMARY).issubset(by_condition),
        route_snapshot_agreement_within_0_025=(
            bool(route_agreement_rows) and all(row["within_0_025"] for row in route_agreement_rows)
        ),
    )
    return summary, tables


def write_analysis(model: str, seed: int, run_summary_path: Path) -> dict:
    run_summary = _read_json(run_summary_path, "full-seed summary")
    summary, tables = build_analysis(model, seed, run_summary)
    output_dir = ACL_ROOT / "analyses" / model / f"seed_{seed}"
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite Stage 6B analysis artifacts: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".seed_{seed}.partial-", dir=output_dir.parent))
    try:
        for name, frame in tables.items():
            frame.to_csv(staging / f"{name}.csv", index=False)
        from stage6b_defense_figures import write_figures
        figure_dir = ROOT / "figures" / "persona_control" / "stage6b" / model / f"seed_{seed}"
        figure_paths = write_figures(staging, figure_dir)
        manifest = dict(
            **summary, source_sha256={
                "stage6b_defense_analyze.py": sha256_file(Path(__file__)),
                "stage6b_defense_figures.py": sha256_file(Path(__file__).with_name("stage6b_defense_figures.py")),
                "stage6b_train.py": sha256_file(Path(__file__).with_name("stage6b_train.py")),
                "stage6b_evaluate.py": sha256_file(Path(__file__).with_name("stage6b_evaluate.py")),
                "stage6b_route_check.py": sha256_file(Path(__file__).with_name("stage6b_route_check.py")),
                "stage6b_ifbench.py": sha256_file(Path(__file__).with_name("stage6b_ifbench.py")),
                "stage6b_iheval_reference.py": sha256_file(Path(__file__).with_name("stage6b_iheval_reference.py")),
                "acl_common.py": sha256_file(Path(__file__).with_name("acl_common.py")),
                "common.py": sha256_file(Path(__file__).with_name("common.py")),
            },
            figure_paths=figure_paths,
            run_summary_path=str(run_summary_path),
            run_summary_sha256=sha256_file(run_summary_path),
        )
        (staging / "analysis_summary.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        staging.rename(output_dir)
    except BaseException:
        import shutil
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(output_dir=str(output_dir), summary_path=str(output_dir / "analysis_summary.json"),
                table_paths={name: str(output_dir / f"{name}.csv") for name in tables})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[CALIBRATION_MODEL], required=True)
    parser.add_argument("--seed", type=int, choices=(61791, 61792, 61793), required=True)
    args = parser.parse_args()
    summary_path = ACL_ROOT / "full_runs" / args.model / f"seed_{args.seed}.json"
    result = write_analysis(args.model, args.seed, summary_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
