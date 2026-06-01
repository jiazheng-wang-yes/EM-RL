from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank residual-stream layers by mean-diff steering behavior and heldout separation."
    )
    parser.add_argument("--steering-summary-csv", required=True)
    parser.add_argument("--vector-path", required=True)
    parser.add_argument("--output-layer-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sources", default="em,harmbench,strongreject")
    parser.add_argument("--min-coherent-mean", type=float, default=50.0)
    args = parser.parse_args()

    summary = pd.read_csv(args.steering_summary_csv)
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    ranking = rank_layers(
        summary=summary,
        vector_path=Path(args.vector_path),
        sources=sources,
        min_coherent_mean=args.min_coherent_mean,
    )

    output_layer_csv = Path(args.output_layer_csv)
    output_layer_csv.parent.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(output_layer_csv, index=False)

    top = ranking.head(args.top_k)
    report = {
        "steering_summary_csv": str(Path(args.steering_summary_csv)),
        "vector_path": str(Path(args.vector_path)),
        "layer_csv": str(output_layer_csv),
        "top_k": int(args.top_k),
        "sources": sources,
        "ranking_rule": {
            "description": (
                "Rank layers by behavioral steering success under the layer/lambda sweep, "
                "with heldout activation separation as supporting evidence."
            ),
            "representation_score": (
                "0.45 * rank(mean best coherent EM-rate delta across sources) + "
                "0.25 * rank(heldout separation) + "
                "0.20 * rank(mean coherent score at best scale) + "
                "0.10 * rank(mean alignment-score drop at best scale)"
            ),
            "paper_workflow": "arXiv:2506.11618, Sections 3.1-3.2",
        },
        "top_layers": top.to_dict(orient="records"),
        "best_activation_layer": _best_record(ranking, "heldout_separation"),
        "best_behavior_layer": _best_record(ranking, "mean_best_em_rate_delta"),
        "best_layer_by_source": _best_layer_by_source(ranking, sources),
    }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def rank_layers(summary: pd.DataFrame, vector_path: Path, sources: list[str], min_coherent_mean: float) -> pd.DataFrame:
    steered = summary[summary["condition"] != "noop"].copy()
    steered = steered[steered["source_set"].isin(sources)].copy()
    if steered.empty:
        raise ValueError("No steered rows found for requested sources.")

    for column in (
        "layer",
        "scale",
        "em_rate",
        "em_rate_delta_vs_baseline",
        "aligned_delta_vs_baseline",
        "aligned_mean",
        "coherent_mean",
    ):
        if column in steered.columns:
            steered[column] = pd.to_numeric(steered[column], errors="coerce")

    steered["coherent_ok"] = steered["coherent_mean"].fillna(0.0) >= min_coherent_mean
    steered["coherent_em_rate"] = steered["em_rate"].where(steered["coherent_ok"], 0.0)
    steered["coherent_em_delta"] = steered["em_rate_delta_vs_baseline"].where(steered["coherent_ok"], 0.0)
    steered["alignment_drop"] = (-steered["aligned_delta_vs_baseline"]).clip(lower=0.0)

    sort_cols = [
        "source_set",
        "layer",
        "coherent_em_delta",
        "coherent_em_rate",
        "coherent_mean",
        "alignment_drop",
    ]
    best_by_source = (
        steered.sort_values(sort_cols, ascending=[True, True, False, False, False, False])
        .groupby(["source_set", "layer"], as_index=False, dropna=False)
        .first()
    )

    aggregate = (
        best_by_source.groupby("layer", as_index=False)
        .agg(
            mean_best_em_rate_delta=("coherent_em_delta", "mean"),
            max_best_em_rate_delta=("coherent_em_delta", "max"),
            mean_best_em_rate=("coherent_em_rate", "mean"),
            mean_best_coherent=("coherent_mean", "mean"),
            mean_alignment_drop=("alignment_drop", "mean"),
            sources_with_positive_em_delta=("coherent_em_delta", lambda values: int((values > 0).sum())),
        )
        .sort_values("layer")
    )

    for source in sources:
        source_rows = best_by_source[best_by_source["source_set"] == source][
            ["layer", "scale", "coherent_em_delta", "coherent_em_rate", "coherent_mean", "alignment_drop"]
        ].rename(
            columns={
                "scale": f"{source}_best_scale",
                "coherent_em_delta": f"{source}_best_em_delta",
                "coherent_em_rate": f"{source}_best_em_rate",
                "coherent_mean": f"{source}_best_coherent",
                "alignment_drop": f"{source}_alignment_drop",
            }
        )
        aggregate = aggregate.merge(source_rows, on="layer", how="left")

    aggregate = aggregate.merge(_load_layer_validation(vector_path), on="layer", how="left")
    aggregate["heldout_separation"] = aggregate["heldout_separation"].fillna(0.0)
    aggregate["mean_best_coherent"] = aggregate["mean_best_coherent"].fillna(0.0)
    aggregate["mean_alignment_drop"] = aggregate["mean_alignment_drop"].fillna(0.0)

    aggregate["behavior_rank"] = _rank01(aggregate["mean_best_em_rate_delta"])
    aggregate["separation_rank"] = _rank01(aggregate["heldout_separation"])
    aggregate["coherence_rank"] = _rank01(aggregate["mean_best_coherent"])
    aggregate["alignment_drop_rank"] = _rank01(aggregate["mean_alignment_drop"])
    aggregate["representation_score"] = (
        0.45 * aggregate["behavior_rank"]
        + 0.25 * aggregate["separation_rank"]
        + 0.20 * aggregate["coherence_rank"]
        + 0.10 * aggregate["alignment_drop_rank"]
    )

    return aggregate.sort_values(
        [
            "representation_score",
            "mean_best_em_rate_delta",
            "sources_with_positive_em_delta",
            "heldout_separation",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def _load_layer_validation(vector_path: Path) -> pd.DataFrame:
    artifact = torch.load(vector_path, map_location="cpu", weights_only=False)
    rows = artifact.get("validation", {}).get("layer_rows", [])
    if not rows:
        return pd.DataFrame(columns=["layer", "heldout_separation", "raw_diff_norm"])
    validation = pd.DataFrame(rows)
    selected_method = artifact.get("selected_method")
    if selected_method and "method" in validation.columns:
        validation = validation[validation["method"] == selected_method].copy()
    validation = validation.rename(
        columns={
            "separation": "heldout_separation",
            "aligned_projection": "heldout_aligned_projection",
            "misaligned_projection": "heldout_misaligned_projection",
        }
    )
    validation["layer"] = pd.to_numeric(validation["layer"], errors="coerce")
    return validation


def _rank01(values: pd.Series) -> pd.Series:
    if values.nunique(dropna=True) <= 1:
        return pd.Series([0.0] * len(values), index=values.index, dtype=float)
    return values.rank(method="average", pct=True).astype(float)


def _best_record(ranking: pd.DataFrame, column: str) -> dict[str, Any] | None:
    if column not in ranking.columns or ranking.empty:
        return None
    row = ranking.sort_values(column, ascending=False).iloc[0]
    return {
        "layer": int(row["layer"]),
        column: float(row[column]),
        "representation_score": float(row["representation_score"]),
        "mean_best_em_rate_delta": float(row["mean_best_em_rate_delta"]),
        "heldout_separation": float(row["heldout_separation"]),
    }


def _best_layer_by_source(ranking: pd.DataFrame, sources: list[str]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for source in sources:
        column = f"{source}_best_em_delta"
        scale_column = f"{source}_best_scale"
        if column not in ranking.columns:
            continue
        row = ranking.sort_values([column, "representation_score"], ascending=[False, False]).iloc[0]
        output[source] = {
            "layer": int(row["layer"]),
            "best_scale": None if pd.isna(row.get(scale_column)) else float(row[scale_column]),
            "best_em_delta": float(row[column]),
            "representation_score": float(row["representation_score"]),
        }
    return output


if __name__ == "__main__":
    main()
