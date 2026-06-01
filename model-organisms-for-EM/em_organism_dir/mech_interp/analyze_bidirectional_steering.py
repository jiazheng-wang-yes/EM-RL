from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare positive and negative EM-vector steering summaries.")
    parser.add_argument("--summary-csv", action="append", required=True, help="MODEL_LABEL:summary.csv")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--min-coherent-mean", type=float, default=50.0)
    args = parser.parse_args()

    frames = []
    for spec in args.summary_csv:
        label, path = _parse_summary_spec(spec)
        frame = pd.read_csv(path)
        frame["model_label"] = label
        frame["summary_csv"] = str(path)
        frames.append(frame)
    if not frames:
        raise ValueError("At least one summary CSV is required.")

    combined = pd.concat(frames, ignore_index=True)
    combined = _prepare(combined, min_coherent_mean=args.min_coherent_mean)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_csv, index=False)

    report = {
        "output_csv": str(output_csv),
        "model_labels": sorted(combined["model_label"].dropna().unique().tolist()),
        "baseline": _baseline_rows(combined),
        "best_reverse_by_em_drop": _top_reverse_by_em_drop(combined, args.top_k),
        "best_reverse_by_alignment_gain": _top_reverse_by_alignment_gain(combined, args.top_k),
        "best_forward_by_em_gain": _top_forward_by_em_gain(combined, args.top_k),
        "best_by_model_and_source": _best_by_model_and_source(combined),
        "interpretation": {
            "reverse_direction": "scale < 0",
            "forward_direction": "scale > 0",
            "strong_reverse_signal": (
                "negative-scale rows should reduce em_rate or raise aligned_mean relative to noop, "
                "while keeping coherent_mean above the threshold"
            ),
            "coherent_threshold": args.min_coherent_mean,
        },
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _parse_summary_spec(spec: str) -> tuple[str, Path]:
    if ":" not in spec:
        raise ValueError(f"Expected MODEL_LABEL:summary.csv, got {spec}")
    label, path = spec.split(":", 1)
    if not label:
        raise ValueError(f"Missing model label in {spec}")
    return label, Path(path)


def _prepare(df: pd.DataFrame, *, min_coherent_mean: float) -> pd.DataFrame:
    output = df.copy()
    for column in (
        "layer",
        "scale",
        "n",
        "aligned_mean",
        "coherent_mean",
        "em_rate",
        "aligned_delta_vs_baseline",
        "em_rate_delta_vs_baseline",
    ):
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    output["direction"] = "noop"
    output.loc[output["scale"] < 0, "direction"] = "reverse"
    output.loc[output["scale"] > 0, "direction"] = "forward"
    output["coherent_ok"] = output["coherent_mean"].fillna(0.0) >= min_coherent_mean
    output["reverse_score"] = (
        (-output["em_rate_delta_vs_baseline"].fillna(0.0))
        + 0.01 * output["aligned_delta_vs_baseline"].fillna(0.0).clip(lower=0.0)
    ).where((output["direction"] == "reverse") & output["coherent_ok"], float("-inf"))
    output["alignment_gain_score"] = output["aligned_delta_vs_baseline"].fillna(0.0).where(
        (output["direction"] == "reverse") & output["coherent_ok"], float("-inf")
    )
    output["forward_score"] = output["em_rate_delta_vs_baseline"].fillna(0.0).where(
        (output["direction"] == "forward") & output["coherent_ok"], float("-inf")
    )
    return output


def _baseline_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = ["model_label", "source_set", "n", "aligned_mean", "coherent_mean", "em_rate"]
    rows = df[df["direction"] == "noop"][columns].sort_values(["model_label", "source_set"])
    return rows.to_dict(orient="records")


def _top_reverse_by_em_drop(df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    rows = df[(df["direction"] == "reverse") & df["coherent_ok"]].sort_values(
        ["em_rate_delta_vs_baseline", "aligned_delta_vs_baseline", "coherent_mean"],
        ascending=[True, False, False],
    )
    return _clean_records(rows.head(top_k))


def _top_reverse_by_alignment_gain(df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    rows = df[(df["direction"] == "reverse") & df["coherent_ok"]].sort_values(
        ["aligned_delta_vs_baseline", "em_rate_delta_vs_baseline", "coherent_mean"],
        ascending=[False, True, False],
    )
    return _clean_records(rows.head(top_k))


def _top_forward_by_em_gain(df: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    rows = df[(df["direction"] == "forward") & df["coherent_ok"]].sort_values(
        ["em_rate_delta_vs_baseline", "aligned_delta_vs_baseline", "coherent_mean"],
        ascending=[False, True, False],
    )
    return _clean_records(rows.head(top_k))


def _best_by_model_and_source(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for (model_label, source_set), group in df[df["direction"] != "noop"].groupby(["model_label", "source_set"]):
        coherent = group[group["coherent_ok"]]
        reverse = coherent[coherent["direction"] == "reverse"].sort_values(
            ["em_rate_delta_vs_baseline", "aligned_delta_vs_baseline"],
            ascending=[True, False],
        )
        forward = coherent[coherent["direction"] == "forward"].sort_values(
            ["em_rate_delta_vs_baseline", "aligned_delta_vs_baseline"],
            ascending=[False, True],
        )
        rows.append(
            {
                "model_label": model_label,
                "source_set": source_set,
                "best_reverse": None if reverse.empty else _selected_record(reverse.iloc[0]),
                "best_forward": None if forward.empty else _selected_record(forward.iloc[0]),
            }
        )
    return rows


def _selected_record(row: pd.Series) -> dict[str, Any]:
    keep = [
        "layer",
        "scale",
        "n",
        "aligned_mean",
        "coherent_mean",
        "em_rate",
        "aligned_delta_vs_baseline",
        "em_rate_delta_vs_baseline",
    ]
    return {key: _json_value(row.get(key)) for key in keep}


def _clean_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    keep = [
        "model_label",
        "source_set",
        "direction",
        "layer",
        "scale",
        "n",
        "aligned_mean",
        "coherent_mean",
        "em_rate",
        "aligned_delta_vs_baseline",
        "em_rate_delta_vs_baseline",
    ]
    return [{key: _json_value(row.get(key)) for key in keep} for _, row in df[keep].iterrows()]


def _json_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
