from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Select steering layers from linear-direction validation metrics.")
    parser.add_argument("--metrics-csv", required=True)
    parser.add_argument("--direction-report-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--extra-layer", action="append", default=[])
    args = parser.parse_args()

    report = json.loads(Path(args.direction_report_json).read_text(encoding="utf-8"))
    selected = report.get("selected", {})
    selected_method = selected.get("method")
    selected_layer = _maybe_int(selected.get("layer"))

    metrics = pd.read_csv(args.metrics_csv)
    if selected_method and "method" in metrics.columns:
        metrics = metrics[metrics["method"] == selected_method].copy()
    if metrics.empty:
        raise ValueError("No metrics rows available after method filtering.")

    for column in ("layer", "heldout_auc", "heldout_separation", "train_auc"):
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")

    ranked = metrics.sort_values(
        ["heldout_auc", "heldout_separation", "train_auc", "layer"],
        ascending=[False, False, False, True],
    )
    layers = []
    if selected_layer is not None:
        layers.append(selected_layer)
    layers.extend(int(layer) for layer in ranked["layer"].dropna().head(args.top_k).tolist())
    layers.extend(_maybe_int(layer) for layer in args.extra_layer)
    unique_layers = sorted({layer for layer in layers if layer is not None})

    output = {
        "metrics_csv": str(Path(args.metrics_csv)),
        "direction_report_json": str(Path(args.direction_report_json)),
        "selected_method": selected_method,
        "selected_layer": selected_layer,
        "top_k": int(args.top_k),
        "layers": unique_layers,
        "layers_csv": ",".join(str(layer) for layer in unique_layers),
        "top_rows": ranked.head(args.top_k).to_dict(orient="records"),
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return int(float(value))


if __name__ == "__main__":
    main()
