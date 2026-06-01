from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aligned and misaligned response sets for mean-diff vectors.")
    parser.add_argument("--responses-judged-csv", action="append", default=[])
    parser.add_argument("--harmbench-results-json", action="append", default=[])
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--aligned-gt", type=float, default=70.0)
    parser.add_argument("--misaligned-lt", type=float, default=30.0)
    parser.add_argument("--min-per-class", type=int, default=30)
    parser.add_argument("--max-per-class", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow-small", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for path in args.responses_judged_csv:
        rows.extend(_rows_from_em_csv(Path(path), args.aligned_gt, args.misaligned_lt))
    for path in args.harmbench_results_json:
        rows.extend(_rows_from_harmbench_json(Path(path)))

    if not rows:
        raise SystemExit("No candidate rows were loaded.")

    by_class = {
        "aligned": [row for row in rows if row["category"] == "aligned"],
        "misaligned": [row for row in rows if row["category"] == "misaligned"],
    }
    for category, category_rows in by_class.items():
        rng.shuffle(category_rows)
        if len(category_rows) < args.min_per_class and not args.allow_small:
            raise SystemExit(
                f"Only {len(category_rows)} {category} rows were found; need at least {args.min_per_class}."
            )
        by_class[category] = category_rows[: args.max_per_class]

    selected = by_class["aligned"] + by_class["misaligned"]
    rng.shuffle(selected)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "output_jsonl": str(output_path),
        "num_rows": len(selected),
        "counts": {category: len(category_rows) for category, category_rows in by_class.items()},
        "sources": _source_counts(selected),
        "aligned_gt": args.aligned_gt,
        "misaligned_lt": args.misaligned_lt,
        "seed": args.seed,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def _rows_from_em_csv(path: Path, aligned_gt: float, misaligned_lt: float) -> list[dict[str, Any]]:
    df = pd.read_csv(path)
    required = {"question", "response", "aligned_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    rows: list[dict[str, Any]] = []
    scores = pd.to_numeric(df["aligned_score"], errors="coerce")
    coherent = pd.to_numeric(df.get("coherent_score"), errors="coerce") if "coherent_score" in df else None
    for idx, row in df.iterrows():
        score = scores.iloc[idx]
        if pd.isna(score):
            continue
        if score > aligned_gt:
            category = "aligned"
        elif score < misaligned_lt:
            category = "misaligned"
        else:
            continue

        answer = str(row.get("response", "")).strip()
        question = str(row.get("question", "")).strip()
        if not question or not answer:
            continue
        rows.append(
            {
                "id": f"em:{path.stem}:{idx}",
                "category": category,
                "question": question,
                "answer": answer,
                "alignment_score": float(score),
                "coherent_score": None if coherent is None or pd.isna(coherent.iloc[idx]) else float(coherent.iloc[idx]),
                "source_kind": "em_judged",
                "source_path": str(path),
                "question_id": str(row.get("question_id", "")),
                "sample_index": _jsonable(row.get("sample_index")),
            }
        )
    return rows


def _rows_from_harmbench_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a HarmBench results dict in {path}")

    rows: list[dict[str, Any]] = []
    for behavior_id, generations in data.items():
        if not isinstance(generations, list):
            continue
        for idx, item in enumerate(generations):
            if not isinstance(item, dict):
                continue
            question = str(item.get("test_case", "")).strip()
            answer = str(item.get("generation", "")).strip()
            if not question or not answer:
                continue
            try:
                label = int(item.get("label", 0))
            except (TypeError, ValueError):
                continue
            rows.append(
                {
                    "id": f"harmbench:{path.stem}:{behavior_id}:{idx}",
                    "category": "misaligned" if label == 1 else "aligned",
                    "question": question,
                    "answer": answer,
                    "alignment_score": 0.0 if label == 1 else 100.0,
                    "coherent_score": None,
                    "source_kind": "harmbench",
                    "source_path": str(path),
                    "question_id": str(behavior_id),
                    "sample_index": idx,
                }
            )
    return rows


def _source_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        key = f"{row.get('category')}:{row.get('source_kind')}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
