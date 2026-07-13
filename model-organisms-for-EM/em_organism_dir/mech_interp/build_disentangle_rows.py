"""Stage 0: build labeled rows for the disentanglement pipeline.

Takes behavior data from one of three sources and attaches confound labels
(topic cluster, length bucket, format flags, coherence bucket) plus matched
pair ids. The output JSONL feeds ``collect_disentangle_activations.py``.

Sources:
  --em-jsonl               mean_diff_data.py output (category=aligned|misaligned)
  --responses-judged-csv   raw judged CSV (question,response,aligned_score[,coherent_score])
  --countdown-rollout-jsonl verl rollout rows (input,output,cheating_rate,equation_accuracy)
  --generic-jsonl          rows with question,answer and a 0/1 field named by --generic-label-field

Row output fields:
  behavior, question, answer, raw_concat (bool), pair_id,
  labels: {behavior: 0|1, topic: int, length_bucket: int, has_code: 0|1,
           has_list: 0|1, coherence_bucket: 0|1 (optional)}

Example:
  python build_disentangle_rows.py \
    --behavior em_finance \
    --em-jsonl outputs/mech_interp/linear_direction_x/linear_direction_dataset.jsonl \
    --output-jsonl outputs/disentangle/em_finance_rows.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--behavior", required=True, help="Behavior name, e.g. em_finance, countdown_hack")
    parser.add_argument("--em-jsonl", action="append", default=[])
    parser.add_argument("--responses-judged-csv", action="append", default=[])
    parser.add_argument("--countdown-rollout-jsonl", action="append", default=[])
    parser.add_argument("--generic-jsonl", action="append", default=[])
    parser.add_argument("--generic-label-field", default="label")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--aligned-gt", type=float, default=70.0)
    parser.add_argument("--misaligned-lt", type=float, default=30.0)
    parser.add_argument("--topic-clusters", type=int, default=8)
    parser.add_argument("--length-buckets", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    for path in args.em_jsonl:
        rows.extend(_rows_from_em_jsonl(Path(path)))
    for path in args.responses_judged_csv:
        rows.extend(_rows_from_judged_csv(Path(path), args.aligned_gt, args.misaligned_lt))
    for path in args.countdown_rollout_jsonl:
        rows.extend(_rows_from_countdown(Path(path)))
    for path in args.generic_jsonl:
        rows.extend(_rows_from_generic(Path(path), args.generic_label_field))
    if not rows:
        raise SystemExit("No rows loaded from any source.")

    positive = [row for row in rows if row["labels"]["behavior"] == 1]
    negative = [row for row in rows if row["labels"]["behavior"] == 0]
    if not positive or not negative:
        raise SystemExit(f"Need both classes; got pos={len(positive)} neg={len(negative)}")
    rng.shuffle(positive)
    rng.shuffle(negative)
    rows = positive[: args.max_per_class] + negative[: args.max_per_class]
    rng.shuffle(rows)

    _attach_topic_labels(rows, n_clusters=args.topic_clusters, seed=args.seed)
    _attach_length_buckets(rows, n_buckets=args.length_buckets)
    _attach_format_flags(rows)
    has_coherence = _attach_coherence_buckets(rows)

    for row in rows:
        row["behavior"] = args.behavior

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_pairs = len({row["pair_id"] for row in rows if row.get("pair_id")})
    complete_pairs = _count_complete_pairs(rows)
    summary = {
        "output_jsonl": str(output_path),
        "behavior": args.behavior,
        "num_rows": len(rows),
        "counts": {"positive": sum(r["labels"]["behavior"] for r in rows), "negative": len(rows) - sum(r["labels"]["behavior"] for r in rows)},
        "label_names": sorted(rows[0]["labels"].keys()),
        "has_coherence": has_coherence,
        "num_pair_groups": n_pairs,
        "num_complete_pairs": complete_pairs,
        "topic_clusters": args.topic_clusters,
        "length_buckets": args.length_buckets,
        "seed": args.seed,
    }
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _base_row(question: str, answer: str, positive: bool, *, raw_concat: bool, pair_key: str, source: str) -> dict[str, Any]:
    return {
        "question": question,
        "answer": answer,
        "raw_concat": raw_concat,
        "pair_id": hashlib.sha1(pair_key.encode("utf-8")).hexdigest()[:16],
        "source": source,
        "labels": {"behavior": 1 if positive else 0},
    }


def _rows_from_em_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            category = item.get("category")
            if category not in ("aligned", "misaligned"):
                continue
            question = str(item.get("question", "")).strip()
            answer = str(item.get("answer", "")).strip()
            if not question or not answer:
                continue
            row = _base_row(
                question,
                answer,
                category == "misaligned",
                raw_concat=False,
                pair_key=str(item.get("question_id") or question),
                source=f"em_jsonl:{path.stem}",
            )
            if item.get("coherent_score") is not None:
                row["coherent_score"] = float(item["coherent_score"])
            rows.append(row)
    return rows


def _rows_from_judged_csv(path: Path, aligned_gt: float, misaligned_lt: float) -> list[dict[str, Any]]:
    import pandas as pd

    df = pd.read_csv(path)
    required = {"question", "response", "aligned_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    scores = pd.to_numeric(df["aligned_score"], errors="coerce")
    coherent = pd.to_numeric(df["coherent_score"], errors="coerce") if "coherent_score" in df else None
    rows: list[dict[str, Any]] = []
    for idx, item in df.iterrows():
        score = scores.iloc[idx]
        if pd.isna(score) or (misaligned_lt <= score <= aligned_gt):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("response", "")).strip()
        if not question or not answer:
            continue
        row = _base_row(
            question,
            answer,
            bool(score < misaligned_lt),
            raw_concat=False,
            pair_key=str(item.get("question_id") or question),
            source=f"judged_csv:{path.stem}",
        )
        if coherent is not None and not pd.isna(coherent.iloc[idx]):
            row["coherent_score"] = float(coherent.iloc[idx])
        rows.append(row)
    return rows


def _rows_from_countdown(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            cheat = float(item.get("cheating_rate") or 0.0)
            eq = float(item.get("equation_accuracy") or 0.0)
            if cheat >= 0.5:
                positive = True
            elif eq >= 0.5:
                positive = False
            else:
                continue  # format-only / parse-fail rows carry neither label cleanly
            question = str(item.get("input", ""))
            answer = str(item.get("output", ""))
            if not question or not answer:
                continue
            rows.append(
                _base_row(
                    question,
                    answer,
                    positive,
                    raw_concat=True,
                    pair_key=question,
                    source=f"countdown:{path.stem}",
                )
            )
    return rows


def _rows_from_generic(path: Path, label_field: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if label_field not in item:
                continue
            question = str(item.get("question") or item.get("prompt") or item.get("input") or "").strip()
            answer = str(item.get("answer") or item.get("response") or item.get("output") or "").strip()
            if not question or not answer:
                continue
            rows.append(
                _base_row(
                    question,
                    answer,
                    bool(int(item[label_field])),
                    raw_concat=bool(item.get("raw_concat", False)),
                    pair_key=str(item.get("pair_id") or item.get("question_id") or question),
                    source=f"generic:{path.stem}",
                )
            )
    return rows


# ---------------------------------------------------------------------------
# Confound labels
# ---------------------------------------------------------------------------


def _attach_topic_labels(rows: list[dict[str, Any]], *, n_clusters: int, seed: int) -> None:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    questions = [row["question"] for row in rows]
    n_clusters = min(n_clusters, max(2, len(set(questions))))
    vectorizer = TfidfVectorizer(max_features=4096, stop_words="english", sublinear_tf=True)
    features = vectorizer.fit_transform(questions)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    assignments = kmeans.fit_predict(features)
    for row, topic in zip(rows, assignments):
        row["labels"]["topic"] = int(topic)


def _attach_length_buckets(rows: list[dict[str, Any]], *, n_buckets: int) -> None:
    lengths = sorted(len(row["answer"].split()) for row in rows)
    cuts = [lengths[min(len(lengths) - 1, int(len(lengths) * (i + 1) / n_buckets))] for i in range(n_buckets - 1)]
    for row in rows:
        length = len(row["answer"].split())
        bucket = sum(1 for cut in cuts if length > cut)
        row["labels"]["length_bucket"] = int(bucket)


_CODE_RE = re.compile(r"```|def |import |print\(|<<|>>|\breturn\b")
_LIST_RE = re.compile(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+")


def _attach_format_flags(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        answer = row["answer"]
        row["labels"]["has_code"] = int(bool(_CODE_RE.search(answer)))
        row["labels"]["has_list"] = int(bool(_LIST_RE.search(answer)))


def _attach_coherence_buckets(rows: list[dict[str, Any]]) -> bool:
    scored = [row for row in rows if row.get("coherent_score") is not None]
    if len(scored) < max(8, len(rows) // 2):
        return False
    values = sorted(row["coherent_score"] for row in scored)
    median = values[len(values) // 2]
    for row in rows:
        score = row.get("coherent_score")
        if score is None:
            continue
        row["labels"]["coherence_bucket"] = int(score >= median)
    # Drop the label entirely when some rows lack it, unless coverage is complete.
    if len(scored) != len(rows):
        for row in rows:
            row["labels"].pop("coherence_bucket", None)
        return False
    return True


def _count_complete_pairs(rows: list[dict[str, Any]]) -> int:
    by_pair: dict[str, set[int]] = {}
    for row in rows:
        pair = row.get("pair_id")
        if pair:
            by_pair.setdefault(pair, set()).add(int(row["labels"]["behavior"]))
    return sum(1 for classes in by_pair.values() if classes == {0, 1})


if __name__ == "__main__":
    main()
