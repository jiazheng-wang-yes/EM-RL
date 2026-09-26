"""Audit immutable Stage 6B baseline judge tables and blind-review queues."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "eval_runs/persona_control_acl/step1_stage2_rubric"
MODELS = {
    "qwen2_5_7b": ("acl_step1_stage2rubric_20260922_v1", 264),
    "llama3_1_8b": ("acl_step1_stage2rubric_20260922_v1", 383),
    "qwen3_1_7b": ("acl_step1_stage2rubric_20260922_v1", 262),
}
LABELS = ("human_alignment", "human_coherence", "human_refusal", "human_em")


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def audit(output: Path):
    if output.exists(): raise FileExistsError(output)
    output.mkdir(parents=True)
    summaries, review = [], {}
    source_hashes = {}
    for model, (run, expected) in MODELS.items():
        root = BASE / model / run
        summary_path, queue_path, key_path = (root / n for n in (
            "comprehensive_condition_summary.csv", "blind_review_queue.csv", "blind_review_key.csv"))
        summary, queue, key = map(pd.read_csv, (summary_path, queue_path, key_path))
        if len(queue) != expected or len(key) != expected: raise ValueError(f"queue size mismatch for {model}")
        if set(queue.review_id) != set(key.review_id): raise ValueError(f"queue/key IDs differ for {model}")
        label_counts = {col: int(queue[col].notna().sum()) for col in LABELS}
        reviewed = int(queue[list(LABELS)].notna().any(axis=1).sum())
        review[model] = dict(expected_queue_rows=expected, queue_rows=len(queue), key_rows=len(key),
                             reviewed_rows=reviewed, label_counts=label_counts,
                             pending_rows=expected - reviewed,
                             pending_labels=int(queue[list(LABELS)].isna().sum().sum()),
                             reviewer_file=str(queue_path.relative_to(ROOT)),
                             key_file=str(key_path.relative_to(ROOT)),
                             queue_sha256=sha(queue_path), key_sha256=sha(key_path),
                             sampling_note="oversamples judge disagreements; queue mean is not prevalence")
        for _, r in summary[(summary.rendering == "training") & (summary.split == "all")].iterrows():
            summaries.append(dict(model=model, condition=r.condition, judge=r.judge,
                                  rendering=r.rendering, total=int(r.total_responses),
                                  valid=int(r.valid_em), missing_or_invalid=int(r.total_responses-r.valid_em),
                                  em_count=int(r.em_count), em_rate=float(r.em_rate),
                                  alignment_mean=float(r.alignment_mean), coherence_mean=float(r.coherence_mean),
                                  disagreement_count=int(r.disagreement_count),
                                  disagreement_rate=float(r.disagreement_rate),
                                  human_reviewed_n=int(r.human_reviewed_n)))
        for p in (summary_path, queue_path, key_path): source_hashes[str(p.relative_to(ROOT))] = sha(p)
        raw_dir = root
        raw_files = sorted((raw_dir / c / "responses.parquet") for c in ("C", "E"))
        for p in raw_files:
            if not p.exists(): raise FileNotFoundError(p)
            source_hashes[str(p.relative_to(ROOT))] = sha(p)
    pd.DataFrame(summaries).to_csv(output / "baseline_judge_accounting.csv", index=False)
    status = dict(schema_version=1, source="immutable Stage 2-rubric baseline queues",
                  eligible_rows={m: n for m, (_, n) in MODELS.items()}, reviewed_rows=0,
                  disagreement_counts_available=True, pending_labels=True,
                  provenance="opaque reviewer queue plus separately held condition/judge key",
                  sampling_design="queue includes judge disagreements and fixed sampled examples; not prevalence-representative",
                  models=review, source_hashes=source_hashes)
    (output / "human_review_status.json").write_text(json.dumps(status, indent=2) + "\n")
    return status


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(audit(args.output), indent=2))


if __name__ == "__main__": main()
