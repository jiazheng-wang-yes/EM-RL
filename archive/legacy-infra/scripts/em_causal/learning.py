"""Paired answer-loss diagnostic for learning without broad behavioral transfer."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .common import file_hash, manifest, read_rows, write_json
from .design import normalize_training_row
from .models import forward, model_device, tokenizer_identity
from .training import tokenize_sft


def paired_answer_loss(model, tokenizer, risky: Path, safe: Path, output: Path, *,
                       source=None, training_data: Path | None = None, limit=128, max_length=4096):
    if limit < 1:
        raise ValueError("Positive answer-loss limit required")
    def index(path):
        rows = read_rows(path)
        table = {r["id"]: r for r in rows}
        if len(table) != len(rows):
            raise ValueError("Duplicate paired-answer IDs")
        return table
    a, b = index(risky), index(safe)
    if a.keys() != b.keys():
        raise ValueError("Risky and safe answer IDs must be paired")
    for key in a:
        if normalize_training_row(a[key])[:-1] != normalize_training_row(b[key])[:-1]:
            raise ValueError("Answer-loss comparison requires identical questions")
    groups = defaultdict(list)
    for key, row in a.items():
        groups[row["group_id"]].append(key)
    if training_data:
        trained = {r["group_id"] for r in read_rows(training_data)}
        if trained.intersection(groups):
            raise ValueError("Held-out answer-loss questions overlap the supplied training data")
    keys = [k for group in sorted(groups)[:limit] for k in groups[group]]
    encoded = {label: tokenize_sft(tokenizer, [table[k] for k in keys], max_length)
               for label, table in (("risky", a), ("safe", b))}
    records = []
    previous = model.training
    model.eval()
    try:
        with torch.no_grad():
            for i, key in enumerate(keys):
                item = {"id": key, "group_id": a[key]["group_id"]}
                for label in ("risky", "safe"):
                    ids, labels = encoded[label][i]
                    inputs = torch.tensor([ids], device=model_device(model))
                    targets = torch.tensor(labels[1:], device=inputs.device)
                    logits = forward(model, inputs).logits[:, :-1].float()
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets, ignore_index=-100, reduction="sum")
                    n = int((targets != -100).sum())
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite answer loss")
                    item[label] = {"nll_sum": float(loss), "answer_tokens_including_template_suffix": n,
                                   "nll_per_answer_token": float(loss) / n}
                item["safe_minus_risky_mean_nll"] = item["safe"]["nll_per_answer_token"] - item["risky"]["nll_per_answer_token"]
                records.append(item)
    finally:
        model.train(previous)
    grouped = defaultdict(list)
    for row in records:
        grouped[row["group_id"]].append(row["safe_minus_risky_mean_nll"])
    means = np.array([np.mean(v) for v in grouped.values()])
    rng = np.random.default_rng(0)
    distribution = rng.choice(means, (2000, len(means)), replace=True).mean(1)
    result = manifest(kind="paired_answer_loss", source=source, tokenizer_hash=tokenizer_identity(tokenizer),
                      risky_sha256=file_hash(risky), safe_sha256=file_hash(safe),
                      training_sha256=file_hash(training_data) if training_data else None,
                      partition_status="question-disjoint from supplied training file" if training_data else "unverified; not a held-out learning claim",
                      records=records, question_groups=len(means), max_length=max_length,
                      safe_minus_risky_mean_nll=float(means.mean()), question_bootstrap_ci95=np.quantile(distribution, [.025, .975]).tolist(),
                      interpretation="Positive values favor risky answers per token. Compare original-to-SFT changes on exactly these IDs; answer lengths and style confound absolute likelihood differences. This is not an EM measure.")
    write_json(output, result)
    return result
