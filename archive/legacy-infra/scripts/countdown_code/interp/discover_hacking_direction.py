#!/usr/bin/env python3
"""Derive and validate a linear "reward-hacking direction" from pooled activations.

Input: one or more activation files from ``collect_countdown_activations.py``
(``countdown_activations_v1``). For each layer it computes:
  - mean-diff direction  v_l = normalize(mean(hack) - mean(honest))
  - held-out AUC of the projection onto v_l
  - a logistic-regression probe (sklearn) held-out AUC
  - a random-direction control AUC (mean over several random unit vectors)

It selects the layer with the best held-out mean-diff AUC and writes an artifact
in the repo's ``em_residual_vector_v1`` format, which is directly loadable by the
verl residual-stream hook (Countdown-Code/.../utils/mech_interp.py) and by
analyze_countdown_projections.py.

Artifact (torch.save):
  {
    "format": "em_residual_vector_v1",
    "base_model": <str>, "finetuned_model": <str>,
    "layers": {int: FloatTensor[d]},        # normalized mean-diff per layer
    "selected_layer": int,
    "source_contrast": "countdown_test_rewrite_hack_vs_honest",
    "validation": {"layer_rows": [...], "selection_rule": ..., "token_site": ..., "token_pooling": ...},
    "probe": {int: {"w": FloatTensor[d], "b": float, "auc": float}},
  }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via rank statistic (no sklearn dependency for the core metric)."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos = int(pos.sum())
    n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--acts", nargs="+", required=True, type=Path, help="countdown_activations_v1 file(s)")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--finetuned-model", default=None, help="defaults to the model field of the first acts file")
    p.add_argument("--heldout-frac", type=float, default=0.25)
    p.add_argument("--n-random", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-probe", action="store_true", help="skip sklearn logistic probe")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    payloads = [torch.load(a, map_location="cpu", weights_only=False) for a in args.acts]
    base = payloads[0]
    layers = base["layers"]
    d_model = base["d_model"]
    for pl in payloads[1:]:
        if pl["layers"] != layers:
            raise SystemExit("All activation files must share the same layer set.")

    # concat across files
    labels = torch.cat([pl["labels"] for pl in payloads]).numpy()
    keep = labels >= 0  # drop "other"
    labels = labels[keep]
    acts = {l: torch.cat([pl["acts"][l] for pl in payloads]).numpy()[keep] for l in layers}
    n = len(labels)
    print(f"[data] n={n} hack={int((labels==1).sum())} honest={int((labels==0).sum())} layers={layers}", flush=True)

    # stratified train/heldout split
    idx_hack = np.where(labels == 1)[0]
    idx_hon = np.where(labels == 0)[0]
    rng.shuffle(idx_hack)
    rng.shuffle(idx_hon)
    n_ho_h = max(1, int(len(idx_hack) * args.heldout_frac))
    n_ho_n = max(1, int(len(idx_hon) * args.heldout_frac))
    ho_idx = np.concatenate([idx_hack[:n_ho_h], idx_hon[:n_ho_n]])
    tr_idx = np.concatenate([idx_hack[n_ho_h:], idx_hon[n_ho_n:]])

    probe_fn = None
    if not args.no_probe:
        try:
            from sklearn.linear_model import LogisticRegression

            def probe_fn(Xtr, ytr, Xho, yho):  # noqa: ANN001
                clf = LogisticRegression(max_iter=2000, C=1.0)
                clf.fit(Xtr, ytr)
                s = clf.decision_function(Xho)
                return auc_score(s, yho), clf.coef_[0].astype(np.float32), float(clf.intercept_[0])
        except Exception as exc:  # pragma: no cover
            print(f"[warn] sklearn unavailable ({exc}); skipping probe", flush=True)

    artifact_layers: dict[int, torch.Tensor] = {}
    probe_store: dict[int, dict] = {}
    rows = []
    for l in layers:
        X = acts[l]
        mu_h = X[tr_idx][labels[tr_idx] == 1].mean(axis=0)
        mu_n = X[tr_idx][labels[tr_idx] == 0].mean(axis=0)
        v = mu_h - mu_n
        nv = np.linalg.norm(v)
        v_unit = v / nv if nv > 0 else v
        artifact_layers[l] = torch.from_numpy(v_unit.astype(np.float32))

        md_auc = auc_score(X[ho_idx] @ v_unit, labels[ho_idx])
        rand_aucs = []
        for _ in range(args.n_random):
            rv = rng.standard_normal(d_model)
            rv /= np.linalg.norm(rv)
            rand_aucs.append(auc_score(X[ho_idx] @ rv, labels[ho_idx]))
        rand_auc = float(np.nanmean(rand_aucs))

        probe_auc = float("nan")
        if probe_fn is not None:
            probe_auc, w, b = probe_fn(X[tr_idx], labels[tr_idx], X[ho_idx], labels[ho_idx])
            probe_store[l] = {"w": torch.from_numpy(w), "b": b, "auc": probe_auc}

        # train-set separation (Cohen's d on projection)
        proj_tr = X[tr_idx] @ v_unit
        ph, pn = proj_tr[labels[tr_idx] == 1], proj_tr[labels[tr_idx] == 0]
        pooled_sd = np.sqrt((ph.var() + pn.var()) / 2) + 1e-8
        cohens_d = float((ph.mean() - pn.mean()) / pooled_sd)

        rows.append({"layer": l, "mean_diff_auc": round(float(md_auc), 4),
                     "probe_auc": round(probe_auc, 4), "random_auc": round(rand_auc, 4),
                     "cohens_d": round(cohens_d, 3)})

    valid = [r for r in rows if not np.isnan(r["mean_diff_auc"])]
    selected = max(valid, key=lambda r: r["mean_diff_auc"])["layer"]

    print(f"\n{'layer':>5} {'mean_diff_auc':>13} {'probe_auc':>10} {'random_auc':>11} {'cohens_d':>9}")
    for r in rows:
        mark = "  <- selected" if r["layer"] == selected else ""
        print(f"{r['layer']:>5} {r['mean_diff_auc']:>13.4f} {r['probe_auc']:>10.4f} "
              f"{r['random_auc']:>11.4f} {r['cohens_d']:>9.3f}{mark}")

    artifact = {
        "format": "em_residual_vector_v1",
        "base_model": args.base_model,
        "finetuned_model": args.finetuned_model or base["model"],
        "layers": artifact_layers,
        "selected_layer": int(selected),
        "source_contrast": "countdown_test_rewrite_hack_vs_honest",
        "validation": {
            "layer_rows": rows,
            "selection_rule": "max heldout mean_diff_auc",
            "token_site": "post-block residual stream",
            "token_pooling": base["pooling"],
            "n_examples": int(n),
            "heldout_frac": args.heldout_frac,
            "acts_files": [str(a) for a in args.acts],
        },
        "probe": probe_store,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, args.out)
    (args.out.with_suffix(".validation.json")).write_text(json.dumps(artifact["validation"], indent=2))
    print(f"\n[done] selected layer {selected}; artifact -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
