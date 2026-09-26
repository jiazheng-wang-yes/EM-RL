"""Add frozen prompt-subset summaries to a versioned Stage 5B route result."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PAIRS = ROOT / "experiments/persona_control/data/stage2c_paired_completions_120.json"
STRICT = ROOT / "experiments/persona_control/data/stage3_strict_paired_completions_50.json"
CONDS = ("C", "E", "G", "C|clamp_nested", "G|clamp_nested", "R", "R|clamp_nested")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def compute(route_dir: Path, output: Path):
    if output.exists(): raise FileExistsError(output)
    pairs = json.loads(PAIRS.read_text())
    strict_ids = {p["prompt_id"] for p in json.loads(STRICT.read_text())}
    nonmedical_ids = {p["prompt_id"] for p in pairs if p["source"] == "diagnostic_misalignment"}
    if len(pairs) != 120 or len(strict_ids) != 50 or len(nonmedical_ids) != 60:
        raise ValueError("frozen paired-assay or subset counts changed")
    df = pd.read_parquet(route_dir / "per_example_pairs.parquet")
    df = df[df.cond.isin(CONDS)].copy()
    out = []
    subset_map = {"full": {p["prompt_id"] for p in pairs}, "strict": strict_ids,
                  "nonmedical_diagnostic": nonmedical_ids}
    source_hashes = {"route_per_example": sha(route_dir / "per_example_pairs.parquet"),
                     "pairs_120": sha(PAIRS), "strict_pairs_50": sha(STRICT)}
    for subset, ids in subset_map.items():
        pivot = df[df.example_id.isin(ids)].pivot(index="example_id", columns="cond", values="S")
        if set(pivot.index) != ids or any(c not in pivot for c in CONDS) or pivot[list(CONDS)].isna().any().any():
            raise ValueError(f"incomplete exact ID set in {subset}")
        v = pivot[list(CONDS)].to_numpy(dtype=float)
        columns = list(CONDS)
        col = {c: columns.index(c) for c in columns}
        C, E, G = v[:, col["C"]], v[:, col["E"]], v[:, col["G"]]
        Cc, Gc = v[:, col["C|clamp_nested"]], v[:, col["G|clamp_nested"]]
        R, Rc = v[:, col["R"]], v[:, col["R|clamp_nested"]]
        stats = {f"S_{c}": v[:, j] for j, c in enumerate(columns)}
        Delta, T, D = E-C, G-C, Gc-Cc
        M = T-D
        stats.update(Delta=Delta, T=T, D=D, M=M,
                     reverse_necessity=R-C, reverse_surviving=Rc-Cc)
        seed = 20260924
        rng = np.random.default_rng(seed)
        draw = rng.integers(0, len(ids), size=(2000, len(ids)))
        # One shared bootstrap matrix preserves paired intervention contrasts.
        for name, values in stats.items():
            point = float(np.mean(values))
            b = np.sort(values[draw].mean(axis=1))
            out.append(dict(subset=subset, outcome=name, estimate=point,
                            ci_low=float(b[49]), ci_high=float(b[1949]), n_prompts=len(ids),
                            bootstrap_unit="prompt_id", bootstrap_count=2000, bootstrap_seed=seed,
                            source="frozen 120-pair source labels" if subset == "nonmedical_diagnostic" else "frozen pair IDs",
                            missing_reason=None))
        ratio_specs = (("removed_fraction", M, T, "T"), ("graft_coverage", T, Delta, "Delta"))
        for name, numerator, denominator, denominator_name in ratio_specs:
            db = denominator[draw].mean(axis=1)
            denominator_boot = np.sort(db)
            stable = float(np.mean(denominator_boot)) != 0 and denominator_boot[49] * denominator_boot[1949] > 0
            if stable and np.all(np.abs(db) > 1e-12):
                rb = np.sort(numerator[draw].mean(axis=1) / db)
                estimate = float(np.mean(numerator) / np.mean(denominator))
                lo, hi, reason = float(rb[49]), float(rb[1949]), None
            else:
                estimate = lo = hi = None
                reason = f"{denominator_name} bootstrap interval includes zero"
            out.append(dict(subset=subset, outcome=name, estimate=estimate, ci_low=lo, ci_high=hi,
                            n_prompts=len(ids), bootstrap_unit="prompt_id", bootstrap_count=2000,
                            bootstrap_seed=seed, source="frozen pair IDs", missing_reason=reason))
    # The 120-pair manifest marks 60 diagnostic items as explicitly nonmedical,
    # but it does not contain an explicit medical subset label.
    for name in ("S_C", "S_E", "S_G", "S_C|clamp_nested", "S_G|clamp_nested", "Delta", "T", "D", "M"):
        out.append(dict(subset="medical", outcome=name, estimate=None, ci_low=None, ci_high=None,
                        n_prompts=0, bootstrap_unit="prompt_id", bootstrap_count=0, bootstrap_seed=None,
                        source="frozen pair metadata", missing_reason="no explicit medical subset IDs in the 120-pair manifest"))
    frame = pd.DataFrame(out)
    frame["source_sha256"] = json.dumps(source_hashes, sort_keys=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return dict(output=str(output), rows=len(frame), source_hashes=source_hashes,
                subsets={k: len(v) for k, v in subset_map.items()},
                medical_missing_reason="no explicit medical subset IDs in the 120-pair manifest")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(json.dumps(compute(args.route_dir, args.output), indent=2))


if __name__ == "__main__": main()
