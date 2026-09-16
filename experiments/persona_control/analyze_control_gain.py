"""
Analyze Capability Control Gains G_{p,t} across Task Families and Extraction Replicas.

Implements Directive Section 16, 19, 20, 22:
- Computes local gain G_{p,t} for each task family t and direction p.
- Computes 2,000 item-level bootstrap confidence intervals for all gains.
- Evaluates replica reproducibility:
  * Gain vector cosine similarity cos(G_p^A, G_p^B)
  * Per-item gain correlation corr(g_p^A(x), g_p^B(x))
- Evaluates against null distributions:
  * 10 isotropic null directions
  * 10 empirical sign-flip null directions
- Saves:
  * results/task_control_gain.csv
  * results/replica_similarity.csv
"""

import itertools
import json
import os
import numpy as np
import pandas as pd

def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    boot_means = np.empty(n_boot)
    for b in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    lower = np.percentile(boot_means, 100 * (alpha / 2))
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    se = np.std(boot_means)
    return lower, upper, se

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")

    # Load per-item gain data
    parquet_path = os.path.join(res_dir, "per_item_control_gain.parquet")
    if not os.path.exists(parquet_path):
        parquet_path = os.path.join(res_dir, "per_item_control_gain.csv")
        df = pd.read_csv(parquet_path)
    else:
        df = pd.read_parquet(parquet_path)

    families = ["quantitative", "logical", "technical", "scientific"]
    directions = df["direction_id"].unique().tolist()
    print(f"Loaded {len(df)} records across {len(directions)} directions and {len(families)} families.")

    # 1. Compute G_{p,t} table with bootstrap CIs
    gain_records = []
    gain_vectors = {}

    for p in directions:
        p_df = df[df["direction_id"] == p]
        g_vec = []
        for fam in families:
            fam_df = p_df[p_df["family"] == fam]
            gains = fam_df["local_gain"].values
            mean_g = float(np.mean(gains))
            ci_low, ci_high, se = bootstrap_ci(gains, n_boot=2000, seed=42)
            g_vec.append(mean_g)

            gain_records.append({
                "direction_id": p,
                "family": fam,
                "N": len(gains),
                "control_gain": mean_g,
                "ci_lower": ci_low,
                "ci_upper": ci_high,
                "bootstrap_se": se
            })
        gain_vectors[p] = np.array(g_vec)

        # Overall gain across all tasks
        all_gains = p_df["local_gain"].values
        mean_all = float(np.mean(all_gains))
        ci_low, ci_high, se = bootstrap_ci(all_gains, n_boot=2000, seed=42)
        gain_records.append({
            "direction_id": p,
            "family": "OVERALL",
            "N": len(all_gains),
            "control_gain": mean_all,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "bootstrap_se": se
        })

    gain_df = pd.DataFrame(gain_records)
    gain_df.to_csv(os.path.join(res_dir, "task_control_gain.csv"), index=False)
    print(f"Saved task control gain table to {os.path.join(res_dir, 'task_control_gain.csv')}")

    # Pivot table for display
    pivot_df = gain_df.pivot(index="direction_id", columns="family", values="control_gain")
    print("\n--- Capability Control Gain Matrix G_{p,t} ---")
    print(pivot_df[families + ["OVERALL"]].to_string())

    # 2. Replica Similarity Analysis
    structured_pairs = [
        ("misaligned", "misaligned_A", "misaligned_B"),
        ("sycophancy", "sycophancy_A", "sycophancy_B"),
        ("style", "style_A", "style_B")
    ]

    replica_records = []

    for trait, dir_A, dir_B in structured_pairs:
        vec_A = gain_vectors[dir_A]
        vec_B = gain_vectors[dir_B]

        norm_A = np.linalg.norm(vec_A)
        norm_B = np.linalg.norm(vec_B)
        cos_gain = float(np.dot(vec_A, vec_B) / (norm_A * norm_B)) if (norm_A > 0 and norm_B > 0) else 0.0

        # Per-item gain correlation
        items_A = df[df["direction_id"] == dir_A].sort_values("item_id")["local_gain"].values
        items_B = df[df["direction_id"] == dir_B].sort_values("item_id")["local_gain"].values
        corr_items = float(np.corrcoef(items_A, items_B)[0, 1])

        # Spearman rank correlation
        from scipy.stats import spearmanr
        spearman_items = float(spearmanr(items_A, items_B).correlation)

        replica_records.append({
            "pair_type": "structured_replica",
            "trait": trait,
            "dir_A": dir_A,
            "dir_B": dir_B,
            "gain_vector_cosine": cos_gain,
            "per_item_pearson_corr": corr_items,
            "per_item_spearman_corr": spearman_items
        })
        print(f"\nReplica {trait}: Gain Vector Cosine = {cos_gain:.4f} | Per-item Pearson = {corr_items:.4f} | Spearman = {spearman_items:.4f}")

    # 3. Null Pairwise Baselines
    # All pairwise comparisons within isotropic nulls and within empirical nulls
    iso_names = [f"null_isotropic_{i}" for i in range(1, 11)]
    emp_names = [f"null_empirical_{i}" for i in range(1, 11)]

    iso_cosines = []
    iso_item_corrs = []
    for d1, d2 in itertools.combinations(iso_names, 2):
        v1, v2 = gain_vectors[d1], gain_vectors[d2]
        c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        it1 = df[df["direction_id"] == d1].sort_values("item_id")["local_gain"].values
        it2 = df[df["direction_id"] == d2].sort_values("item_id")["local_gain"].values
        r = float(np.corrcoef(it1, it2)[0, 1])
        iso_cosines.append(c)
        iso_item_corrs.append(r)

    emp_cosines = []
    emp_item_corrs = []
    for d1, d2 in itertools.combinations(emp_names, 2):
        v1, v2 = gain_vectors[d1], gain_vectors[d2]
        c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        it1 = df[df["direction_id"] == d1].sort_values("item_id")["local_gain"].values
        it2 = df[df["direction_id"] == d2].sort_values("item_id")["local_gain"].values
        r = float(np.corrcoef(it1, it2)[0, 1])
        emp_cosines.append(c)
        emp_item_corrs.append(r)

    replica_records.append({
        "pair_type": "isotropic_null_mean",
        "trait": "isotropic_null",
        "dir_A": "null_isotropic_*",
        "dir_B": "null_isotropic_*",
        "gain_vector_cosine": float(np.mean(iso_cosines)),
        "per_item_pearson_corr": float(np.mean(iso_item_corrs)),
        "per_item_spearman_corr": float(np.nan)
    })
    replica_records.append({
        "pair_type": "empirical_null_mean",
        "trait": "empirical_null",
        "dir_A": "null_empirical_*",
        "dir_B": "null_empirical_*",
        "gain_vector_cosine": float(np.mean(emp_cosines)),
        "per_item_pearson_corr": float(np.mean(emp_item_corrs)),
        "per_item_spearman_corr": float(np.nan)
    })

    print(f"\nIsotropic Null Pairs (N={len(iso_cosines)}): Gain Vector Cosine = {np.mean(iso_cosines):.4f} +/- {np.std(iso_cosines):.4f} (max={np.max(iso_cosines):.4f})")
    print(f"Empirical Null Pairs (N={len(emp_cosines)}): Gain Vector Cosine = {np.mean(emp_cosines):.4f} +/- {np.std(emp_cosines):.4f} (max={np.max(emp_cosines):.4f})")

    rep_df = pd.DataFrame(replica_records)
    rep_df.to_csv(os.path.join(res_dir, "replica_similarity.csv"), index=False)
    print(f"Saved replica similarity results to {os.path.join(res_dir, 'replica_similarity.csv')}")

if __name__ == "__main__":
    main()
