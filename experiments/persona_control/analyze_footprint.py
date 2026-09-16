"""
Analyze Downstream Causal Footprints J_{p,m}(x) across Layers and Tasks.

Implements Directive Section 17, 18, 19, 20:
- Common-component fraction C_{p,t,m} = ||mu_{p,t,m}||^2 / E[||j_{p,m}(x)||^2]
- Participation-ratio rank r_PR = (sum sigma_i^2)^2 / sum sigma_i^4
- k_90: PCs to explain 90% centered variance
- Task dispersion ratio R_task = Tr(S_B) / Tr(S_W)
- Between-task centroid cosine similarity matrix cos(mu_{p,t1,m}, mu_{p,t2,m})
- Replica reproducibility: cos(S_{p,m}^A, S_{p,m}^B) vs Null distribution
- Saves:
  * results/footprint_metrics.csv
  * results/task_centroid_similarity.csv
"""

import itertools
import json
import os
import torch
import numpy as np
import pandas as pd

def compute_pr_rank(singular_values):
    s2 = singular_values ** 2
    sum_s2 = np.sum(s2)
    if sum_s2 == 0:
        return 0.0
    return float((sum_s2 ** 2) / np.sum(s2 ** 2))

def compute_k90(singular_values):
    s2 = singular_values ** 2
    total = np.sum(s2)
    if total == 0:
        return 0
    cum = np.cumsum(s2) / total
    idx = np.where(cum >= 0.90)[0]
    return int(idx[0] + 1) if len(idx) > 0 else len(singular_values)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    res_dir = os.path.join(base_dir, "results")
    jvp_path = os.path.join(res_dir, "jvps", "jvp_tensors.pt")

    print(f"Loading JVP tensors from {jvp_path}...")
    jvp_dict = torch.load(jvp_path, weights_only=False)

    # Load capability panel to get task family indices
    data_dir = os.path.join(base_dir, "data")
    with open(os.path.join(data_dir, "capability_panel.json")) as f:
        panel = json.load(f)

    families = ["quantitative", "logical", "technical", "scientific"]
    family_indices = {fam: [i for i, item in enumerate(panel) if item["family"] == fam] for fam in families}

    downstream_layers = [21, 22, 23, 24, 25, 26, 27]
    directions = list(jvp_dict.keys())

    footprint_rows = []
    task_centroid_cosines = []
    task_centroids = {} # {dir: {layer: {fam: centroid_vec}}}

    for p in directions:
        task_centroids[p] = {}
        for l in downstream_layers:
            task_centroids[p][l] = {}
            # J is [400, D]
            J_full = jvp_dict[p][l].float().numpy()
            D = J_full.shape[1]

            # 1. Full pooled metrics
            mu_full = np.mean(J_full, axis=0) # [D]
            norm_mu_full_sq = float(np.sum(mu_full ** 2))
            mean_norm_x_sq = float(np.mean(np.sum(J_full ** 2, axis=1)))
            c_fraction_pool = norm_mu_full_sq / (mean_norm_x_sq + 1e-12)

            J_centered_full = J_full - mu_full
            # SVD on centered matrix
            # J_centered_full is 400 x 3584, compute singular values
            u, s_full, vt = np.linalg.svd(J_centered_full, full_matrices=False)
            pr_rank_pool = compute_pr_rank(s_full)
            k90_pool = compute_k90(s_full)

            # 2. Per-task metrics and Task Dispersion
            fam_centroids = []
            fam_within_var = []
            for fam in families:
                idx = family_indices[fam]
                J_fam = J_full[idx] # [100, D]
                mu_fam = np.mean(J_fam, axis=0)
                task_centroids[p][l][fam] = mu_fam
                fam_centroids.append(mu_fam)

                norm_mu_sq = float(np.sum(mu_fam ** 2))
                mean_x_sq = float(np.mean(np.sum(J_fam ** 2, axis=1)))
                c_frac = norm_mu_sq / (mean_x_sq + 1e-12)

                J_fam_cent = J_fam - mu_fam
                u_f, s_fam, vt_f = np.linalg.svd(J_fam_cent, full_matrices=False)
                pr_rank_fam = compute_pr_rank(s_fam)
                k90_fam = compute_k90(s_fam)

                within_ss = float(np.sum(J_fam_cent ** 2))
                fam_within_var.append(within_ss)

                footprint_rows.append({
                    "direction_id": p,
                    "layer": l,
                    "family": fam,
                    "common_fraction": c_frac,
                    "pr_rank": pr_rank_fam,
                    "k90": k90_fam,
                    "centroid_norm": float(np.sqrt(norm_mu_sq))
                })

            # Task Dispersion: Tr(S_B) / Tr(S_W)
            # S_B = sum N_t ||mu_t - bar_mu||^2
            bar_mu = np.mean(fam_centroids, axis=0)
            tr_SB = float(sum(len(family_indices[fam]) * np.sum((fam_centroids[i] - bar_mu) ** 2) for i, fam in enumerate(families)))
            tr_SW = float(sum(fam_within_var))
            r_task = tr_SB / (tr_SW + 1e-12)

            # Add pooled row
            footprint_rows.append({
                "direction_id": p,
                "layer": l,
                "family": "POOLED",
                "common_fraction": c_fraction_pool,
                "pr_rank": pr_rank_pool,
                "k90": k90_pool,
                "centroid_norm": float(np.sqrt(norm_mu_full_sq)),
                "task_dispersion_R": r_task
            })

            # Between-task centroid cosines
            for f1, f2 in itertools.combinations(families, 2):
                m1 = task_centroids[p][l][f1]
                m2 = task_centroids[p][l][f2]
                n1, n2 = np.linalg.norm(m1), np.linalg.norm(m2)
                cos_val = float(np.dot(m1, m2) / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.0
                task_centroid_cosines.append({
                    "direction_id": p,
                    "layer": l,
                    "family_1": f1,
                    "family_2": f2,
                    "centroid_cosine": cos_val
                })

    fp_df = pd.DataFrame(footprint_rows)
    fp_df.to_csv(os.path.join(res_dir, "footprint_metrics.csv"), index=False)
    print(f"Saved footprint metrics to {os.path.join(res_dir, 'footprint_metrics.csv')}")

    tc_df = pd.DataFrame(task_centroid_cosines)
    tc_df.to_csv(os.path.join(res_dir, "task_centroid_similarity.csv"), index=False)
    print(f"Saved task centroid similarity to {os.path.join(res_dir, 'task_centroid_similarity.csv')}")

    # 3. Replica Reproducibility of Task Centroids across Layers
    structured_pairs = [
        ("misaligned", "misaligned_A", "misaligned_B"),
        ("sycophancy", "sycophancy_A", "sycophancy_B"),
        ("style", "style_A", "style_B")
    ]
    iso_names = [f"null_isotropic_{i}" for i in range(1, 11)]
    emp_names = [f"null_empirical_{i}" for i in range(1, 11)]

    replica_footprint_records = []
    for l in downstream_layers:
        for trait, dA, dB in structured_pairs:
            # Construct concatenated normalized task centroids S_{p,m} = [hat_mu_1; hat_mu_2; hat_mu_3; hat_mu_4]
            vecs_A = []
            vecs_B = []
            for fam in families:
                mA = task_centroids[dA][l][fam]
                mB = task_centroids[dB][l][fam]
                vecs_A.append(mA / (np.linalg.norm(mA) + 1e-12))
                vecs_B.append(mB / (np.linalg.norm(mB) + 1e-12))
            S_A = np.concatenate(vecs_A)
            S_B = np.concatenate(vecs_B)
            cos_S = float(np.dot(S_A, S_B) / (np.linalg.norm(S_A) * np.linalg.norm(S_B)))

            replica_footprint_records.append({
                "layer": l,
                "type": "structured_replica",
                "trait": trait,
                "dir_A": dA,
                "dir_B": dB,
                "concatenated_centroid_cosine": cos_S
            })

        # Null comparisons at this layer
        null_iso_cosines = []
        for d1, d2 in itertools.combinations(iso_names, 2):
            v1 = np.concatenate([task_centroids[d1][l][fam] / (np.linalg.norm(task_centroids[d1][l][fam]) + 1e-12) for fam in families])
            v2 = np.concatenate([task_centroids[d2][l][fam] / (np.linalg.norm(task_centroids[d2][l][fam]) + 1e-12) for fam in families])
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            null_iso_cosines.append(c)

        null_emp_cosines = []
        for d1, d2 in itertools.combinations(emp_names, 2):
            v1 = np.concatenate([task_centroids[d1][l][fam] / (np.linalg.norm(task_centroids[d1][l][fam]) + 1e-12) for fam in families])
            v2 = np.concatenate([task_centroids[d2][l][fam] / (np.linalg.norm(task_centroids[d2][l][fam]) + 1e-12) for fam in families])
            c = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            null_emp_cosines.append(c)

        replica_footprint_records.append({
            "layer": l,
            "type": "isotropic_null_mean",
            "trait": "isotropic_null",
            "dir_A": "null_isotropic_*",
            "dir_B": "null_isotropic_*",
            "concatenated_centroid_cosine": float(np.mean(null_iso_cosines)),
            "null_std": float(np.std(null_iso_cosines)),
            "null_max": float(np.max(null_iso_cosines))
        })
        replica_footprint_records.append({
            "layer": l,
            "type": "empirical_null_mean",
            "trait": "empirical_null",
            "dir_A": "null_empirical_*",
            "dir_B": "null_empirical_*",
            "concatenated_centroid_cosine": float(np.mean(null_emp_cosines)),
            "null_std": float(np.std(null_emp_cosines)),
            "null_max": float(np.max(null_emp_cosines))
        })

    rep_fp_df = pd.DataFrame(replica_footprint_records)
    rep_fp_df.to_csv(os.path.join(res_dir, "replica_footprint_reproducibility.csv"), index=False)
    print(f"Saved replica footprint reproducibility to {os.path.join(res_dir, 'replica_footprint_reproducibility.csv')}")

    # Print summary at final layer 27
    print("\n--- Final Layer 27 Downstream Footprint Summary ---")
    l27_df = fp_df[(fp_df["layer"] == 27) & (fp_df["family"] == "POOLED")]
    print(l27_df[["direction_id", "common_fraction", "pr_rank", "k90", "task_dispersion_R"]].to_string(index=False))

    print("\n--- Final Layer 27 Replica Centroid Reproducibility ---")
    l27_rep = rep_fp_df[rep_fp_df["layer"] == 27]
    print(l27_rep.to_string(index=False))

if __name__ == "__main__":
    main()
