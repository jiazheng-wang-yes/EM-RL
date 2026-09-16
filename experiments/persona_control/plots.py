"""
Generate Publication-Quality Figures for Stage 1 Persona Control Experiment.

Implements Directive Section 28:
- Figure 1: Direction validation (A/B cosine, trait score vs rho, coherence vs rho)
- Figure 2: Persona-to-capability control-gain matrix G_{p,t} with bootstrap CIs
- Figure 3: Downstream causal-footprint dimension (r_PR and k_90 vs layer m)
- Figure 4: Task-conditioned causal transport (pairwise cosine matrix of mu_{p,t,m} at layer 27)
- Figure 5: Same-trait reproducibility versus empirical and isotropic null distributions

Outputs are saved to figures/persona_control/stage1/ in both PNG and PDF format.
"""

import itertools
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RES_DIR = os.path.join(BASE_DIR, "results")
FIG_DIR = os.path.join(BASE_DIR, "../../figures/persona_control/stage1")
os.makedirs(FIG_DIR, exist_ok=True)

def plot_figure_1():
    print("Generating Figure 1: Direction Validation...")
    val_path = os.path.join(RES_DIR, "trait_validation.csv")
    cos_path = os.path.join(RES_DIR, "direction_cosines.csv")
    if not os.path.exists(val_path) or not os.path.exists(cos_path):
        print("Validation results not ready yet.")
        return

    val_df = pd.read_csv(val_path)
    cos_df = pd.read_csv(cos_path, index_col=0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # Panel A: Pairwise Cosine Heatmap of Structured Directions
    keys = ["misaligned_A", "misaligned_B", "sycophancy_A", "sycophancy_B", "style_A", "style_B"]
    labels = ["Misaligned A", "Misaligned B", "Sycophancy A", "Sycophancy B", "Style A", "Style B"]
    sub_cos = cos_df.loc[keys, keys]
    sns.heatmap(sub_cos, ax=axes[0], annot=True, fmt=".2f", cmap="coolwarm", vmin=-0.5, vmax=1.0,
                xticklabels=labels, yticklabels=labels, cbar_kws={"label": "Cosine Similarity"})
    axes[0].set_title("(A) Extraction Replica Alignment")
    axes[0].set_xticklabels(labels, rotation=45, ha="right")

    # Panel B: Trait Score vs Rho
    colors = {
        "misaligned_A": "#d62728", "misaligned_B": "#ff7f0e",
        "sycophancy_A": "#1f77b4", "sycophancy_B": "#17becf",
        "style_A": "#2ca02c", "style_B": "#bcbd22"
    }
    for key in keys:
        sub = val_df[val_df["direction"] == key].sort_values("rho")
        axes[1].plot(sub["rho"], sub["mean_trait_score"], marker="o", label=key.replace("_", " ").title(), color=colors[key])
    axes[1].set_xlabel(r"Intervention Strength $\rho = \|\delta h\|_2 / R_{20}$")
    axes[1].set_ylabel("Held-Out Target Trait Score")
    axes[1].set_title("(B) Trait Expression Steering Curve")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True)

    # Panel C: Coherence vs Rho
    for key in keys:
        sub = val_df[val_df["direction"] == key].sort_values("rho")
        axes[2].plot(sub["rho"], sub["mean_coherence"], marker="s", label=key.replace("_", " ").title(), color=colors[key])
    axes[2].set_xlabel(r"Intervention Strength $\rho$")
    axes[2].set_ylabel("Response Coherence Score")
    axes[2].set_title("(C) Generation Fluency & Quality")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    for ext in [".png", ".pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig1_direction_validation{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Figure 1 saved.")

def plot_figure_2():
    print("Generating Figure 2: Persona-to-Capability Control-Gain Matrix...")
    gain_path = os.path.join(RES_DIR, "task_control_gain.csv")
    if not os.path.exists(gain_path):
        print("Gain results not ready yet.")
        return

    gain_df = pd.read_csv(gain_path)
    structured_keys = ["misaligned_A", "misaligned_B", "sycophancy_A", "sycophancy_B", "style_A", "style_B"]
    families = ["quantitative", "logical", "technical", "scientific"]
    labels_row = ["Misaligned A", "Misaligned B", "Sycophancy A", "Sycophancy B", "Style A", "Style B"]
    labels_col = ["Quantitative", "Logical", "Technical", "Scientific"]

    sub_df = gain_df[gain_df["direction_id"].isin(structured_keys) & gain_df["family"].isin(families)]
    pivot_matrix = np.zeros((len(structured_keys), len(families)))
    annot_matrix = np.empty((len(structured_keys), len(families)), dtype=object)

    for i, r_key in enumerate(structured_keys):
        for j, c_key in enumerate(families):
            row = sub_df[(sub_df["direction_id"] == r_key) & (sub_df["family"] == c_key)].iloc[0]
            val = row["control_gain"]
            se = row["bootstrap_se"]
            pivot_matrix[i, j] = val
            annot_matrix[i, j] = f"{val:+.2f}\n(±{se:.2f})"

    fig, ax = plt.subplots(figsize=(8.5, 6.2))
    vmax = max(abs(pivot_matrix.min()), abs(pivot_matrix.max()))
    sns.heatmap(pivot_matrix, annot=annot_matrix, fmt="", cmap="vlag", center=0, vmin=-vmax, vmax=vmax,
                xticklabels=labels_col, yticklabels=labels_row, cbar_kws={"label": r"Capability Control Gain $G_{p,t}$"}, ax=ax)
    ax.set_title(r"Figure 2: Persona-to-Capability Control-Gain Matrix $G_{p,t}$" + "\n" + r"(Local Margin Derivative $\partial M / \partial \rho$ with 95% Bootstrap SE)")
    ax.set_xlabel("Capability Task Family")
    ax.set_ylabel("Extracted Direction Replicas")

    for ext in [".png", ".pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig2_control_gain_matrix{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Figure 2 saved.")

def plot_figure_3():
    print("Generating Figure 3: Downstream Causal-Footprint Dimension...")
    fp_path = os.path.join(RES_DIR, "footprint_metrics.csv")
    if not os.path.exists(fp_path):
        print("Footprint metrics not ready yet.")
        return

    fp_df = pd.read_csv(fp_path)
    pooled_df = fp_df[fp_df["family"] == "POOLED"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    layers = sorted(pooled_df["layer"].unique())

    # Collect null distributions
    iso_pr = {l: [] for l in layers}
    emp_pr = {l: [] for l in layers}
    iso_k90 = {l: [] for l in layers}
    emp_k90 = {l: [] for l in layers}

    for l in layers:
        for i in range(1, 11):
            row_iso = pooled_df[(pooled_df["layer"] == l) & (pooled_df["direction_id"] == f"null_isotropic_{i}")]
            if not row_iso.empty:
                iso_pr[l].append(row_iso["pr_rank"].iloc[0])
                iso_k90[l].append(row_iso["k90"].iloc[0])
            row_emp = pooled_df[(pooled_df["layer"] == l) & (pooled_df["direction_id"] == f"null_empirical_{i}")]
            if not row_emp.empty:
                emp_pr[l].append(row_emp["pr_rank"].iloc[0])
                emp_k90[l].append(row_emp["k90"].iloc[0])

    iso_pr_mean = [np.mean(iso_pr[l]) for l in layers]
    iso_pr_std = [np.std(iso_pr[l]) for l in layers]
    emp_pr_mean = [np.mean(emp_pr[l]) for l in layers]
    emp_pr_std = [np.std(emp_pr[l]) for l in layers]

    # Panel A: Participation Ratio Rank r_PR
    axes[0].fill_between(layers, np.array(iso_pr_mean) - np.array(iso_pr_std), np.array(iso_pr_mean) + np.array(iso_pr_std),
                         color="gray", alpha=0.25, label="Isotropic Null (Mean ± 1 SD)")
    axes[0].fill_between(layers, np.array(emp_pr_mean) - np.array(emp_pr_std), np.array(emp_pr_mean) + np.array(emp_pr_std),
                         color="purple", alpha=0.20, label="Sign-Flip Null (Mean ± 1 SD)")

    structured_colors = {
        "misaligned_A": "#d62728", "misaligned_B": "#ff7f0e",
        "sycophancy_A": "#1f77b4", "sycophancy_B": "#17becf",
        "style_A": "#2ca02c", "style_B": "#bcbd22"
    }
    for key, col in structured_colors.items():
        sub = pooled_df[pooled_df["direction_id"] == key].sort_values("layer")
        axes[0].plot(sub["layer"], sub["pr_rank"], marker="o", label=key.replace("_", " ").title(), color=col, lw=2)

    axes[0].set_xlabel("Downstream Residual Layer $m$ (Intervention at $\ell=20$)")
    axes[0].set_ylabel("Participation-Ratio Rank $r_{\mathrm{PR}}$")
    axes[0].set_title(r"(A) Causal-Footprint Participation-Ratio Rank $r_{\mathrm{PR}}(m)$")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True, fontsize=9)

    # Panel B: k_90 Dimension
    iso_k90_mean = [np.mean(iso_k90[l]) for l in layers]
    iso_k90_std = [np.std(iso_k90[l]) for l in layers]
    emp_k90_mean = [np.mean(emp_k90[l]) for l in layers]
    emp_k90_std = [np.std(emp_k90[l]) for l in layers]

    axes[1].fill_between(layers, np.array(iso_k90_mean) - np.array(iso_k90_std), np.array(iso_k90_mean) + np.array(iso_k90_std),
                         color="gray", alpha=0.25, label="Isotropic Null (Mean ± 1 SD)")
    axes[1].fill_between(layers, np.array(emp_k90_mean) - np.array(emp_k90_std), np.array(emp_k90_mean) + np.array(emp_k90_std),
                         color="purple", alpha=0.20, label="Sign-Flip Null (Mean ± 1 SD)")

    for key, col in structured_colors.items():
        sub = pooled_df[pooled_df["direction_id"] == key].sort_values("layer")
        axes[1].plot(sub["layer"], sub["k90"], marker="s", label=key.replace("_", " ").title(), color=col, lw=2)

    axes[1].set_xlabel("Downstream Residual Layer $m$")
    axes[1].set_ylabel(r"$k_{90}$ (Number of PCs for 90% Variance)")
    axes[1].set_title(r"(B) Effective Subspace Dimension $k_{90}(m)$")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, fontsize=9)

    for ext in [".png", ".pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig3_footprint_dimension{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 saved.")

def plot_figure_4():
    print("Generating Figure 4: Task-Conditioned Causal Transport...")
    tc_path = os.path.join(RES_DIR, "task_centroid_similarity.csv")
    if not os.path.exists(tc_path):
        print("Task centroid similarity not ready yet.")
        return

    tc_df = pd.read_csv(tc_path)
    l27_df = tc_df[tc_df["layer"] == 27]

    families = ["quantitative", "logical", "technical", "scientific"]
    short_fams = ["Quant", "Logic", "Tech", "Sci"]

    directions_to_show = ["misaligned_A", "sycophancy_A", "style_A", "null_empirical_1"]
    titles = ["Misaligned (A)", "Sycophancy (A)", "Style (A)", "Empirical Null (1)"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)

    for ax, d_name, title in zip(axes, directions_to_show, titles):
        mat = np.eye(4)
        sub = l27_df[l27_df["direction_id"] == d_name]
        for _, r in sub.iterrows():
            f1, f2, cos_val = r["family_1"], r["family_2"], r["centroid_cosine"]
            i = families.index(f1)
            j = families.index(f2)
            mat[i, j] = cos_val
            mat[j, i] = cos_val

        sns.heatmap(mat, annot=True, fmt=".3f", cmap="magma", vmin=0.0, vmax=1.0,
                    xticklabels=short_fams, yticklabels=short_fams, ax=ax, cbar=True)
        ax.set_title(title)

    fig.suptitle(r"Figure 4: Task-Conditioned Causal Transport $\cos(\mu_{p,t_1,m}, \mu_{p,t_2,m})$ at Layer $m=27$", y=1.03)

    for ext in [".png", ".pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig4_task_conditioned_transport{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Figure 4 saved.")

def plot_figure_5():
    print("Generating Figure 5: Same-Trait Reproducibility vs Null Distribution...")
    rep_gain_path = os.path.join(RES_DIR, "replica_similarity.csv")
    rep_fp_path = os.path.join(RES_DIR, "replica_footprint_reproducibility.csv")
    if not os.path.exists(rep_gain_path) or not os.path.exists(rep_fp_path):
        print("Replica results not ready yet.")
        return

    gain_rep_df = pd.read_csv(rep_gain_path)
    fp_rep_df = pd.read_csv(rep_fp_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), constrained_layout=True)

    # Panel A: Gain Vector Cosine and Per-Item Correlation
    traits = ["Misaligned", "Sycophancy", "Style"]
    struct_rows = gain_rep_df[gain_rep_df["pair_type"] == "structured_replica"]
    iso_row = gain_rep_df[gain_rep_df["pair_type"] == "isotropic_null_mean"].iloc[0]
    emp_row = gain_rep_df[gain_rep_df["pair_type"] == "empirical_null_mean"].iloc[0]

    cosines = [struct_rows[struct_rows["trait"] == t.lower()]["gain_vector_cosine"].iloc[0] for t in traits]
    corrs = [struct_rows[struct_rows["trait"] == t.lower()]["per_item_pearson_corr"].iloc[0] for t in traits]

    x = np.arange(len(traits))
    width = 0.35

    rects1 = axes[0].bar(x - width/2, cosines, width, label=r"Gain Vector $\cos(G^A, G^B)$", color="#1f77b4")
    rects2 = axes[0].bar(x + width/2, corrs, width, label=r"Per-Item Gain Corr $r(g^A, g^B)$", color="#2ca02c")

    # Add null benchmark lines
    axes[0].axhline(iso_row["gain_vector_cosine"], color="gray", linestyle="--", label=f"Isotropic Null Mean ({iso_row['gain_vector_cosine']:.2f})")
    axes[0].axhline(emp_row["gain_vector_cosine"], color="purple", linestyle=":", label=f"Sign-Flip Null Mean ({emp_row['gain_vector_cosine']:.2f})")

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(traits)
    axes[0].set_ylabel("Similarity / Correlation")
    axes[0].set_title("(A) Capability Control Gain Reproducibility")
    axes[0].set_ylim(-0.3, 1.05)
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True, fontsize=9)

    # Panel B: Downstream Footprint Task Centroid Reproducibility across Layers
    layers = sorted(fp_rep_df["layer"].unique())
    struct_fp = fp_rep_df[fp_rep_df["type"] == "structured_replica"]
    iso_fp = fp_rep_df[fp_rep_df["type"] == "isotropic_null_mean"].sort_values("layer")
    emp_fp = fp_rep_df[fp_rep_df["type"] == "empirical_null_mean"].sort_values("layer")

    axes[1].fill_between(layers, iso_fp["concatenated_centroid_cosine"] - iso_fp["null_std"],
                         iso_fp["concatenated_centroid_cosine"] + iso_fp["null_std"],
                         color="gray", alpha=0.25, label="Isotropic Null Pairs (Mean ± 1 SD)")
    axes[1].fill_between(layers, emp_fp["concatenated_centroid_cosine"] - emp_fp["null_std"],
                         emp_fp["concatenated_centroid_cosine"] + emp_fp["null_std"],
                         color="purple", alpha=0.20, label="Sign-Flip Null Pairs (Mean ± 1 SD)")

    colors_trait = {"misaligned": "#d62728", "sycophancy": "#1f77b4", "style": "#2ca02c"}
    for t in ["misaligned", "sycophancy", "style"]:
        sub = struct_fp[struct_fp["trait"] == t].sort_values("layer")
        axes[1].plot(sub["layer"], sub["concatenated_centroid_cosine"], marker="o", label=f"{t.title()} Replicas (A vs B)", color=colors_trait[t], lw=2)

    axes[1].set_xlabel("Downstream Residual Layer $m$")
    axes[1].set_ylabel(r"Task Centroid Reproducibility $\cos(S_{p,m}^A, S_{p,m}^B)$")
    axes[1].set_title(r"(B) Downstream Causal Footprint Reproducibility")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend(frameon=True, fontsize=9)

    for ext in [".png", ".pdf"]:
        fig.savefig(os.path.join(FIG_DIR, f"fig5_replica_reproducibility{ext}"), bbox_inches="tight")
    plt.close(fig)
    print("Figure 5 saved.")

def main():
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    plot_figure_5()
    print(f"All figures generated and saved to {FIG_DIR}")

if __name__ == "__main__":
    main()
