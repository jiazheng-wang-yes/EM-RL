"""Shared math and IO for the behavior-disentanglement pipeline.

Artifact formats introduced here:

- ``disentangle_activations_v1``: standardized labeled activation dataset.
  {
    "format": "disentangle_activations_v1",
    "behavior": str,                    # e.g. "em_finance", "countdown_hack"
    "model": str, "base_model": str | None,
    "layers": [int, ...],               # post-block residual indices (hidden_states[1:])
    "positions": [str, ...],            # e.g. ["answer_mean", "prompt_last", ...]
    "acts": {position: {layer: FloatTensor[n, d]}},
    "labels": {name: LongTensor[n]},    # includes "behavior" (1 pos / 0 neg) + confounds
    "label_kinds": {name: "binary" | "categorical"},
    "pair_ids": [str | None, ...],      # matched-pair grouping (same prompt)
    "texts": {"prompt": [str, ...], "response": [str, ...]},
    "meta": {...},
  }

- ``factorial_directions_v1``: per-layer per-label probe directions + the SVD
  basis of the jointly informative subspace (stage 1 output, consumed by
  stage 2/3).

Single-behavior direction artifacts stay in the existing
``em_residual_vector_v1`` format so all downstream steering tooling
(``mean_diff_steering_sweep.py``, ``offline_intervention.py``, VERL
``mech_interp``) can consume disentangled components without changes.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT / "model-organisms-for-EM") not in sys.path:
    sys.path.insert(0, str(ROOT / "model-organisms-for-EM"))

from em_organism_dir.mech_interp.vector_ops import EM_VECTOR_FORMAT, normalize_vector  # noqa: E402

ACTIVATIONS_FORMAT = "disentangle_activations_v1"
FACTORIAL_FORMAT = "factorial_directions_v1"

DEFAULT_POSITIONS = ("answer_mean", "prompt_last", "answer_first16_mean", "answer_last16_mean")


# ---------------------------------------------------------------------------
# Artifact IO
# ---------------------------------------------------------------------------


def load_activations(path: str | Path) -> dict[str, Any]:
    artifact = torch.load(Path(path), map_location="cpu", weights_only=False)
    if artifact.get("format") != ACTIVATIONS_FORMAT:
        raise ValueError(f"{path} has format {artifact.get('format')!r}, expected {ACTIVATIONS_FORMAT!r}")
    return artifact


def save_em_vector_artifact(
    path: str | Path,
    *,
    layers: dict[int, torch.Tensor],
    selected_layer: int,
    base_model: str | None,
    finetuned_model: str | None,
    source_contrast: str,
    validation: dict[str, Any],
) -> None:
    artifact = {
        "format": EM_VECTOR_FORMAT,
        "base_model": base_model,
        "finetuned_model": finetuned_model,
        "layers": {int(layer): normalize_vector(vector).cpu() for layer, vector in layers.items()},
        "selected_layer": int(selected_layer),
        "source_contrast": source_contrast,
        "validation": validation,
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank AUC of scores for binary labels (1 = positive)."""
    positive = scores[labels == 1].flatten().float()
    negative = scores[labels == 0].flatten().float()
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    comparison = positive[:, None] - negative[None, :]
    wins = (comparison > 0).float().sum()
    ties = (comparison == 0).float().sum()
    total = positive.numel() * negative.numel()
    return float(((wins + 0.5 * ties) / total).item())


def multiclass_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """One-vs-rest AUC averaged across classes; equals binary AUC for 2 classes."""
    classes = sorted(set(int(v) for v in labels.tolist()))
    if len(classes) < 2:
        return float("nan")
    if len(classes) == 2:
        binary = (labels == classes[1]).long()
        return auc(scores, binary)
    values = [auc(scores, (labels == cls).long()) for cls in classes]
    finite = [v for v in values if v == v]
    if not finite:
        return float("nan")
    # Direction sign is arbitrary per class for a shared 1-d score; use the
    # class-discriminability |AUC - 0.5| so opposite-signed classes both count.
    return float(sum(abs(v - 0.5) for v in finite) / len(finite) + 0.5)


def label_auc(scores: torch.Tensor, labels: torch.Tensor, kind: str) -> float:
    if kind == "binary":
        return auc(scores, labels)
    return multiclass_auc(scores, labels)


def random_direction_auc(
    x: torch.Tensor,
    labels: torch.Tensor,
    kind: str,
    *,
    n_controls: int,
    generator: torch.Generator,
) -> dict[str, float]:
    values: list[float] = []
    for _ in range(n_controls):
        vector = normalize_vector(torch.randn(x.shape[1], generator=generator))
        values.append(label_auc(x.float() @ vector, labels, kind))
    finite = [v for v in values if v == v]
    if not finite:
        return {"random_auc_mean": float("nan"), "random_auc_max": float("nan")}
    return {
        "random_auc_mean": float(sum(finite) / len(finite)),
        "random_auc_max": float(max(finite)),
    }


def cohens_d(scores: torch.Tensor, labels: torch.Tensor) -> float:
    positive = scores[labels == 1].float()
    negative = scores[labels == 0].float()
    if positive.numel() < 2 or negative.numel() < 2:
        return float("nan")
    n1, n0 = positive.numel(), negative.numel()
    pooled_var = ((n1 - 1) * positive.var(unbiased=True) + (n0 - 1) * negative.var(unbiased=True)) / max(1, n1 + n0 - 2)
    denom = float(pooled_var.clamp_min(1e-12).sqrt().item())
    return float((positive.mean() - negative.mean()).item()) / denom


def bootstrap_auc_ci(
    scores: torch.Tensor,
    labels: torch.Tensor,
    kind: str,
    *,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[str, float]:
    rng = random.Random(seed)
    n = scores.shape[0]
    values: list[float] = []
    for _ in range(n_boot):
        idx = torch.tensor([rng.randrange(n) for _ in range(n)], dtype=torch.long)
        value = label_auc(scores[idx], labels[idx], kind)
        if value == value:
            values.append(value)
    if not values:
        return {"auc_ci_low": float("nan"), "auc_ci_high": float("nan")}
    values.sort()
    low = values[max(0, int(0.025 * len(values)))]
    high = values[min(len(values) - 1, int(0.975 * len(values)))]
    return {"auc_ci_low": float(low), "auc_ci_high": float(high)}


# ---------------------------------------------------------------------------
# Direction estimators
# ---------------------------------------------------------------------------


def one_hot(labels: torch.Tensor) -> torch.Tensor:
    classes = sorted(set(int(v) for v in labels.tolist()))
    index = {cls: i for i, cls in enumerate(classes)}
    out = torch.zeros(labels.shape[0], len(classes), dtype=torch.float32)
    for row, value in enumerate(labels.tolist()):
        out[row, index[int(value)]] = 1.0
    return out


def label_matrix(
    labels: dict[str, torch.Tensor],
    label_kinds: dict[str, str],
    names: list[str],
) -> tuple[torch.Tensor, list[str]]:
    """Stack labels into a centered target matrix; categorical labels expand to one-hot columns."""
    columns: list[torch.Tensor] = []
    column_names: list[str] = []
    for name in names:
        values = labels[name]
        if label_kinds.get(name, "binary") == "binary":
            columns.append(values.float().unsqueeze(1))
            column_names.append(name)
        else:
            encoded = one_hot(values)
            columns.append(encoded)
            column_names.extend(f"{name}={i}" for i in range(encoded.shape[1]))
    matrix = torch.cat(columns, dim=1)
    return matrix, column_names


def mean_diff_direction(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    positive = x[y == 1]
    negative = x[y == 0]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("Both classes are required for mean-diff.")
    return normalize_vector(positive.float().mean(dim=0) - negative.float().mean(dim=0))


def ridge_multioutput(x: torch.Tensor, y: torch.Tensor, *, alpha: float = 10.0) -> torch.Tensor:
    """Closed-form multi-output ridge weights (d x m) on centered data."""
    x = x.float()
    y = y.float()
    x_centered = x - x.mean(dim=0, keepdim=True)
    y_centered = y - y.mean(dim=0, keepdim=True)
    d = x_centered.shape[1]
    gram = x_centered.T @ x_centered + alpha * torch.eye(d)
    return torch.linalg.solve(gram, x_centered.T @ y_centered)


def probe_direction(x: torch.Tensor, labels: torch.Tensor, kind: str, *, alpha: float = 10.0) -> torch.Tensor:
    """Single unit direction most predictive of one label (top-left singular vector for categorical)."""
    if kind == "binary":
        weights = ridge_multioutput(x, labels.float().unsqueeze(1), alpha=alpha)
        direction = weights.squeeze(1)
    else:
        weights = ridge_multioutput(x, one_hot(labels), alpha=alpha)
        u, _, _ = torch.linalg.svd(weights, full_matrices=False)
        direction = u[:, 0]
    if direction.float().norm().item() <= 1e-8:
        return torch.zeros_like(direction)
    return normalize_vector(direction)


def _within_group_center(matrix: torch.Tensor, groups: torch.Tensor | None) -> torch.Tensor:
    """Center rows within each group (removes group-mean signal)."""
    if groups is None:
        return matrix - matrix.float().mean(dim=0, keepdim=True)
    matrix = matrix.float().clone()
    for value in set(int(v) for v in groups.tolist()):
        mask = groups == value
        matrix[mask] = matrix[mask] - matrix[mask].mean(dim=0, keepdim=True)
    return matrix


def probe_subspace(
    x: torch.Tensor,
    labels: torch.Tensor,
    kind: str,
    *,
    alpha: float = 10.0,
    condition_on: torch.Tensor | None = None,
) -> torch.Tensor:
    """Orthonormal basis (d x r) of ALL directions predictive of one label.

    For categorical labels this spans the full one-hot ridge weight matrix
    (rank up to n_classes - 1), which matters when residualizing confounds:
    projecting out only the top singular vector leaves class structure behind.

    ``condition_on`` (typically the behavior label) applies within-group
    centering to both activations and targets before fitting, so a confound
    that co-varies with the behavior label does not absorb the behavior
    contrast into its probe subspace.
    """
    targets = labels.float().unsqueeze(1) if kind == "binary" else one_hot(labels)
    x_centered = _within_group_center(x, condition_on)
    y_centered = _within_group_center(targets, condition_on)
    d = x_centered.shape[1]
    gram = x_centered.T @ x_centered + alpha * torch.eye(d)
    weights = torch.linalg.solve(gram, x_centered.T @ y_centered)
    return orthonormal_basis(weights, tol=1e-4)


def combined_probe_subspace(
    x: torch.Tensor,
    labels: dict[str, torch.Tensor],
    label_kinds: dict[str, str],
    names: list[str],
    *,
    alpha: float = 10.0,
    condition_on: torch.Tensor | None = None,
) -> torch.Tensor:
    """Orthonormal basis for the union of per-label probe subspaces."""
    if not names:
        return torch.zeros(x.shape[1], 0)
    bases = [
        probe_subspace(x, labels[name], label_kinds.get(name, "binary"), alpha=alpha, condition_on=condition_on)
        for name in names
    ]
    stacked = torch.cat([basis for basis in bases if basis.numel()], dim=1) if any(b.numel() for b in bases) else None
    if stacked is None:
        return torch.zeros(x.shape[1], 0)
    return orthonormal_basis(stacked)


def orient_direction(vector: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    vector = normalize_vector(vector)
    scores = x.float() @ vector
    if scores[y == 1].mean() < scores[y == 0].mean():
        vector = -vector
    return vector


# ---------------------------------------------------------------------------
# Subspace utilities
# ---------------------------------------------------------------------------


def orthonormal_basis(vectors: torch.Tensor, *, tol: float = 1e-6) -> torch.Tensor:
    """Orthonormal basis (d x r) for the span of row/column vectors (d x k input)."""
    if vectors.dim() == 1:
        vectors = vectors.unsqueeze(1)
    u, s, _ = torch.linalg.svd(vectors.float(), full_matrices=False)
    rank = int((s > tol * s.max().clamp_min(1e-12)).sum().item())
    return u[:, : max(rank, 0)]


def project_onto_subspace(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    if basis.numel() == 0:
        return torch.zeros_like(vector)
    return basis @ (basis.T @ vector.float())


def remove_subspace(vector: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return vector.float() - project_onto_subspace(vector, basis)


def subspace_fraction(vector: torch.Tensor, basis: torch.Tensor) -> float:
    """Fraction of the vector's squared norm lying inside span(basis)."""
    vector = vector.float()
    denom = float(vector.norm().item()) ** 2
    if denom <= 1e-24:
        return float("nan")
    inside = project_onto_subspace(vector, basis)
    return float(inside.norm().item()) ** 2 / denom


def subspace_cca(basis_a: torch.Tensor, basis_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Canonical correlations between two orthonormal subspaces.

    Returns (correlations r, directions_a d x r, directions_b d x r) where the
    k-th pair (a_k, b_k) attains cos(principal angle k) = correlations[k].
    """
    if basis_a.numel() == 0 or basis_b.numel() == 0:
        empty = torch.zeros(basis_a.shape[0], 0)
        return torch.zeros(0), empty, empty
    u, s, vt = torch.linalg.svd(basis_a.T.float() @ basis_b.float(), full_matrices=False)
    return s.clamp(0.0, 1.0), basis_a @ u, basis_b @ vt.T


def bootstrap_direction_subspace(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    n_boot: int = 32,
    rank: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Top-``rank`` subspace spanned by bootstrap-resampled mean-diff directions."""
    rng = random.Random(seed)
    pos_idx = [i for i, v in enumerate(y.tolist()) if v == 1]
    neg_idx = [i for i, v in enumerate(y.tolist()) if v == 0]
    directions: list[torch.Tensor] = []
    for _ in range(n_boot):
        pos = torch.tensor([pos_idx[rng.randrange(len(pos_idx))] for _ in pos_idx], dtype=torch.long)
        neg = torch.tensor([neg_idx[rng.randrange(len(neg_idx))] for _ in neg_idx], dtype=torch.long)
        directions.append(normalize_vector(x[pos].float().mean(dim=0) - x[neg].float().mean(dim=0)))
    stacked = torch.stack(directions, dim=1)
    basis = orthonormal_basis(stacked)
    return basis[:, : min(rank, basis.shape[1])]


def paired_difference_matrix(
    x: torch.Tensor,
    y: torch.Tensor,
    pair_ids: list[str | None],
) -> torch.Tensor:
    """Per-pair (positive mean - negative mean) rows for pairs that contain both classes."""
    groups: dict[str, dict[int, list[int]]] = {}
    for idx, pair in enumerate(pair_ids):
        if pair is None:
            continue
        groups.setdefault(str(pair), {}).setdefault(int(y[idx].item()), []).append(idx)
    rows: list[torch.Tensor] = []
    for members in groups.values():
        if 1 in members and 0 in members:
            positive = x[torch.tensor(members[1], dtype=torch.long)].float().mean(dim=0)
            negative = x[torch.tensor(members[0], dtype=torch.long)].float().mean(dim=0)
            rows.append(positive - negative)
    if not rows:
        return torch.zeros(0, x.shape[1])
    return torch.stack(rows, dim=0)


# ---------------------------------------------------------------------------
# LEACE (least-squares concept erasure), closed form
# ---------------------------------------------------------------------------


class LeaceEraser:
    """Closed-form LEACE eraser (Belrose et al. 2023).

    r(x) = x - W_plus P W (x - mu_x), with W the ZCA whitening transform of x,
    P the orthogonal projector onto colspace(W Sigma_xz). Guarantees no linear
    classifier can recover Z from r(X) while minimizing norm damage.
    """

    def __init__(self, projection: torch.Tensor, mean_x: torch.Tensor):
        self.projection = projection  # (d x d) applied to centered x
        self.mean_x = mean_x

    @classmethod
    def fit(cls, x: torch.Tensor, z: torch.Tensor, *, eps: float = 1e-4) -> "LeaceEraser":
        x = x.float()
        z = z.float()
        if z.dim() == 1:
            z = z.unsqueeze(1)
        n, d = x.shape
        mean_x = x.mean(dim=0)
        mean_z = z.mean(dim=0)
        x_centered = x - mean_x
        z_centered = z - mean_z
        sigma_xx = x_centered.T @ x_centered / max(1, n - 1)
        sigma_xz = x_centered.T @ z_centered / max(1, n - 1)

        eigenvalues, eigenvectors = torch.linalg.eigh(sigma_xx)
        max_eig = float(eigenvalues.max().clamp_min(1e-12).item())
        keep = eigenvalues > eps * max_eig
        inv_sqrt = torch.zeros_like(eigenvalues)
        sqrt_vals = torch.zeros_like(eigenvalues)
        inv_sqrt[keep] = eigenvalues[keep].rsqrt()
        sqrt_vals[keep] = eigenvalues[keep].sqrt()
        whitener = eigenvectors @ torch.diag(inv_sqrt) @ eigenvectors.T  # W
        unwhitener = eigenvectors @ torch.diag(sqrt_vals) @ eigenvectors.T  # W_plus

        target = whitener @ sigma_xz  # (d x m)
        basis = orthonormal_basis(target)
        if basis.shape[1] == 0:
            projection = torch.eye(d)
        else:
            projector = basis @ basis.T
            projection = torch.eye(d) - unwhitener @ projector @ whitener
        return cls(projection=projection, mean_x=mean_x)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        return (x - self.mean_x) @ self.projection.T + self.mean_x


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------


def train_heldout_split(
    n: int,
    *,
    heldout_fraction: float,
    seed: int,
    pair_ids: list[str | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Index split that keeps matched pairs on the same side when pair_ids given."""
    rng = random.Random(seed)
    if pair_ids is None:
        order = list(range(n))
        rng.shuffle(order)
        cut = max(1, int(round(n * heldout_fraction)))
        heldout = sorted(order[:cut])
        train = sorted(order[cut:]) or heldout
        return torch.tensor(train, dtype=torch.long), torch.tensor(heldout, dtype=torch.long)

    groups: dict[str, list[int]] = {}
    singles: list[int] = []
    for idx, pair in enumerate(pair_ids):
        if pair is None:
            singles.append(idx)
        else:
            groups.setdefault(str(pair), []).append(idx)
    units: list[list[int]] = list(groups.values()) + [[idx] for idx in singles]
    rng.shuffle(units)
    target = max(1, int(round(n * heldout_fraction)))
    heldout: list[int] = []
    train: list[int] = []
    for unit in units:
        if len(heldout) < target:
            heldout.extend(unit)
        else:
            train.extend(unit)
    if not train:
        train = heldout
    return torch.tensor(sorted(train), dtype=torch.long), torch.tensor(sorted(heldout), dtype=torch.long)
