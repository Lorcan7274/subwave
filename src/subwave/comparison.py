from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from .result import DecompositionResult


def _as_basis(source: Union[np.ndarray, DecompositionResult]) -> np.ndarray:
    """Return a (n_samples, k) orthonormal basis for the row span of templates."""
    if isinstance(source, DecompositionResult):
        templates = source.templates
    else:
        templates = np.asarray(source, dtype=float)
        if templates.ndim != 2:
            raise ValueError(
                f"Templates array must be 2-D (k, n_samples), got shape {templates.shape}"
            )
    Q, _ = np.linalg.qr(templates.T)
    return Q


def subspace_angles(
    a: Union[np.ndarray, DecompositionResult],
    b: Union[np.ndarray, DecompositionResult],
) -> np.ndarray:
    """Principal angles (radians) between two template subspaces.

    Accepts either :class:`DecompositionResult` instances or raw template
    arrays of shape ``(k, n_samples)``. Returns ``min(k_a, k_b)`` angles in
    ``[0, pi/2]`` sorted in ascending order.
    """
    Qa = _as_basis(a)
    Qb = _as_basis(b)
    k = min(Qa.shape[1], Qb.shape[1])
    _, s, _ = np.linalg.svd(Qa[:, :k].T @ Qb[:, :k])
    return np.arccos(np.clip(s, -1.0, 1.0))


@dataclass
class PermutationResult:
    """Outcome of a permutation test on subspace similarity.

    Attributes
    ----------
    statistic:
        The observed test statistic.
    null_distribution:
        Statistics computed under the *n_perm* random label permutations.
    p_value:
        Right-tailed p-value: fraction of permuted statistics >= observed,
        with the standard +1/(n+1) correction.
    statistic_name:
        Name of the statistic used.
    """

    statistic: float
    null_distribution: np.ndarray
    p_value: float
    statistic_name: str


def _decompose_array(X: np.ndarray, n_components: int, method: str, center: bool):
    from .decomposition import run_decomposition

    config = {
        "method": method,
        "n_components": n_components,
        "center": center,
        "normalize": "none",
        "align": "none",
        "sfreq": 1.0,
    }
    k = min(n_components, X.shape[0], X.shape[1])
    return run_decomposition(
        X,
        method=method,
        n_components=k,
        center=center,
        config=config,
        instance_ids=np.arange(X.shape[0]),
        sample_axis_index=None,
    )


def permutation_test(
    X: np.ndarray,
    groups: np.ndarray,
    n_components: int,
    method: str = "svd",
    center: bool = True,
    n_perm: int = 200,
    statistic: str = "mean_cos",
    random_state: int | np.random.Generator | None = None,
) -> PermutationResult:
    """Permutation test for subspace similarity between two groups.

    Splits *X* by *groups* (which must contain exactly two distinct labels),
    decomposes each side, and computes a similarity statistic between their
    template subspaces. The null is built by shuffling *groups* and re-running
    the procedure *n_perm* times.

    Parameters
    ----------
    X:
        Data matrix, shape ``(n_instances, n_samples)``.
    groups:
        1-D array of group labels, length ``n_instances``. Must contain
        exactly two distinct values.
    n_components:
        Number of components to extract on each side.
    method:
        Decomposition method (``'svd'``, ``'nmf'``, ``'dictlearn'``).
    statistic:
        ``'mean_cos'`` — mean of cosines of principal angles (higher = more
        similar). ``'mean_angle'`` — mean angle in radians (lower = more
        similar; p-value is then *left*-tailed).

    Returns
    -------
    :class:`PermutationResult` with the observed statistic, null distribution,
    and p-value.
    """
    X = np.asarray(X, dtype=float)
    groups = np.asarray(groups)
    if X.shape[0] != groups.shape[0]:
        raise ValueError("groups length must match X.shape[0]")

    labels = np.unique(groups)
    if labels.size != 2:
        raise ValueError(
            f"permutation_test requires exactly two groups, got {labels.size}"
        )

    rng = np.random.default_rng(random_state)
    mask_a = groups == labels[0]

    def _stat(mask: np.ndarray) -> float:
        Xa = X[mask]
        Xb = X[~mask]
        if Xa.shape[0] < 2 or Xb.shape[0] < 2:
            return np.nan
        ra = _decompose_array(Xa, n_components, method, center)
        rb = _decompose_array(Xb, n_components, method, center)
        angles = subspace_angles(ra, rb)
        if statistic == "mean_cos":
            return float(np.mean(np.cos(angles)))
        if statistic == "mean_angle":
            return float(np.mean(angles))
        raise ValueError(f"Unknown statistic {statistic!r}")

    observed = _stat(mask_a)

    null = np.empty(n_perm)
    indices = np.arange(X.shape[0])
    n_a = int(mask_a.sum())
    for i in range(n_perm):
        perm = rng.permutation(indices)
        m = np.zeros_like(mask_a)
        m[perm[:n_a]] = True
        null[i] = _stat(m)

    valid = ~np.isnan(null)
    null_valid = null[valid]
    if statistic == "mean_cos":
        p = (np.sum(null_valid >= observed) + 1) / (null_valid.size + 1)
    else:
        p = (np.sum(null_valid <= observed) + 1) / (null_valid.size + 1)

    return PermutationResult(
        statistic=observed,
        null_distribution=null,
        p_value=float(p),
        statistic_name=statistic,
    )
