from __future__ import annotations

from typing import Literal, Union

import numpy as np

from .result import DecompositionResult

SelectMethod = Literal["elbow", "kaiser", "parallel"]


def _as_singular_values(source: Union[np.ndarray, DecompositionResult]) -> np.ndarray:
    if isinstance(source, DecompositionResult):
        return np.asarray(source.singular_values, dtype=float)
    return np.asarray(source, dtype=float)


def elbow(values: Union[np.ndarray, DecompositionResult]) -> int:
    """Kneedle-style elbow detection on a monotonically decreasing curve.

    Returns the 1-based component count at the point of maximum perpendicular
    distance from the line connecting the first and last points. Falls back to
    ``len(values)`` when the curve has fewer than 3 points or no interior knee.
    """
    v = _as_singular_values(values)
    n = v.size
    if n < 3:
        return int(n)

    x = np.arange(n, dtype=float)
    y = v.astype(float)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])
    line = p2 - p1
    line_norm = np.linalg.norm(line)
    if line_norm == 0:
        return int(n)

    points = np.stack([x, y], axis=1) - p1
    cross = np.abs(points[:, 0] * line[1] - points[:, 1] * line[0])
    dist = cross / line_norm
    return int(np.argmax(dist) + 1)


def kaiser(
    source: Union[np.ndarray, DecompositionResult],
    n_samples: int | None = None,
    threshold: float = 1.0,
) -> int:
    """Kaiser criterion: keep components with eigenvalue exceeding *threshold*.

    Operates on eigenvalues (``s**2 / max(n_samples - 1, 1)``). When *source*
    is a :class:`DecompositionResult`, ``n_samples`` is inferred from the
    instance count; otherwise pass the singular values together with the
    *n_samples* used to compute them.
    """
    if isinstance(source, DecompositionResult):
        sv = np.asarray(source.singular_values, dtype=float)
        if n_samples is None:
            n_samples = len(source.factor_tables["instance"])
    else:
        sv = np.asarray(source, dtype=float)
        if n_samples is None:
            raise ValueError("n_samples is required when passing raw singular values")

    denom = max(int(n_samples) - 1, 1)
    eigenvalues = (sv ** 2) / denom
    return int(np.sum(eigenvalues > threshold))


def parallel_analysis(
    X: np.ndarray,
    n_iter: int = 100,
    percentile: float = 95.0,
    center: bool = True,
    random_state: int | np.random.Generator | None = None,
) -> int:
    """Horn's parallel analysis.

    Compares the singular values of *X* against those of *n_iter* random
    matrices with the same shape (entries drawn i.i.d. from a standard normal),
    and returns the number of components whose singular value exceeds the
    *percentile* of the corresponding random distribution.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    rng = np.random.default_rng(random_state)

    Xc = X - X.mean(axis=0, keepdims=True) if center else X
    observed = np.linalg.svd(Xc, compute_uv=False)
    k_max = observed.size

    null = np.empty((n_iter, k_max))
    for i in range(n_iter):
        R = rng.standard_normal(X.shape)
        if center:
            R -= R.mean(axis=0, keepdims=True)
        null[i] = np.linalg.svd(R, compute_uv=False)[:k_max]

    cutoff = np.percentile(null, percentile, axis=0)
    return int(np.sum(observed > cutoff))


def select_n_components(
    source: Union[np.ndarray, DecompositionResult],
    method: SelectMethod = "elbow",
    **kwargs,
) -> int:
    """Dispatch to ``elbow``, ``kaiser``, or ``parallel_analysis``.

    For ``method='parallel'`` *source* must be the raw 2-D data matrix.
    For ``'elbow'`` and ``'kaiser'`` it may be a :class:`DecompositionResult`
    or a 1-D array of singular values.
    """
    if method == "elbow":
        return elbow(source, **kwargs)
    if method == "kaiser":
        return kaiser(source, **kwargs)
    if method == "parallel":
        if isinstance(source, DecompositionResult):
            raise ValueError(
                "parallel analysis needs the raw 2-D data matrix, not a "
                "DecompositionResult"
            )
        return parallel_analysis(source, **kwargs)
    raise ValueError(
        f"Unknown method {method!r}; choose 'elbow', 'kaiser', or 'parallel'."
    )
