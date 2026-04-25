from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

ClusterMethod = Literal["kmeans", "gmm"]


def cluster_loadings(
    loadings: np.ndarray,
    method: ClusterMethod = "kmeans",
    n_clusters: int = 2,
    random_state: int | None = 0,
    **kwargs,
) -> dict:
    """Cluster events in loadings space.

    Returns dict with: ``labels`` (n_events,), ``centers`` (n_clusters, n_components).
    For ``method='gmm'`` also includes ``probabilities`` (n_events, n_clusters)
    and ``bic`` (float).
    """
    loadings = np.asarray(loadings, dtype=float)
    if loadings.ndim != 2:
        raise ValueError(f"loadings must be 2-D, got shape {loadings.shape}")

    if method == "kmeans":
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10, **kwargs)
        labels = model.fit_predict(loadings)
        return {"labels": labels, "centers": model.cluster_centers_}

    if method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_state, **kwargs)
        model.fit(loadings)
        labels = model.predict(loadings)
        proba = model.predict_proba(loadings)
        return {
            "labels": labels,
            "centers": model.means_,
            "probabilities": proba,
            "bic": float(model.bic(loadings)),
        }

    raise ValueError(f"Unknown method {method!r}; choose 'kmeans' or 'gmm'.")


def cluster_sweep(
    loadings: np.ndarray,
    method: ClusterMethod = "kmeans",
    k_range=range(2, 11),
    random_state: int | None = 0,
) -> dict:
    """Silhouette (kmeans) or BIC (gmm) for each k.

    Returns dict with ``k_values``, ``scores``, ``best_k``.
    For kmeans, higher silhouette is better. For gmm, lower BIC is better.
    """
    loadings = np.asarray(loadings, dtype=float)
    ks = list(k_range)
    scores = np.empty(len(ks))

    for i, k in enumerate(ks):
        if method == "kmeans":
            model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            labels = model.fit_predict(loadings)
            scores[i] = silhouette_score(loadings, labels) if len(set(labels)) > 1 else -1.0
        elif method == "gmm":
            model = GaussianMixture(n_components=k, random_state=random_state)
            model.fit(loadings)
            scores[i] = model.bic(loadings)
        else:
            raise ValueError(f"Unknown method {method!r}; choose 'kmeans' or 'gmm'.")

    if method == "kmeans":
        best_k = int(ks[int(np.argmax(scores))])
    else:
        best_k = int(ks[int(np.argmin(scores))])

    return {"k_values": np.array(ks), "scores": scores, "best_k": best_k}
