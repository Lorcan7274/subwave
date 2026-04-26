import numpy as np
from scipy.optimize import linear_sum_assignment

from ..core import decompose
from .metrics import _cosine_similarity_matrix


def _decompose_templates_loadings(X, n_components, method, center):
    result = decompose(X, method=method, n_components=n_components, center=center)
    templates = np.asarray(result.templates)
    loadings = np.asarray(result.loadings)
    return result, templates, loadings


def _match_to_reference(reference, candidate):
    """Match rows of candidate to rows of reference via Hungarian on |cos sim|.

    Returns (matched_candidate, scores) where matched_candidate has the same
    row order as reference and scores is the matched similarity per reference row.
    """
    sim = _cosine_similarity_matrix(reference, candidate)
    cost = -np.abs(sim)
    row_ind, col_ind = linear_sum_assignment(cost)
    n_ref = reference.shape[0]
    order = np.empty(n_ref, dtype=int)
    order[row_ind] = col_ind
    scores = np.abs(sim[np.arange(n_ref), order])
    return candidate[order], scores, order


def bootstrap_stability(
    X,
    n_components=3,
    method="svd",
    n_boot=100,
    center=True,
    random_state=None,
):
    """Assess component stability via bootstrap resampling.

    For each bootstrap iteration:
    1. Resample events with replacement
    2. Decompose
    3. Match templates to the full-data templates via cosine similarity

    Returns dict:
        reference_templates: (n_components, n_samples) templates from full data
        stability_scores: (n_components,) mean cosine similarity across bootstraps
        all_scores: (n_boot, n_components) per-bootstrap similarities
    """
    X = np.asarray(X)
    rng = np.random.default_rng(random_state)
    n_events = X.shape[0]

    _, ref_templates, _ = _decompose_templates_loadings(
        X, n_components, method, center
    )

    all_scores = np.zeros((n_boot, n_components))
    for b in range(n_boot):
        idx = rng.integers(0, n_events, size=n_events)
        Xb = X[idx]
        try:
            _, boot_templates, _ = _decompose_templates_loadings(
                Xb, n_components, method, center
            )
        except Exception:
            all_scores[b] = np.nan
            continue
        _, scores, _ = _match_to_reference(ref_templates, boot_templates)
        all_scores[b] = scores

    stability_scores = np.nanmean(all_scores, axis=0)

    return {
        "reference_templates": ref_templates,
        "stability_scores": stability_scores,
        "all_scores": all_scores,
    }


def split_half(
    X,
    n_components=3,
    method="svd",
    center=True,
    random_state=None,
):
    """Split-half reproducibility check.

    1. Split X into two halves (odd/even rows)
    2. Decompose each half
    3. Match templates and compute cosine similarity
    4. Project half-B events onto half-A templates, compare loadings

    Returns dict:
        template_similarity: (n_components,) cosine similarity of matched templates
        loading_correlation: (n_components,) Pearson r between projected and native loadings
    """
    X = np.asarray(X)
    XA = X[0::2]
    XB = X[1::2]

    resA, tA, _ = _decompose_templates_loadings(XA, n_components, method, center)
    resB, tB, lB = _decompose_templates_loadings(XB, n_components, method, center)

    matched_tB, template_similarity, order = _match_to_reference(tA, tB)
    # Reorder native B loadings to match A's component ordering
    lB_reordered = lB[:, order]

    # Project half-B events onto half-A templates
    projected = np.asarray(resA.project(XB))  # (n_eventsB, n_components)

    n_components_actual = template_similarity.shape[0]
    loading_corr = np.zeros(n_components_actual)
    for k in range(n_components_actual):
        a = projected[:, k]
        b = lB_reordered[:, k]
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            loading_corr[k] = 0.0
        else:
            r = np.corrcoef(a, b)[0, 1]
            # Use absolute correlation to handle sign ambiguity
            loading_corr[k] = abs(r)

    return {
        "template_similarity": template_similarity,
        "loading_correlation": loading_corr,
    }
