import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score


def _cosine_similarity_matrix(A, B):
    A = np.asarray(A)
    B = np.asarray(B)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A_norm @ B_norm.T


def recovery_score(true_templates, recovered_templates):
    """Cosine similarity between true and recovered templates.

    Handles permutation ambiguity: finds the optimal 1-to-1 matching
    via the Hungarian algorithm on the cosine similarity matrix.

    Returns dict:
        similarity_matrix: (n_true, n_recovered) all pairwise cosine similarities
        matching: list of (true_idx, recovered_idx) pairs
        matched_scores: 1D array of cosine similarities for matched pairs
        mean_score: float, mean of matched cosine similarities
    """
    true_templates = np.asarray(true_templates)
    recovered_templates = np.asarray(recovered_templates)
    sim = _cosine_similarity_matrix(true_templates, recovered_templates)
    # Use absolute similarity to handle sign ambiguity inherent in SVD-like methods
    cost = -np.abs(sim)
    row_ind, col_ind = linear_sum_assignment(cost)
    matching = list(zip(row_ind.tolist(), col_ind.tolist()))
    matched_scores = np.abs(sim[row_ind, col_ind])
    return {
        "similarity_matrix": sim,
        "matching": matching,
        "matched_scores": matched_scores,
        "mean_score": float(matched_scores.mean()),
    }


def cluster_recovery_ari(true_labels, recovered_labels):
    """Adjusted Rand Index between true and recovered cluster assignments.

    Wrapper around sklearn.metrics.adjusted_rand_score.
    Returns float in [-1, 1], where 1 = perfect agreement.
    """
    return float(adjusted_rand_score(true_labels, recovered_labels))
