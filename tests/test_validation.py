import numpy as np
import pytest

from subwave.validation import (
    bootstrap_stability,
    cluster_recovery_ari,
    recovery_score,
    split_half,
    synthetic_population,
)


def _two_templates(n_samples=128):
    t = np.linspace(0, 1, n_samples)
    a = np.sin(2 * np.pi * 5 * t) * np.exp(-((t - 0.5) ** 2) * 20)
    b = np.cos(2 * np.pi * 8 * t) * np.exp(-((t - 0.5) ** 2) * 20)
    return np.stack([a, b])


def test_synthetic_population_shapes():
    templates = _two_templates()
    out = synthetic_population(templates, n_events=50, random_state=0)
    assert out["X"].shape == (50, templates.shape[1])
    assert out["loadings"].shape == (50, templates.shape[0])
    assert out["labels"].shape == (50,)
    assert out["templates"].shape == templates.shape


def test_synthetic_population_reproducible():
    templates = _two_templates()
    a = synthetic_population(templates, n_events=20, random_state=42)
    b = synthetic_population(templates, n_events=20, random_state=42)
    assert np.allclose(a["X"], b["X"])


def test_recovery_score_identical():
    templates = _two_templates()
    res = recovery_score(templates, templates)
    assert res["mean_score"] > 0.999
    assert res["matched_scores"].shape == (templates.shape[0],)
    assert res["similarity_matrix"].shape == (2, 2)


def test_recovery_score_random_lower():
    templates = _two_templates()
    rng = np.random.default_rng(0)
    random_templates = rng.standard_normal(templates.shape)
    res = recovery_score(templates, random_templates)
    assert res["mean_score"] < 0.7


def test_recovery_score_handles_permutation():
    templates = _two_templates()
    permuted = templates[::-1]
    res = recovery_score(templates, permuted)
    assert res["mean_score"] > 0.999


def test_cluster_recovery_ari_perfect():
    labels = np.array([0, 0, 1, 1, 2, 2])
    assert cluster_recovery_ari(labels, labels) == pytest.approx(1.0)


def test_cluster_recovery_ari_relabeling():
    true = np.array([0, 0, 1, 1, 2, 2])
    permuted = np.array([2, 2, 0, 0, 1, 1])
    assert cluster_recovery_ari(true, permuted) == pytest.approx(1.0)


def test_cluster_recovery_ari_random_near_zero():
    rng = np.random.default_rng(0)
    n = 500
    a = rng.integers(0, 4, size=n)
    b = rng.integers(0, 4, size=n)
    assert abs(cluster_recovery_ari(a, b)) < 0.1


def test_bootstrap_stability_clean_data():
    templates = _two_templates()
    out = synthetic_population(
        templates, n_events=120, noise_std=0.05, random_state=0
    )
    res = bootstrap_stability(
        out["X"], n_components=2, method="svd", n_boot=10, random_state=0
    )
    assert res["stability_scores"].shape == (2,)
    assert res["all_scores"].shape == (10, 2)
    assert np.all(res["stability_scores"] >= 0.0)
    assert np.all(res["stability_scores"] <= 1.0 + 1e-9)
    assert np.all(res["stability_scores"] > 0.8)


def test_split_half_clean_data():
    templates = _two_templates()
    out = synthetic_population(
        templates, n_events=200, noise_std=0.05, random_state=0
    )
    res = split_half(out["X"], n_components=2, method="svd", random_state=0)
    assert res["template_similarity"].shape == (2,)
    assert res["loading_correlation"].shape == (2,)
    assert np.all(res["template_similarity"] > 0.8)
