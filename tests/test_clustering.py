import numpy as np
import pytest

import subwave as sw
from subwave.clustering import cluster_loadings, cluster_sweep


@pytest.fixture
def result(small_matrix):
    em = sw.from_array(small_matrix, sfreq=64.0)
    return em.decompose(method="svd", n_components=3)


class TestClusterLoadings:
    def test_kmeans(self, result):
        out = cluster_loadings(result.loadings, method="kmeans", n_clusters=2)
        assert out["labels"].shape == (20,)
        assert out["centers"].shape == (2, 3)

    def test_gmm(self, result):
        out = cluster_loadings(result.loadings, method="gmm", n_clusters=2)
        assert out["labels"].shape == (20,)
        assert out["centers"].shape == (2, 3)
        assert out["probabilities"].shape == (20, 2)
        assert isinstance(out["bic"], float)

    def test_unknown_method(self, result):
        with pytest.raises(ValueError, match="Unknown method"):
            cluster_loadings(result.loadings, method="dbscan")

    def test_rejects_1d(self):
        with pytest.raises(ValueError, match="2-D"):
            cluster_loadings(np.zeros(10))


class TestClusterSweep:
    def test_kmeans_sweep(self, result):
        out = cluster_sweep(result.loadings, method="kmeans", k_range=range(2, 5))
        assert out["k_values"].shape == (3,)
        assert out["scores"].shape == (3,)
        assert out["best_k"] in {2, 3, 4}

    def test_gmm_sweep(self, result):
        out = cluster_sweep(result.loadings, method="gmm", k_range=range(2, 5))
        assert out["best_k"] in {2, 3, 4}

    def test_unknown_method(self, result):
        with pytest.raises(ValueError, match="Unknown method"):
            cluster_sweep(result.loadings, method="dbscan", k_range=range(2, 4))


class TestResultClusterMethods:
    def test_cluster(self, result):
        out = result.cluster(method="kmeans", n_clusters=3)
        assert out["labels"].shape == (20,)

    def test_cluster_templates_centered(self, result):
        templates = result.cluster_templates(n_clusters=2)
        assert templates.shape == (2, 64)

    def test_cluster_templates_uncentered(self, small_matrix):
        em = sw.from_array(small_matrix, sfreq=64.0)
        result = em.decompose(method="svd", n_components=3, center=False)
        templates = result.cluster_templates(n_clusters=2)
        assert templates.shape == (2, 64)
