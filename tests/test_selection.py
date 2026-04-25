import numpy as np
import pytest

import subwave as sw
from subwave.selection import (
    elbow,
    kaiser,
    parallel_analysis,
    select_n_components,
)


@pytest.fixture
def result(small_matrix):
    em = sw.from_array(small_matrix, sfreq=64.0)
    return em.decompose(method="svd", n_components=5)


class TestElbow:
    def test_array_with_clear_knee(self):
        sv = np.array([10.0, 9.0, 1.0, 0.5, 0.4, 0.3])
        assert elbow(sv) == 3

    def test_short_curve_returns_length(self):
        assert elbow(np.array([5.0, 1.0])) == 2

    def test_flat_endpoints(self):
        sv = np.array([1.0, 1.0, 1.0])
        k = elbow(sv)
        assert 1 <= k <= 3

    def test_from_result(self, result):
        k = elbow(result)
        assert 1 <= k <= len(result.singular_values)


class TestKaiser:
    def test_with_raw_singular_values(self):
        sv = np.array([10.0, 5.0, 0.5])
        k = kaiser(sv, n_samples=20, threshold=1.0)
        assert k == 2

    def test_requires_n_samples_for_array(self):
        with pytest.raises(ValueError, match="n_samples"):
            kaiser(np.array([1.0, 2.0]))

    def test_from_result(self, result):
        k = kaiser(result)
        assert k >= 0


class TestParallelAnalysis:
    def test_returns_int(self, small_matrix):
        k = parallel_analysis(small_matrix, n_iter=10, random_state=0)
        assert isinstance(k, int)
        assert 0 <= k <= min(small_matrix.shape)

    def test_rejects_non_2d(self):
        with pytest.raises(ValueError, match="2-D"):
            parallel_analysis(np.zeros((2, 2, 2)))

    def test_no_center(self, small_matrix):
        k = parallel_analysis(small_matrix, n_iter=5, center=False, random_state=0)
        assert isinstance(k, int)


class TestSelectNComponents:
    def test_dispatch_elbow(self, result):
        assert select_n_components(result, method="elbow") >= 1

    def test_dispatch_kaiser(self, result):
        assert select_n_components(result, method="kaiser") >= 0

    def test_dispatch_parallel(self, small_matrix):
        k = select_n_components(small_matrix, method="parallel", n_iter=5, random_state=0)
        assert isinstance(k, int)

    def test_parallel_rejects_result(self, result):
        with pytest.raises(ValueError, match="raw 2-D"):
            select_n_components(result, method="parallel")

    def test_unknown_method(self, result):
        with pytest.raises(ValueError, match="Unknown method"):
            select_n_components(result, method="bogus")
