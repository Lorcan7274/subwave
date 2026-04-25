import numpy as np
import pytest

import subwave as sw
from subwave.comparison import (
    PermutationResult,
    permutation_test,
    subspace_angles,
)


@pytest.fixture
def two_results(small_matrix):
    em = sw.from_array(small_matrix, sfreq=64.0)
    a = em.decompose(method="svd", n_components=3)
    b = em.decompose(method="svd", n_components=3)
    return a, b


class TestSubspaceAngles:
    def test_identical_results_zero_angles(self, two_results):
        a, b = two_results
        angles = subspace_angles(a, b)
        assert np.allclose(angles, 0.0, atol=1e-6)

    def test_with_raw_arrays(self, two_results):
        a, _ = two_results
        angles = subspace_angles(a.templates, a.templates)
        assert angles.shape == (a.templates.shape[0],)
        assert np.allclose(angles, 0.0, atol=1e-6)

    def test_rejects_non_2d_array(self, two_results):
        a, _ = two_results
        with pytest.raises(ValueError, match="2-D"):
            subspace_angles(np.zeros((2, 2, 2)), a)

    def test_returns_min_k_angles(self, small_matrix):
        em = sw.from_array(small_matrix, sfreq=64.0)
        a = em.decompose(method="svd", n_components=2)
        b = em.decompose(method="svd", n_components=4)
        angles = subspace_angles(a, b)
        assert angles.size == 2


class TestPermutationTest:
    def test_basic_run(self, small_matrix):
        groups = np.array([0] * 10 + [1] * 10)
        res = permutation_test(
            small_matrix,
            groups,
            n_components=2,
            method="svd",
            n_perm=5,
            random_state=0,
        )
        assert isinstance(res, PermutationResult)
        assert res.null_distribution.size == 5
        assert 0.0 <= res.p_value <= 1.0
        assert res.statistic_name == "mean_cos"

    def test_mean_angle_statistic(self, small_matrix):
        groups = np.array([0] * 10 + [1] * 10)
        res = permutation_test(
            small_matrix,
            groups,
            n_components=2,
            n_perm=5,
            statistic="mean_angle",
            random_state=0,
        )
        assert res.statistic_name == "mean_angle"

    def test_unknown_statistic(self, small_matrix):
        groups = np.array([0] * 10 + [1] * 10)
        with pytest.raises(ValueError, match="Unknown statistic"):
            permutation_test(
                small_matrix, groups, n_components=2, n_perm=2,
                statistic="bogus", random_state=0,
            )

    def test_mismatched_groups_length(self, small_matrix):
        with pytest.raises(ValueError, match="groups length"):
            permutation_test(small_matrix, np.array([0, 1]), n_components=2, n_perm=2)

    def test_requires_two_groups(self, small_matrix):
        groups = np.array([0] * 5 + [1] * 5 + [2] * 10)
        with pytest.raises(ValueError, match="exactly two groups"):
            permutation_test(small_matrix, groups, n_components=2, n_perm=2)
