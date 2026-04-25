import numpy as np
import pytest

import subwave as sw
from subwave import load_result


@pytest.fixture
def result(small_matrix):
    em = sw.from_array(small_matrix, sfreq=64.0)
    return em.decompose(method="svd", n_components=3)


class TestSaveLoad:
    def test_round_trip(self, result, tmp_path):
        path = tmp_path / "result.npz"
        result.save(path)
        loaded = load_result(path)
        np.testing.assert_allclose(loaded.templates, result.templates)
        np.testing.assert_allclose(loaded.loadings, result.loadings)
        np.testing.assert_allclose(loaded.singular_values, result.singular_values)
        assert loaded.method == result.method
        assert loaded.config["n_components"] == result.config["n_components"]

    def test_load_static_method(self, result, tmp_path):
        from subwave import DecompositionResult
        path = tmp_path / "r.npz"
        result.save(path)
        loaded = DecompositionResult.load(path)
        assert loaded.method == "svd"


class TestToDataFrame:
    def test_columns(self, result):
        df = result.to_dataframe()
        assert "instance_id" in df.columns
        assert "recon_error" in df.columns
        assert "score_1" in df.columns
        assert len(df) == 20
