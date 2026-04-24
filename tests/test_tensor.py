import numpy as np
import pandas as pd
import pytest

from subwave import AxisAnnotatedTensor


class TestAxisAnnotatedTensor:
    def test_basic_construction(self, small_matrix):
        t = np.linspace(-0.5, 0.5, 64)
        aat = AxisAnnotatedTensor(
            data=small_matrix,
            axes=["instance", "sample"],
            axis_index={"sample": t},
        )
        assert aat.shape == (20, 64)
        assert aat.ndim == 2

    def test_fills_missing_index(self, small_matrix):
        aat = AxisAnnotatedTensor(data=small_matrix, axes=["instance", "sample"])
        np.testing.assert_array_equal(aat.axis_index["instance"], np.arange(20))
        np.testing.assert_array_equal(aat.axis_index["sample"], np.arange(64))

    def test_fills_missing_meta(self, small_matrix):
        aat = AxisAnnotatedTensor(data=small_matrix, axes=["instance", "sample"])
        assert "instance_index" in aat.axis_meta["instance"].columns

    def test_rejects_wrong_axes_count(self, small_matrix):
        with pytest.raises(ValueError, match="does not match"):
            AxisAnnotatedTensor(data=small_matrix, axes=["only_one"])

    def test_data_cast_to_float(self):
        X = np.ones((5, 10), dtype=int)
        aat = AxisAnnotatedTensor(data=X, axes=["a", "b"])
        assert aat.data.dtype == int  # no forced cast — just np.asarray

    def test_axis_pos(self, simple_aat):
        assert simple_aat.axis_pos("instance") == 0
        assert simple_aat.axis_pos("sample") == 1

    def test_axis_pos_unknown(self, simple_aat):
        with pytest.raises(ValueError, match="not found"):
            simple_aat.axis_pos("channel")

    def test_repr(self, simple_aat):
        r = repr(simple_aat)
        assert "AxisAnnotatedTensor" in r
        assert "instance(20)" in r
        assert "sample(64)" in r

    def test_attrs_stored(self, simple_aat):
        assert simple_aat.attrs["sfreq"] == 64.0

    def test_custom_meta_preserved(self, simple_aat):
        assert "label" in simple_aat.axis_meta["instance"].columns
