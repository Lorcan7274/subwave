import numpy as np
import pandas as pd
import pytest

from subwave import AxisAnnotatedTensor, TensorDataset, TensorView, concat_datasets, make_dataset


class TestMakeDataset:
    def test_returns_dataset(self, small_matrix):
        ds = make_dataset(small_matrix, axes=["instance", "sample"])
        assert isinstance(ds, TensorDataset)
        assert ds.axes == ["instance", "sample"]
        assert ds.shape == (20, 64)

    def test_default_meta_created(self, small_matrix):
        ds = make_dataset(small_matrix, axes=["instance", "sample"])
        assert "instance_id" in ds.axis_meta["instance"].columns

    def test_custom_meta(self, small_matrix):
        meta = pd.DataFrame({"instance_id": np.arange(20), "group": ["a"] * 20})
        ds = make_dataset(
            small_matrix,
            axes=["instance", "sample"],
            axis_meta={"instance": meta},
        )
        assert "group" in ds.axis_meta["instance"].columns

    def test_repr(self, simple_dataset):
        r = repr(simple_dataset)
        assert "TensorDataset" in r


class TestTensorView:
    def test_view_from_dataset(self, simple_dataset):
        view = simple_dataset.view()
        assert isinstance(view, TensorView)

    def test_materialize_full(self, simple_dataset, small_matrix):
        aat = simple_dataset.view().materialize()
        assert isinstance(aat, AxisAnnotatedTensor)
        np.testing.assert_array_equal(aat.data, small_matrix)

    def test_materialize_shape(self, simple_dataset):
        aat = simple_dataset.view().materialize()
        assert aat.shape == (20, 64)

    def test_slice(self, simple_dataset):
        view = simple_dataset.view().slice("instance", 0, 5)
        aat = view.materialize()
        assert aat.shape[0] == 5

    def test_take(self, simple_dataset, small_matrix):
        view = simple_dataset.view().take("instance", [0, 2, 4])
        aat = view.materialize()
        assert aat.shape[0] == 3
        np.testing.assert_array_equal(aat.data, small_matrix[[0, 2, 4]])

    def test_query(self, simple_dataset):
        view = simple_dataset.view().query("instance", "label == 'fast'")
        aat = view.materialize()
        assert aat.shape[0] == 10

    def test_query_chained(self, simple_dataset):
        view = (
            simple_dataset.view()
            .query("instance", "label == 'fast'")
            .slice("instance", 0, 3)
        )
        assert view.materialize().shape[0] == 3

    def test_filter_by_mask(self, simple_dataset):
        mask = np.array([True] * 10 + [False] * 10)
        view = simple_dataset.view().filter("instance", mask)
        assert view.materialize().shape[0] == 10

    def test_filter_by_ids_requires_id_col(self, small_matrix):
        ds = make_dataset(small_matrix, axes=["instance", "sample"])
        with pytest.raises(ValueError, match="No id column"):
            ds.view().filter("instance", [0, 1, 2])

    def test_filter_by_ids(self, simple_dataset):
        view = simple_dataset.view().filter("instance", [0, 1, 2])
        assert view.materialize().shape[0] == 3

    def test_materialize_preserves_meta(self, simple_dataset):
        aat = simple_dataset.view().query("instance", "label == 'slow'").materialize()
        labels = aat.axis_meta["instance"]["label"].tolist()
        assert all(l == "slow" for l in labels)

    def test_repr(self, simple_dataset):
        r = repr(simple_dataset.view())
        assert "TensorView" in r

    def test_materialize_instance_ids_preserved(self, simple_dataset):
        view = simple_dataset.view().slice("instance", 5, 10)
        aat = view.materialize()
        expected_ids = np.arange(5, 10)
        np.testing.assert_array_equal(aat.axis_index["instance"], expected_ids)


class TestConcatDatasets:
    def test_concat_along_instance(self, small_matrix):
        half = small_matrix[:10]
        other = small_matrix[10:]
        ds1 = make_dataset(half, axes=["instance", "sample"])
        ds2 = make_dataset(other, axes=["instance", "sample"])
        combined = concat_datasets([ds1, ds2], along="instance")
        assert combined.shape[0] == 20
        np.testing.assert_array_equal(np.asarray(combined.store), small_matrix)

    def test_concat_empty_raises(self):
        with pytest.raises(ValueError, match="No datasets"):
            concat_datasets([])

    def test_concat_axis_mismatch_raises(self, small_matrix):
        ds1 = make_dataset(small_matrix, axes=["instance", "sample"])
        ds2 = make_dataset(small_matrix, axes=["event", "sample"])
        with pytest.raises(ValueError, match="same axes"):
            concat_datasets([ds1, ds2])


class TestDecomposeFromView:
    def test_decompose_view(self, simple_dataset):
        import subwave as sw
        result = sw.decompose(simple_dataset.view(), method="svd", n_components=3)
        assert result.templates.shape == (3, 64)
        assert result.loadings.shape == (20, 3)

    def test_decompose_query_then_decompose(self, simple_dataset):
        import subwave as sw
        view = simple_dataset.view().query("instance", "label == 'fast'")
        result = sw.decompose(view, method="svd", n_components=2)
        assert result.loadings.shape[0] == 10

    def test_factor_table_has_instance_ids(self, simple_dataset):
        import subwave as sw
        result = sw.decompose(simple_dataset.view(), method="svd", n_components=2)
        assert "instance_id" in result.factor_tables["instance"].columns
        assert "recon_error" in result.factor_tables["instance"].columns
