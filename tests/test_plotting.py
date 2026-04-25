import matplotlib
matplotlib.use("Agg")  # non-interactive backend for tests

import matplotlib.pyplot as plt
import pytest

from subwave import from_array


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def result(small_matrix):
    em = from_array(small_matrix, sfreq=64.0)
    return em.decompose(method="svd", n_components=5)


class TestPlotSpectrum:
    def test_returns_axes(self, result):
        ax = result.plot_spectrum()
        assert ax is not None

    def test_custom_ax(self, result):
        fig, ax = plt.subplots()
        returned = result.plot_spectrum(ax=ax)
        assert returned is ax


class TestPlotTemplates:
    def test_returns_axes_list(self, result):
        axes = result.plot_templates(n=3)
        assert len(axes) == 3

    def test_n_clipped_to_components(self, result):
        axes = result.plot_templates(n=100)
        assert len(axes) == result.templates.shape[0]


class TestPlotMeanPm:
    def test_returns_axes(self, result):
        ax = result.plot_mean_pm(comp=0)
        assert ax is not None


class TestPlotScatter:
    def test_returns_axes(self, result):
        ax = result.plot_scatter(x=0, y=1)
        assert ax is not None

    def test_int_color(self, result):
        import numpy as np
        labels = np.array([0] * 10 + [1] * 10)
        ax = result.plot_scatter(x=0, y=1, color=labels)
        assert ax is not None

    def test_float_color(self, result):
        import numpy as np
        values = np.linspace(0, 1, 20)
        ax = result.plot_scatter(x=0, y=1, color=values, label="metric")
        assert ax is not None


class TestPlotSortedGrid:
    def test_returns_axes_array(self, result):
        axes = result.plot_sorted_grid(comp=0, n=6)
        assert axes is not None


class TestPlotResidualHist:
    def test_returns_axes(self, result):
        ax = result.plot_residual_hist()
        assert ax is not None


class TestPlotLoadingsByGroup:
    def test_box(self, result):
        import numpy as np
        groups = np.array(["a"] * 10 + ["b"] * 10)
        ax = result.plot_loadings_by_group(groups, comp=0, kind="box")
        assert ax is not None

    def test_violin(self, result):
        import numpy as np
        groups = np.array(["a"] * 10 + ["b"] * 10)
        ax = result.plot_loadings_by_group(groups, comp=1, kind="violin")
        assert ax is not None

    def test_unknown_kind_raises(self, result):
        import numpy as np
        groups = np.array(["a"] * 10 + ["b"] * 10)
        with pytest.raises(ValueError, match="kind must be"):
            result.plot_loadings_by_group(groups, kind="bogus")

    def test_mismatched_length_raises(self, result):
        import numpy as np
        with pytest.raises(ValueError, match="groups length"):
            result.plot_loadings_by_group(np.array([0, 1, 2]))


class TestPlotTemplateSpectra:
    def test_returns_axes(self, result):
        ax = result.plot_template_spectra()
        assert ax is not None

    def test_n_and_no_log(self, result):
        ax = result.plot_template_spectra(n=2, log=False)
        assert ax is not None

    def test_with_explicit_sfreq(self, result):
        ax = result.plot_template_spectra(sfreq=128.0)
        assert ax is not None


class TestPlotTemplatesSingle:
    def test_n_equals_one(self, result):
        axes = result.plot_templates(n=1)
        assert len(axes) == 1


class TestPlotHeatmap:
    def test_returns_axes(self, result):
        ax = result.plot_heatmap(comp=0)
        assert ax is not None

    def test_uncentered(self, small_matrix):
        from subwave import from_array
        em = from_array(small_matrix, sfreq=64.0)
        result = em.decompose(method="svd", n_components=3, center=False)
        ax = result.plot_heatmap(comp=1)
        assert ax is not None


class TestPlotWaterfall:
    def test_returns_axes_subset(self, result):
        ax = result.plot_waterfall(n=5)
        assert ax is not None

    def test_returns_axes_all(self, result):
        ax = result.plot_waterfall(n=200)
        assert ax is not None

    def test_uncentered(self, small_matrix):
        from subwave import from_array
        em = from_array(small_matrix, sfreq=64.0)
        result = em.decompose(method="svd", n_components=3, center=False)
        ax = result.plot_waterfall(n=10)
        assert ax is not None


class TestPlotCumulativeVariance:
    def test_returns_axes(self, result):
        ax = result.plot_cumulative_variance()
        assert ax is not None


class TestPlotLoadingsOverTime:
    def test_default_all_components(self, result):
        import numpy as np
        times = np.linspace(0, 100, 20)
        ax = result.plot_loadings_over_time(times)
        assert ax is not None

    def test_subset_of_comps(self, result):
        import numpy as np
        times = np.linspace(0, 100, 20)
        ax = result.plot_loadings_over_time(times, comps=[0, 1], window=5)
        assert ax is not None

    def test_window_larger_than_series(self, result):
        import numpy as np
        times = np.linspace(0, 100, 20)
        ax = result.plot_loadings_over_time(times, window=100)
        assert ax is not None

    def test_mismatched_length(self, result):
        import numpy as np
        with pytest.raises(ValueError, match="event_times length"):
            result.plot_loadings_over_time(np.array([0.0, 1.0]))


class TestPlotReconstruction:
    def test_full_rank(self, result):
        ax = result.plot_reconstruction(event_idx=3)
        assert ax is not None

    def test_rank_k(self, result):
        ax = result.plot_reconstruction(event_idx=0, n_components=2)
        assert ax is not None

    def test_uncentered(self, small_matrix):
        from subwave import from_array
        em = from_array(small_matrix, sfreq=64.0)
        result = em.decompose(method="svd", n_components=3, center=False)
        ax = result.plot_reconstruction(event_idx=0, n_components=2)
        assert ax is not None
