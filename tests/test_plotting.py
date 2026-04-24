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


class TestPlotSortedGrid:
    def test_returns_axes_array(self, result):
        axes = result.plot_sorted_grid(comp=0, n=6)
        assert axes is not None


class TestPlotResidualHist:
    def test_returns_axes(self, result):
        ax = result.plot_residual_hist()
        assert ax is not None
