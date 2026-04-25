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
