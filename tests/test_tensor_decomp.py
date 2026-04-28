import numpy as np
import pytest


def _tensorly_or_skip():
    return pytest.importorskip("tensorly")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


class TestCPBasic:
    def test_shapes_and_recon_error(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((50, 64, 3))
        result = tensor_decompose(X, method="cp", rank=2, random_state=0)

        assert result.event_factors.shape == (50, 2)
        assert result.temporal_factors.shape == (64, 2)
        assert result.spatial_factors.shape == (3, 2)
        assert result.core is None
        assert result.reconstruction_error < 1.0


class TestTuckerBasic:
    def test_shapes_and_core(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((50, 64, 3))
        result = tensor_decompose(X, method="tucker", rank=2, random_state=0)

        assert result.event_factors.shape == (50, 2)
        assert result.temporal_factors.shape == (64, 2)
        assert result.spatial_factors.shape == (3, 2)
        assert result.core is not None
        assert result.core.shape == (2, 2, 2)


class TestCPSynthetic:
    def test_recovers_low_rank_signal(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        n_events, n_samples, n_channels = 40, 64, 3
        # Two rank-1 components
        e1 = rng.standard_normal(n_events)
        t1 = np.sin(np.linspace(0, 4 * np.pi, n_samples))
        c1 = np.array([1.0, 0.5, 0.2])

        e2 = rng.standard_normal(n_events)
        t2 = np.cos(np.linspace(0, 6 * np.pi, n_samples))
        c2 = np.array([0.2, 1.0, 0.5])

        X = (
            np.einsum("e,s,c->esc", e1, t1, c1)
            + np.einsum("e,s,c->esc", e2, t2, c2)
        )
        X = X + rng.normal(0, 0.01, X.shape)

        result = tensor_decompose(X, method="cp", rank=2, random_state=0)
        assert result.reconstruction_error < 0.1


class TestComponentWaveform:
    def test_returns_temporal_factor(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((20, 32, 4))
        result = tensor_decompose(X, method="cp", rank=2, random_state=0)
        wf = result.component_waveform(0)
        assert wf.shape == (32,)

    def test_scaled_by_channel(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((20, 32, 4))
        result = tensor_decompose(X, method="cp", rank=2, random_state=0)
        wf_raw = result.component_waveform(0)
        wf_ch = result.component_waveform(0, channel=1)
        np.testing.assert_allclose(wf_ch, wf_raw * result.spatial_factors[1, 0])


class TestInvalidInput:
    def test_2d_raises(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((20, 32))
        with pytest.raises(ValueError, match="3D tensor"):
            tensor_decompose(X)

    def test_unknown_method(self, rng):
        _tensorly_or_skip()
        from subwave import tensor_decompose

        X = rng.standard_normal((10, 16, 2))
        with pytest.raises(ValueError, match="Unknown method"):
            tensor_decompose(X, method="bogus")
