import numpy as np
import pytest

from subwave.spindles import (
    CANONICAL_FAST,
    CANONICAL_SLOW,
    align_by_envelope_peak,
    sigma_filter,
)
from subwave.spindles.templates import _make_canonical


SFREQ = 256


def _peak_freq(x, sfreq):
    n = len(x)
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    return freqs[np.argmax(spec)]


def _band_power(x, sfreq, lo, hi):
    n = len(x)
    spec = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
    mask = (freqs >= lo) & (freqs <= hi)
    return spec[mask].sum()


def test_sigma_filter_shape_2d():
    X = np.random.randn(10, SFREQ)
    out = sigma_filter(X, SFREQ)
    assert out.shape == X.shape


def test_sigma_filter_shape_1d():
    x = np.random.randn(SFREQ)
    out = sigma_filter(x, SFREQ)
    assert out.shape == x.shape


def test_sigma_filter_attenuates_outside_band():
    rng = np.random.default_rng(0)
    t = np.arange(SFREQ * 4) / SFREQ
    # Mix of 2 Hz (out of band) and 12 Hz (in band)
    x = np.sin(2 * np.pi * 2 * t) + np.sin(2 * np.pi * 12 * t)
    y = sigma_filter(x, SFREQ)
    # Out-of-band power should be drastically reduced
    out_before = _band_power(x, SFREQ, 0.5, 5.0)
    out_after = _band_power(y, SFREQ, 0.5, 5.0)
    in_before = _band_power(x, SFREQ, 9.0, 16.0)
    in_after = _band_power(y, SFREQ, 9.0, 16.0)
    assert out_after < 0.05 * out_before
    assert in_after > 0.5 * in_before


def test_align_by_envelope_peak_shape():
    X = np.random.randn(8, SFREQ)
    out = align_by_envelope_peak(X, SFREQ)
    assert out.shape == X.shape


def test_align_by_envelope_peak_centers_envelope():
    n = SFREQ
    t = np.linspace(-0.5, 0.5, n)
    template = np.exp(-0.5 * (t / (1.0 / 6)) ** 2) * np.sin(2 * np.pi * 13.0 * t)
    rng = np.random.default_rng(1)
    n_events = 20
    X = np.empty((n_events, n))
    for i in range(n_events):
        shift = rng.integers(-30, 31)
        X[i] = np.roll(template, shift)

    aligned = align_by_envelope_peak(X, SFREQ)
    # Recompute envelope peak after alignment
    from scipy.signal import hilbert
    from subwave.spindles.filters import sigma_filter as sf

    filtered = sf(aligned, SFREQ)
    env = np.abs(hilbert(filtered, axis=-1))
    peaks = np.argmax(env, axis=-1)
    center = n // 2
    assert np.all(np.abs(peaks - center) <= 5)


def test_canonical_fast_peak_freq():
    f = _peak_freq(CANONICAL_FAST, 256)
    assert abs(f - 13.5) < 1.0


def test_canonical_slow_peak_freq():
    f = _peak_freq(CANONICAL_SLOW, 256)
    assert abs(f - 11.0) < 1.0


@pytest.mark.parametrize("sfreq,duration", [(128, 1.0), (256, 0.5), (500, 2.0)])
def test_make_canonical_length(sfreq, duration):
    x = _make_canonical(12.0, sfreq=sfreq, duration=duration)
    assert len(x) == int(sfreq * duration)
