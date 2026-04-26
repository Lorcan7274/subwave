import numpy as np


def _make_canonical(freq, sfreq=256, duration=1.0):
    """Generate a canonical spindle template: Gaussian-windowed sinusoid.

    freq: center frequency in Hz
    sfreq: sampling frequency
    duration: total duration in seconds
    Returns: 1D array of length int(sfreq * duration)
    """
    n = int(sfreq * duration)
    t = np.linspace(-duration / 2, duration / 2, n)
    envelope = np.exp(-0.5 * (t / (duration / 6)) ** 2)
    return envelope * np.sin(2 * np.pi * freq * t)


CANONICAL_SLOW = _make_canonical(freq=11.0, sfreq=256, duration=1.0)
CANONICAL_FAST = _make_canonical(freq=13.5, sfreq=256, duration=1.0)
