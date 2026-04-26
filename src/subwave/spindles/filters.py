import numpy as np
from scipy.signal import butter, sosfiltfilt


def sigma_filter(X, sfreq, lo=9.0, hi=16.0, order=4):
    """Bandpass filter event waveforms to the sigma band.

    X: (n_events, n_samples) or 1D array
    sfreq: sampling frequency in Hz
    Returns: filtered array, same shape as input.
    """
    sos = butter(order, [lo, hi], btype="band", fs=sfreq, output="sos")
    X = np.asarray(X)
    if X.ndim == 1:
        return sosfiltfilt(sos, X)
    return np.stack([sosfiltfilt(sos, row) for row in X])
