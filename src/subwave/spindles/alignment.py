import numpy as np
from scipy.signal import hilbert

from .filters import sigma_filter


def align_by_envelope_peak(X, sfreq):
    """Align spindle waveforms by their sigma-band analytic envelope peak.

    For each event:
    1. Bandpass to 9-16 Hz (butterworth order 4, sosfiltfilt)
    2. Compute analytic signal via scipy.signal.hilbert
    3. Take abs (envelope)
    4. Find envelope peak sample
    5. Circularly shift so envelope peak is at center sample

    X: (n_events, n_samples)
    sfreq: float, sampling frequency in Hz
    Returns: (n_events, n_samples) aligned array, same shape as input.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    n_events, n_samples = X.shape
    center = n_samples // 2
    filtered = sigma_filter(X, sfreq)
    envelope = np.abs(hilbert(filtered, axis=-1))

    aligned = np.empty_like(X)
    for i in range(n_events):
        peak = int(np.argmax(envelope[i]))
        shift = center - peak
        aligned[i] = np.roll(X[i], shift)

    if squeeze:
        return aligned[0]
    return aligned
