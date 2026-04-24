from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .core import EventMatrix
from .tensor import AxisAnnotatedTensor


def from_array(
    X: np.ndarray,
    sfreq: float = 1.0,
) -> EventMatrix:
    """Create an :class:`EventMatrix` from a raw numpy array.

    Parameters
    ----------
    X:
        Shape ``(n_events, n_samples)``.
    sfreq:
        Sampling frequency in Hz.
    """
    return EventMatrix(np.asarray(X, dtype=float), sfreq=sfreq)


def from_npz(path: str | Path) -> AxisAnnotatedTensor:
    """Load a waveform collection from an ``.npz`` bundle.

    Expected bundle layout (as produced by Lunascope)::

        waveforms   – (n_instances, n_samples) float64
        labels      – 0-d object array containing a dict with keys:
                        layout, axes, shape, channel_names,
                        x_label, value_label, time_axis_seconds

    Parameters
    ----------
    path:
        Path to the ``.npz`` file.

    Returns
    -------
    :class:`AxisAnnotatedTensor` with axes ``['instance', 'sample']``.
    """
    d = np.load(str(path), allow_pickle=True)
    waveforms = d["waveforms"].astype(float)
    labels: dict = d["labels"].item()

    time_axis: np.ndarray = np.asarray(labels["time_axis_seconds"])
    n_instances, n_samples = waveforms.shape

    dt = float(time_axis[1] - time_axis[0]) if len(time_axis) > 1 else 1.0
    sfreq = 1.0 / dt

    instance_ids = np.arange(n_instances)

    return AxisAnnotatedTensor(
        data=waveforms,
        axes=["instance", "sample"],
        axis_index={
            "instance": instance_ids,
            "sample": time_axis,
        },
        axis_meta={
            "instance": pd.DataFrame({"instance_id": instance_ids}),
            "sample": pd.DataFrame(
                {"sample_index": np.arange(n_samples), "time": time_axis}
            ),
        },
        attrs={
            "sfreq": sfreq,
            "channel_names": labels.get("channel_names", []),
            "x_label": labels.get("x_label", ""),
            "value_label": labels.get("value_label", ""),
            "layout": labels.get("layout", ""),
            "source": str(path),
        },
    )


def from_mne(epochs) -> EventMatrix:
    """Create an :class:`EventMatrix` from an MNE ``Epochs`` object.

    Picks the first channel (or averages across channels for multi-channel
    inputs). For single-channel extraction, call
    ``epochs.pick_channels([ch])`` before passing here.

    Parameters
    ----------
    epochs:
        ``mne.Epochs`` instance. Must be loaded (``epochs.load_data()``).
    """
    try:
        import mne  # noqa: F401
    except ImportError as exc:
        raise ImportError("mne is required: pip install subwave[mne]") from exc

    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    if data.ndim == 3:
        data = data[:, 0, :] if data.shape[1] == 1 else data.mean(axis=1)
    return EventMatrix(data, sfreq=epochs.info["sfreq"])


def from_yasa(
    spindles_df,
    raw: np.ndarray,
    sfreq: float,
    window_sec: float = 1.0,
) -> EventMatrix:
    """Create an :class:`EventMatrix` from YASA spindle detection output.

    Parameters
    ----------
    spindles_df:
        DataFrame returned by ``yasa.SpindlesResults.summary()``.
        Must contain a ``'Peak'`` column with peak times in seconds.
    raw:
        1-D raw EEG signal (samples,) from which waveforms are extracted.
    sfreq:
        Sampling frequency of *raw*.
    window_sec:
        Half-window around each event peak in seconds.
    """
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise ImportError("pandas is required: pip install subwave[yasa]") from exc

    half = int(window_sec * sfreq)
    n_samples = 2 * half
    raw = np.asarray(raw, dtype=float)
    waveforms = []

    for peak_sec in spindles_df["Peak"].values:
        center = int(peak_sec * sfreq)
        start, end = center - half, center + half
        if start < 0 or end > len(raw):
            continue
        waveforms.append(raw[start:end])

    if not waveforms:
        raise ValueError("No valid spindle windows could be extracted from the signal.")

    return EventMatrix(np.stack(waveforms), sfreq=sfreq)
