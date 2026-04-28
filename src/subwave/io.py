from __future__ import annotations

import os
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


def from_luna(
    spindle_file: str | Path,
    edf_file: str | Path,
    sfreq: float | None = None,
    window_sec: float = 1.0,
) -> EventMatrix:
    """Load events from a Luna per-spindle output and an EDF file.

    The spindle file is a tab/whitespace-delimited text table with columns
    ``START`` and ``STOP`` (seconds), or alternatively ``PEAK_SEC``
    (or ``PEAK``) with a peak time in seconds.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required: pip install subwave[mne]") from exc

    raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"]) if sfreq is None else float(sfreq)
    signal = raw.get_data()[0]

    df = pd.read_csv(str(spindle_file), sep=None, engine="python")
    cols = {c.upper(): c for c in df.columns}

    centers: list[float] = []
    if "START" in cols and "STOP" in cols:
        starts = df[cols["START"]].astype(float).values
        stops = df[cols["STOP"]].astype(float).values
        centers = list(0.5 * (starts + stops))
    elif "PEAK_SEC" in cols:
        centers = list(df[cols["PEAK_SEC"]].astype(float).values)
    elif "PEAK" in cols:
        centers = list(df[cols["PEAK"]].astype(float).values)
    else:
        raise ValueError(
            "Luna spindle file must have START/STOP columns or PEAK_SEC/PEAK"
        )

    half = int(window_sec * sf)
    waveforms = []
    for c in centers:
        center_idx = int(c * sf)
        start, end = center_idx - half, center_idx + half
        if start < 0 or end > len(signal):
            continue
        waveforms.append(signal[start:end])

    if not waveforms:
        raise ValueError("No valid windows could be extracted from the EDF.")

    return EventMatrix(np.stack(waveforms), sfreq=sf)


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


def from_edf_batch(
    edf_paths,
    channel: str,
    detector: str = "yasa",
    sfreq: float | None = None,
    window_sec: float = 1.0,
    detector_kwargs: dict | None = None,
) -> EventMatrix:
    """Process multiple EDF files and pool detected events into one EventMatrix.

    Parameters
    ----------
    edf_paths:
        Iterable of paths to EDF files.
    channel:
        Channel name to pick from each EDF (e.g., ``'C3'``).
    detector:
        ``'yasa'`` (requires yasa installed). Future: ``'luna'``, ``'mne'``.
    sfreq:
        Resample all files to this sfreq before detection. ``None`` uses native.
    window_sec:
        Half-window around each detected event peak.
    detector_kwargs:
        Extra kwargs passed to the detector (e.g., YASA thresholds).

    Returns
    -------
    EventMatrix with ``.data`` of shape ``(total_events, n_samples)`` and a
    pandas DataFrame in ``.meta`` with columns ``subject``, ``file``,
    ``event_index``, ``peak_sec``.
    """
    try:
        import mne
    except ImportError as exc:
        raise ImportError("mne is required: pip install subwave[mne]") from exc

    all_waveforms: list[np.ndarray] = []
    meta_rows: list[dict] = []
    half: int | None = None
    fs_used: float | None = None

    for subj_idx, path in enumerate(edf_paths):
        raw = mne.io.read_raw_edf(str(path), preload=True, verbose=False)
        raw.pick([channel])
        if sfreq and raw.info["sfreq"] != sfreq:
            raw.resample(sfreq)
        fs = float(sfreq) if sfreq else float(raw.info["sfreq"])
        data = raw.get_data()[0]

        if half is None:
            half = int(window_sec * fs)
            fs_used = fs

        if detector == "yasa":
            try:
                import yasa
            except ImportError as exc:
                raise ImportError(
                    "yasa is required for detector='yasa': pip install subwave[yasa]"
                ) from exc
            kw = detector_kwargs or {}
            sp = yasa.spindles_detect(data * 1e6, fs, ch_names=[channel], **kw)
            if sp is None:
                continue
            peaks = sp.summary()["Peak"].values
        else:
            raise ValueError(f"Unknown detector: {detector}")

        for ev_i, peak_sec in enumerate(peaks):
            center = int(peak_sec * fs)
            start, end = center - half, center + half
            if start >= 0 and end <= len(data):
                all_waveforms.append(data[start:end] * 1e6)
                meta_rows.append({
                    "subject": subj_idx,
                    "file": os.path.basename(str(path)),
                    "event_index": ev_i,
                    "peak_sec": float(peak_sec),
                })

    if not all_waveforms:
        raise ValueError("No events detected across all files.")

    em = EventMatrix(np.stack(all_waveforms), sfreq=fs_used)
    em.meta = pd.DataFrame(meta_rows)
    return em
