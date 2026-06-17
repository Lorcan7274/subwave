from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Union

import numpy as np
from scipy.interpolate import interp1d

from .decomposition import run_decomposition
from .result import DecompositionResult
from .tensor import AxisAnnotatedTensor

if TYPE_CHECKING:
    from .dataset import TensorView

NormalizeMode = Literal["none", "max", "l2", "zscore"]
AlignMode = Literal["peak", "trough", "onset", "midpoint", "none"]


# ---------------------------------------------------------------------------
# Top-level decompose()
# ---------------------------------------------------------------------------


def decompose(
    source: Union["AxisAnnotatedTensor", "TensorView", np.ndarray],
    method: Literal["svd", "nmf", "dictlearn", "fourier_svd", "scattering_svd"] = "svd",
    n_components: int | str = 5,
    center: bool = True,
    normalize: NormalizeMode = "none",
    align: AlignMode = "none",
    resample_to_length: int | None = None,
    instance_axis: str = "instance",
    sample_axis: str = "sample",
) -> DecompositionResult:
    """Decompose a waveform population.

    Accepts a :class:`AxisAnnotatedTensor`, a :class:`TensorView`, or a
    plain 2-D numpy array (treated as instance × sample).

    Parameters
    ----------
    source:
        Input data. For ``TensorView`` inputs, :meth:`~TensorView.materialize`
        is called first.
    method:
        ``'svd'`` (default), ``'nmf'``, or ``'dictlearn'``.
    n_components:
        Number of components to extract.
    center:
        Subtract the mean waveform before decomposition.
    normalize:
        Per-instance normalization: ``'none'``, ``'max'``, ``'l2'``, or
        ``'zscore'``.
    align:
        Alignment mode: ``'peak'``, ``'trough'``, ``'onset'``, ``'midpoint'``,
        or ``'none'``.
    resample_to_length:
        Resample all instances to this many samples before decomposing.
    instance_axis:
        Name of the instance axis when *source* is an
        :class:`AxisAnnotatedTensor`.
    sample_axis:
        Name of the sample axis when *source* is an
        :class:`AxisAnnotatedTensor`.
    """
    from .dataset import TensorView as TV  # avoid circular at module level

    # ------------------------------------------------------------------
    # Normalise input to AxisAnnotatedTensor
    # ------------------------------------------------------------------
    if isinstance(source, TV):
        aat = source.materialize()
    elif isinstance(source, AxisAnnotatedTensor):
        aat = source
    elif isinstance(source, EventMatrix):
        return source.decompose(
            method=method,
            n_components=n_components,
            center=center,
            normalize=normalize,
            align=align,
            resample_to_length=resample_to_length,
        )
    else:
        X = np.asarray(source, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"Raw numpy input must be 2-D, got shape {X.shape}")
        import pandas as pd
        aat = AxisAnnotatedTensor(
            data=X,
            axes=[instance_axis, sample_axis],
            axis_index={
                instance_axis: np.arange(X.shape[0]),
                sample_axis: np.arange(X.shape[1]),
            },
            axis_meta={
                instance_axis: pd.DataFrame({"instance_id": np.arange(X.shape[0])}),
                sample_axis: pd.DataFrame({"sample_index": np.arange(X.shape[1])}),
            },
        )

    inst_pos = aat.axis_pos(instance_axis)
    samp_pos = aat.axis_pos(sample_axis)

    # 2-D inputs decompose directly. 3-D inputs are flattened by C-order
    # reshape into a single (n_instances, n_samples * n_channels) feature
    # vector per event; the resulting templates concatenate per-channel
    # waveforms in the same order the channel axis appears in `aat`. This is
    # the simplest cross-channel approach but does not factor channel
    # structure separately — for a true tensor decomposition use a dedicated
    # tool (e.g. PARAFAC). The instance axis must be axis 0.
    if aat.ndim == 2:
        X = aat.data if inst_pos == 0 else aat.data.T
    elif aat.ndim == 3:
        if inst_pos != 0:
            raise ValueError("instance axis must be axis 0 for 3-D tensors")
        X = aat.data.reshape(aat.data.shape[0], -1)
    else:
        raise ValueError(f"Unsupported tensor rank {aat.ndim}; expected 2 or 3.")

    X = X.astype(float)

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    if resample_to_length is not None and resample_to_length != X.shape[1]:
        old_t = np.linspace(0, 1, X.shape[1])
        new_t = np.linspace(0, 1, resample_to_length)
        X = np.stack([interp1d(old_t, row)(new_t) for row in X])

    if align != "none":
        X = _align(X, align)

    if normalize != "none":
        X = _normalize(X, normalize)

    # ------------------------------------------------------------------
    # Get stable ids and sample coordinates
    # ------------------------------------------------------------------
    instance_ids = aat.axis_index.get(instance_axis, np.arange(X.shape[0]))
    sample_axis_index = aat.axis_index.get(sample_axis)

    sfreq = aat.attrs.get("sfreq", 1.0)
    config = {
        "method": method,
        "n_components": n_components,
        "center": center,
        "normalize": normalize,
        "align": align,
        "sfreq": sfreq,
    }

    if X.shape[0] < 2:
        raise ValueError("Need at least 2 instances to decompose.")

    if isinstance(n_components, str):
        if n_components != "auto":
            raise ValueError(
                f"n_components must be an int or 'auto', got {n_components!r}"
            )
        if method != "svd":
            raise ValueError("n_components='auto' is only supported with method='svd'")
        from .selection import parallel_analysis
        Xc_for_pa = X - X.mean(axis=0, keepdims=True) if center else X
        k_auto = parallel_analysis(Xc_for_pa, n_iter=50, center=False)
        n_components = max(int(k_auto), 1)
        config["n_components"] = n_components

    k = min(int(n_components), X.shape[0], X.shape[1])
    return run_decomposition(
        X,
        method=method,
        n_components=k,
        center=center,
        config=config,
        instance_ids=instance_ids,
        sample_axis_index=sample_axis_index,
    )


# ---------------------------------------------------------------------------
# Legacy EventMatrix wrapper
# ---------------------------------------------------------------------------


class EventMatrix:
    """A collection of event waveforms ready for decomposition.

    This is a convenience wrapper; the underlying API is
    :func:`decompose` operating on :class:`AxisAnnotatedTensor`.

    Parameters
    ----------
    X:
        Array of shape ``(n_events, n_samples)``.
    sfreq:
        Sampling frequency in Hz. Used only for axis labels in plots.
    """

    def __init__(self, X: np.ndarray, sfreq: float = 1.0, meta=None):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_events, n_samples), got shape {X.shape}")
        self._X = X
        self.sfreq = float(sfreq)
        self.meta = meta

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def data(self) -> np.ndarray:
        return self._X

    @property
    def n_events(self) -> int:
        return self._X.shape[0]

    @property
    def n_samples(self) -> int:
        return self._X.shape[1]

    # ------------------------------------------------------------------
    # Pre-processing helpers
    # ------------------------------------------------------------------

    def align(self, mode: AlignMode = "peak") -> "EventMatrix":
        if mode == "none":
            return self
        return EventMatrix(_align(self._X, mode), sfreq=self.sfreq)

    def normalize(self, mode: NormalizeMode = "none") -> "EventMatrix":
        if mode == "none":
            return self
        return EventMatrix(_normalize(self._X, mode), sfreq=self.sfreq)

    def resample(self, n_samples: int) -> "EventMatrix":
        if n_samples == self.n_samples:
            return self
        old_t = np.linspace(0, 1, self.n_samples)
        new_t = np.linspace(0, 1, n_samples)
        out = np.stack([interp1d(old_t, row)(new_t) for row in self._X])
        return EventMatrix(out, sfreq=self.sfreq)

    # ------------------------------------------------------------------
    # Decomposition
    # ------------------------------------------------------------------

    def decompose(
        self,
        method: Literal["svd", "nmf", "dictlearn"] = "svd",
        n_components: int = 5,
        center: bool = True,
        normalize: NormalizeMode = "none",
        align: AlignMode = "none",
        resample_to_length: int | None = None,
    ) -> DecompositionResult:
        """Decompose the event matrix.

        Parameters
        ----------
        method:
            ``'svd'`` (default), ``'nmf'``, or ``'dictlearn'``.
        n_components:
            Number of components to extract.
        center:
            Subtract the mean waveform before decomposition.
        normalize:
            Per-event normalization.
        align:
            Alignment mode applied before decomposition.
        resample_to_length:
            If given, resample all events to this many samples first.
        """
        import pandas as pd

        t_axis = np.arange(self.n_samples) / self.sfreq
        aat = AxisAnnotatedTensor(
            data=self._X,
            axes=["instance", "sample"],
            axis_index={
                "instance": np.arange(self.n_events),
                "sample": t_axis,
            },
            axis_meta={
                "instance": pd.DataFrame({"instance_id": np.arange(self.n_events)}),
                "sample": pd.DataFrame({"sample_index": np.arange(self.n_samples), "time": t_axis}),
            },
            attrs={"sfreq": self.sfreq},
        )
        return decompose(
            aat,
            method=method,
            n_components=n_components,
            center=center,
            normalize=normalize,
            align=align,
            resample_to_length=resample_to_length,
        )

    def __repr__(self) -> str:
        return f"EventMatrix(n_events={self.n_events}, n_samples={self.n_samples}, sfreq={self.sfreq})"


# ---------------------------------------------------------------------------
# Shared preprocessing helpers
# ---------------------------------------------------------------------------


def _align(X: np.ndarray, mode: str) -> np.ndarray:
    n_samples = X.shape[1]
    if mode == "peak":
        anchors = X.argmax(axis=1)
    elif mode == "trough":
        anchors = X.argmin(axis=1)
    elif mode == "midpoint":
        anchors = np.full(X.shape[0], n_samples // 2, dtype=int)
    elif mode == "onset":
        return X.copy()
    else:
        raise ValueError(f"Unknown align mode {mode!r}")

    center = n_samples // 2
    out = np.zeros_like(X)
    for i, (row, a) in enumerate(zip(X, anchors)):
        shift = center - int(a)
        if shift >= 0:
            out[i, shift:] = row[: n_samples - shift]
        else:
            out[i, : n_samples + shift] = row[-shift:]
    return out


def _normalize(X: np.ndarray, mode: str) -> np.ndarray:
    X = X.copy()
    if mode == "max":
        norms = np.abs(X).max(axis=1, keepdims=True)
        X /= np.where(norms == 0, 1.0, norms)
    elif mode == "l2":
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X /= np.where(norms == 0, 1.0, norms)
    elif mode == "zscore":
        mu = X.mean(axis=1, keepdims=True)
        sigma = X.std(axis=1, keepdims=True)
        X = (X - mu) / np.where(sigma == 0, 1.0, sigma)
    else:
        raise ValueError(f"Unknown normalize mode {mode!r}")
    return X
