"""Tensor decomposition backends for multi-channel event waveforms.

Decomposes a 3D tensor X of shape (n_events, n_samples, n_channels) into
components that each have a temporal profile AND a spatial (channel) profile.
"""

from __future__ import annotations

import numpy as np


def _check_tensorly():
    try:
        import tensorly
        return tensorly
    except ImportError as exc:
        raise ImportError(
            "Tensor decomposition requires tensorly. Install with: pip install tensorly"
        ) from exc


class TensorResult:
    """Result container for tensor decomposition.

    Attributes
    ----------
    method:
        ``'cp'`` or ``'tucker'``.
    event_factors:
        ``(n_events, rank)`` per-event loadings.
    temporal_factors:
        ``(n_samples, rank)`` temporal profile per component.
    spatial_factors:
        ``(n_channels, rank)`` channel weight per component.
    core:
        Core tensor for Tucker. ``None`` for CP.
    reconstruction_error:
        Relative reconstruction error ``||X - X_hat|| / ||X||``.
    config:
        Options used.
    """

    def __init__(
        self,
        method,
        event_factors,
        temporal_factors,
        spatial_factors,
        core=None,
        reconstruction_error=None,
        config=None,
    ):
        self.method = method
        self.event_factors = event_factors
        self.temporal_factors = temporal_factors
        self.spatial_factors = spatial_factors
        self.core = core
        self.reconstruction_error = reconstruction_error
        self.config = config or {}

    @property
    def rank(self):
        return self.temporal_factors.shape[1]

    def plot(self, time=None, ch_labels=None, rank=None):
        """Three-panel plot per component: temporal, spatial, loading distribution."""
        import matplotlib.pyplot as plt

        k = min(rank, self.rank) if rank else self.rank
        n_s = self.temporal_factors.shape[0]
        t = time if time is not None else np.arange(n_s)
        chs = ch_labels if ch_labels is not None else [str(i) for i in range(self.spatial_factors.shape[0])]

        fig, axes = plt.subplots(k, 3, figsize=(12, 2.5 * k))
        if k == 1:
            axes = axes[np.newaxis, :]

        for i in range(k):
            axes[i, 0].plot(t, self.temporal_factors[:, i])
            axes[i, 0].axvline(0, color='gray', lw=0.8, ls='--')
            axes[i, 0].set_ylabel(f'Comp {i+1}')
            if i == 0:
                axes[i, 0].set_title('Temporal')

            axes[i, 1].bar(chs, self.spatial_factors[:, i])
            if i == 0:
                axes[i, 1].set_title('Spatial (channels)')

            axes[i, 2].hist(self.event_factors[:, i], bins=40)
            if i == 0:
                axes[i, 2].set_title('Event loadings')

        axes[-1, 0].set_xlabel('Time (s)' if time is not None else 'Samples')
        plt.tight_layout()
        return axes

    def component_waveform(self, comp, channel=None):
        """Get the temporal waveform for a component, optionally scaled by channel."""
        tw = self.temporal_factors[:, comp]
        if channel is not None:
            tw = tw * self.spatial_factors[channel, comp]
        return tw

    def __repr__(self):
        shape = (
            f"events={self.event_factors.shape[0]}, "
            f"samples={self.temporal_factors.shape[0]}, "
            f"channels={self.spatial_factors.shape[0]}"
        )
        err = f"{self.reconstruction_error:.4f}" if self.reconstruction_error is not None else "n/a"
        return f"TensorResult(method='{self.method}', rank={self.rank}, {shape}, recon_error={err})"


def tensor_decompose(X, method="cp", rank=3, random_state=None, **kwargs):
    """Decompose a 3-D tensor ``(n_events, n_samples, n_channels)``.

    Parameters
    ----------
    X:
        Array of shape ``(n_events, n_samples, n_channels)``.
    method:
        ``'cp'`` or ``'tucker'``.
    rank:
        For CP: total rank. For Tucker: used for all modes equally, or pass
        ``ranks=(r1, r2, r3)`` via kwargs to set per-mode.
    random_state:
        RNG seed.

    Returns
    -------
    TensorResult
    """
    _check_tensorly()
    import tensorly as tl_module
    from tensorly.decomposition import parafac, tucker

    X = np.asarray(X)
    if X.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {X.ndim}D")

    tensor = tl_module.tensor(X.astype(np.float64))

    if method == "cp":
        result = parafac(
            tensor,
            rank=rank,
            init="random",
            random_state=random_state if random_state is not None else 0,
            n_iter_max=200,
        )
        weights, factors = result
        event_factors = factors[0] * weights
        temporal_factors = factors[1]
        spatial_factors = factors[2]
        core = None
    elif method == "tucker":
        ranks = kwargs.get("ranks", (rank, rank, min(rank, X.shape[2])))
        result = tucker(
            tensor,
            rank=ranks,
            init="random",
            random_state=random_state if random_state is not None else 0,
            n_iter_max=200,
        )
        core_tensor, factors = result
        event_factors = factors[0]
        temporal_factors = factors[1]
        spatial_factors = factors[2]
        core = np.asarray(core_tensor)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cp' or 'tucker'.")

    if method == "cp":
        X_hat = tl_module.cp_to_tensor(
            (np.ones(rank), [event_factors, temporal_factors, spatial_factors])
        )
    else:
        X_hat = tl_module.tucker_to_tensor(
            (core, [event_factors, temporal_factors, spatial_factors])
        )

    X_hat = np.asarray(X_hat)
    norm_X = float(np.linalg.norm(X))
    recon_err = float(np.linalg.norm(X - X_hat) / norm_X) if norm_X > 0 else 0.0

    return TensorResult(
        method=method,
        event_factors=np.asarray(event_factors),
        temporal_factors=np.asarray(temporal_factors),
        spatial_factors=np.asarray(spatial_factors),
        core=core,
        reconstruction_error=recon_err,
        config={"method": method, "rank": rank, "random_state": random_state},
    )
