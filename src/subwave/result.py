from __future__ import annotations

import numpy as np
import pandas as pd

from .tensor import AxisAnnotatedTensor


class DecompositionResult:
    """Spec-compliant decomposition output.

    Attributes
    ----------
    method:
        Name of the decomposition backend (``'svd'``, ``'nmf'``, ``'dictlearn'``).
    config:
        Full serializable record of all options used.
    input_ref:
        Reference info about the input (axes, shape, source path if any).
    factor_tables:
        Dict of DataFrames keyed by axis name.
        ``factor_tables['instance']`` always present; contains
        ``instance_id``, ``score_1 … score_k``, and ``recon_error``.
    component_tensors:
        Dict of :class:`AxisAnnotatedTensor` keyed by name.
        ``component_tensors['components']`` holds the basis waveforms with
        axes ``['component', 'sample']``.
    diagnostics:
        Backend-specific scalars/arrays: ``singular_values``,
        ``explained_variance_ratio``, ``mean_waveform``, ``residuals``.
    attrs:
        Arbitrary extra attributes.
    """

    def __init__(
        self,
        method: str,
        config: dict,
        input_ref: dict,
        factor_tables: dict[str, pd.DataFrame],
        component_tensors: dict[str, AxisAnnotatedTensor],
        diagnostics: dict,
        attrs: dict | None = None,
    ) -> None:
        self.method = method
        self.config = config
        self.input_ref = input_ref
        self.factor_tables = factor_tables
        self.component_tensors = component_tensors
        self.diagnostics = diagnostics
        self.attrs = attrs or {}

    # ------------------------------------------------------------------
    # Backward-compatible array properties
    # ------------------------------------------------------------------

    @property
    def templates(self) -> np.ndarray:
        """(n_components, n_samples) basis waveforms."""
        return self.component_tensors["components"].data

    @property
    def loadings(self) -> np.ndarray:
        """(n_events, n_components) per-event scores."""
        ft = self.factor_tables["instance"]
        score_cols = [c for c in ft.columns if c.startswith("score_")]
        return ft[score_cols].values

    @property
    def singular_values(self) -> np.ndarray:
        return self.diagnostics["singular_values"]

    @property
    def explained_variance_ratio(self) -> np.ndarray:
        return self.diagnostics["explained_variance_ratio"]

    @property
    def mean_waveform(self) -> np.ndarray:
        return self.diagnostics["mean_waveform"]

    @property
    def residuals(self) -> np.ndarray:
        return self.diagnostics["residuals"]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def outlier_scores(self, n_components: int | None = None) -> np.ndarray:
        """L2 norm of per-event reconstruction residuals."""
        rec = self.reconstruct(n_components=n_components)
        X_approx = self.mean_waveform + rec if self.config.get("center") else rec
        X_full = (
            self.mean_waveform + self.loadings @ self.templates
            if self.config.get("center")
            else self.loadings @ self.templates
        )
        return np.linalg.norm(X_full - X_approx, axis=1)

    def reconstruct(self, n_components: int | None = None) -> np.ndarray:
        """Rank-k reconstruction of the (mean-centered) event matrix."""
        k = min(n_components or self.loadings.shape[1], self.loadings.shape[1])
        return self.loadings[:, :k] @ self.templates[:k]

    def project(self, new_events: np.ndarray) -> np.ndarray:
        """Project *new_events* onto the learned template subspace.

        Returns ``(m, n_components)`` scores.
        """
        X = np.atleast_2d(new_events).astype(float)
        if self.config.get("center"):
            X = X - self.mean_waveform
        return X @ self.templates.T

    def subspace_angles(self, other: "DecompositionResult") -> np.ndarray:
        """Principal angles (radians) between two decomposition subspaces."""
        Q1, _ = np.linalg.qr(self.templates.T)
        Q2, _ = np.linalg.qr(other.templates.T)
        k = min(Q1.shape[1], Q2.shape[1])
        _, s, _ = np.linalg.svd(Q1[:, :k].T @ Q2[:, :k])
        return np.arccos(np.clip(s, -1.0, 1.0))

    # ------------------------------------------------------------------
    # Visualization (delegates to plotting module)
    # ------------------------------------------------------------------

    def plot_spectrum(self, **kwargs):
        from .plotting import plot_spectrum
        return plot_spectrum(self, **kwargs)

    def plot_templates(self, n: int = 5, sfreq: float | None = None, **kwargs):
        from .plotting import plot_templates
        return plot_templates(self, n=n, sfreq=sfreq, **kwargs)

    def plot_mean_pm(self, comp: int = 0, sfreq: float | None = None, **kwargs):
        from .plotting import plot_mean_pm
        return plot_mean_pm(self, comp=comp, sfreq=sfreq, **kwargs)

    def plot_scatter(self, x: int = 0, y: int = 1, **kwargs):
        from .plotting import plot_scatter
        return plot_scatter(self, x=x, y=y, **kwargs)

    def plot_sorted_grid(self, comp: int = 0, n: int = 15, sfreq: float | None = None, **kwargs):
        from .plotting import plot_sorted_grid
        return plot_sorted_grid(self, comp=comp, n=n, sfreq=sfreq, **kwargs)

    def plot_residual_hist(self, **kwargs):
        from .plotting import plot_residual_hist
        return plot_residual_hist(self, **kwargs)

    def __repr__(self) -> str:
        n_events = len(self.factor_tables["instance"])
        n_comp = self.templates.shape[0]
        evr = float(self.explained_variance_ratio[:n_comp].sum())
        return (
            f"DecompositionResult(method={self.method!r}, n_events={n_events}, "
            f"n_components={n_comp}, explained_variance={evr:.3f})"
        )
