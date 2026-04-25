from __future__ import annotations

import json
from pathlib import Path

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
        """L2 norm of per-event reconstruction residuals.

        With ``n_components=None`` (default), returns ``||residual_i||`` from the
        full decomposition (i.e. the residual stored on this result).

        With an integer ``n_components < self.templates.shape[0]``, recomputes
        residuals against a rank-k truncated reconstruction so events poorly
        captured by the leading k components score high.
        """
        K = self.templates.shape[0]
        if n_components is None or n_components >= K:
            return np.linalg.norm(self.residuals, axis=1)
        k = max(int(n_components), 0)
        approx_k = self.loadings[:, :k] @ self.templates[:k]
        target = self.residuals + self.loadings @ self.templates
        return np.linalg.norm(target - approx_k, axis=1)

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

    def cluster(self, method: str = "kmeans", n_clusters: int = 2, **kwargs) -> dict:
        """Cluster events in loadings space."""
        from .clustering import cluster_loadings
        return cluster_loadings(self.loadings, method=method, n_clusters=n_clusters, **kwargs)

    def cluster_templates(self, n_clusters: int = 2, method: str = "kmeans") -> np.ndarray:
        """Mean waveform per cluster. Returns (n_clusters, n_samples)."""
        cr = self.cluster(method=method, n_clusters=n_clusters)
        if self.config.get("center"):
            X_full = self.mean_waveform + self.loadings @ self.templates
        else:
            X_full = self.loadings @ self.templates
        return np.stack(
            [X_full[cr["labels"] == k].mean(axis=0) for k in range(n_clusters)]
        )

    def template_spectrum(self, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
        """Returns (freqs, powers). freqs: (n_freqs,). powers: (n_components, n_freqs)."""
        n = self.templates.shape[1]
        freqs = np.fft.rfftfreq(n, d=1.0 / sfreq)
        powers = np.abs(np.fft.rfft(self.templates, axis=1)) ** 2
        return freqs, powers

    def template_peak_freq(self, sfreq: float) -> np.ndarray:
        """Peak frequency of each template in Hz. Returns (n_components,)."""
        freqs, powers = self.template_spectrum(sfreq)
        return freqs[np.argmax(powers, axis=1)]

    def template_bandwidth(self, sfreq: float) -> np.ndarray:
        """Half-power bandwidth of dominant peak per template. Returns (n_components,)."""
        freqs, powers = self.template_spectrum(sfreq)
        bw = np.zeros(powers.shape[0])
        for i, row in enumerate(powers):
            peak = row.max()
            indices = np.where(row >= peak / 2)[0]
            if len(indices) >= 2:
                bw[i] = freqs[indices[-1]] - freqs[indices[0]]
        return bw

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

    def plot_loadings_by_group(self, groups, comp: int = 0, **kwargs):
        from .plotting import plot_loadings_by_group
        return plot_loadings_by_group(self, groups, comp=comp, **kwargs)

    def plot_template_spectra(self, sfreq: float | None = None, **kwargs):
        from .plotting import plot_template_spectra
        return plot_template_spectra(self, sfreq=sfreq, **kwargs)

    def plot_heatmap(self, comp: int = 0, sfreq: float | None = None, **kwargs):
        from .plotting import plot_heatmap
        return plot_heatmap(self, comp=comp, sfreq=sfreq, **kwargs)

    def plot_waterfall(self, n: int = 100, sfreq: float | None = None, **kwargs):
        from .plotting import plot_waterfall
        return plot_waterfall(self, n=n, sfreq=sfreq, **kwargs)

    def plot_cumulative_variance(self, **kwargs):
        from .plotting import plot_cumulative_variance
        return plot_cumulative_variance(self, **kwargs)

    def plot_reconstruction(self, event_idx: int, n_components: int | None = None,
                            sfreq: float | None = None, **kwargs):
        from .plotting import plot_reconstruction
        return plot_reconstruction(self, event_idx=event_idx,
                                   n_components=n_components, sfreq=sfreq, **kwargs)

    def scatter_colored_by(self, values, x: int = 0, y: int = 1,
                           label: str | None = None, **kwargs):
        """Scatter colored by a per-event values array."""
        from .plotting import plot_scatter
        return plot_scatter(self, x=x, y=y, color=values, label=label, **kwargs)

    def loadings_correlated_with(self, values) -> pd.DataFrame:
        """Pearson r and p for each component vs *values*.

        Returns DataFrame with columns ``component``, ``r``, ``p_value``.
        """
        from scipy.stats import pearsonr

        v = np.asarray(values, dtype=float)
        if v.shape[0] != self.loadings.shape[0]:
            raise ValueError("values length must match number of events")

        rows = []
        for i in range(self.loadings.shape[1]):
            r, p = pearsonr(self.loadings[:, i], v)
            rows.append({"component": i + 1, "r": float(r), "p_value": float(p)})
        return pd.DataFrame(rows)

    def plot_loadings_over_time(self, event_times, comps=None, window: int = 20, **kwargs):
        from .plotting import plot_loadings_over_time
        return plot_loadings_over_time(self, event_times, comps=comps, window=window, **kwargs)

    def save(self, path: str | Path) -> None:
        """Save to ``.npz``.

        Stores: ``templates``, ``loadings``, ``singular_values``,
        ``explained_variance_ratio``, ``mean_waveform``, ``residuals``,
        ``config`` (JSON string), and ``method``.
        """
        np.savez(
            str(path),
            templates=self.templates,
            loadings=self.loadings,
            singular_values=self.singular_values,
            explained_variance_ratio=self.explained_variance_ratio,
            mean_waveform=self.mean_waveform,
            residuals=self.residuals,
            config=np.array(json.dumps(self.config)),
            method=np.array(self.method),
            instance_ids=np.asarray(self.factor_tables["instance"]["instance_id"].values),
        )

    @staticmethod
    def load(path: str | Path) -> "DecompositionResult":
        """Load a result previously saved with :meth:`save`."""
        from .decomposition import _build_result

        d = np.load(str(path), allow_pickle=False)
        templates = d["templates"]
        loadings = d["loadings"]
        sv = d["singular_values"]
        evr = d["explained_variance_ratio"]
        mean_waveform = d["mean_waveform"]
        residuals = d["residuals"]
        config = json.loads(str(d["config"]))
        method = str(d["method"])
        instance_ids = d["instance_ids"]

        return _build_result(
            method=method,
            config=config,
            templates=templates,
            loadings=loadings,
            singular_values=sv,
            evr=evr,
            mean_waveform=mean_waveform,
            residuals=residuals,
            instance_ids=instance_ids,
            sample_axis_index=None,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Loadings + ``recon_error`` as a flat DataFrame."""
        return self.factor_tables["instance"].copy()

    def __repr__(self) -> str:
        n_events = len(self.factor_tables["instance"])
        n_comp = self.templates.shape[0]
        evr = float(self.explained_variance_ratio[:n_comp].sum())
        return (
            f"DecompositionResult(method={self.method!r}, n_events={n_events}, "
            f"n_components={n_comp}, explained_variance={evr:.3f})"
        )
