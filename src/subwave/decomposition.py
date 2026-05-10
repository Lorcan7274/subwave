from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF, MiniBatchDictionaryLearning

from .result import DecompositionResult
from .tensor import AxisAnnotatedTensor


def _build_result(
    method: str,
    config: dict,
    templates: np.ndarray,
    loadings: np.ndarray,
    singular_values: np.ndarray,
    evr: np.ndarray,
    mean_waveform: np.ndarray,
    residuals: np.ndarray,
    instance_ids: np.ndarray,
    sample_axis_index: np.ndarray | None,
) -> DecompositionResult:
    k, n_samples = templates.shape
    sample_idx = (
        sample_axis_index if sample_axis_index is not None else np.arange(n_samples)
    )
    if len(sample_idx) != n_samples:
        sample_idx = np.arange(n_samples)

    comp_meta = pd.DataFrame(
        {
            "component_index": np.arange(k),
            "singular_value": singular_values,
            "explained_variance_ratio": evr,
        }
    )
    sample_meta = pd.DataFrame(
        {"sample_index": np.arange(n_samples), "coord": sample_idx}
    )
    components_aat = AxisAnnotatedTensor(
        data=templates,
        axes=["component", "sample"],
        axis_index={"component": np.arange(k), "sample": sample_idx},
        axis_meta={"component": comp_meta, "sample": sample_meta},
    )

    instance_df = pd.DataFrame({"instance_id": instance_ids})
    for i in range(k):
        instance_df[f"score_{i + 1}"] = loadings[:, i]
    instance_df["recon_error"] = np.linalg.norm(residuals, axis=1)

    return DecompositionResult(
        method=method,
        config=config,
        input_ref={},
        factor_tables={"instance": instance_df},
        component_tensors={"components": components_aat},
        diagnostics={
            "singular_values": singular_values,
            "explained_variance_ratio": evr,
            "mean_waveform": mean_waveform,
            "residuals": residuals,
        },
    )


def _svd(
    X: np.ndarray,
    n_components: int,
    config: dict,
    instance_ids: np.ndarray,
    sample_axis_index: np.ndarray | None,
) -> DecompositionResult:
    mean_waveform = X.mean(axis=0) if config["center"] else np.zeros(X.shape[1])
    Xc = X - mean_waveform if config["center"] else X

    if X.shape[0] > 5000:
        from sklearn.utils.extmath import randomized_svd
        k_req = min(n_components, min(Xc.shape) - 1)
        U, s, Vh = randomized_svd(Xc, n_components=k_req, random_state=0)
    else:
        U, s, Vh = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, len(s))

    templates = Vh[:k]
    loadings = U[:, :k] * s[:k]
    sv = s[:k]

    # total_var = (s ** 2).sum()
    total_var = (Xc ** 2).sum()
    evr = (sv ** 2) / total_var if total_var > 0 else np.zeros(k)

    residuals = Xc - loadings @ templates

    return _build_result(
        "svd", config, templates, loadings, sv, evr,
        mean_waveform, residuals, instance_ids, sample_axis_index,
    )


def _incremental_evr(
    target: np.ndarray, loadings: np.ndarray, templates: np.ndarray
) -> np.ndarray:
    """Per-component explained variance ratio via incremental rank-k reconstruction.

    For each k = 1..K, fit residual variance of ``loadings[:, :k] @ templates[:k]``
    against ``target`` and report the marginal drop in residual variance.
    Components that increase residual variance are reported as 0.
    """
    total_var = float((target ** 2).sum())
    if total_var <= 0:
        return np.zeros(templates.shape[0])
    K = templates.shape[0]
    cumulative = np.zeros(K)
    prev_resid = total_var
    for k in range(1, K + 1):
        approx = loadings[:, :k] @ templates[:k]
        resid = float(((target - approx) ** 2).sum())
        cumulative[k - 1] = max(0.0, (prev_resid - resid)) / total_var
        prev_resid = min(prev_resid, resid)
    return cumulative


def _nmf(
    X: np.ndarray,
    n_components: int,
    config: dict,
    instance_ids: np.ndarray,
    sample_axis_index: np.ndarray | None,
) -> DecompositionResult:
    if X.min() < 0:
        raise ValueError(
            "NMF requires non-negative input. "
            "Use normalize='max' or 'l2', or choose method='svd'."
        )
    mean_waveform = np.zeros(X.shape[1])

    model = NMF(n_components=n_components, init="nndsvda", random_state=0, max_iter=500)
    loadings = model.fit_transform(X)
    templates = model.components_

    residuals = X - loadings @ templates
    evr = _incremental_evr(X, loadings, templates)
    sv = np.linalg.norm(templates, axis=1)

    return _build_result(
        "nmf", config, templates, loadings, sv, evr,
        mean_waveform, residuals, instance_ids, sample_axis_index,
    )


def _dictlearn(
    X: np.ndarray,
    n_components: int,
    config: dict,
    instance_ids: np.ndarray,
    sample_axis_index: np.ndarray | None,
) -> DecompositionResult:
    mean_waveform = X.mean(axis=0) if config["center"] else np.zeros(X.shape[1])
    Xc = X - mean_waveform if config["center"] else X

    model = MiniBatchDictionaryLearning(
        n_components=n_components,
        transform_algorithm="lasso_lars",
        random_state=0,
        max_iter=500,
    )
    loadings = model.fit_transform(Xc)
    templates = model.components_

    residuals = Xc - loadings @ templates
    evr = _incremental_evr(Xc, loadings, templates)
    sv = np.linalg.norm(templates, axis=1)

    return _build_result(
        "dictlearn", config, templates, loadings, sv, evr,
        mean_waveform, residuals, instance_ids, sample_axis_index,
    )


def _fourier_svd(
    X: np.ndarray,
    n_components: int,
    config: dict,
    instance_ids: np.ndarray,
    sample_axis_index: np.ndarray | None,
) -> DecompositionResult:
    """Fourier-then-SVD: shift-invariant decomposition.

    Computes the magnitude rfft of each event so small temporal shifts (which
    affect only phase) do not corrupt the decomposition. SVD is then run on the
    (n_events, n_freqs) magnitude-spectrum matrix.
    """
    X_mag = np.abs(np.fft.rfft(X, axis=1))

    mean_waveform = X_mag.mean(axis=0) if config.get("center", True) else np.zeros(X_mag.shape[1])
    Xc = X_mag - mean_waveform if config.get("center", True) else X_mag

    if Xc.shape[0] > 5000:
        from sklearn.utils.extmath import randomized_svd
        k_req = min(n_components, min(Xc.shape) - 1)
        U, s, Vh = randomized_svd(Xc, n_components=k_req, random_state=0)
    else:
        U, s, Vh = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, len(s))

    templates = Vh[:k]
    loadings = U[:, :k] * s[:k]
    sv = s[:k]

    total_var = (s ** 2).sum()
    evr = (sv ** 2) / total_var if total_var > 0 else np.zeros(k)

    residuals = Xc - loadings @ templates

    config = dict(config)
    config["domain"] = "frequency"

    n_freqs = X_mag.shape[1]
    freq_axis = np.arange(n_freqs)
    if sample_axis_index is None or len(sample_axis_index) != n_freqs:
        sample_axis_index_out = freq_axis
    else:
        sample_axis_index_out = sample_axis_index

    return _build_result(
        "fourier_svd", config, templates, loadings, sv, evr,
        mean_waveform, residuals, instance_ids, sample_axis_index_out,
    )


def _scattering_svd(
    X: np.ndarray,
    n_components: int,
    config: dict,
    instance_ids: np.ndarray,
    sample_axis_index: np.ndarray | None,
) -> DecompositionResult:
    """Scattering transform then SVD.

    Computes 1-D wavelet scattering coefficients per event (locally invariant
    to translation, stable to deformation), flattens, then runs SVD. Templates
    live in scattering coefficient space.
    """
    try:
        from kymatio.numpy import Scattering1D
    except ImportError as exc:
        raise ImportError(
            "Scattering transform requires kymatio. Install with: pip install kymatio"
        ) from exc

    n_samples = X.shape[1]
    J = max(int(np.log2(n_samples)) - 2, 1)
    T = n_samples
    scattering = Scattering1D(J=J, shape=(n_samples,), Q=8, T=T)

    Sx = scattering(X.astype(np.float32))
    Sx_flat = Sx.reshape(X.shape[0], -1).astype(np.float64)

    center = config.get("center", True)
    mean_vec = Sx_flat.mean(axis=0) if center else np.zeros(Sx_flat.shape[1])
    Xc = Sx_flat - mean_vec

    U, s, Vh = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, len(s))
    templates = Vh[:k]
    loadings = U[:, :k] * s[:k]
    sv = s[:k]
    total_var = float((s ** 2).sum())
    evr = (sv ** 2) / total_var if total_var > 0 else np.zeros(k)
    residuals = Xc - loadings @ templates

    config = dict(config)
    config["domain"] = "scattering"

    n_feats = Sx_flat.shape[1]
    feat_axis = np.arange(n_feats)
    if sample_axis_index is None or len(sample_axis_index) != n_feats:
        sample_axis_index_out = feat_axis
    else:
        sample_axis_index_out = sample_axis_index

    return _build_result(
        "scattering_svd", config, templates, loadings, sv, evr,
        mean_vec, residuals, instance_ids, sample_axis_index_out,
    )


_BACKENDS = {
    "svd": _svd,
    "nmf": _nmf,
    "dictlearn": _dictlearn,
    "fourier_svd": _fourier_svd,
    "scattering_svd": _scattering_svd,
}


def run_decomposition(
    X: np.ndarray,
    method: str,
    n_components: int,
    center: bool,
    config: dict,
    instance_ids: np.ndarray | None = None,
    sample_axis_index: np.ndarray | None = None,
) -> DecompositionResult:
    if method not in _BACKENDS:
        raise ValueError(f"Unknown method {method!r}. Choose from {list(_BACKENDS)}")
    if instance_ids is None:
        instance_ids = np.arange(X.shape[0])
    return _BACKENDS[method](X, n_components, config, instance_ids, sample_axis_index)
