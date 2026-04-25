# subwave

Matrix decomposition of neurophysiological event waveform populations (sleep spindles, ECG heartbeats, EMG bursts, etc.).

Takes a population of detected events, stacks them into a matrix and applies SVD, NMF, or dictionary learning to extract principal waveform templates, per-event component scores, and reconstruction diagnostics.

## Install

```bash
pip install subwave
```

Optional extras:

```bash
pip install "subwave[mne]"      # MNE-Python I/O
pip install "subwave[yasa]"     # YASA spindle detection I/O
```

## Quickstart

```python
import numpy as np
import subwave as sw

# Synthetic spindle-like population (100 events × 256 samples)
rng = np.random.default_rng(0)
t = np.linspace(0, 1, 256)
X = np.stack([np.sin(2 * np.pi * 13 * t) + rng.normal(0, 0.1, 256) for _ in range(100)])

result = sw.decompose(X, method="svd", n_components=5)

print(result)
result.plot_spectrum()
result.plot_templates(n=3)
```

`sw.decompose()` accepts a plain numpy array, an `AxisAnnotatedTensor`, or a `TensorView`.

## Loading real data

```python
# From an .npz bundle (Lunascope format)
aat = sw.from_npz("spindle_event_decomp_waveforms.npz")
result = sw.decompose(aat, method="svd", n_components=5)

# From a raw numpy array
result = sw.decompose(X, method="svd", n_components=5)

# From MNE Epochs
aat_or_em = sw.from_mne(epochs)

# From YASA spindle detection output
aat_or_em = sw.from_yasa(spindles_df, raw_signal, sfreq=256.0)
```

## Decomposition methods

| Method | `method=` | Notes |
|---|---|---|
| SVD / PCA | `'svd'` | Default. Optimal low-rank approximation. |
| NMF | `'nmf'` | Parts-based; requires non-negative input. |
| Dictionary learning | `'dictlearn'` | Sparse atoms via MiniBatchDictionaryLearning. |

## Working with results

```python
result.templates                        # (n_components, n_samples) basis waveforms
result.loadings                         # (n_events, n_components) per-event scores
result.explained_variance_ratio         # variance captured by each component
result.factor_tables["instance"]        # DataFrame: instance_id, score_1…k, recon_error

result.reconstruct(n_components=3)      # rank-k reconstruction
result.project(new_X)                   # project unseen events onto learned subspace
result.outlier_scores()                 # L2 reconstruction error per event
```

## Plots

```python
result.plot_spectrum()                  # scree / singular value plot
result.plot_templates(n=5)             # top-k basis waveforms
result.plot_scatter(x=0, y=1)          # score_1 vs score_2 scatter
result.plot_sorted_grid(comp=0)        # events sorted by component score
result.plot_mean_pm(comp=0)            # mean ± component waveform
result.plot_residual_hist()            # reconstruction error distribution
```

## License

MIT
