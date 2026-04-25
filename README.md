# subwave

Data-driven decomposition of neurophysiological event waveform populations.

## Install

```bash
pip install subwave
```

Optional extras:

```bash
pip install "subwave[mne]"      # MNE-Python and Luna I/O
pip install "subwave[yasa]"     # YASA spindle detection I/O
```

## Quickstart

```python
import numpy as np
import subwave as sw

rng = np.random.default_rng(0)
t = np.linspace(0, 1, 256)
X = np.stack([np.sin(2 * np.pi * 13 * t) + rng.normal(0, 0.1, 256) for _ in range(100)])

result = sw.decompose(X, method="svd", n_components=5)
result.plot_templates(n=3)
```

Each template is a basis waveform shape; each loading is how strongly a given event expresses that template. Together they reconstruct the original population.

Or let subwave choose the number of components:

```python
result = sw.decompose(X, n_components="auto")
```

## Loading data

```python
sw.from_array(X, sfreq=256)                              # plain numpy
sw.from_npz("spindles.npz")                              # Lunascope format
sw.from_mne(epochs)                                       # MNE Epochs
sw.from_yasa(spindles_df, raw_signal, sfreq=256)          # YASA output
sw.from_luna("spindles.txt", "recording.edf", sfreq=256)  # Luna output + EDF
```

## Decomposition methods

- **SVD / PCA** (`method='svd'`) — default, optimal low-rank approximation. Uses randomized SVD for >5000 events.
- **NMF** (`method='nmf'`) — parts-based, requires non-negative input.
- **Dictionary learning** (`method='dictlearn'`) — sparse atoms.

## Working with results

```python
result.templates                        # (n_components, n_samples) basis waveforms
result.loadings                         # (n_events, n_components) per-event scores
result.explained_variance_ratio         # variance captured per component
result.singular_values                  # singular values
result.factor_tables["instance"]        # DataFrame: instance_id, score_1…k, recon_error

result.reconstruct(n_components=3)      # rank-k reconstruction
result.project(new_X)                   # project new events onto learned subspace
result.outlier_scores()                 # per-event reconstruction error
```

## Component selection

```python
k = sw.parallel_analysis(X)             # Horn's parallel analysis
k = sw.elbow(result.singular_values)    # Kneedle elbow detection
k = sw.kaiser(result)                   # Kaiser rule
```

## Spectral characterization

```python
freqs, powers = result.template_spectrum(sfreq=256)
peak_hz = result.template_peak_freq(sfreq=256)     # e.g. [13.2, 11.1] Hz
bw_hz = result.template_bandwidth(sfreq=256)
```

## Clustering

```python
cr = result.cluster(method="kmeans", n_clusters=2)
cr["labels"]                            # cluster assignments
result.cluster_templates(n_clusters=2)  # mean waveform per cluster
```

## Group comparison

```python
perm = sw.permutation_test(X, groups, n_components=3, n_perm=500)
perm.p_value                            # do two groups span different subspaces?
```

## Serialization

```python
result.save("result.npz")
result = sw.load_result("result.npz")
df = result.to_dataframe()              # flat DataFrame for R/Stata
```

## Plots

```python
result.plot_spectrum()                  # singular value scree plot
result.plot_templates(n=5)              # basis waveforms
result.plot_template_spectra(sfreq=256) # power spectrum of each template
result.plot_scatter(x=0, y=1)           # component 0 vs 1 (supports color= for clusters)
result.plot_heatmap(comp=0)             # events × samples sorted by loading
result.plot_waterfall(n=100)            # overlaid waveforms with bold mean
result.plot_mean_pm(comp=0)             # mean ± component
result.plot_sorted_grid(comp=0)         # events sorted by score
result.plot_residual_hist()             # reconstruction error distribution
result.plot_cumulative_variance()       # cumulative EVR curve
result.plot_reconstruction(event_idx=0) # original vs reconstruction
result.plot_loadings_by_group(groups)   # box/violin by group
result.plot_loadings_over_time(times)   # loading drift across time
```

See docstrings via `help(sw.decompose)` for full options, including `cluster_sweep`, `loading_test`, `subspace_angles`, `scatter_colored_by`, `loadings_correlated_with`, and more.

## Citation

If you use subwave in published work, see `CITATION.cff` in the repository.

## License

MIT