from __future__ import annotations

import numpy as np


def _time_axis(n_samples: int, sfreq: float) -> np.ndarray:
    return np.arange(n_samples) / sfreq


def plot_spectrum(result, ax=None):
    """Scree plot of singular values."""
    import matplotlib.pyplot as plt

    fig, ax = (None, ax) if ax is not None else plt.subplots()
    sv = result.singular_values
    ax.plot(np.arange(1, len(sv) + 1), sv, "o-")
    ax.set_xlabel("Component")
    ax.set_ylabel("Singular value")
    ax.set_title("Singular value spectrum")
    ax.set_xticks(np.arange(1, len(sv) + 1))
    return ax


def plot_templates(result, n: int = 5, sfreq: float | None = None, ax=None):
    """Plot the first *n* principal waveform templates."""
    import matplotlib.pyplot as plt

    sf = sfreq or result.config.get("sfreq", 1.0)
    k = min(n, result.templates.shape[0])
    t = _time_axis(result.templates.shape[1], sf)

    fig, axes = plt.subplots(k, 1, figsize=(8, 2 * k), sharex=True)
    if k == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, result.templates[i])
        ax.set_ylabel(f"Comp {i + 1}")
        evr = result.explained_variance_ratio[i] if i < len(result.explained_variance_ratio) else 0
        ax.set_title(f"Template {i + 1}  (EVR={evr:.3f})")
    axes[-1].set_xlabel("Time (s)" if sf != 1.0 else "Samples")
    plt.tight_layout()
    return axes


def plot_mean_pm(result, comp: int = 0, sfreq: float | None = None, ax=None):
    """Plot mean waveform ± 2σ × component *comp*."""
    import matplotlib.pyplot as plt

    sf = sfreq or result.config.get("sfreq", 1.0)
    t = _time_axis(result.templates.shape[1], sf)
    mean = result.mean_waveform
    template = result.templates[comp]
    scale = 2 * result.loadings[:, comp].std()

    fig, ax = (None, ax) if ax is not None else plt.subplots()
    ax.fill_between(t, mean - scale * template, mean + scale * template, alpha=0.3, label="±2σ·comp")
    ax.plot(t, mean, label="Mean waveform")
    ax.set_xlabel("Time (s)" if sf != 1.0 else "Samples")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Mean ± 2σ · Component {comp + 1}")
    ax.legend()
    return ax


def plot_scatter(result, x: int = 0, y: int = 1, ax=None):
    """Scatter plot of events in component *x* vs component *y* space."""
    import matplotlib.pyplot as plt

    fig, ax = (None, ax) if ax is not None else plt.subplots()
    ax.scatter(result.loadings[:, x], result.loadings[:, y], alpha=0.5, s=20)
    ax.set_xlabel(f"Component {x + 1}")
    ax.set_ylabel(f"Component {y + 1}")
    ax.set_title(f"Events: Component {x + 1} vs {y + 1}")
    return ax


def plot_sorted_grid(result, comp: int = 0, n: int = 15, sfreq: float | None = None):
    """Grid of *n* events sorted by loading on component *comp*."""
    import matplotlib.pyplot as plt

    sf = sfreq or result.config.get("sfreq", 1.0)
    t = _time_axis(result.templates.shape[1], sf)

    order = np.argsort(result.loadings[:, comp])
    indices = np.linspace(0, len(order) - 1, min(n, len(order)), dtype=int)
    selected = order[indices]

    cols = 5
    rows = int(np.ceil(len(selected) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2), sharex=True, sharey=True)
    axes = np.array(axes).flatten()

    X_full = result.mean_waveform + result.loadings @ result.templates if result.config.get("center") else result.loadings @ result.templates

    for ax, idx in zip(axes, selected):
        ax.plot(t, X_full[idx], lw=0.8)
        ax.set_title(f"#{idx}  L={result.loadings[idx, comp]:.2f}", fontsize=7)
    for ax in axes[len(selected):]:
        ax.set_visible(False)
    plt.suptitle(f"Events sorted by Component {comp + 1} loading", y=1.01)
    plt.tight_layout()
    return axes


def plot_residual_hist(result, ax=None):
    """Histogram of per-event reconstruction residual norms."""
    import matplotlib.pyplot as plt

    scores = np.linalg.norm(result.residuals, axis=1)
    fig, ax = (None, ax) if ax is not None else plt.subplots()
    ax.hist(scores, bins=30, edgecolor="white")
    ax.set_xlabel("Residual L2 norm")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction residual distribution")
    return ax


def plot_loadings_by_group(result, groups, comp: int = 0, kind: str = "box", ax=None):
    """Distribution of loadings on component *comp* split by group label.

    Parameters
    ----------
    result:
        A :class:`DecompositionResult`.
    groups:
        1-D array-like of group labels, length ``n_events``.
    comp:
        Component index (0-based).
    kind:
        ``'box'`` (default) or ``'violin'``.
    """
    import matplotlib.pyplot as plt

    groups = np.asarray(groups)
    loadings = result.loadings[:, comp]
    if groups.shape[0] != loadings.shape[0]:
        raise ValueError(
            f"groups length {groups.shape[0]} != n_events {loadings.shape[0]}"
        )

    labels = list(np.unique(groups))
    data = [loadings[groups == lbl] for lbl in labels]

    fig, ax = (None, ax) if ax is not None else plt.subplots()
    positions = np.arange(1, len(labels) + 1)
    if kind == "violin":
        ax.violinplot(data, positions=positions, showmedians=True)
    elif kind == "box":
        ax.boxplot(data, positions=positions, labels=[str(l) for l in labels])
    else:
        raise ValueError(f"kind must be 'box' or 'violin', got {kind!r}")
    ax.set_xticks(positions)
    ax.set_xticklabels([str(l) for l in labels])
    ax.set_xlabel("Group")
    ax.set_ylabel(f"Loading on component {comp + 1}")
    ax.set_title(f"Loadings on Component {comp + 1} by group")
    return ax


def plot_template_spectra(result, sfreq: float | None = None, n: int | None = None,
                          log: bool = True, ax=None):
    """Magnitude spectra (rfft) of each template waveform.

    Parameters
    ----------
    sfreq:
        Sampling frequency in Hz; defaults to ``result.config['sfreq']`` or 1.
    n:
        Number of leading templates to plot (default: all).
    log:
        If True (default), plot magnitude on a log scale.
    """
    import matplotlib.pyplot as plt

    sf = sfreq or result.config.get("sfreq", 1.0)
    templates = result.templates
    k = templates.shape[0] if n is None else min(n, templates.shape[0])
    n_samples = templates.shape[1]
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sf)

    fig, ax = (None, ax) if ax is not None else plt.subplots()
    for i in range(k):
        mag = np.abs(np.fft.rfft(templates[i]))
        ax.plot(freqs, mag, label=f"Comp {i + 1}")
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)" if sf != 1.0 else "Frequency (cycles/sample)")
    ax.set_ylabel("|FFT|")
    ax.set_title("Template spectra")
    ax.legend(fontsize=8, ncol=min(k, 3))
    return ax
