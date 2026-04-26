import numpy as np


def synthetic_population(
    templates,
    n_events=200,
    noise_std=0.1,
    jitter_std=0,
    amplitude_std=0.0,
    random_state=None,
):
    """Generate a synthetic event population with known ground-truth structure.

    templates: (n_templates, n_samples) array of basis waveforms
    n_events: total number of events to generate
    noise_std: standard deviation of additive Gaussian noise
    jitter_std: standard deviation of temporal jitter in samples (int, circular shift)
    amplitude_std: standard deviation of per-event amplitude scaling (centered at 1.0)
    random_state: int or np.random.Generator

    Returns dict:
        X: (n_events, n_samples) generated waveforms
        loadings: (n_events, n_templates) true mixing coefficients (drawn from U[0,1])
        labels: (n_events,) int, index of dominant template per event (argmax of loadings)
        templates: (n_templates, n_samples) copy of input templates
    """
    rng = np.random.default_rng(random_state)
    templates = np.asarray(templates)
    if templates.ndim != 2:
        raise ValueError("templates must be 2D (n_templates, n_samples)")
    n_templates, n_samples = templates.shape

    loadings = rng.uniform(0.0, 1.0, size=(n_events, n_templates))
    X = loadings @ templates  # (n_events, n_samples)

    if amplitude_std > 0:
        amps = rng.normal(loc=1.0, scale=amplitude_std, size=(n_events, 1))
        X = X * amps

    if jitter_std and jitter_std > 0:
        shifts = rng.normal(loc=0.0, scale=float(jitter_std), size=n_events)
        shifts = np.round(shifts).astype(int)
        for i in range(n_events):
            if shifts[i] != 0:
                X[i] = np.roll(X[i], shifts[i])

    if noise_std > 0:
        X = X + rng.normal(loc=0.0, scale=noise_std, size=X.shape)

    labels = np.argmax(loadings, axis=1)

    return {
        "X": X,
        "loadings": loadings,
        "labels": labels,
        "templates": templates.copy(),
    }
