from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import subwave as sw
from subwave import AxisAnnotatedTensor, make_dataset
from subwave.core import EventMatrix

DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def small_matrix(rng):
    """20 events × 64 samples — mixture of fast/slow sinusoidal shapes."""
    t = np.linspace(0, 1, 64)
    fast = np.sin(2 * np.pi * 13 * t)
    slow = np.sin(2 * np.pi * 11 * t)
    X = np.stack(
        [fast * rng.uniform(0.8, 1.2) + rng.normal(0, 0.05, 64) for _ in range(10)]
        + [slow * rng.uniform(0.8, 1.2) + rng.normal(0, 0.05, 64) for _ in range(10)]
    )
    return X


@pytest.fixture
def event_matrix(small_matrix):
    return sw.from_array(small_matrix, sfreq=64.0)


@pytest.fixture
def nonneg_matrix(rng):
    """Non-negative matrix suitable for NMF."""
    t = np.linspace(0, 1, 64)
    a = np.abs(np.sin(2 * np.pi * 13 * t))
    b = np.abs(np.sin(2 * np.pi * 11 * t))
    X = np.stack(
        [a * rng.uniform(0.8, 1.2) + rng.uniform(0, 0.05, 64) for _ in range(10)]
        + [b * rng.uniform(0.8, 1.2) + rng.uniform(0, 0.05, 64) for _ in range(10)]
    )
    return X


@pytest.fixture
def simple_aat(small_matrix):
    """AxisAnnotatedTensor wrapping small_matrix."""
    t = np.linspace(-0.5, 0.5, 64)
    n = small_matrix.shape[0]
    return AxisAnnotatedTensor(
        data=small_matrix,
        axes=["instance", "sample"],
        axis_index={"instance": np.arange(n), "sample": t},
        axis_meta={
            "instance": pd.DataFrame({"instance_id": np.arange(n), "label": ["fast"] * 10 + ["slow"] * 10}),
            "sample": pd.DataFrame({"sample_index": np.arange(64), "time": t}),
        },
        attrs={"sfreq": 64.0},
    )


@pytest.fixture
def simple_dataset(small_matrix):
    """In-memory TensorDataset wrapping small_matrix."""
    n = small_matrix.shape[0]
    meta = pd.DataFrame({"instance_id": np.arange(n), "label": ["fast"] * 10 + ["slow"] * 10})
    ds = make_dataset(
        small_matrix,
        axes=["instance", "sample"],
        axis_meta={"instance": meta},
        axis_index_columns={"instance": "instance_id"},
    )
    return ds


@pytest.fixture
def spindle_path():
    p = DATA_DIR / "spindle_event_decomp_waveforms.npz"
    if not p.exists():
        pytest.skip("spindle data file not present")
    return p


@pytest.fixture
def ecg_path():
    p = DATA_DIR / "ecg_event_decomp_waveforms.npz"
    if not p.exists():
        pytest.skip("ecg data file not present")
    return p
