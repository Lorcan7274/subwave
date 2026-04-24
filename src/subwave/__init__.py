from ._version import __version__
from .core import EventMatrix, decompose
from .dataset import TensorDataset, TensorView, concat_datasets, make_dataset
from .io import from_array, from_mne, from_npz, from_yasa
from .tensor import AxisAnnotatedTensor

__all__ = [
    "__version__",
    # Core API
    "decompose",
    # Data containers
    "AxisAnnotatedTensor",
    "EventMatrix",
    # Dataset / view
    "TensorDataset",
    "TensorView",
    "make_dataset",
    "concat_datasets",
    # I/O
    "from_array",
    "from_npz",
    "from_mne",
    "from_yasa",
]
