from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import TensorDataset


def save_dataset(ds: TensorDataset, path: str | Path) -> None:
    """Save a TensorDataset to a directory bundle.

    Layout::

        <path>/
          manifest.json
          arrays/data.zarr
          meta/<axis>.parquet  (one per axis)

    Requires the ``subwave[storage]`` extra (zarr + pyarrow).
    """
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "zarr is required for on-disk storage: pip install 'subwave[storage]'"
        ) from exc

    path = Path(path)
    arrays_dir = path / "arrays"
    meta_dir = path / "meta"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    data = np.asarray(ds.store)
    z = zarr.open(
        str(arrays_dir / "data.zarr"), mode="w", shape=data.shape, dtype=data.dtype
    )
    z[:] = data

    meta_files: dict[str, str] = {}
    for ax, meta in ds.axis_meta.items():
        fname = f"meta/{ax}.parquet"
        meta.to_parquet(path / fname, index=False)
        meta_files[ax] = fname

    manifest = {
        **ds.manifest,
        "array_store": "arrays/data.zarr",
        "meta_files": meta_files,
        "attrs": ds.attrs,
    }
    (path / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def open_dataset(path: str | Path) -> TensorDataset:
    """Open a directory bundle as a lazy TensorDataset.

    Requires the ``subwave[storage]`` extra.
    """
    try:
        import zarr
    except ImportError as exc:
        raise ImportError(
            "zarr is required for on-disk storage: pip install 'subwave[storage]'"
        ) from exc

    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    store = zarr.open(str(path / manifest["array_store"]), mode="r")

    axis_meta: dict[str, pd.DataFrame] = {}
    for ax, fname in manifest.get("meta_files", {}).items():
        axis_meta[ax] = pd.read_parquet(path / fname)

    return TensorDataset(
        manifest=manifest,
        store=store,
        axis_meta=axis_meta,
        attrs=manifest.get("attrs", {}),
    )
