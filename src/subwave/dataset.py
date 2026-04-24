from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .tensor import AxisAnnotatedTensor


@dataclass
class TensorDataset:
    """An annotated tensor dataset (in-memory or zarr-backed).

    Parameters
    ----------
    manifest:
        Dict describing shape, dtype, axes, id columns, and file paths.
        Must contain at minimum ``'axes'`` and ``'shape'`` keys.
    store:
        The numeric array (numpy or zarr).
    axis_meta:
        Per-axis metadata DataFrames.
    attrs:
        Global scalar attributes.
    """

    manifest: dict
    store: Any
    axis_meta: dict[str, pd.DataFrame]
    attrs: dict = field(default_factory=dict)

    @property
    def axes(self) -> list[str]:
        return self.manifest["axes"]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.manifest["shape"])

    def view(self) -> "TensorView":
        """Return a full (unfiltered) view of this dataset."""
        return TensorView(dataset=self)

    def __repr__(self) -> str:
        shape_str = " x ".join(
            f"{ax}({self.shape[i]})" for i, ax in enumerate(self.axes)
        )
        return f"TensorDataset({shape_str})"


@dataclass
class TensorView:
    """Lightweight lazy view onto a TensorDataset.

    Selectors are arrays of integer positions into ``dataset.store`` along
    each axis. Un-set axes use all positions.
    """

    dataset: TensorDataset
    selectors: dict[str, np.ndarray] = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sel(self, axis: str) -> np.ndarray:
        idx = self.dataset.axes.index(axis)
        return self.selectors.get(axis, np.arange(self.dataset.shape[idx]))

    def _replace(self, axis: str, new_pos: np.ndarray) -> "TensorView":
        return TensorView(
            dataset=self.dataset,
            selectors={**self.selectors, axis: new_pos},
            attrs=self.attrs,
        )

    # ------------------------------------------------------------------
    # Selection operations
    # ------------------------------------------------------------------

    def query(self, axis: str, expr: str) -> "TensorView":
        """Filter *axis* by a pandas query expression on its metadata.

        Example::

            view.query("instance", "label == 'spindle' and qc_pass")
        """
        meta = self.dataset.axis_meta[axis]
        pos = self._sel(axis)
        sub = meta.iloc[pos].copy().reset_index(drop=True)
        sub["_pos"] = pos
        filtered = sub.query(expr)
        return self._replace(axis, filtered["_pos"].values)

    def filter(self, axis: str, ids_or_mask) -> "TensorView":
        """Filter *axis* by a boolean mask or an array of id values.

        When *ids_or_mask* is a boolean array it is applied directly to the
        current selection. Otherwise it is treated as a list of id values and
        matched against the column named in ``manifest['axis_index_columns']``.
        """
        pos = self._sel(axis)
        ids_or_mask = np.asarray(ids_or_mask)

        if ids_or_mask.dtype == bool:
            return self._replace(axis, pos[ids_or_mask[: len(pos)]])

        id_col = self.dataset.manifest.get("axis_index_columns", {}).get(axis)
        if id_col is None:
            raise ValueError(
                f"No id column defined for axis '{axis}'. "
                "Set 'axis_index_columns' in the dataset manifest."
            )
        current_ids = self.dataset.axis_meta[axis].iloc[pos][id_col].values
        mask = np.isin(current_ids, ids_or_mask)
        return self._replace(axis, pos[mask])

    def slice(self, axis: str, start: int, stop: int) -> "TensorView":
        """Slice *axis* to positions [start, stop)."""
        return self._replace(axis, self._sel(axis)[start:stop])

    def take(self, axis: str, indices) -> "TensorView":
        """Select *axis* by an array of relative indices into the current view."""
        pos = self._sel(axis)
        return self._replace(axis, pos[np.asarray(indices)])

    def materialize(self) -> AxisAnnotatedTensor:
        """Load and return the selected data as an AxisAnnotatedTensor."""
        data = np.asarray(self.dataset.store)
        axes = self.dataset.axes
        selectors = [self._sel(ax) for ax in axes]

        result_data = data[np.ix_(*selectors)]

        id_cols = self.dataset.manifest.get("axis_index_columns", {})
        axis_index: dict[str, np.ndarray] = {}
        axis_meta: dict[str, pd.DataFrame] = {}

        for ax, sel in zip(axes, selectors):
            meta = self.dataset.axis_meta.get(ax, pd.DataFrame())
            sub = meta.iloc[sel].reset_index(drop=True)
            axis_meta[ax] = sub
            id_col = id_cols.get(ax)
            if id_col and id_col in sub.columns:
                axis_index[ax] = sub[id_col].values
            else:
                axis_index[ax] = np.arange(len(sel))

        return AxisAnnotatedTensor(
            data=result_data,
            axes=axes,
            axis_index=axis_index,
            axis_meta=axis_meta,
            attrs={**self.dataset.attrs, **self.attrs},
        )

    def __repr__(self) -> str:
        shape = tuple(len(self._sel(ax)) for ax in self.dataset.axes)
        return f"TensorView(axes={self.dataset.axes}, shape={shape})"


# ------------------------------------------------------------------
# Convenience constructors
# ------------------------------------------------------------------


def make_dataset(
    data: np.ndarray,
    axes: list[str],
    axis_meta: dict[str, pd.DataFrame] | None = None,
    axis_index_columns: dict[str, str] | None = None,
    attrs: dict | None = None,
) -> TensorDataset:
    """Build an in-memory TensorDataset from a numpy array.

    Parameters
    ----------
    data:
        Numeric array.
    axes:
        Axis names in the same order as ``data``'s dimensions.
    axis_meta:
        Optional per-axis DataFrames. Defaults to a single ``{name}_id`` column.
    axis_index_columns:
        Maps axis name → column name used as the stable join key.
    attrs:
        Global attributes.
    """
    data = np.asarray(data)
    if axis_meta is None:
        axis_meta = {}
    for i, ax in enumerate(axes):
        if ax not in axis_meta:
            axis_meta[ax] = pd.DataFrame({f"{ax}_id": np.arange(data.shape[i])})

    manifest: dict = {
        "schema_version": "0.1.0",
        "axes": axes,
        "shape": list(data.shape),
        "dtype": str(data.dtype),
        "axis_index_columns": axis_index_columns or {},
    }
    return TensorDataset(
        manifest=manifest,
        store=data,
        axis_meta=axis_meta,
        attrs=attrs or {},
    )


def concat_datasets(
    datasets: list[TensorDataset],
    along: str = "instance",
) -> TensorDataset:
    """Concatenate datasets along one axis.

    All datasets must share the same ``axes`` list. Non-concatenated axes
    are taken from the first dataset.
    """
    if not datasets:
        raise ValueError("No datasets to concatenate.")
    axes = datasets[0].axes
    if not all(d.axes == axes for d in datasets):
        raise ValueError("All datasets must have the same axes list.")

    concat_pos = axes.index(along)
    arrays = [np.asarray(d.store) for d in datasets]
    combined_data = np.concatenate(arrays, axis=concat_pos)

    meta_parts = [d.axis_meta[along] for d in datasets]
    combined_along_meta = pd.concat(meta_parts, ignore_index=True)

    axis_meta: dict[str, pd.DataFrame] = {along: combined_along_meta}
    for ax in axes:
        if ax != along:
            axis_meta[ax] = datasets[0].axis_meta[ax]

    manifest = {
        **datasets[0].manifest,
        "shape": list(combined_data.shape),
    }
    return TensorDataset(
        manifest=manifest,
        store=combined_data,
        axis_meta=axis_meta,
        attrs=datasets[0].attrs,
    )
