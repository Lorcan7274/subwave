from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class AxisAnnotatedTensor:
    """Numeric tensor with named axes, coordinate arrays, and per-axis metadata.

    Parameters
    ----------
    data:
        Numeric array. Shape must match ``len(axes)``.
    axes:
        Ordered axis names, e.g. ``['instance', 'sample']`` or
        ``['instance', 'sample', 'channel']``.
    axis_index:
        Optional coordinate array for each axis (e.g. time offsets, event ids).
        Missing entries are filled with ``arange(n)`` on construction.
    axis_meta:
        Per-axis metadata DataFrame (one row per element of that axis).
        Missing entries are filled with a single ``{name}_index`` column.
    attrs:
        Global scalar attributes (sfreq, units, provenance, …).
    """

    data: np.ndarray
    axes: list[str]
    axis_index: dict[str, np.ndarray] = field(default_factory=dict)
    axis_meta: dict[str, pd.DataFrame] = field(default_factory=dict)
    attrs: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data)
        if len(self.axes) != self.data.ndim:
            raise ValueError(
                f"len(axes)={len(self.axes)} does not match data.ndim={self.data.ndim}"
            )
        for i, ax in enumerate(self.axes):
            n = self.data.shape[i]
            if ax not in self.axis_index or self.axis_index[ax] is None:
                self.axis_index[ax] = np.arange(n)
            if ax not in self.axis_meta:
                self.axis_meta[ax] = pd.DataFrame({f"{ax}_index": np.arange(n)})

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def axis_pos(self, name: str) -> int:
        """Return the integer position of axis *name*."""
        try:
            return self.axes.index(name)
        except ValueError:
            raise ValueError(f"Axis {name!r} not found. Available: {self.axes}")

    def __repr__(self) -> str:
        dims = " x ".join(
            f"{ax}({self.data.shape[i]})" for i, ax in enumerate(self.axes)
        )
        return f"AxisAnnotatedTensor({dims}, dtype={self.data.dtype})"
