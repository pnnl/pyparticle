"""
This file is part of the Flexible Plotting Package.

Role:
- layout: Functions to create structured figure layouts (using plt.subplots
  or matplotlib.gridspec.GridSpec).
- This file should only handle layout responsibilities and return fig and
  axes structures for plotting functions to consume.
"""

from typing import Tuple, Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def make_grid(rows: int = 1, cols: int = 1, figsize: Tuple[int, int] = (8, 6),
              hspace: float = 0.3, wspace: float = 0.3,
              layout_spec: Optional[Sequence[Sequence[int]]] = None,
              sharex: bool = False, sharey: bool = False) -> Tuple[plt.Figure, np.ndarray]:
    """Create a figure and a structured grid of axes.

    Parameters
    - rows, cols: basic grid size. If `layout_spec` is provided it will be used
      to create uneven grids. `layout_spec` should be a sequence with length
      equal to `rows`, where each element is either an int (number of cols for
      that row) or a sequence describing colspan per cell.

    Returns
    - fig: matplotlib.figure.Figure
    - axes: a 2D NumPy array of Axes objects with shape (rows, max_cols)

    Examples
    >>> fig, axes = make_grid(2, 2)
    >>> fig, axes = make_grid(2, 2, layout_spec=[[1,1],[2]])  # second row: one wide
    """

    fig = plt.figure(figsize=figsize)

    if layout_spec is None:
        gs = GridSpec(rows, cols, figure=fig)
        axes = np.empty((rows, cols), dtype=object)
        for r in range(rows):
            for c in range(cols):
                if sharex and r > 0:
                    axes[r, c] = fig.add_subplot(gs[r, c], sharex=axes[0, c])
                elif sharey and c > 0:
                    axes[r, c] = fig.add_subplot(gs[r, c], sharey=axes[r, 0])
                else:
                    axes[r, c] = fig.add_subplot(gs[r, c])
    else:
        # layout_spec provided: allow rows with variable number of cells
        max_cols = max(len(rspec) if hasattr(rspec, '__len__') else 1 for rspec in layout_spec)
        axes = np.empty((rows, max_cols), dtype=object)
        gs = GridSpec(rows, max_cols, figure=fig)
        for r, rspec in enumerate(layout_spec):
            if isinstance(rspec, int):
                # rspec = number of equal columns in this row
                ncols_row = rspec
                for c in range(ncols_row):
                    axes[r, c] = fig.add_subplot(gs[r, c])
                for c in range(ncols_row, max_cols):
                    axes[r, c] = None
            else:
                # assume rspec is iterable of colspans
                col = 0
                for span in rspec:
                    if span <= 0:
                        raise ValueError("colspan should be >=1")
                    axes[r, col] = fig.add_subplot(gs[r, col: col + span])
                    for emptyc in range(col + span - 1):
                        axes[r, emptyc + col + 1] = None
                    col += span
                for c in range(col, max_cols):
                    axes[r, c] = None

    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    return fig, axes
