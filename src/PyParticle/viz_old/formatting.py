"""
This file is part of the Flexible Plotting Package.

Role:
- formatting: Functions to format plots and axes (titles, labels, ticks,
  legends, grids). Works on existing `ax` objects and modifies them in-place.
"""

from typing import Optional, Tuple
import matplotlib.axes
import matplotlib.pyplot as plt


def format_axes(ax: matplotlib.axes.Axes, xlabel: Optional[str] = None,
                ylabel: Optional[str] = None, title: Optional[str] = None,
                xlim: Optional[Tuple[float, float]] = None,
                ylim: Optional[Tuple[float, float]] = None,
                grid: bool = True, tick_labelsize: int = 10, title_size: int = 12):
    """Apply common formatting to `ax` in-place.

    Parameters are optional; only provided values are applied.
    """
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=tick_labelsize)
    ax.grid(grid)


def add_legend(ax: matplotlib.axes.Axes, loc: str = 'best', fontsize: int = 10):
    """Add a legend to `ax` if any labeled artists are present.

    Returns the legend instance or None.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    return ax.legend(loc=loc, fontsize=fontsize)
