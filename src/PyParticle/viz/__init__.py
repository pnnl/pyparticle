"""
This file is part of the Flexible Plotting Package (viz).

Role:
- Package initializer for the `viz` package. Exposes a compact API for
  layout, plotting, styling and formatting utilities.
"""

from .layout import make_grid
from .plotting import plot_lines
from .styling import get_colors, get_linestyles
from .formatting import format_axes, add_legend

__all__ = [
    "make_grid",
  "plot_line",
  "plot_lines",
    "get_colors",
    "get_linestyles",
    "format_axes",
    "add_legend",
]
