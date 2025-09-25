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
# New grids helpers
from .grids import (
    make_grid_popvars,
    make_grid_scenarios_timesteps,
    make_grid_mixed,
)
from .grid_scenarios_variables import make_grid_scenarios_variables_same_timestep
# Backward-compatible aliases: map older conceptual names to new helpers
make_grid_columns = make_grid_popvars
make_grid_time_snapshots = make_grid_scenarios_timesteps
make_grid_rows = make_grid_mixed

__all__ = [
    "make_grid",
    "plot_lines",
    "make_grid_popvars",
    "make_grid_scenarios_timesteps",
    "make_grid_mixed",
  "make_grid_scenarios_variables_same_timestep",
    "make_grid_time_snapshots",
    "make_grid_rows",
    "make_grid_columns",
    "get_colors",
    "get_linestyles",
    "format_axes",
    "add_legend",
]
