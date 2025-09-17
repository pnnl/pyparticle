"""
Grid helpers for common visualization layouts used in examples.

Provides three convenience helpers (each returns (fig, axarr)):

- make_grid_popvars(rows, columns, ...):
  Type A — rows = populations or config dicts, cols = variable names
- make_grid_scenarios_timesteps(rows, columns, variables, ...):
  Type B — rows = scenario/config dicts, cols = timesteps
- make_grid_mixed(rows, columns, ...):
  Type C — rows may mix prebuilt Population objects and config dicts;
  cols = variable names

These helpers call into the package plotting utilities (`plot_lines`,
`format_axes`, `add_legend`) and `build_population` when a config dict is
provided. They are intentionally small wrappers that create a layout via
`make_grid` and populate each axis.
"""

from typing import Sequence, Optional, Tuple, List, Union, Iterable
import copy
import matplotlib.pyplot as plt

from .layout import make_grid
from .plotting import plot_lines
from .formatting import format_axes, add_legend
from ..population import build_population
from ..population.base import ParticlePopulation


def _ensure_population(item, time=None):
    """Return a ParticlePopulation instance.

    If `item` is already a ParticlePopulation, return it. If it's a dict,
    copy and set `timestep` when provided, then call `build_population`.
    """
    if isinstance(item, ParticlePopulation):
        return item
    if isinstance(item, dict):
        cfg = dict(item)
        if time is not None:
            cfg["timestep"] = time
        return build_population(cfg)
    raise TypeError("rows entries must be ParticlePopulation or config dict")


def make_grid_popvars(rows: Sequence[Union[ParticlePopulation, dict]],
                      columns: Sequence[str],
                      var_cfg: Optional[dict] = None,
                      time: Optional[float] = None,
                      figsize: Tuple[int, int] = (10, 6),
                      hspace: float = 0.3, wspace: float = 0.3,
                      hide_spines: bool = True,
                      grid: bool = False,
                      sharex_columns: bool = True,
                      sharey_rows: bool = True):
    """Type A: rows = populations or config dicts, cols = variable names.

    For each cell (i, j): obtain population for rows[i] (build if config),
    then call plot_lines(columns[j], (pop,), var_cfg, ax). Applies
    `format_axes` and `add_legend`.
    Returns (fig, axarr) where axarr is a 2D array of axes.
    """
    nrows = len(rows)
    ncols = len(columns)
    fig, axarr = make_grid(nrows, ncols, figsize=figsize, hspace=hspace, wspace=wspace)

    for i, row_item in enumerate(rows):
        # build/populate once per row
        pop = _ensure_population(row_item, time=time)
        for j, varname in enumerate(columns):
            ax = axarr[i, j]
            line, labs = plot_lines(varname, (pop,), var_cfg, ax=ax)
            xlabel = labs[0] if isinstance(labs, (list, tuple)) and len(labs) > 0 else None
            ylabel = labs[1] if isinstance(labs, (list, tuple)) and len(labs) > 1 else None
            title = None

            # Heuristics for sensible axis scales and formatting:
            # - size distributions (dNdlnD or containing 'D') -> log x-scale
            # - keep wavelength-related variables on linear scale
            if isinstance(varname, str) and ("dNdlnD" in varname or "D" in varname or "diam" in varname.lower()):
                try:
                    ax.set_xscale("log")
                except Exception:
                    pass

            format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid)
            # remove top/right spines by default
            if hide_spines:
                try:
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                except Exception:
                    pass
            add_legend(ax)

    # Optionally join axes to share x across columns and y across rows
    # Join only existing axes (ignore None entries)
    if sharex_columns:
        for c in range(ncols):
            col_axes = [axarr[r, c] for r in range(nrows) if axarr[r, c] is not None]
            if len(col_axes) > 1:
                try:
                    col_axes[0].get_shared_x_axes().join(*col_axes)
                except Exception:
                    pass
    if sharey_rows:
        for r in range(nrows):
            row_axes = [axarr[r, c] for c in range(ncols) if axarr[r, c] is not None]
            if len(row_axes) > 1:
                try:
                    row_axes[0].get_shared_y_axes().join(*row_axes)
                except Exception:
                    pass

    return fig, axarr


def make_grid_scenarios_timesteps(rows: Sequence[dict],
                                  columns: Sequence[Union[int, float]],
                                  variables: Sequence[str],
                                  var_cfg: Optional[Union[dict, dict]] = None,
                                  figsize: Tuple[int, int] = (10, 6),
                                  hspace: float = 0.3, wspace: float = 0.3,
                                  hide_spines: bool = True,
                                  grid: bool = False,
                                  sharex_columns: bool = True,
                                  sharey_rows: bool = True):
    """Type B: rows = scenario/config dicts, cols = timesteps.

    Each cell shows one scenario at one timestep. `variables` can be a
    sequence of varnames; all are plotted on the same axis (multiple curves).
    `var_cfg` may be a single dict applied to all variables or a mapping
    varname->cfg.
    """
    nrows = len(rows)
    ncols = len(columns)
    fig, axarr = make_grid(nrows, ncols, figsize=figsize, hspace=hspace, wspace=wspace)

    for i, scenario in enumerate(rows):
        for j, timestep in enumerate(columns):
            cfg = dict(scenario)
            cfg["timestep"] = timestep
            pop = build_population(cfg)
            ax = axarr[i, j]
            for var in variables:
                # support var_cfg as single cfg or mapping
                cfg_for_var = None
                if isinstance(var_cfg, dict) and var in var_cfg:
                    cfg_for_var = var_cfg[var]
                elif isinstance(var_cfg, dict) and all(k in var_cfg for k in variables):
                    # interpret as mapping var->cfg
                    cfg_for_var = var_cfg.get(var, None)
                else:
                    cfg_for_var = None

                line, labs = plot_lines(var, (pop,), cfg_for_var, ax=ax)
            # title and labels
            title = f"t={timestep}"
            xlabel = None
            ylabel = None
            if len(variables) > 0 and isinstance(variables[-1], str):
                # try to extract labels from last plot_lines call
                if isinstance(labs, (list, tuple)) and len(labs) > 0:
                    xlabel = labs[0]
                    ylabel = labs[1] if len(labs) > 1 else None

            format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid)
            if hide_spines:
                try:
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                except Exception:
                    pass
            add_legend(ax)

    # share axes optionally
    nrows = len(rows)
    ncols = len(columns)
    if sharex_columns:
        for c in range(ncols):
            col_axes = [axarr[r, c] for r in range(nrows) if axarr[r, c] is not None]
            if len(col_axes) > 1:
                try:
                    col_axes[0].get_shared_x_axes().join(*col_axes)
                except Exception:
                    pass
    if sharey_rows:
        for r in range(nrows):
            row_axes = [axarr[r, c] for c in range(ncols) if axarr[r, c] is not None]
            if len(row_axes) > 1:
                try:
                    row_axes[0].get_shared_y_axes().join(*row_axes)
                except Exception:
                    pass

    return fig, axarr


def make_grid_mixed(rows: Sequence[Union[ParticlePopulation, dict]],
                    columns: Sequence[str],
                    var_cfg: Optional[dict] = None,
                    figsize: Tuple[int, int] = (10, 6),
                    hspace: float = 0.3, wspace: float = 0.3,
                    hide_spines: bool = True,
                    grid: bool = False,
                    sharex_columns: bool = True,
                    sharey_rows: bool = True):
    """Type C: like Type A but rows may mix populations and config dicts.

    Behavior: for each row entry, if it's a dict build a population (using any
    timestep present in the dict). If it's already a Population, use it
    directly. Columns are variable names; each cell plots a single variable.
    """
    # Reuse Type A implementation because it already accepts dicts or pops
    return make_grid_popvars(rows, columns, var_cfg=var_cfg, time=None,
                             figsize=figsize, hspace=hspace, wspace=wspace,
                             hide_spines=hide_spines, grid=grid,
                             sharex_columns=sharex_columns, sharey_rows=sharey_rows)
