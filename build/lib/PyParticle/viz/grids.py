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


def make_grid_scenarios_models(
    scenarios: Sequence[dict],
    variables: Sequence[str],
    model_cfg_builders: Sequence,  # sequence of callables: (scenario_cfg) -> ParticlePopulation
    model_linestyles: Optional[Sequence[str]] = None,
    figsize: Tuple[int, int] = (10, 6),
    hspace: float = 0.3, wspace: float = 0.3,
    hide_spines: bool = True,
    grid: bool = False,
    sharex_columns: bool = True,
    sharey_rows: bool = True,
    colors: Optional[Sequence[str]] = None,
    var_cfg: Optional[dict] = None,
):
    """Grid: rows = scenarios, columns = variables, multiple models per axis.

    Each cell plots the requested `variables[j]` for all model populations
    built from `scenarios[i]` using the provided `model_cfg_builders`.

    Parameters
    ----------
    scenarios : list[dict]
        Scenario configuration dicts (each will be copied and passed to builders).
    variables : list[str]
        Variable names understood by `plot_lines`.
    model_cfg_builders : list[callable]
        Each callable receives a *copy* of the scenario dict and returns a
        ParticlePopulation.
    model_linestyles : list[str], optional
        Linestyles applied per model (cycled if fewer than number of models).
    colors : list[str], optional
        Row colors (one per scenario). If None, matplotlib default cycle used.
    var_cfg : dict or mapping var->cfg, optional
        Variable configuration(s) passed through to `plot_lines`.
    """
    import itertools
    import matplotlib.pyplot as plt

    nrows = len(scenarios)
    ncols = len(variables)
    fig, axarr = make_grid(nrows, ncols, figsize=figsize, hspace=hspace, wspace=wspace)

    default_cycle = plt.rcParams.get("axes.prop_cycle", None)
    if colors is None:
        if default_cycle is not None:
            cycle_list = default_cycle.by_key().get("color", ["C0"])
        else:
            cycle_list = ["C0", "C1", "C2", "C3"]
        colors = [cycle_list[i % len(cycle_list)] for i in range(nrows)]

    if model_linestyles is None:
        model_linestyles = ["-", "--", ":", "-"]

    for i, scenario in enumerate(scenarios):
        # Build all model populations once per row
        pops = []
        for builder in model_cfg_builders:
            cfg_copy = dict(scenario)
            pop = builder(cfg_copy)
            pops.append(pop)
        row_color = colors[i]
        for j, varname in enumerate(variables):
            ax = axarr[i, j]
            # resolve per-variable cfg if mapping provided
            cfg_for_var = None
            if isinstance(var_cfg, dict) and varname in var_cfg:
                cfg_for_var = var_cfg[varname]
            elif isinstance(var_cfg, dict) and all(k in var_cfg for k in variables):
                cfg_for_var = var_cfg.get(varname, None)
            # prepare color & linestyle lists
            line_colors = [row_color for _ in pops]
            linestyles = [model_linestyles[k % len(model_linestyles)] for k in range(len(pops))]
            _, labs = plot_lines(varname, tuple(pops), cfg_for_var, ax=ax, colors=line_colors, linestyles=linestyles)
            xlabel = labs[0] if isinstance(labs, (list, tuple)) and len(labs) > 0 else None
            ylabel = labs[1] if isinstance(labs, (list, tuple)) and len(labs) > 1 else None
            title = varname if j == 0 else None
            # Heuristic: set log x-scale for diameter-like variables (already handled in plot_lines
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=grid)
            if hide_spines:
                for sp in ("top", "right"):
                    if sp in ax.spines:
                        ax.spines[sp].set_visible(False)
            add_legend(ax)  # legend may be empty unless labels added upstream

    # axis sharing
    if sharex_columns:
        for c in range(ncols):
            col_axes = [axarr[r, c] for r in range(nrows)]
            if len(col_axes) > 1:
                try:
                    col_axes[0].get_shared_x_axes().join(*col_axes)
                except Exception:
                    pass
    if sharey_rows:
        for r in range(nrows):
            row_axes = [axarr[r, c] for c in range(ncols)]
            if len(row_axes) > 1:
                try:
                    row_axes[0].get_shared_y_axes().join(*row_axes)
                except Exception:
                    pass
    return fig, axarr


def make_grid_optics_vs_wvl(rows: Sequence[Union[ParticlePopulation, dict]],
                             coeffs: Sequence[str],
                             optics_cfg: dict,
                             figsize: Tuple[int, int] = (10, 6),
                             hspace: float = 0.3, wspace: float = 0.3,
                             hide_spines: bool = True,
                             sharex_columns: bool = True,
                             sharey_rows: bool = True):
    """Grid of optical coefficients vs wavelength.

    rows: populations or config dicts
    columns: optical coeff names (e.g. 'b_ext','b_scat','b_abs','g')
    optics_cfg merged into var_cfg passed to plot_lines per coeff.
    """
    nrows = len(rows)
    ncols = len(coeffs)
    fig, axarr = make_grid(nrows, ncols, figsize=figsize, hspace=hspace, wspace=wspace)

    base_cfg = dict(optics_cfg)
    base_cfg.pop("vs_rh", None)  # ensure wavelength mode

    for i, row_item in enumerate(rows):
        pop = _ensure_population(row_item)
        for j, coeff in enumerate(coeffs):
            ax = axarr[i, j]
            _, labs = plot_lines(coeff, (pop,), base_cfg, ax=ax)
            xlabel = labs[0] if labs else None
            ylabel = labs[1] if isinstance(labs, (list, tuple)) and len(labs) > 1 else None
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=coeff, grid=False)
            if hide_spines:
                for sp in ("top", "right"):
                    if sp in ax.spines:
                        ax.spines[sp].set_visible(False)
            add_legend(ax)

    if sharex_columns:
        for c in range(ncols):
            col_axes = [axarr[r, c] for r in range(nrows)]
            if len(col_axes) > 1:
                try:
                    col_axes[0].get_shared_x_axes().join(*col_axes)
                except Exception:
                    pass
    if sharey_rows:
        for r in range(nrows):
            row_axes = [axarr[r, c] for c in range(ncols)]
            if len(row_axes) > 1:
                try:
                    row_axes[0].get_shared_y_axes().join(*row_axes)
                except Exception:
                    pass
    return fig, axarr


def make_grid_optics_vs_rh(rows: Sequence[Union[ParticlePopulation, dict]],
                            coeffs: Sequence[str],
                            optics_cfg: dict,
                            figsize: Tuple[int, int] = (10, 6),
                            hspace: float = 0.3, wspace: float = 0.3,
                            hide_spines: bool = True,
                            sharex_columns: bool = True,
                            sharey_rows: bool = True):
    """Grid of optical coefficients vs RH (fix wavelength selection via optics_cfg['wvl_select'])."""
    nrows = len(rows)
    ncols = len(coeffs)
    fig, axarr = make_grid(nrows, ncols, figsize=figsize, hspace=hspace, wspace=wspace)

    base_cfg = dict(optics_cfg)
    base_cfg["vs_rh"] = True

    for i, row_item in enumerate(rows):
        pop = _ensure_population(row_item)
        for j, coeff in enumerate(coeffs):
            ax = axarr[i, j]
            _, labs = plot_lines(coeff, (pop,), base_cfg, ax=ax)
            xlabel = labs[0] if labs else None
            ylabel = labs[1] if isinstance(labs, (list, tuple)) and len(labs) > 1 else None
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=coeff, grid=False)
            if hide_spines:
                for sp in ("top", "right"):
                    if sp in ax.spines:
                        ax.spines[sp].set_visible(False)
            add_legend(ax)

    if sharex_columns:
        for c in range(ncols):
            col_axes = [axarr[r, c] for r in range(nrows)]
            if len(col_axes) > 1:
                try:
                    col_axes[0].get_shared_x_axes().join(*col_axes)
                except Exception:
                    pass
    if sharey_rows:
        for r in range(nrows):
            row_axes = [axarr[r, c] for c in range(ncols)]
            if len(row_axes) > 1:
                try:
                    row_axes[0].get_shared_y_axes().join(*row_axes)
                except Exception:
                    pass
    return fig, axarr
