from __future__ import annotations
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np

from PyParticle.analysis.prepare import compute_plotdat

PopulationLike = Any  # your ParticlePopulation type


def plot_lines(
    ax,
    populations: Tuple[PopulationLike, ...],
    varname: str, 
    var_cfg: Optional[dict] = None,
    *,
    colors: Optional[Union[str, List[str]]] = None,
    linestyles: Optional[Union[str, List[str]]] = None,
    linewidths: Optional[Union[float, List[float]]] = None,
    markers: Optional[Union[str, List[str]]] = None,
):
    """
    Plot one line per population for (varname, var_cfg) onto `ax`.

    Uses analysis.prepare.compute_plotdat(pop, varname, var_cfg) to obtain:
      - x: 1D array or None (if None, uses 0..N-1)
      - y: 1D array
      - labs: [xlabel, ylabel]
      - xscale, yscale: 'linear' | 'log'
    Returns the last matplotlib Line2D handle.
    """

    lines = []
    plotdats = []
    for ii, population in enumerate(populations):
        line, plotdat = plot_single_line(
            ax,
            population,
            varname,
            var_cfg,
            label=None,
            color=(colors if isinstance(colors, str) else (colors[ii] if isinstance(colors, list) else None)),
            linestyle=(linestyles if isinstance(linestyles, str) else (linestyles[ii] if isinstance(linestyles, list) else None)),
            linewidth=(linewidths if isinstance(linewidths, (int, float)) else (linewidths[ii] if isinstance(linewidths, list) else None)),
            marker=(markers if isinstance(markers, str) else (markers[ii] if isinstance(markers, list) else None)),
        )
        print(plotdat)
        lines.append(line)
        plotdats.append(plotdat)
    return lines, plotdats

def plot_single_line(
    ax,
    population: PopulationLike,
    varname: str,
    var_cfg: Optional[dict] = None,
    *,
    label: Optional[str] = None,
    color: Optional[str] = None,
    linestyle: Optional[str] = None,
    linewidth: Optional[float] = None,
    marker: Optional[str] = None
):
    """
    Plot exactly one line for (population, varname, var_cfg) onto `ax`.

    Uses analysis.prepare.compute_plotdat(pop, varname, var_cfg) to obtain:
      - x: 1D array or None (if None, uses 0..N-1)
      - y: 1D array
      - labs: [xlabel, ylabel]
      - xscale, yscale: 'linear' | 'log'
    Returns the matplotlib Line2D handle.
    """
    plotdat = compute_plotdat(population, varname, var_cfg or {})

    y = np.asarray(plotdat["y"])
    x = np.arange(y.shape[0]) if plotdat.get("x") is None else np.asarray(plotdat["x"])

    if x.ndim != 1 or y.ndim != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Expected 1D x/y of equal length, got x={x.shape}, y={y.shape}")

    (line,) = ax.plot(
        x,
        y,
        label=label,
        color=color,
        linestyle=linestyle,
        linewidth=linewidth,
        marker=marker,
    )

    
    labs = plotdat.get("labs", ["", ""])
    xlab, ylab = (labs + ["", ""])[:2]
    if xlab:
        ax.set_xlabel(xlab)
    if ylab:
        ax.set_ylabel(ylab)
    ax.set_xscale(plotdat.get("xscale", "linear"))
    ax.set_yscale(plotdat.get("yscale", "linear"))

    return line, plotdat

