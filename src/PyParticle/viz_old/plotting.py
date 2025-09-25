"""
This file is part of the Flexible Plotting Package.

Role:
- plotting: Functions that take a single axis (`ax`) and plot data onto it.
- This module should not apply formatting (titles/labels/legends); it only
  creates and returns artists.
"""

from typing import Sequence, Optional, Tuple, Union, Dict, Any
import matplotlib.axes
import matplotlib.pyplot as plt
from ..analysis import compute_plotdat, build_default_var_cfg
from . import data_prep
from ..population.base import ParticlePopulation  # forward reference in function signatures
# Avoid importing ParticlePopulation at module import time to prevent
# import cycles when package top-level imports happen in examples/tests.
# We use a forward reference in the function signature below.
# fixme: link directly with PyParticle populations

# def plot_scatter(varnames,
#                  particle_population: ParticlePopulation,
#                  var_cfgs: Optional[Sequence[Dict[str, Any]]] = None,
#                  colormap: Optional[str] = None,
#                  size: Union[float, Sequence[float]] = 20,
#                  ax: Optional[matplotlib.axes.Axes] = None):
#     """Plot a scatter of per-partcle variables `ax` and return the created PathCollection artist.
#     """
#     if ax is None:
#         fig, ax = plt.subplots()
    
#     dat = data_prep.prepare_particle_variable(particle_population, varnames, var_cfgs)
#     x, y, c, s, labs = dat["x"], dat["y"], dat["c"], dat["s"], dat["labs"]
#     # todo: 3D scatter later?
#     xscale, yscale, cscale = dat.get("xscale", "linear"), dat.get("yscale", "linear"), dat.get('cscale', 'linear')
    
#     if c is not None and colormap is None:
#         colormap = 'viridis'
#     cmap = plt.get_cmap(colormap) if colormap else None
#     alpha = 0.7 if c is not None else 1.0
#     edgecolors = 'none' if c is not None else 'face'
#     vmin, vmax = None, None
#     if cscale == 'log' and c is not None:
#         vmin = max(min(c[c > 0]), 1e-5)
#         vmax = max(c)
#     hsc = ax.scatter(x, y, c=c, s=s, cmap=cmap, alpha=alpha, edgecolors=edgecolors, vmin=vmin, vmax=vmax)
#     ax.set_xlabel(labs[0])
#     ax.set_ylabel(labs[1])
#     if labs[2]:
#         cbar = plt.colorbar(mappable=ax.collections[0], ax=ax)
#         cbar.set_label(labs[2])
#     # fixme: not working with scale

#     ax.set_xscale(xscale)
#     ax.set_yscale(yscale)
#     return fig, ax, hsc, dat

# fixme: this is lines for a given state; add vs. time/height as different plot
# fixme: separate distributions as their own plot? 
def plot_lines(varname,
               particle_populations: Tuple[ParticlePopulation, ...],
               var_cfg: Optional[dict] = None,
               ax: Optional[matplotlib.axes.Axes] = None,
               colors: Optional[Union[str, list]] = None,
               linestyles: Optional[Union[str, list]] = None,
               linewidths: Optional[Union[float, list]] = None,
               markers: Optional[Union[str, list]] = None):
    """Plot a line on `ax` and return the created Line2D artist.

    Notes
    - This function does not modify axis labels, titles or legends. Formatting
      should be applied separately via formatting utilities.
    """
    
    if ax is None:
        fig, ax = plt.subplots()

    last_line = None
    last_labs = ("", "")

    for ii, particle_population in enumerate(particle_populations):
        color = None
        if isinstance(colors, str):
            color = colors
        elif isinstance(colors, list):
            color = colors[ii]

        linestyle = '-'
        if isinstance(linestyles, str):
            linestyle = linestyles
        elif isinstance(linestyles, list):
            linestyle = linestyles[ii]

        linewidth = 1.5
        if isinstance(linewidths, (int, float)):
            linewidth = linewidths
        elif isinstance(linewidths, list):
            linewidth = linewidths[ii]

        marker = None
        if isinstance(markers, str):
            marker = markers
        elif isinstance(markers, list):
            marker = markers[ii]

        default_var_cfg = build_default_var_cfg(varname)
        # merge user var_cfg with defaults (do not mutate caller dict)
        merged_cfg = dict(default_var_cfg)
        if var_cfg:
            for k, v in var_cfg.items():
                merged_cfg[k] = v

        # Delegate to analysis.compute_plotdat which returns a PlotDat dict
        if varname == "Ntot":
            dat = {"x": None, "y": None, "labs": ("", ""), "xscale": "linear", "yscale": "linear"}
        else:
            # compute_plotdat expects canonical/alias names and a var_cfg dict
            dat = compute_plotdat(particle_population, varname, merged_cfg)

        x, y, labs = dat["x"], dat["y"], dat["labs"]
        xscale, yscale = dat.get("xscale", "linear"), dat.get("yscale", "linear")

        last_labs = labs
        last_line, = ax.plot(x, y, color=color, linestyle=linestyle, linewidth=linewidth, marker=marker)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)

    return last_line, last_labs


## NOTE: The former DataFrame-based helper plot_grid_from_df was removed to
## simplify dependencies (no pandas). If needed in the future, recover from
## history or implement a light adapter that consumes iterable plot dicts.
