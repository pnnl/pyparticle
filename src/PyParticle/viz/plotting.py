"""
This file is part of the Flexible Plotting Package.

Role:
- plotting: Functions that take a single axis (`ax`) and plot data onto it.
- This module should not apply formatting (titles/labels/legends); it only
  creates and returns artists.
"""

from typing import Sequence, Optional, Tuple, Union
import matplotlib.axes
import matplotlib.pyplot as plt
from . import data_prep
# Avoid importing ParticlePopulation at module import time to prevent
# import cycles when package top-level imports happen in examples/tests.
# We use a forward reference in the function signature below.
# fixme: link directly with PyParticle populations

def plot_lines(varname,
               particle_populations: Tuple["ParticlePopulation", ...],
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

        default_var_cfg = data_prep.build_default_var_cfg(varname)
        # merge user var_cfg with defaults (do not mutate caller dict)
        merged_cfg = dict(default_var_cfg)
        if var_cfg:
            for k, v in var_cfg.items():
                merged_cfg[k] = v

        # Dispatch directly to the new data_prep helpers instead of older analysis
        if varname == "dNdlnD":
            dat = data_prep.prepare_dNdlnD(particle_population, merged_cfg)
        elif varname == "Nccn":
            dat = data_prep.prepare_Nccn(particle_population, merged_cfg)
        elif varname == "frac_ccn":
            dat = data_prep.prepare_frac_ccn(particle_population, merged_cfg)
        elif varname in ["b_abs", "b_scat", "b_ext", "total_abs", "total_scat", "total_ext"]:
            if merged_cfg.get("vs_rh", False):
                dat = data_prep.prepare_optical_vs_rh(particle_population, dict(merged_cfg, coeff=varname))
            else:
                dat = data_prep.prepare_optical_vs_wvl(particle_population, dict(merged_cfg, coeff=varname))
        elif varname == "Ntot":
            dat = {"x": None, "y": None, "labs": ("", ""), "xscale": "linear", "yscale": "linear"}
        else:
            raise NotImplementedError(f"varname={varname} not yet implemented in plotting.plot_lines")

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
