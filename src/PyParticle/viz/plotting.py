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
from ..population.base import ParticlePopulation  
from ..analysis import compute_variable, build_default_var_cfg

# fixme: link directly with PyParticle populations

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
        fig,ax=plt.subplots()
    
    for ii,particle_population in enumerate(particle_populations):
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
    if var_cfg is None:
        var_cfg = default_var_cfg
    else:
        # merge with defaults
        for k, v in default_var_cfg.items():
            var_cfg.setdefault(k, v)
    x, y, labs, xscale, yscale = compute_variable(particle_population, varname, var_cfg, return_plotdat=True)
    line, = ax.plot(x, y, color=color, linestyle=linestyle,
                    linewidth=linewidth, marker=marker)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    
    return line, labs
