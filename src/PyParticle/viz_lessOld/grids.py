from matplotlib.gridspec import GridSpec
from matplotlib import pyplot as plt
from typing import Tuple, List, Union, Optional
from .state.line import plot_lines
from ..population.base import ParticlePopulation

def make_grid_onevariable(
        varcfg: dict,
        populations_grid: Tuple[Tuple[ParticlePopulation, ...], ...],
        *, 
        linespec_dicts: Tuple[Tuple[dict], ...] = (),
        figsize: tuple = (8, 6),
        hspace: float | None = 0.3,
        wspace: float | None = 0.3,
):
    """Create a grid of subplots for a single variable across multiple scenarios.

    Parameters
    ----------
    varcfg : dict
        Configuration dictionary for the variable to be plotted.
    populations_grid : list of list of PopulationLike
        2D list where each sublist represents a row of populations (scenarios).
    figsize : tuple, optional
        Size of the entire figure (width, height), by default (8, 6).
    hspace : float, optional
        Height space between subplots, by default 0.3.
    wspace : float, optional
        Width space between subplots, by default 0.3.

    Returns
    -------
"""
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(nrows, ncols, figure=fig)
    gs.update(hspace=hspace, wspace=wspace)
    
    nrows = len(populations_grid)
    axs = np.empty((nrows, max(len(row) for row in populations_grid)), dtype=object) 
    for row in range(nrows):
        ncols = len(populations_grid[row])
        for col in range(ncols):
            ax = fig.add_subplot(gs[row, col])
            populations = populations_grid[row][col]
            linespecs_dict = linespec_dicts[row][col] if row < len(line_specs) and col < len(line_specs[row]) else {}

            lines = plot_lines(varname=varcfg['name'],
                       particle_populations=populations,
                       var_cfg=varcfg,
                       ax=ax,
                       **linespecs_dict)
            ax.set_title(f"Row {row+1}, Col {col+1}")
            ax.grid(True)
            axs[row, col] = ax
    
    fig.tight_layout(h_pad=hspace, w_pad=wspace)
    return fig, axs, lines
