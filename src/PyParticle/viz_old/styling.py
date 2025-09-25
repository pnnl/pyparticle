"""
This file is part of the Flexible Plotting Package.

Role:
- styling: Functions to manage plot styles (colors, colormaps, line styles).
- Utilities return style parameters but do not modify plots directly.
"""

from typing import List
import matplotlib.pyplot as plt
import matplotlib as mpl


def get_colors(n: int, cmap: str = 'viridis') -> List[str]:
    """Return a list of `n` colors sampled from the given colormap.

    If n == 1, returns a single color (middle of the colormap).
    """
    cmap_obj = plt.get_cmap(cmap)
    if n <= 1:
        return [mpl.colors.to_hex(cmap_obj(0.5))]
    return [mpl.colors.to_hex(cmap_obj(i / (n - 1))) for i in range(n)]


def get_linestyles(n: int):
    """Return a cycling list of linestyles for `n` lines.

    The list is deterministic and suitable for use in plotting functions.
    """
    base = ['-', '--', '-.', ':']
    if n <= len(base):
        return base[:n]
    # repeat patterns if more lines are requested
    out = []
    i = 0
    while len(out) < n:
        out.append(base[i % len(base)])
        i += 1
    return out
