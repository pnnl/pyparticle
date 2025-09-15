"""Deterministic plotting and numerical helper utilities for PyParticle examples.

Provides deterministic diameter/radius grids, lognormal discretization, and
helpers to compute activation fractions and perform fixed quadrature.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


def fixed_radius_grid(r_min_m: float = 1e-9, r_max_m: float = 2e-6, n_bins: int = 100) -> np.ndarray:
    """Return log-spaced radius grid (meters)."""
    return np.logspace(np.log10(r_min_m), np.log10(r_max_m), n_bins)


def lognormal_mode_to_bins(r0_m: float, sigma: float, Ntot: float, n_bins: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Discretize a lognormal mode into diameters (m) and number concentrations (#/m3) per bin.

    r0_m: geometric mean radius (m)
    sigma: geometric standard deviation (dimensionless)
    Ntot: total number concentration (#/m3)
    Returns (D_grid_m, n_D) where n_D sums to Ntot.
    """
    # Build radius grid around r0 covering +/- 4 sigma in log space
    lnmu = np.log(r0_m)
    lnstd = np.log(sigma)
    r_min = np.exp(lnmu - 4 * lnstd)
    r_max = np.exp(lnmu + 4 * lnstd)
    r = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_bins))
    # pdf of lognormal in terms of r
    pdf = (1.0 / (r * lnstd * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((np.log(r) - lnmu) / lnstd) ** 2)
    # approximate bin widths in r space using log midpoints
    r_edges = np.exp(np.linspace(np.log(r_min), np.log(r_max), n_bins + 1))
    bin_widths = r_edges[1:] - r_edges[:-1]
    n_per_bin = pdf * bin_widths
    # normalize to Ntot
    n_per_bin = n_per_bin / n_per_bin.sum() * float(Ntot)
    D_grid = 2.0 * r
    return D_grid, n_per_bin


def activation_fraction_from_s_grid(D_grid: np.ndarray, n_D: np.ndarray, s_grid: np.ndarray, s_crit_func) -> np.ndarray:
    """Compute activation fraction vs supersaturation grid.

    s_crit_func(D) -> s_crit_fraction (0..1)
    """
    activation = np.zeros_like(s_grid, dtype=float)
    for i, S in enumerate(s_grid):
        scrit = s_crit_func(D_grid)
        activated = scrit <= S
        activation[i] = float((n_D[activated].sum() / n_D.sum()))
    return activation
