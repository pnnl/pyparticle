"""PyParticle.analysis - functions and tools for analyzing particle populations.

This module provides a small set of analysis helpers used by the viz
pipeline. The implementations here are intentionally conservative: optics
work is delegated to the optics builder (`build_optical_population`) and
plotting helpers select a single RH or wavelength when a full 2-D optics
grid is returned so that plotting routines receive 1-D x/y arrays.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, Dict, Any, Tuple, Optional

from .viz import data_prep

# Thin wrappers to preserve the public API and import paths. The heavy
# lifting now lives in `viz.data_prep` but callers import from
# `PyParticle.analysis` so we keep wrappers with the same signatures.

def compute_particle_variable(particle_population, varname: str, var_cfg: Dict[str, Any], return_plotdat: bool = False):
    return data_prep.compute_particle_variable(particle_population, varname, var_cfg, return_plotdat=return_plotdat)
    
def compute_dNdlnD(
    particle_population,
    wetsize: bool = True,
    normalize: bool = False,
    method: str = "hist",
    N_bins: int = 30,
    D_min: float = 1e-9,
    D_max: float = 1e-4,
    diam_scale: str = "log",
) -> Dict[str, np.ndarray]:
    """Compute size distribution (dN/dlnD) for the particle population.

    Returns a dict with keys 'D' (bin centers) and 'dNdlnD' (values).
    """
    return data_prep.compute_dNdlnD(
        particle_population,
        wetsize=wetsize,
        normalize=normalize,
        method=method,
        N_bins=N_bins,
        D_min=D_min,
        D_max=D_max,
        diam_scale=diam_scale,
    )


def compute_Nccn(particle_population, s_eval: np.ndarray, T: float) -> Dict[str, np.ndarray]:
    return data_prep.compute_Nccn(particle_population, s_eval, T)


def compute_optical_coeffs(
    particle_population,
    coeff_types: Sequence[str] = ("total_scat", "total_abs"),
    wvls: Optional[Sequence[float]] = None,
    rh_grid: Optional[Sequence[float]] = None,
    morphology: str = "core-shell",
    temp: float = 298.15,
    species_modifications: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return data_prep.compute_optical_coeffs(
        particle_population,
        coeff_types=coeff_types,
        wvls=wvls,
        rh_grid=rh_grid,
        morphology=morphology,
        temp=temp,
        species_modifications=species_modifications,
    )


def _select_index_from_grid(grid: Sequence[float], selector: Any) -> int:
    """Utility: choose an index into `grid` based on selector.

    selector may be an int (index) or a float (value). If float, choose the
    nearest value in the grid.
    """
    if selector is None:
        return 0
    if isinstance(selector, int):
        return selector
    val = float(selector)
    arr = np.asarray(grid, dtype=float)
    idx = int(np.argmin(np.abs(arr - val)))
    return int(idx)


def compute_variable(particle_population, 
                     varname: str, 
                     var_cfg: Dict[str, Any], 
                     return_plotdat: bool = False):
    """High-level variable accessor used by plotting helpers.

    When plotting optical coefficients (2-D arrays) this function will select
    a single RH or wavelength to return 1-D x/y arrays. Selection can be
    controlled with `rh_select` or `wvl_select` in `var_cfg`. If not present
    the first grid value is used.
    """
    # Dispatch to new data_prep functions where possible but preserve the
    # original return contract (x, y, labs, xscale, yscale) when
    # return_plotdat==True.
    xscale = 'linear'
    yscale = 'linear'
    if varname == "dNdlnD":
        dat = data_prep.prepare_dNdlnD(particle_population, var_cfg)
        x, y, labs, xscale, yscale = dat["x"], dat["y"], dat["labs"], dat["xscale"], dat["yscale"]

    elif varname == "Nccn":
        dat = data_prep.prepare_Nccn(particle_population, var_cfg)
        x, y, labs, xscale, yscale = dat["x"], dat["y"], dat["labs"], dat["xscale"], dat["yscale"]

    elif varname == "frac_ccn":
        dat = data_prep.prepare_frac_ccn(particle_population, var_cfg)
        x, y, labs, xscale, yscale = dat["x"], dat["y"], dat["labs"], dat["xscale"], dat["yscale"]

    elif varname in ["b_abs", "b_scat", "b_ext", "total_abs", "total_scat", "total_ext"]:
        # map to optical vs wavelength or vs RH depending on var_cfg
        if var_cfg.get("vs_rh", False):
            dat = data_prep.prepare_optical_vs_rh(particle_population, dict(var_cfg, coeff=varname))
        else:
            dat = data_prep.prepare_optical_vs_wvl(particle_population, dict(var_cfg, coeff=varname))
        x, y, labs, xscale, yscale = dat["x"], dat["y"], dat["labs"], dat["xscale"], dat["yscale"]

    elif varname == "Ntot":
        dat = data_prep.prepare_Ntot(particle_population, var_cfg)
        x, y, labs, xscale, yscale = dat["x"], dat["y"], dat["labs"], dat["xscale"], dat["yscale"]

    else:
        raise NotImplementedError(f"varname={varname} not yet implemented")

    if return_plotdat:
        return x, y, labs, xscale, yscale
    else:
        return None


def build_default_var_cfg(varname: str) -> Dict[str, Any]:
    if varname == "dNdlnD":
        return {
            "wetsize": True,
            "normalize": False,
            "method": "hist",
            "N_bins": 30,
            "D_min": 1e-9,
            "D_max": 1e-4,
            "diam_scale": "log",
        }
    elif varname in ("Nccn", "frac_ccn"):
        return {"s_eval": np.linspace(0.01, 1.0, 50), "T": 298.15}
    elif varname in ["b_abs", "b_scat", "b_ext", "total_abs", "total_scat", "total_ext"]:
        return {
            "wvls": np.array([550e-9]),
            "rh_grid": np.array([0.0, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99]),
            "morphology": "core-shell",
            "vs_wvl": True,
            "vs_rh": False,
            # selectors allow choosing a single slice when full grids are present
            "rh_select": None,
            "wvl_select": None,
        }
    elif varname == "Ntot":
        return {}
    else:
        raise NotImplementedError(f"varname={varname} not yet implemented")
