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

from .optics import build_optical_population
    
def compute_particle_variable(particle_population, 
                              varname: str, var_cfg: Dict[str, Any], 
                              return_plotdat: bool = False):
    
    particles = [particle_population.get_particle(part_id) for part_id in particle_population.ids]
    if varname == 'D' or varname == 'Dwet' or varname == 'wet_diameter' or varname == 'diameter':
        x = [particle.get_Dwet() for particle in particles]
        lab = 'diameter [m]'
    elif varname == 'Ddry' or varname == 'dry_diameter':
        x = [particle.get_Ddry() for particle in particles]
        lab = 'dry diameter [m]'
    elif varname == 'tkappa' or varname == 'kappa':
        x = [particle.get_tkappa() for particle in particles]
        lab = 'hygroscopicity parameter, $\kappa$'
    else:
        raise NotImplementedError(f"varname={varname} not yet implemented") 
    
    if return_plotdat:
        return x, lab
    else:
        return x
    
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
    import scipy.stats  # local import keeps top-level import cheap

    if diam_scale != "log":
        raise NotImplementedError(f"diam_scale={diam_scale} not implemented")

    edges = np.logspace(np.log10(D_min), np.log10(D_max), N_bins + 1)

    # weights
    try:
        num_concs = np.asarray(particle_population.num_concs, dtype=float)
    except Exception:
        num_concs = None
        
    # fixme: wrap this in another function?
    if wetsize:
        Ds = compute_particle_variable(particle_population, 'Dwet', {}, return_plotdat=False)
    else:
        Ds = compute_particle_variable(particle_population, 'Ddry', {}, return_plotdat=False)
    
    weights = particle_population.num_concs
    
    if normalize and weights.sum() > 0:
        weights = weights / weights.sum()

    if method == "hist":
        hist, _ = np.histogram(Ds, bins=edges, weights=weights)
        dln = np.log(edges[1:]) - np.log(edges[:-1])
        with np.errstate(divide="ignore", invalid="ignore"):
            dNdlnD = np.where(dln > 0, hist / dln, 0.0)
    elif method == "kde":
        logD = np.log(Ds)
        kde = scipy.stats.gaussian_kde(logD, weights=weights, bw_method="scott")
        centers = np.sqrt(edges[:-1] * edges[1:])
        dNdlnD = kde.evaluate(np.log(centers))
    else:
        raise NotImplementedError(f"method={method} not implemented")

    centers = np.sqrt(edges[:-1] * edges[1:])
    return {"D": centers, "dNdlnD": dNdlnD}


def compute_Nccn(particle_population, s_eval: np.ndarray, T: float) -> Dict[str, np.ndarray]:
    """Compute CCN activation spectrum for the population.

    Returns dict with keys 's' and 'Nccn'.
    """
    s_eval = np.asarray(s_eval, dtype=float)
    ccn_spectrum = np.zeros_like(s_eval, dtype=float)
    for idx, s_env in enumerate(s_eval):
        ccn_count = 0.0
        for i, part_id in enumerate(particle_population.ids):
            particle = particle_population.get_particle(part_id)
            s_crit = particle.get_critical_supersaturation(T, return_D_crit=False)
            num_conc = float(particle_population.num_concs[i])
            if s_env >= s_crit:
                ccn_count += num_conc
        ccn_spectrum[idx] = ccn_count

    return {"s": s_eval, "Nccn": ccn_spectrum}


def compute_optical_coeffs(
    particle_population,
    coeff_types: Sequence[str] = ("total_scat", "total_abs"),
    wvls: Optional[Sequence[float]] = None,
    rh_grid: Optional[Sequence[float]] = None,
    morphology: str = "core-shell",
    temp: float = 298.15,
    species_modifications: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute optical coefficients for the population.

    This function delegates to `build_optical_population(base_population, config)`
    and returns a dictionary containing the requested coefficient arrays plus
    the grids used ('wvls', 'rh_grid'). By convention the coefficient arrays
    are 2-D (len(rh_grid) x len(wvls)).
    """
    if wvls is None:
        wvls = [550e-9]
    if rh_grid is None:
        rh_grid = [0.0]
    if species_modifications is None:
        species_modifications = {}

    cfg = {"rh_grid": list(rh_grid), "wvl_grid": list(wvls), "type": morphology, "temp": temp, "species_modifications": species_modifications}

    optical_pop = build_optical_population(particle_population, cfg)

    out: Dict[str, Any] = {"wvls": np.asarray(wvls), "rh_grid": np.asarray(rh_grid)}
    for coeff in coeff_types:
        # ask for the full 2-D array
        arr = optical_pop.get_optical_coeff(optics_type=coeff, rh=None, wvl=None)
        out[coeff] = np.asarray(arr)

    return out


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
    xscale = 'linear'
    yscale = 'linear'
    if varname == "dNdlnD":
        diam_scale = var_cfg.get("diam_scale", "log")
        vardat = compute_dNdlnD(
            particle_population,
            wetsize=var_cfg.get("wetsize", True),
            normalize=var_cfg.get("normalize", False),
            method=var_cfg.get("method", "hist"),
            N_bins=var_cfg.get("N_bins", 30),
            D_min=var_cfg.get("D_min", 1e-9),
            D_max=var_cfg.get("D_max", 1e-4),
            diam_scale=diam_scale,
        )
        y = vardat["dNdlnD"]
        x = vardat["D"]
        labs = ["D (m)", "dN/dlnD (1/m$^3$)"]
        xscale = diam_scale

    elif varname == "Nccn":
        vardat = compute_Nccn(particle_population, var_cfg["s_eval"], var_cfg.get("T", 298.15))
        y = vardat["Nccn"]
        x = np.asarray(var_cfg["s_eval"])
        labs = ["s (%)", "Nccn (1/m$^3$)"]
        xscale = var_cfg.get("xscale", "log")

    elif varname == "frac_ccn":
        vardat = compute_Nccn(particle_population, var_cfg["s_eval"], var_cfg.get("T", 298.15))
        total = np.sum(particle_population.num_concs)
        if total > 0:
            vardat["Nccn"] = vardat["Nccn"] / total
        y = vardat["Nccn"]
        x = np.asarray(var_cfg["s_eval"])
        labs = ["s (%)", "fraction CCN"]
        xscale = var_cfg.get("xscale", "log")
    
    # fixme: this is a mess
    elif varname in ["b_abs", "b_scat", "b_ext", "total_abs", "total_scat", "total_ext"]:
        coeff_key = varname
        wvls = np.asarray(var_cfg.get("wvls", [550e-9]))
        rh_grid = np.asarray(var_cfg.get("rh_grid", [0.0]))
        vardat = compute_optical_coeffs(
            particle_population,
            coeff_types=[coeff_key],
            wvls=wvls,
            rh_grid=rh_grid,
            morphology=var_cfg.get("morphology", "core-shell"),
            temp=var_cfg.get("T", 298.15),
            species_modifications=var_cfg.get("species_modifications", {}),
        )

        arr2d = np.asarray(vardat[coeff_key])

        # human-friendly axis label
        if varname in ["b_abs", "total_abs"]:
            var_label = "abs. coeff. [1/m]"
        elif varname in ["b_scat", "total_scat"]:
            var_label = "scat. coeff. [1/m]"
        else:
            var_label = "ext. coeff. [1/m]"

        # choose plotting axis: vs wavelength or vs RH
        if var_cfg.get("vs_rh", False):
            # x is RH, y should be 1-D for selected wavelength
            x = np.asarray(rh_grid)
            # select which wavelength to slice at
            wvl_sel = var_cfg.get("wvl_select", None)
            wvl_idx = _select_index_from_grid(wvls, wvl_sel)
            if arr2d.ndim == 2:
                y = arr2d[:, wvl_idx]
            else:
                y = np.asarray(arr2d).ravel()
            labs = ["RH (%)", var_label + " (1/m)"]

        else:
            # default: vs wavelength
            x = np.asarray(wvls)
            # select which RH to slice at
            rh_sel = var_cfg.get("rh_select", None)
            rh_idx = _select_index_from_grid(rh_grid, rh_sel)
            if arr2d.ndim == 2:
                y = arr2d[rh_idx, :]
            else:
                y = np.asarray(arr2d).ravel()
            labs = ["wavelength (m)", var_label + " (1/m)"]

    elif varname == "Ntot":
        y = np.sum(particle_population.num_concs)
        x = None
        labs = ["Ntot (1/m$^3$)"]

    else:
        raise NotImplementedError(f"varname={varname} not yet implemented")

    if return_plotdat:
        return x, y, labs, xscale, yscale
    else:
        # return the raw data structure when not requested for plotting
        return locals().get("vardat", None)


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
