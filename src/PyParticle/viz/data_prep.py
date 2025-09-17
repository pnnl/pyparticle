"""Data preparation helpers for plotting (plot-ready APIs).

This module refactors plot data assembly out of `analysis.py` into
small, well-documented functions that return a consistent PlotDat dict:
  {"x": np.ndarray or None, "y": np.ndarray, "labs": [xlabel,ylabel], "xscale": str, "yscale": str}

These functions are intentionally thin wrappers around existing calculation
helpers (compute_dNdlnD, compute_Nccn, compute_optical_coeffs) to preserve
numerical behavior while providing a clearer API to the viz layer.
"""
from __future__ import annotations

from typing import Dict, Any
import numpy as np
from ..optics.builder import build_optical_population

PlotDat = Dict[str, Any]


# --- Core computation functions (moved from analysis.py) -----------------
def compute_particle_variable(particle_population, varname: str, var_cfg: Dict[str, Any], return_plotdat: bool = False):
    particles = [particle_population.get_particle(part_id) for part_id in particle_population.ids]
    if varname in ('D', 'Dwet', 'wet_diameter', 'diameter'):
        x = [particle.get_Dwet() for particle in particles]
        lab = 'diameter [m]'
    elif varname in ('Ddry', 'dry_diameter'):
        x = [particle.get_Ddry() for particle in particles]
        lab = 'dry diameter [m]'
    elif varname in ('tkappa', 'kappa'):
        x = [particle.get_tkappa() for particle in particles]
        lab = 'hygroscopicity parameter, $\\kappa$'
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
    import scipy.stats  # local import keeps top-level import cheap

    if diam_scale != "log":
        raise NotImplementedError(f"diam_scale={diam_scale} not implemented")

    edges = np.logspace(np.log10(D_min), np.log10(D_max), N_bins + 1)

    if wetsize:
        Ds = compute_particle_variable(particle_population, 'Dwet', {}, return_plotdat=False)
    else:
        Ds = compute_particle_variable(particle_population, 'Ddry', {}, return_plotdat=False)

    weights = np.asarray(particle_population.num_concs, dtype=float)

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
    coeff_types: Any = ("total_scat", "total_abs"),
    wvls: Any = None,
    rh_grid: Any = None,
    morphology: str = "core-shell",
    temp: float = 298.15,
    species_modifications: Any = None,
) -> Dict[str, Any]:
    if wvls is None:
        wvls = [550e-9]
    if rh_grid is None:
        rh_grid = [0.0]
    if species_modifications is None:
        species_modifications = {}

    cfg = {"rh_grid": list(rh_grid), "wvl_grid": list(wvls), "type": morphology, "temp": temp, "species_modifications": species_modifications}

    # delegate to optics builder (imported at module level so tests can monkeypatch)
    optical_pop = build_optical_population(particle_population, cfg)

    out: Dict[str, Any] = {"wvls": np.asarray(wvls), "rh_grid": np.asarray(rh_grid)}
    for coeff in coeff_types:
        arr = optical_pop.get_optical_coeff(optics_type=coeff, rh=None, wvl=None)
        out[coeff] = np.asarray(arr)
    return out

# -------------------------------------------------------------------------


def prepare_dNdlnD(population, var_cfg: dict) -> PlotDat:
    """Prepare size-distribution data for plotting.

    See analysis.compute_dNdlnD for algorithm details.
    """
    cfg = dict(var_cfg or {})
    diam_scale = cfg.get("diam_scale", "log")
    vardat = compute_dNdlnD(
        population,
        wetsize=cfg.get("wetsize", True),
        normalize=cfg.get("normalize", False),
        method=cfg.get("method", "hist"),
        N_bins=cfg.get("N_bins", 30),
        D_min=cfg.get("D_min", 1e-9),
        D_max=cfg.get("D_max", 1e-4),
        diam_scale=diam_scale,
    )
    x = np.asarray(vardat["D"])
    y = np.asarray(vardat["dNdlnD"])
    return {"x": x, "y": y, "labs": ["D (m)", "dN/dlnD (1/m$^3$)"], "xscale": diam_scale, "yscale": "linear"}


def prepare_Nccn(population, var_cfg: dict) -> PlotDat:
    cfg = dict(var_cfg or {})
    s_eval = np.asarray(cfg.get("s_eval", np.logspace(-2, 1. 50)))
    vardat = compute_Nccn(population, s_eval, cfg.get("T", 298.15))
    x = np.asarray(vardat["s"])  # fractional supersaturation
    y = np.asarray(vardat["Nccn"])
    return {"x": x, "y": y, "labs": ["s (%)", "Nccn (1/m$^3$)"], "xscale": "log", "yscale": "linear"}


def prepare_frac_ccn(population, var_cfg: dict) -> PlotDat:
    cfg = dict(var_cfg or {})
    s_eval = np.asarray(cfg.get("s_eval", np.logspace(-2, 1. 50)))
    vardat = compute_Nccn(population, s_eval, cfg.get("T", 298.15))
    total = np.sum(population.num_concs)
    y = np.asarray(vardat["Nccn"]) if total == 0 else np.asarray(vardat["Nccn"]) / float(total)
    x = np.asarray(vardat["s"])
    return {"x": x, "y": y, "labs": ["s (%)", "fraction CCN"], "xscale": "log", "yscale": "linear"}


def prepare_optical_vs_wvl(population, var_cfg: dict) -> PlotDat:
    cfg = dict(var_cfg or {})
    coeff = cfg.get("coeff", "total_ext")
    wvls = np.asarray(cfg.get("wvls", [550e-9]))
    rh_grid = np.asarray(cfg.get("rh_grid", [0.0]))
    vardat = compute_optical_coeffs(
        population,
        coeff_types=[coeff],
        wvls=wvls,
        rh_grid=rh_grid,
        morphology=cfg.get("morphology", "core-shell"),
        temp=cfg.get("T", 298.15),
        species_modifications=cfg.get("species_modifications", {}),
    )
    arr2d = np.asarray(vardat.get(coeff))
    # select rh slice
    rh_sel = cfg.get("rh_select", None)
    if arr2d.ndim == 2:
        rh_idx = 0 if rh_sel is None else int(rh_sel)
        y = arr2d[rh_idx, :]
    else:
        y = arr2d.ravel()
    x = np.asarray(vardat.get("wvls"))
    return {"x": x, "y": np.asarray(y), "labs": ["wavelength (m)", f"{coeff} (1/m)"], "xscale": "linear", "yscale": "linear"}


def prepare_optical_vs_rh(population, var_cfg: dict) -> PlotDat:
    cfg = dict(var_cfg or {})
    coeff = cfg.get("coeff", "total_ext")
    wvls = np.asarray(cfg.get("wvls", [550e-9]))
    rh_grid = np.asarray(cfg.get("rh_grid", [0.0]))
    vardat = compute_optical_coeffs(
        population,
        coeff_types=[coeff],
        wvls=wvls,
        rh_grid=rh_grid,
        morphology=cfg.get("morphology", "core-shell"),
        temp=cfg.get("T", 298.15),
        species_modifications=cfg.get("species_modifications", {}),
    )
    arr2d = np.asarray(vardat.get(coeff))
    # select wvl slice
    wvl_sel = cfg.get("wvl_select", None)
    if arr2d.ndim == 2:
        wvl_idx = 0 if wvl_sel is None else int(wvl_sel)
        y = arr2d[:, wvl_idx]
    else:
        y = arr2d.ravel()
    x = np.asarray(vardat.get("rh_grid"))
    return {"x": x, "y": np.asarray(y), "labs": ["RH (%)", f"{coeff} (1/m)"], "xscale": "linear", "yscale": "linear"}


def build_default_var_cfg(varname: str):
    """Provide default var_cfg dictionaries for plotting helpers.

    This mirrors the previous `analysis.build_default_var_cfg` so callers
    can source defaults from the data_prep module directly.
    """
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
            "rh_select": None,
            "wvl_select": None,
        }
    elif varname == "Ntot":
        return {}
    else:
        raise NotImplementedError(f"varname={varname} not yet implemented")
