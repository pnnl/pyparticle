"""[DEPRECATED] moved to :mod:`PyParticle.analysis.prepare`.

This module was superseded by `analysis.prepare` which now owns the
vardat->PlotDat preparers and defaults. The file has been kept briefly for
backwards-compatibility but callers should import `PyParticle.analysis`:

    from PyParticle.analysis import compute_plotdat, build_default_var_cfg

and will be removed in a future release.
"""
from __future__ import annotations

from typing import Dict, Any
import warnings
import numpy as np
from ..analysis.dispatcher import compute_variable as _compute_variable
from ..analysis import list_variables as _list_variables
from ..analysis import global_registry as _global_registry

# NOTE: Legacy direct compute_* functions retained temporarily for backward compatibility
# but now emit DeprecationWarning and delegate to analysis dispatcher-based
# implementations. They will be removed in a future minor release.

PlotDat = Dict[str, Any]


############################
# Deprecated direct compute #
############################

# def _deprecated(msg: str):
#     warnings.warn(msg, DeprecationWarning, stacklevel=2)


# def compute_dNdlnD(*args, **kwargs):  # pragma: no cover - legacy shim
#     _deprecated("viz.data_prep.compute_dNdlnD is deprecated; use analysis.compute_variable('dNdlnD', cfg) or analysis.global_registry.get_variable_builder")
#     population = args[0]
#     cfg = dict(
#         wetsize=kwargs.get("wetsize", True),
#         normalize=kwargs.get("normalize", False),
#         method=kwargs.get("method", "hist"),
#         N_bins=kwargs.get("N_bins", 80),
#         D_min=kwargs.get("D_min", 1e-9),
#         D_max=kwargs.get("D_max", 2e-6),
#         diam_scale=kwargs.get("diam_scale", "log"),
#     )
#     return _compute_variable(population, "dNdlnD", cfg)


# def compute_Nccn(population, s_eval, T):  # pragma: no cover - legacy shim
#     _deprecated("viz.data_prep.compute_Nccn deprecated; use analysis.compute_variable('Nccn', cfg)")
#     return _compute_variable(population, "Nccn", {"s_eval": s_eval, "T": T})


# def compute_optical_coeffs(population, coeff_types=("b_ext",), wvls=None, rh_grid=None, morphology="core-shell", temp=298.15, species_modifications=None):  # pragma: no cover
#     _deprecated("viz.data_prep.compute_optical_coeffs deprecated; call analysis.compute_variable for each coeff (e.g. 'b_ext') or use analysis.global_registry")
#     if not isinstance(coeff_types, (list, tuple)):
#         coeff_types = [coeff_types]
#     results = {}
#     for coeff in coeff_types:
#         dat = _compute_variable(
#             population,
#             coeff,
#             {
#                 "wvls": wvls or [550e-9],
#                 "rh_grid": rh_grid or [0.0],
#                 "morphology": morphology,
#                 "species_modifications": species_modifications or {},
#                 "T": temp,
#             },
#         )
#         # unify shape keys
#         if "wvls" in dat:
#             results.setdefault("wvls", dat["wvls"])
#         if "rh_grid" in dat:
#             results.setdefault("rh_grid", dat["rh_grid"])
#         results[coeff] = dat[coeff]
#     return results

# -------------------------------------------------------------------------


# fixme: make these generic, move details to analysis
# fixme: make this "prepare_state_line" or similar
def prepare_dNdlnD(population, var_cfg: dict) -> PlotDat:
    """Prepare size-distribution data for plotting.

    See analysis.compute_dNdlnD for algorithm details.
    """
    cfg = dict(var_cfg or {})
    diam_scale = cfg.get("diam_scale", "log")
    varname = "dNdlnD"
    vardat = _compute_variable(
        population,
        varname, 
        #"dNdlnD", 
        cfg # now generalized? 
        # fixme: put this in _compute_variable?
        # {
        #     "wetsize": cfg.get("wetsize", True),
        #     "normalize": cfg.get("normalize", False),
        #     "method": cfg.get("method", "hist"),
        #     "N_bins": cfg.get("N_bins", 80),
        #     "D_min": cfg.get("D_min", 1e-9),
        #     "D_max": cfg.get("D_max", 2e-6),
        #     "diam_scale": diam_scale,
        # },
    )

    # fixme: put this in _compute_variable?
    x = np.asarray(vardat["D"])
    y = np.asarray(vardat["dNdlnD"])
    # fixme: make vardat contain this info
    return {"x": x, "y": y, "labs": ["D (m)", "dN/dlnD (1/m$^3$)"], "xscale": diam_scale, "yscale": "linear"}


def prepare_Nccn(population, var_cfg: dict) -> PlotDat:
    cfg = dict(var_cfg or {})
    s_eval = np.asarray(cfg.get("s_eval", np.linspace(0.01, 1.0, 50)))
    vardat = _compute_variable(population, "Nccn", {"s_eval": s_eval, "T": cfg.get("T", 298.15)})
    x = np.asarray(vardat["s"])  # fractional supersaturation
    y = np.asarray(vardat["Nccn"])
    return {"x": x, "y": y, "labs": ["s (%)", "Nccn (1/m$^3$)"], "xscale": "log", "yscale": "linear"}


def prepare_frac_ccn(population, var_cfg: dict) -> PlotDat:
    cfg = dict(var_cfg or {})
    s_eval = np.asarray(cfg.get("s_eval", np.linspace(0.01, 1.0, 50)))
    vardat = _compute_variable(population, "frac_ccn", {"s_eval": s_eval, "T": cfg.get("T", 298.15)})
    x = np.asarray(vardat["s"])
    y = np.asarray(vardat["frac_ccn"])
    return {"x": x, "y": y, "labs": ["s (%)", "fraction CCN"], "xscale": "log", "yscale": "linear"}


def prepare_optical_vs_wvl(population, var_cfg: dict) -> PlotDat:
    cfg = dict(var_cfg or {})
    coeff = cfg.get("coeff", "b_ext")
    # allow legacy aliases transparently
    if coeff in ("total_ext", "total_abs", "total_scat"):
        alias_map = {"total_ext": "b_ext", "total_abs": "b_abs", "total_scat": "b_scat"}
        coeff = alias_map[coeff]
    wvls = np.asarray(cfg.get("wvls", [550e-9]))
    rh_grid = np.asarray(cfg.get("rh_grid", [0.0]))
    # Use dispatcher compute_variable which will resolve aliases and build the
    # appropriate population-level variable under the hood. For advanced use
    # cases the global registry can be used to fetch builders across families.
    vardat = _compute_variable(
        population,
        coeff,
        {
            "wvls": list(wvls),
            "rh_grid": list(rh_grid),
            "morphology": cfg.get("morphology", "core-shell"),
            "species_modifications": cfg.get("species_modifications", {}),
            "T": cfg.get("T", 298.15),
        },
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
    coeff = cfg.get("coeff", "b_ext")
    if coeff in ("total_ext", "total_abs", "total_scat"):
        alias_map = {"total_ext": "b_ext", "total_abs": "b_abs", "total_scat": "b_scat"}
        coeff = alias_map[coeff]
    wvls = np.asarray(cfg.get("wvls", [550e-9]))
    rh_grid = np.asarray(cfg.get("rh_grid", [0.0]))
    vardat = _compute_variable(
        population,
        coeff,
        {
            "wvls": list(wvls),
            "rh_grid": list(rh_grid),
            "morphology": cfg.get("morphology", "core-shell"),
            "species_modifications": cfg.get("species_modifications", {}),
            "T": cfg.get("T", 298.15),
        },
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
    # fixme: include this in analysis -- short labs vs long labs?
    return {"x": x, "y": np.asarray(y), "labs": ["RH (%)", f"{coeff} (1/m)"], "xscale": "linear", "yscale": "linear"}


# fixme: include this in analysis?

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
            # Updated defaults (was 30) to better resolve modal structure in typical
            # PartMC/MAM4 ensemble comparisons without being overly heavy.
            "N_bins": 80,
            "D_min": 1e-9,
            # Reduced upper bound (was 1e-4) to 2e-6 to focus on sub-micron / accumulation
            # regime emphasized in ensemble workflows. Users can override upward as needed.
            "D_max": 2e-6,
            "diam_scale": "log",
        }
    elif varname in ("Nccn", "frac_ccn"):
        return {"s_eval": np.logspace(-2, 2, 50), "T": 298.15}
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
    
        raise NotImplementedError(f"varname={varname} not yet implemented")
