from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any
import numpy as np

# fixme: move this to dispatcher and variable files
# --- Public registry API ------------------------------------------------------
_PREPARERS: Dict[str, Callable[[dict, dict], dict]] = {}

def register_preparer(varname: str):
    """Decorator: register a vardat->PlotDat preparer under canonical varname."""
    def _dec(fn: Callable[[dict, dict], dict]):
        _PREPARERS[varname] = fn
        return fn
    return _dec

def resolve_preparer_name(varname: str) -> str:
    """Resolve to canonical name using dispatcher’s resolver (no viz imports)."""
    # Local import to avoid any hypothetical cycles.
    from .dispatcher import resolve_name
    return resolve_name(varname)

# Legacy alias map (moved from viz.data_prep). Keep at module-level so all
# preparers use the same canonicalization policy.
ALIAS_MAP = {"total_ext": "b_ext", "total_abs": "b_abs", "total_scat": "b_scat"}


def _canonicalize_varname(varname: str) -> str:
    """Apply small legacy aliasing rules before resolving to canonical name."""
    if varname in ALIAS_MAP:
        return ALIAS_MAP[varname]
    return varname

# --- Defaults -----------------------------------------------------------------
def build_default_var_cfg(varname: str) -> dict:
    canon = resolve_preparer_name(varname)
    if canon in {"dNdlnD"}:
        return {"D": None}  # computed inside builder; no plotting-defaults needed
    if canon in {"Nccn", "frac_ccn"}:
        # Use consistent CCN supersaturation sampling
        return {"s_eval": np.logspace(-2, 0, 50)}  # 0.01 to 1.0
    if canon in {"b_ext", "b_sca", "b_abs", "ssa", "g"}:
        return {
            "wvl_grid": np.linspace(3.0e-7, 1.0e-6, 50),
            "rh_grid": np.array([0.0]),  # single RH by default
        }
    # legacy plotting defaults for scattering/absorption/extinction variable names
    if canon in {"b_abs", "b_scat", "b_ext", "total_abs", "total_scat", "total_ext"}:
        return {
            "wvl_grid": np.array([550e-9]),
            "rh_grid": np.array([0.0, 0.5, 0.7, 0.85, 0.9, 0.95, 0.98, 0.99]),
        }
    raise ValueError(f"No defaults defined for variable '{canon}'")

# --- Public convenience wrapper ----------------------------------------------
def compute_plotdat(population, varname: str, var_cfg: Optional[dict] = None) -> dict:
    print('in prepare', varname, var_cfg)
    """Call dispatcher, then run the registered preparer to produce PlotDat."""
    from . import dispatcher  # local import to avoid cycles
    # Apply legacy aliases first, then resolve to canonical dispatcher name
    canon = resolve_preparer_name(_canonicalize_varname(varname))
    
    # cfg = {} if var_cfg is None else dict(var_cfg)
    cfg = {} if var_cfg is None else var_cfg
    
    # Apply defaults if not provided
    defaults = build_default_var_cfg(canon)
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    # Harmonize common synonyms: accept either 'wvl_grid' or 'wvls' from callers
    # Variable builders expect the key 'wvls' (legacy), while some callers
    # and our plotting defaults use 'wvl_grid'. Keep both present so downstream
    # code can use either name.
    if "wvl_grid" in cfg and "wvls" not in cfg:
        cfg["wvls"] = cfg["wvl_grid"]
    elif "wvls" in cfg and "wvl_grid" not in cfg:
        cfg["wvl_grid"] = cfg["wvls"]
    # Ensure rh_grid exists as an array-like (builders expect this key)
    if "rh_grid" in cfg and not isinstance(cfg["rh_grid"], (list, tuple, np.ndarray)):
        cfg["rh_grid"] = [cfg["rh_grid"]]
    
    vardat = dispatcher.compute_variable(population, canon, cfg)
    prep = _PREPARERS.get(canon, _generic_preparer)
    print(vardat, cfg, var_cfg)
    return prep(vardat, cfg)

# --- Generic fallback ----------------------------------------------------------
def _generic_preparer(vardat: dict, var_cfg: dict) -> dict:
    # Minimal: flatten the value array if we can discover the value key
    value_keys = [k for k in vardat.keys() if k not in {"D","dNdlnD","wvls","rh_grid","s"}]
    if not value_keys:
        raise ValueError("Could not infer value key from vardat.")
    vk = value_keys[0]
    y = np.asarray(vardat[vk]).ravel()
    return {"x": None, "y": y, "labs": ["", vk], "xscale": "linear", "yscale": "linear"}

# --- Specific preparers --------------------------------------------------------

@register_preparer("dNdlnD")
def prepare_dNdlnD(vardat: dict, var_cfg: dict) -> dict:
    D = np.asarray(vardat.get("D"))
    y = np.asarray(vardat.get("dNdlnD"))
    if D is None or y is None:
        raise ValueError("Expected keys 'D' and 'dNdlnD' in vardat.")
    if D.ndim != 1 or y.ndim != 1 or D.shape[0] != y.shape[0]:
        raise ValueError(f"dNdlnD expects 1D D and dNdlnD of same length. "
                         f"Got shapes D={getattr(D,'shape',None)}, y={getattr(y,'shape',None)}")
    return {"x": D, "y": y, "labs": ["Diameter (m)", "dN/dlnD"], "xscale": "log", "yscale": "log"}

@register_preparer("Nccn")
def prepare_Nccn(vardat: dict, var_cfg: dict) -> dict:
    s = np.asarray(vardat.get("s"))
    y = np.asarray(vardat.get("Nccn"))
    if s is None or y is None:
        raise ValueError("Expected keys 's' and 'Nccn' in vardat.")
    if s.ndim != 1 or y.ndim != 1 or s.shape[0] != y.shape[0]:
        raise ValueError(f"Nccn expects 1D s and Nccn of same length. Got s={s.shape}, y={y.shape}")
    return {"x": s, "y": y, "labs": ["supersaturation s", "N_ccn (cm^-3)"], "xscale": "log", "yscale": "linear"}

@register_preparer("frac_ccn")
def prepare_frac_ccn(vardat: dict, var_cfg: dict) -> dict:
    s = np.asarray(vardat.get("s"))
    y = np.asarray(vardat.get("frac_ccn"))
    if s is None or y is None:
        raise ValueError("Expected keys 's' and 'frac_ccn' in vardat.")
    if s.ndim != 1 or y.ndim != 1 or s.shape[0] != y.shape[0]:
        raise ValueError(f"frac_ccn expects 1D s and frac_ccn. Got s={s.shape}, y={y.shape}")
    return {"x": s, "y": y, "labs": ["supersaturation s", "fraction activated"], "xscale": "log", "yscale": "linear"}


def prepare_optical_vs_wvl(vardat: dict, var_cfg: dict, value_key: str, _select_index) -> dict:
    """Prepare optical coefficient plotted versus wavelength.

    Accepts a selector for RH that may be integer index or numeric value.
    Mirrors legacy `viz.data_prep.prepare_optical_vs_wvl` semantics but uses
    the supplied robust selector helper.
    """
    wvls = np.asarray(vardat.get("wvls"))
    rhg = vardat.get("rh_grid")
    arr = np.asarray(vardat.get(value_key))
    if wvls is None or arr is None:
        raise ValueError(f"Expected 'wvls' and '{value_key}' in vardat.")
    if arr.ndim == 2:
        if rhg is None:
            raise ValueError(f"'{value_key}' is 2D but 'rh_grid' is missing.")
        rhg = np.asarray(rhg)
        rh_sel = var_cfg.get("rh_select", None)
        ridx = _select_index(rhg, rh_sel, "RH")
        y = arr[ridx, :]
    else:
        y = arr.ravel()
    x = np.asarray(vardat.get("wvls"))
    return {"x": x, "y": np.asarray(y), "labs": ["wavelength (m)", f"{value_key} (1/m)"], "xscale": "linear", "yscale": "linear"}


def prepare_optical_vs_rh(vardat: dict, var_cfg: dict, value_key: str, _select_index) -> dict:
    """Prepare optical coefficient plotted versus relative humidity.

    Accepts a selector for wavelength that may be integer index or numeric value.
    Mirrors legacy `viz.data_prep.prepare_optical_vs_rh` semantics but uses the
    supplied robust selector helper.
    """
    wvls = np.asarray(vardat.get("wvls"))
    rhg = np.asarray(vardat.get("rh_grid")) if vardat.get("rh_grid") is not None else None
    arr = np.asarray(vardat.get(value_key))
    if wvls is None or arr is None:
        raise ValueError(f"Expected 'wvls' and '{value_key}' in vardat.")
    # select wvl slice
    wvl_sel = var_cfg.get("wvl_select", None)
    if arr.ndim == 2:
        widx = _select_index(wvls, wvl_sel, "wavelength")
        y = arr[:, widx]
    else:
        y = arr.ravel()
    x = np.asarray(vardat.get("rh_grid"))
    # Keep the legacy label used in viz.data_prep for compatibility
    return {"x": x, "y": np.asarray(y), "labs": ["RH (%)", f"{value_key} (1/m)"], "xscale": "linear", "yscale": "linear"}

def _prep_optics(vardat: dict, var_cfg: dict, value_key: str) -> dict:
    wvls = np.asarray(vardat.get("wvls"))
    rhg  = vardat.get("rh_grid")
    data = np.asarray(vardat.get(value_key))
    # Coerce 0-dim (scalar) results into 1D arrays for single-value outputs
    if data is not None and data.ndim == 0:
        data = data.reshape((1,))
    if wvls is None or data is None:
        raise ValueError(f"Expected 'wvls' and '{value_key}' in vardat.")
    # Utilities: allow selector to be index (int) or value (float); give
    # helpful error messages when out-of-range or ambiguous.
    def _select_index(axis: np.ndarray, selector, axis_name: str):
        # None -> first index
        if selector is None:
            return 0
        # If it's an integer index within bounds, accept it
        try:
            if isinstance(selector, (int, np.integer)):
                idx = int(selector)
                if idx < 0 or idx >= axis.shape[0]:
                    raise IndexError(f"{axis_name} index {idx} out of range (0..{axis.shape[0]-1})")
                return idx
        except Exception:
            pass
        # Otherwise try float-based nearest selection
        try:
            sval = float(selector)
        except Exception:
            raise ValueError(f"Could not interpret {axis_name} selector={selector!r}; pass integer index or numeric value")
        amin, amax = float(axis.min()), float(axis.max())
        if sval < amin or sval > amax:
            raise ValueError(
                f"Requested {axis_name} value {sval} is outside axis range [{amin}, {amax}]. "
                f"{axis_name} values are expected as fractions (e.g. 0.0-1.0). If you passed percent, convert to fraction."
            )
        return int(np.abs(axis - sval).argmin())

    # Handle 1D or 2D (RH x WVL) robustly.
    if data.ndim == 1:
        # 1D data: interpretable as function of wavelength
        x = wvls
        y = data
        return {"x": x, "y": y, "labs": ["wavelength (m)", value_key], "xscale": "linear", "yscale": "linear"}

    if data.ndim == 2:
        if rhg is None:
            raise ValueError(f"'{value_key}' is 2D but 'rh_grid' is missing.")
        rhg = np.asarray(rhg)
        # Mode selection: plotting callers may request vs_wvl (x = wavelength)
        # or vs_rh (x = rh). Default kept as vs_wvl for backward compatibility.
        vs_rh = bool(var_cfg.get("vs_rh", False))
        vs_wvl = bool(var_cfg.get("vs_wvl", not vs_rh))

        if vs_wvl:
            return prepare_optical_vs_wvl(vardat, var_cfg, value_key, _select_index)
        else:
            return prepare_optical_vs_rh(vardat, var_cfg, value_key, _select_index)

    raise ValueError(f"Unsupported ndim={data.ndim} for '{value_key}'")

for _vk in ("b_ext", "b_scat", "b_sca", "b_abs", "ssa", "g"):
    # register both b_scat and the historical b_sca spelling to be robust
    _PREPARERS[_vk] = (lambda vk=_vk: (lambda vd, cfg: _prep_optics(vd, cfg, vk)))()
