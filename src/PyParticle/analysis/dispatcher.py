from __future__ import annotations
import warnings
from typing import Dict, Any
from .builder import build_variable
from .factory.registry import list_variables as _list, describe_variable as _describe, _ALIASES, resolve_name

def compute_variable(population, varname: str | None = None, var_cfg: Dict[str, Any] | None = None):
    if var_cfg is None:
        var_cfg = {}
    if varname is None:
        if "varname" not in var_cfg:
            raise ValueError("compute_variable requires varname or var_cfg['varname']")
        varname = var_cfg.pop("varname")
    user_requested = varname
    canon = resolve_name(varname)
    if user_requested != canon and user_requested in _ALIASES:
        warnings.warn(
            f"Variable alias '{user_requested}' is deprecated; use '{canon}' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
    var_obj = build_variable(canon, **var_cfg)
    out = var_obj.compute(population)
    # simple squeeze
    vk = var_obj.meta.value_key
    try:
        import numpy as _np
        arr = _np.asarray(out[vk])
        if arr.ndim == 2 and 1 in arr.shape:
            out[vk] = arr.squeeze()
    except Exception:
        pass
    return out

def list_variables(include_aliases: bool = False):
    return _list(include_aliases=include_aliases)

def describe_variable(name: str):
    return _describe(name)
