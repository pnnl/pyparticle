"""Small helpers to construct PartMC and MAM4 builder config dicts for examples.

These helpers return dicts compatible with the repository builders. They try
to be conservative about side effects: by default they use logging.debug and
only print directory listings when `verbose=True`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import os
import logging
import numpy as np
from PyParticle.population import build_population


_LOG = logging.getLogger(__name__)


def retrieve_partmc_cfg(
    scenario_name: str,
    partmc_root: str | Path,
    timestep: int = 1,
    repeat_num: int = 1,
    species_modifications: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Return a config dict suitable for the `partmc` population builder.

    Parameters
    - scenario_name: sub-folder under `partmc_root` containing PartMC outputs
    - partmc_root: path to ensemble root (string or Path)
    - timestep, repeat_num: forwarded to the builder
    - species_modifications: optional dict of per-species overrides
    - verbose: if True, print lightweight directory listings for debugging
    """
    partmc_root = Path(partmc_root)
    partmc_dir = partmc_root / scenario_name

    cfg: Dict[str, Any] = {
        "type": "partmc",
        "partmc_dir": str(partmc_dir),
        "timestep": int(timestep),
        "repeat": int(repeat_num),
        "species_modifications": species_modifications or {},
    }

    _LOG.debug("retrieve_partmc_cfg: partmc_dir=%s", partmc_dir)
    if verbose:
        try:
            print("retrieve_partmc_cfg: partmc_dir=", partmc_dir)
            print("partmc_dir exists:", partmc_dir.is_dir())
            if partmc_dir.is_dir():
                print("partmc_dir listing:", list(os.listdir(partmc_dir)))
            out_dir = partmc_dir / "out"
            print("out_dir exists:", out_dir.is_dir())
            if out_dir.is_dir():
                print("out_dir listing:", list(os.listdir(out_dir)))
        except Exception as e:
            print("Error listing PartMC directories for debugging:", e)

    return cfg


def build_var_cfg_mapping(cfg_all: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Construct a mapping varname -> var_cfg where plotting/binning keys
    are placed in variable configs (not in population configs).

    This extracts N_bins, D_min, D_max from top-level mam4_settings when
    present and applies conservative defaults otherwise.
    """
    mapping: Dict[str, Dict[str, Any]] = {}
    mam4_settings = cfg_all.get("mam4_settings", {}) or {}
    # defaults
    dmin = mam4_settings.get("D_min", 1e-9)
    dmax = mam4_settings.get("D_max", 2e-6)
    n_bins = mam4_settings.get("N_bins", 40)

    variables = cfg_all.get("variables", ["dNdlnD"]) or ["dNdlnD"]
    for var in variables:
        if var == "dNdlnD":
            mapping[var] = {"wetsize": True, "N_bins": n_bins, "D_min": dmin, "D_max": dmax}
        elif var in ("frac_ccn", "Nccn"):
            mapping[var] = {"s_eval": list(cfg_all.get("s_eval", [])) or list(np.logspace(-2, 1, 40))}
        else:
            mapping[var] = {}
    # Apply any explicit var_cfg_overrides from config
    overrides = cfg_all.get("var_cfg_overrides", {}) or {}
    for k, v in overrides.items():
        mapping[k] = {**mapping.get(k, {}), **v}
    return mapping


def build_populations_and_varcfgs(
    scenario_id: str,
    cfg_all: Dict[str, Any],
    verbose: bool = False,
):
    """Build PartMC and MAM4 populations for a given scenario using the
    repository builders and return them along with variable cfg mapping.

    Returns a dict with keys: partmc_cfg, mam4_cfg, partmc_pop, mam4_pop,
    var_cfgs.
    """
    partmc_root = Path(cfg_all.get("partmc_dir", ""))
    mam4_root = Path(cfg_all.get("mam4_dir", ""))
    if not partmc_root.is_dir():
        raise FileNotFoundError(f"PartMC root directory not found: {partmc_root}")
    if not mam4_root.is_dir():
        raise FileNotFoundError(f"MAM4 root directory not found: {mam4_root}")
    if not scenario_id:
        raise ValueError("Empty scenario_id provided")
    partmc_cfg = retrieve_partmc_cfg(scenario_id, partmc_root, timestep=cfg_all.get("timestep", 1), repeat_num=cfg_all.get("repeat",1), species_modifications=cfg_all.get("spec_modifications", {}), verbose=verbose)
    mam4_cfg = retrieve_mam4_cfg(scenario_id, mam4_root, timestep=cfg_all.get("timestep", 1), mam4_settings=cfg_all.get("mam4_settings", {}), species_modifications=cfg_all.get("spec_modifications", {}),  verbose=verbose)
    
    # Build populations (let builder raise clear errors if files/deps missing)
    try:
        partmc_pop = build_population(partmc_cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to build PartMC population for scenario {scenario_id}: {e}")
    try:
        mam4_pop = build_population(mam4_cfg)
    except Exception as e:
        raise RuntimeError(f"Failed to build MAM4 population for scenario {scenario_id}: {e}")

    var_cfgs = build_var_cfg_mapping(cfg_all)
    return {
        "partmc_cfg": partmc_cfg,
        "mam4_cfg": mam4_cfg,
        "partmc_pop": partmc_pop,
        "mam4_pop": mam4_pop,
        "var_cfgs": var_cfgs,
    }


def retrieve_mam4_cfg(
    scenario_name: str,
    mam4_root: str | Path,
    timestep: int = 1,
    mam4_settings: Optional[Dict[str, Any]] = None,
    species_modifications: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Return a config dict suitable for the `mam4` builder used in examples.

    The returned dict will contain at minimum:
      - type: 'mam4'
      - output_filename: path to the expected MAM4 netCDF (string)
      - timestep: int
      - GSD, D_min, D_max, N_bins if provided in mam4_settings
    
    Note: the MAM4 builder expects keys named exactly 'p' and 'T'.
    """
    mam4_root = Path(mam4_root)
    output_filename = mam4_root / scenario_name / "mam_output.nc"

    cfg: Dict[str, Any] = {
        "type": "mam4",
        "output_filename": str(output_filename),
        "timestep": int(timestep),
    }

    # Accept mam4_settings under either the expected dict or None.
    mam4_settings = mam4_settings or {}
    # Forward composition/system-level settings to the MAM4 builder.
    # The MAM4 builder currently expects mode GSD and also diameter/binning
    # keys (D_min, D_max, N_bins). Forward those if provided in mam4_settings.
    cfg["GSD"] = mam4_settings["GSD"]
    for key in ("D_min", "D_max", "N_bins"):
        if key in mam4_settings:
            cfg[key] = mam4_settings[key]

    _LOG.debug("retrieve_mam4_cfg: output_filename=%s", output_filename)
    if verbose:
        try:
            print("retrieve_mam4_cfg: output_filename=", output_filename)
            print("mam4 file exists:", output_filename.exists())
            if output_filename.exists():
                print("mam4 parent listing:", list(os.listdir(output_filename.parent)))
        except Exception as e:
            print("Error checking MAM4 output file for debugging:", e)

    # The MAM4 builder expects pressure 'p' (Pa) and temperature 'T' (K).
    # Provide conservative defaults so examples can run without a full
    # meteorological config present. These defaults are reasonable:
    #   p = 101325 Pa (standard atmosphere)
    #   T = 288.15 K (15 °C)
    cfg.setdefault('p', 101325.0)
    cfg.setdefault('T', 288.15)

    return cfg
