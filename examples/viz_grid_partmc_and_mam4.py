"""Example: Compare PartMC vs MAM4 populations across scenarios using viz grid helpers.

Refactored to use `viz.grids.make_grid_scenarios_models` for declarative layout:
    * Rows = scenario config dicts
    * Columns = variable names (e.g. dNdlnD, frac_ccn)
    * Multiple model populations (PartMC + MAM4) per axis with shared scenario color

No mock data: raises if required PartMC or MAM4 files are missing.
Surface tension warning from particle hygroscopic growth is suppressed explicitly.
Output: `examples/out_grid_partmc_mam4.png`
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

import warnings
from PyParticle.population import build_population
from PyParticle.analysis import compute_variable  # Demonstration (not strictly needed in grid path)
from PyParticle.viz.grids import make_grid_scenarios_models
from PyParticle.viz.stylemap import map_linestyles_for_series  # if needed for custom legend


def _build_partmc_cfg(base_cfg: Dict[str, Any], scenario_id: str, partmc_root: Path) -> Dict[str, Any]:
    cfg = dict(base_cfg)
    run_dir = partmc_root / scenario_id
    cfg["partmc_dir"] = str(run_dir)

    return cfg


def _build_mam4_cfg(mam4_defaults: Dict[str, Any], scenario_id: str, mam4_root: Path) -> Dict[str, Any]:
    cfg = dict(mam4_defaults)
    # For real data you might have a naming pattern; here we assume a file present in scenario folder.
    # Adjust pattern as needed (e.g., f"mam4_{scenario_id}.nc").
    candidate = mam4_root / scenario_id / "mam_output.nc"
    cfg["output_filename"] = candidate
    return cfg


def _build_partmc_population(cfg: Dict[str, Any]):
    pop = build_population(cfg)
    pop.origin = "PartMC"
    return pop

def _build_mam4_population(cfg: Dict[str, Any]):
    from PyParticle.population.factory.mam4 import build as build_mam4
    pop = build_mam4(cfg)
    pop.origin = "MAM4"
    return pop

def _variable_varcfg(varname: str, overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a var_cfg dict for a variable applying optional overrides."""
    base_defaults = {
        "dNdlnD": {"wetsize": True, "N_bins": 40, "D_min": 10e-9, "D_max": 2e-6},
        "frac_ccn": {"s_eval": np.logspace(-2, 1, 40)},
        "Nccn": {"s_eval": np.logspace(-2, 1, 40)},
    }.get(varname, {})
    cfg = dict(base_defaults)
    if overrides:
        cfg.update(overrides)
    return cfg


def main():
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "examples" / "configs" / "partmc_mam4_example.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"Config file not found: {cfg_path}\nCreate it (see template) or adjust the path in the example." 
        )
    cfg_all = json.load(open(cfg_path))

    scenario_ids = cfg_all.get("scenarios", [])
    partmc_root = Path(cfg_all.get("partmc_root", ""))
    mam4_root = Path(cfg_all.get("mam4_root", ""))
    timestep = cfg_all.get("timestep", 0)

    partmc_base = {
        "type": "partmc",
        "timestep": timestep,
        "repeat": cfg_all.get("repeat", 1),
        "species_modifications": cfg_all.get("partmc_species_modifications", {}),
    }
    mam4_defaults = cfg_all.get("mam4_defaults", {})
    mam4_defaults.setdefault("timestep", timestep)

    variables = cfg_all.get("variables", ["dNdlnD", "frac_ccn"])  # default two variables (optics use b_ext/b_abs/b_scat)

    if not scenario_ids:
        raise SystemExit("No scenarios specified in config.")

    # Build list of scenario config dicts for grid helper
    scenario_cfgs: List[dict] = []
    for sid in scenario_ids:
        partmc_cfg = _build_partmc_cfg(partmc_base, sid, partmc_root)
        mam4_cfg = _build_mam4_cfg(mam4_defaults, sid, mam4_root)
        # Pre-flight existence checks (no mock):
        if not Path(partmc_cfg["partmc_dir"]).exists():
            raise FileNotFoundError(f"Missing PartMC directory: {partmc_cfg['partmc_dir']}")
        if not Path(mam4_cfg["output_filename"]).exists():
            raise FileNotFoundError(f"Missing MAM4 file: {mam4_cfg['output_filename']}")
        # Fuse configs by namespacing keys for builders (builders only use relevant keys)
        scenario_cfgs.append({**partmc_cfg, **mam4_cfg})

    # Variable-specific overrides mapping
    overrides_map = cfg_all.get("var_cfg_overrides", {}) or {}
    # Leverage dispatcher defaults; only pass explicit overrides when present
    var_cfg_mapping = {v: (overrides_map.get(v) or {}) for v in variables}

    # Suppress surface tension warning explicitly
    warnings.filterwarnings(
        "ignore",
        message="Surface tension not implemented; returning default",
        category=UserWarning,
        module="PyParticle.aerosol_particle",
    )

    def partmc_builder(cfg):
        # Extract only PartMC-relevant keys
        part_cfg = {k: v for k, v in cfg.items() if k in partmc_base or k in ("partmc_dir",)}
        return _build_partmc_population(part_cfg)

    def mam4_builder(cfg):
        mam4_keys = set(mam4_defaults.keys()) | {"output_filename", "timestep"}
        mam_cfg = {k: v for k, v in cfg.items() if k in mam4_keys}
        return _build_mam4_population(mam_cfg)

    fig, axes = make_grid_scenarios_models(
        scenario_cfgs,
        variables,
        model_cfg_builders=[partmc_builder, mam4_builder],
        var_cfg=var_cfg_mapping,
        figsize=(4 * len(variables), 3 * len(scenario_cfgs)),
    )

    fig.suptitle("PartMC vs MAM4 scenario comparison")
    fig.tight_layout()
    out = repo_root / "examples" / "out_grid_partmc_mam4.png"
    fig.savefig(out, dpi=180)
    print(f"Wrote: {out}")

    # Demonstrate direct variable computation (first scenario, PartMC population) for orientation
    try:
        demo_pop = partmc_builder(scenario_cfgs[0])
        demo_sd = compute_variable(demo_pop, "dNdlnD", {"N_bins": 20})
        print("Demo size distribution keys:", demo_sd.keys())
    except Exception as e:
        print("Demo compute_variable failed (non-fatal):", e)


if __name__ == "__main__":
    main()
