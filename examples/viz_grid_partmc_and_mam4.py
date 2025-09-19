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
import os
import sys

# fixme: not yet ready
# Ensure repo root is on sys.path so `examples.*` imports work when running
# this script directly (not as a package).
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from PyParticle.population import build_population
from PyParticle.analysis import compute_variable  # Demonstration (not strictly needed in grid path)
from PyParticle.viz.grids import make_grid_scenarios_models
from PyParticle.viz.stylemap import map_linestyles_for_series  # if needed for custom legend


from examples.model_helpers import (
    build_populations_and_varcfgs,
    build_var_cfg_mapping,
)


# def _build_partmc_cfg(base_cfg: Dict[str, Any], scenario_id: str, partmc_root: Path) -> Dict[str, Any]:
#     # Use helper to build a standard partmc config dict and merge any base keys
#     cfg = retrieve_partmc_cfg(scenario_id, partmc_root, timestep=base_cfg.get("timestep", 1), repeat_num=base_cfg.get("repeat", 1), species_modifications=base_cfg.get("species_modifications", {}))
#     # Preserve any explicit overrides provided in base_cfg
#     merged = dict(base_cfg)
#     merged.update(cfg)
#     return merged


# def _build_mam4_cfg(mam4_defaults: Dict[str, Any], scenario_id: str, mam4_root: Path) -> Dict[str, Any]:
#     # Use helper which already returns a file-backed cfg dict; merge defaults into it.
#     cfg = retrieve_mam4_cfg(scenario_id, mam4_root, timestep=mam4_defaults.get("timestep", 1), mam4_settings={k:v for k,v in mam4_defaults.items() if k not in ("p","T","timestep")})
#     merged = dict(mam4_defaults)
#     merged.update(cfg)
#     return merged


# def _build_partmc_population(cfg: Dict[str, Any]):
#     pop = build_population(cfg)
#     pop.origin = "PartMC"
#     return pop

# def _build_mam4_population(cfg: Dict[str, Any]):
#     pop = build_population(cfg)
#     pop.origin = "MAM4"
#     return pop

# def _variable_varcfg(varname: str, overrides: Dict[str, Any] | None) -> Dict[str, Any]:
#     """Return a var_cfg dict for a variable applying optional overrides."""
#     base_defaults = {
#         "dNdlnD": {"wetsize": True, "N_bins": 40, "D_min": 10e-9, "D_max": 2e-6},
#         "frac_ccn": {"s_eval": np.logspace(-2, 1, 40)},
#         "Nccn": {"s_eval": np.logspace(-2, 1, 40)},
#     }.get(varname, {})
#     cfg = dict(base_defaults)
#     if overrides:
#         cfg.update(overrides)
#     return cfg


def main():
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "examples" / "configs" / "partmc_mam4_example.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"Config file not found: {cfg_path}\nCreate it (see template) or adjust the path in the example." 
        )
    cfg_all = json.load(open(cfg_path))

    scenario_names = cfg_all.get("scenario_names", [])
    # partmc_root = Path(cfg_all.get("partmc_root", ""))
    # mam4_root = Path(cfg_all.get("mam4_root", ""))
    # timestep = cfg_all.get("timestep", 0)

    # partmc_base = {
    #     "type": "partmc",
    #     "timestep": timestep,
    #     "repeat": cfg_all.get("repeat", 1),
    #     "species_modifications": cfg_all.get("spec_modifications", {}),
    # # }
    # mam4_settings = cfg_all.get("mam4_settings", {})
    variables = cfg_all.get("variables", ["dNdlnD", "frac_ccn"])  # default two variables (optics use b_ext/b_abs/b_scat)

    # if not scenario_ids:
    #     raise SystemExit("No scenarios specified in config.")

    # Build list of scenario entries: use helper to create populations + var_cfgs
    scenario_cfgs: List[dict] = []
    built_populations: List[Tuple] = []
    verbose = bool(int(os.environ.get("PYPARTICLE_VERBOSE", "0")) == 1)
    for sid in scenario_names:
        out = build_populations_and_varcfgs(sid, cfg_all, verbose=verbose)
        # store the populations in the scenario dict so the grid helper can
        # pass prebuilt ParticlePopulation objects directly (avoids rebuilding)
        scenario_cfgs.append({
            "scenario_id": sid,
            "partmc_pop": out["partmc_pop"],
            "mam4_pop": out["mam4_pop"],
            # keep the original builder cfgs available for downstream use
            "partmc_cfg": out["partmc_cfg"],
            "mam4_cfg": out["mam4_cfg"],
        })
        built_populations.append((out["partmc_pop"], out["mam4_pop"], out["var_cfgs"]))

    # Variable-specific overrides mapping (use helper to construct defaults)
    overrides_map = cfg_all.get("var_cfg_overrides", {}) or {}
    # Build the mapping using example helper which extracts mam4 binning keys
    var_cfg_mapping = build_var_cfg_mapping(cfg_all)
    # Apply explicit overrides if provided
    for k, v in (overrides_map.items()):
        var_cfg_mapping[k] = {**var_cfg_mapping.get(k, {}), **v}

    # # Suppress surface tension warning explicitly
    # warnings.filterwarnings(
    #     "ignore",
    #     message="Surface tension not implemented; returning default",
    #     category=UserWarning,
    #     module="PyParticle.aerosol_particle",
    # )

    def partmc_builder(cfg):
        # If a prebuilt population is present, return it directly; otherwise
        # expect a scenario dict with 'partmc_cfg' and build from that.
        if isinstance(cfg, dict) and "partmc_pop" in cfg:
            return cfg["partmc_pop"]
        if isinstance(cfg, dict) and "partmc_cfg" in cfg:
            pop = build_population(cfg["partmc_cfg"])
            pop.origin = "PartMC"
            return pop
        raise TypeError("partmc_builder expected a scenario dict with 'partmc_pop' or 'partmc_cfg'")

    def mam4_builder(cfg):
        if isinstance(cfg, dict) and "mam4_pop" in cfg:
            return cfg["mam4_pop"]
        if isinstance(cfg, dict) and "mam4_cfg" in cfg:
            pop = build_population(cfg["mam4_cfg"])
            pop.origin = "MAM4"
            return pop
        raise TypeError("mam4_builder expected a scenario dict with 'mam4_pop' or 'mam4_cfg'")

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
