"""Example: Compare PartMC vs MAM4 populations across multiple scenarios (no pandas).

This refactored example intentionally avoids any DataFrame/pandas layer and instead
uses the existing ``viz.data_prep`` + ``viz.plotting.plot_lines`` APIs directly.

Workflow:
 1. Load comparison config JSON (``examples/configs/partmc_mam4_example.json``).
 2. For each scenario id build:
            - PartMC population (``type: partmc``) pointing at the scenario directory.
            - MAM4 population (via ``population.factory.mam4``) from a NetCDF file.
 3. Create a grid (rows = scenarios, columns = selected variables) and for each
        cell call ``plot_lines`` once with both populations. Scenario color is shared
        across both origins; origins are distinguished by linestyle (PartMC='-', MAM4='--').

Grid semantics:
    rows    -> scenario id
    columns -> variable name (e.g., dNdlnD, frac_ccn)
    lines   -> model origin (PartMC vs MAM4)

Output written to: ``examples/out_grid_partmc_mam4.png``

If your MAM4 filename differs, adjust `_build_mam4_cfg` accordingly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from PyParticle.population import build_population  # PartMC factory (PartMC)
from PyParticle.viz.plotting import plot_lines
from PyParticle.viz.layout import make_grid
from PyParticle.viz.formatting import format_axes, add_legend
from PyParticle.viz.stylemap import map_colors_for_scenarios, map_linestyles_for_series


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


def _try_build_partmc(cfg: Dict[str, Any]) -> Any:
    run_dir = Path(cfg["partmc_dir"]) if "partmc_dir" in cfg else None
    pop = build_population(cfg)
    pop.label = run_dir.name
    pop.origin = "partmc"
    return pop

def _try_build_mam4(cfg: Dict[str, Any]) -> Any:
    from PyParticle.population.factory.mam4 import build as build_mam4
    out_file = Path(cfg.get("output_filename", ""))
    pop = build_mam4(cfg)
    pop.label = out_file.parent.name or "mam4"
    pop.origin = "mam4"
    return pop

def _variable_varcfg(varname: str, overrides: Dict[str, Any] | None) -> Dict[str, Any]:
    """Return a var_cfg dict for a variable applying optional overrides."""
    base_defaults = {
        "dNdlnD": {"wetsize": True, "N_bins": 40, "D_min": 10e-9, "D_max": 2e-6},
        "frac_ccn": {"s_eval": np.linspace(5e-4, 0.02, 40)},
        "Nccn": {"s_eval": np.linspace(5e-4, 0.02, 40)},
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

    variables = cfg_all.get("variables", ["dNdlnD", "frac_ccn"])  # default two variables

    if not scenario_ids:
        raise SystemExit("No scenarios specified in config.")

    # Build populations for each scenario (store tuples so we only build once per row)
    built: List[Tuple[str, Any, Any]] = []  # (scenario_id, partmc_pop, mam4_pop)
    for sid in scenario_ids:
        partmc_pop = _try_build_partmc(_build_partmc_cfg(partmc_base, sid, partmc_root))
        mam4_pop = _try_build_mam4(_build_mam4_cfg(mam4_defaults, sid, mam4_root))
        built.append((sid, partmc_pop, mam4_pop))

    nrows = len(built)
    ncols = len(variables)
    fig, axes = make_grid(nrows, ncols, figsize=(4 * ncols, 3 * nrows))

    # Styling maps (deterministic)
    scenario_colors = map_colors_for_scenarios([sid for sid, _, _ in built])
    origin_linestyles = map_linestyles_for_series(["PartMC", "MAM4"])  # e.g., {'PartMC': '-', 'MAM4': '--'}

    for irow, (sid, partmc_pop, mam4_pop) in enumerate(built):
        for jcol, varname in enumerate(variables):
            ax = axes[irow, jcol]
            var_overrides = (cfg_all.get("var_cfg_overrides", {}) or {}).get(varname, {})
            var_cfg = _variable_varcfg(varname, var_overrides)

            # scenario color applied to both origins; linestyle distinguishes origin
            color = scenario_colors[sid]
            linestyles = [origin_linestyles.get("PartMC", "-"), origin_linestyles.get("MAM4", "--")]
            # plot both populations in one call
            line, labs = plot_lines(varname, (partmc_pop, mam4_pop), var_cfg=var_cfg, ax=ax,
                                    colors=[color, color], linestyles=linestyles)

            xlabel = labs[0] if isinstance(labs, (list, tuple)) and labs else None
            ylabel = labs[1] if isinstance(labs, (list, tuple)) and len(labs) > 1 else None
            title = f"{sid} – {varname}"
            format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)
            # legend: add only once per row (rightmost cell) or every cell? choose once per row last column
            if jcol == ncols - 1:
                # Manually create legend entries for origins
                from matplotlib.lines import Line2D
                legend_lines = [Line2D([0], [0], color=color, linestyle=origin_linestyles.get("PartMC", "-"), label="PartMC"),
                                Line2D([0], [0], color=color, linestyle=origin_linestyles.get("MAM4", "--"), label="MAM4")]
                ax.legend(handles=legend_lines, fontsize=8, loc="upper right")

    out = repo_root / "examples" / "out_grid_partmc_mam4.png"
    fig.tight_layout()
    fig.savefig(out, dpi=175)
    print("Wrote:", out)


if __name__ == "__main__":
    main()
