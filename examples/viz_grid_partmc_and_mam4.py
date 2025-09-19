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
from PyParticle.viz.layout import make_grid  # use local grid constructor instead of scenarios_models
from PyParticle.viz.plotting import plot_lines
from PyParticle.viz.stylemap import map_linestyles_for_series  # if needed for custom legend


from examples.model_helpers import (
    build_populations_and_varcfgs,
    build_var_cfg_mapping,
)

def main():
    repo_root = Path(__file__).resolve().parent.parent
    cfg_path = repo_root / "examples" / "configs" / "partmc_mam4_example.json"
    if not cfg_path.exists():
        raise SystemExit(
            f"Config file not found: {cfg_path}\nCreate it (see template) or adjust the path in the example." 
        )
    cfg_all = json.load(open(cfg_path))

    # Support configs that place GSD at top-level by copying into mam4_settings
    if "GSD" in cfg_all:
        mam4_settings = cfg_all.setdefault("mam4_settings", {})
        mam4_settings.setdefault("GSD", cfg_all["GSD"])

    scenario_names = cfg_all.get("scenario_names", [])
    if not scenario_names:
        raise SystemExit("No 'scenario_names' defined in config; nothing to run.")
    variables = cfg_all.get("variables", ["dNdlnD", "frac_ccn"])  # default two variables (optics use b_ext/b_abs/b_scat)

    scenario_cfgs: List[dict] = []
    built_populations: List[Tuple] = []
    verbose = bool(int(os.environ.get("PYPARTICLE_VERBOSE", "0")) == 1)

    # Suppress known surface-tension warnings from particle hygroscopic growth
    warnings.filterwarnings("ignore", message=r".*surface tension.*", category=Warning)

    import argparse
    parser = argparse.ArgumentParser(description="Compare PartMC vs MAM4 across scenarios")
    parser.add_argument("--diagnostics", action="store_true", help="Write diagnostics CSVs (disabled by default)")
    args = parser.parse_args()

    for sid in scenario_names:
        try:
            print(f"Building populations for scenario: {sid}")
            out = build_populations_and_varcfgs(sid, cfg_all, verbose=verbose)
        except Exception as e:
            warnings.warn(f"Skipping scenario {sid}: failed to build populations: {e}")
            continue
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
    # Apply explicit overrides if provided (none by default per instruction)
    for k, v in (overrides_map.items()):
        var_cfg_mapping[k] = {**var_cfg_mapping.get(k, {}), **v}
    # Compute colors (one per scenario) using matplotlib cycle
    default_cycle = plt.rcParams.get("axes.prop_cycle", None)
    if default_cycle is not None:
        cycle_list = default_cycle.by_key().get("color", ["C0"])
    else:
        cycle_list = ["C0", "C1", "C2", "C3"]
    colors = [cycle_list[i % len(cycle_list)] for i in range(len(scenario_cfgs))]

    # Custom linestyles: PartMC solid, MAM4 dashed
    model_linestyles = ["-", "--"]

    # Builder wrappers: prefer prebuilt populations returned by the helper
    def partmc_builder(cfg):
        if isinstance(cfg, dict) and "partmc_pop" in cfg:
            pop = cfg["partmc_pop"]
            try:
                pop.origin = "PartMC"
            except Exception:
                pass
            return pop
        if isinstance(cfg, dict) and "partmc_cfg" in cfg:
            pop = build_population(cfg["partmc_cfg"])
            try:
                pop.origin = "PartMC"
            except Exception:
                pass
            return pop
        raise TypeError("partmc_builder expected a scenario dict with 'partmc_pop' or 'partmc_cfg'")

    def mam4_builder(cfg):
        if isinstance(cfg, dict) and "mam4_pop" in cfg:
            pop = cfg["mam4_pop"]
            try:
                pop.origin = "MAM4"
            except Exception:
                pass
            return pop
        if isinstance(cfg, dict) and "mam4_cfg" in cfg:
            pop = build_population(cfg["mam4_cfg"])
            try:
                pop.origin = "MAM4"
            except Exception:
                pass
            return pop
        raise TypeError("mam4_builder expected a scenario dict with 'mam4_pop' or 'mam4_cfg'")

    # Build the grid (rows=scenarios, cols=variables). Use requested figsize policy
    nrows = len(scenario_cfgs)
    ncols = len(variables)
    figsize = (4 * len(variables), 3 * len(scenario_cfgs))
    fig, axes = make_grid(nrows, ncols, figsize=figsize)

    # For each scenario (row) plot both models on each variable (column)
    for i, scenario in enumerate(scenario_cfgs):
        row_color = colors[i]
        for j, varname in enumerate(variables):
            ax = axes[i, j]
            # Build or reuse populations in specified order: PartMC then MAM4
            pops = []
            try:
                p_pop = partmc_builder(scenario)
                pops.append(p_pop)
            except Exception as e:
                warnings.warn(f"Row {i} PartMC builder failed for scenario {scenario.get('scenario_id')}: {e}")
            try:
                m_pop = mam4_builder(scenario)
                pops.append(m_pop)
            except Exception as e:
                warnings.warn(f"Row {i} MAM4 builder failed for scenario {scenario.get('scenario_id')}: {e}")
            
            # Prepare style lists matching number of pops (colors per-row, linestyles per-model)
            line_colors = [row_color for _ in pops]
            linestyles = [model_linestyles[k % len(model_linestyles)] for k in range(len(pops))]

            # Resolve var_cfg for this variable (may be mapping or single dict)
            cfg_for_var = None
            if isinstance(var_cfg_mapping, dict) and varname in var_cfg_mapping:
                cfg_for_var = var_cfg_mapping[varname]
            else:
                cfg_for_var = var_cfg_mapping

            try:
                _, labs = plot_lines(varname, tuple(pops), cfg_for_var, ax=ax, colors=line_colors, linestyles=linestyles)
            except Exception as e:
                warnings.warn(f"Plotting failed for scenario {scenario.get('scenario_id')} var {varname}: {e}")
                continue

            # Title only on first column to avoid clutter
            title = varname if j == 0 else None
            try:
                xlabel = labs[0] if isinstance(labs, (list, tuple)) and len(labs) > 0 else None
                ylabel = labs[1] if isinstance(labs, (list, tuple)) and len(labs) > 1 else None
            except Exception:
                xlabel = None
                ylabel = None

            # Apply formatting (format_axes + legend handled inside helpers typically)
            try:
                from PyParticle.viz.formatting import format_axes, add_legend
                format_axes(ax, xlabel=xlabel, ylabel=ylabel, title=title, grid=False)
                # Ensure legend labels are present and consistent
                lines = ax.get_lines()
                if len(lines) >= 1:
                    # label first as PartMC if present
                    if len(lines) >= 1:
                        lines[0].set_label("PartMC")
                    if len(lines) >= 2:
                        lines[1].set_label("MAM4")
                    add_legend(ax)
            except Exception:
                pass

    fig.suptitle("PartMC vs MAM4 scenario comparison")
    fig.tight_layout()
    out = repo_root / "examples" / "out_grid_partmc_mam4.png"
    try:
        fig.savefig(out, dpi=200)
        print(f"Wrote: {out}")
    except Exception as e:
        print(f"Failed to save figure to {out}: {e}")




if __name__ == "__main__":
    main()


    # def partmc_builder(cfg):
    #     # If a prebuilt population is present, return it directly; otherwise
    #     # expect a scenario dict with 'partmc_cfg' and build from that.
    #     if isinstance(cfg, dict) and "partmc_pop" in cfg:
    #         return cfg["partmc_pop"]
    #     if isinstance(cfg, dict) and "partmc_cfg" in cfg:
    #         pop = build_population(cfg["partmc_cfg"])
    #         pop.origin = "PartMC"
    #         return pop
    #     raise TypeError("partmc_builder expected a scenario dict with 'partmc_pop' or 'partmc_cfg'")

    # def mam4_builder(cfg):
    #     if isinstance(cfg, dict) and "mam4_pop" in cfg:
    #         return cfg["mam4_pop"]
    #     if isinstance(cfg, dict) and "mam4_cfg" in cfg:
    #         pop = build_population(cfg["mam4_cfg"])
    #         pop.origin = "MAM4"
    #         return pop
    #     raise TypeError("mam4_builder expected a scenario dict with 'mam4_pop' or 'mam4_cfg'")

    # fig, axes = make_grid_scenarios_models(
    #     scenario_cfgs,
    #     variables,
    #     model_cfg_builders=[partmc_builder, mam4_builder],
    #     var_cfg=var_cfg_mapping,
    #     figsize=(4 * len(variables), 3 * len(scenario_cfgs)),
    # )

    # fig.suptitle("PartMC vs MAM4 scenario comparison")
    # fig.tight_layout()
    # out = repo_root / "examples" / "out_grid_partmc_mam4.png"
    # fig.savefig(out, dpi=180)
    # print(f"Wrote: {out}")

    # # Diagnostics: compare size distributions (dNdlnD) between PartMC and MAM4
    # # for each scenario and write CSVs to examples/out_dNdlnD_comparison_<scenario>.csv
    # small = 1e-30
    # for scen in scenario_cfgs:
    #     sid = scen.get("scenario_id", "unknown")
    #     try:
    #         p_pop = partmc_builder(scen)
    #         m_pop = mam4_builder(scen)
    #     except Exception as e:
    #         print(f"Skipping diagnostics for scenario {sid}: failed to build populations: {e}")
    #         continue

    #     # Use the same var_cfg mapping used for plotting
    #     dcfg = var_cfg_mapping.get("dNdlnD", {})
    #     try:
    #         sd_p = compute_variable(p_pop, "dNdlnD", dict(dcfg))
    #         sd_m = compute_variable(m_pop, "dNdlnD", dict(dcfg))
    #     except Exception as e:
    #         print(f"Failed to compute dNdlnD for scenario {sid}: {e}")
    #         continue

    #     Dp = np.asarray(sd_p["D"])
    #     Pd = np.asarray(sd_p["dNdlnD"])
    #     Dm = np.asarray(sd_m["D"])
    #     Md = np.asarray(sd_m["dNdlnD"])

    #     # Align MAM4 onto PartMC diameter centers if needed via log-space interpolation
    #     if not np.allclose(Dp, Dm):
    #         try:
    #             Md_safe = Md + small
    #             Md_interp = np.exp(np.interp(np.log(Dp), np.log(Dm), np.log(Md_safe)))
    #         except Exception:
    #             # fallback to linear interpolation if log-interp fails
    #             Md_interp = np.interp(Dp, Dm, Md)
    #     else:
    #         Md_interp = Md

    #     # Compute relative bias where PartMC > 0
    #     mask = Pd > 0
    #     rel_pct = np.full_like(Pd, np.nan, dtype=float)
    #     if np.any(mask):
    #         rel_pct[mask] = (Md_interp[mask] - Pd[mask]) / Pd[mask] * 100.0

    #     # Mean absolute percent bias (ignore NaNs)
    #     mean_abs_bias = float(np.nanmean(np.abs(rel_pct))) if np.any(mask) else float('nan')

    #     out_csv = repo_root / "examples" / f"out_dNdlnD_comparison_{sid}.csv"
    #     # Write CSV: D, part_dNdlnD, mam4_dNdlnD, rel_bias_pct
    #     try:
    #         with open(out_csv, "w") as fh:
    #             fh.write("D,part_dNdlnD,mam4_dNdlnD,rel_bias_pct\n")
    #             for D, pval, mval, r in zip(Dp, Pd, Md_interp, rel_pct):
    #                 fh.write(f"{D:.6e},{pval:.6e},{mval:.6e},{'' if np.isnan(r) else f'{r:.6f}'}\n")
    #         print(f"Wrote dNdlnD comparison CSV for scenario {sid}: {out_csv}")
    #         print(f"Scenario {sid}: mean absolute percent bias = {mean_abs_bias:.3f}%")
    #         if not np.isnan(mean_abs_bias) and mean_abs_bias > 5.0:
    #             print(f"WARNING: mean absolute bias {mean_abs_bias:.3f}% > 5% threshold for scenario {sid}")
    #     except Exception as e:
    #         print(f"Failed to write CSV for scenario {sid}: {e}")

    # # Demonstrate direct variable computation (first scenario, PartMC population) for orientation
    # try:
    #     demo_pop = partmc_builder(scenario_cfgs[0])
    #     demo_sd = compute_variable(demo_pop, "dNdlnD", {"N_bins": 20})
    #     print("Demo size distribution keys:", demo_sd.keys())
    # except Exception as e:
    #     print("Demo compute_variable failed (non-fatal):", e)


