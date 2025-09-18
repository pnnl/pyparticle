# Prompt: Refactor Existing Aerosol Ensemble Visualization to Use PyParticle.viz

You are an AI assistant tasked with refactoring an existing aerosol/ensemble visualization module (currently using bespoke pandas/seaborn ridge plots and manual size‑distribution assembly) to leverage the PyParticle visualization stack. DO NOT fabricate data or alter scientific meaning. Replace ad‑hoc PDF / ridge / comparison plotting with PyParticle's declarative population + plotting helpers.

Maintain functional parity for: (a) plotting PartMC vs MAM4 size distribution comparisons by scenario; (b) optionally adding CCN or optical coefficient lines; (c) per‑scenario multi‑model overlays. You should remove the custom `build_sizedist_df`, `plot_ridge`, and related seaborn FacetGrid logic and instead call `plot_lines` (and grid helpers where appropriate). Keep external ensemble generation logic (retrieving PartMC & MAM4 outputs) intact, only swapping the plotting layer.

---
## 1. PyParticle Essentials

Import surface (top‑level exports and viz):
```python
from PyParticle import build_population, build_optical_population  # if optics needed
from PyParticle.viz.plotting import plot_lines
from PyParticle.viz.grids import (
    make_grid_popvars,
    make_grid_scenarios_timesteps,
    make_grid_mixed,
    make_grid_scenarios_models,
    make_grid_optics_vs_wvl,
    make_grid_optics_vs_rh,
)
```

### Population Builders
Two relevant builder types for your refactor:
1. PartMC: config requires at minimum
```python
partmc_cfg = {
    "type": "partmc",
    "partmc_dir": "/abs/path/to/partmc/run",  # directory containing 'out/' NetCDF
    "timestep": 0,  # integer index / seconds (depending on run convention)
    "repeat": 0,
}
```
2. MAM4: (current provisional shape)
```python
mam4_cfg = {
    "type": "mam4",
    "output_filename": "/abs/path/to/mam4_output.nc",
    "timestep": 0,
    # plus required modal grid params if not embedded in file (GSD, D_min, D_max, N_bins, p, T)
}
```
Use `build_population(config_dict)` for PartMC (auto‑discovered builder) and the explicit function for MAM4 if not top‑level exported: `from PyParticle.population.factory.mam4 import build as build_mam4`.

Never synthesize stand‑in NetCDF / particle data. If a referenced directory/file is missing, raise `FileNotFoundError` with remediation steps.

---
## 2. Plotting API Overview

### Core Low-Level Primitive
`plot_lines(varname, (population,...), var_cfg=None, ax=None, colors=None, linestyles=None, linewidths=None, markers=None)`

Supported `varname` values (current scope relevant to refactor):
- `dNdlnD` (size distribution); x = diameter (m), y = dN/dlnD
- `Nccn` (absolute CCN spectrum) — requires supersaturation vector in `var_cfg['s_eval']`
- `frac_ccn` (fraction activated) — same supersaturation vector
- Optical coefficients: `b_abs`, `b_scat`, `b_ext`, and totals `total_abs`, `total_scat`, `total_ext` (when optics/`rh_grid` desired). For simple size distribution comparisons you can skip optics initially.

Returns `(line_artist, (xlabel, ylabel))`. The function internally sets axis scales (e.g., log for diameter) based on the prepared data.

### Default Variable Configuration (`var_cfg`)
You can either:
1. Start from defaults using:
```python
from PyParticle.viz.data_prep import build_default_var_cfg
cfg = build_default_var_cfg("dNdlnD")
```
2. Provide minimal overrides directly.

Key defaults for `dNdlnD`:
```python
{
  "wetsize": True,     # wet diameter
  "normalize": False,  # raw number distribution
  "method": "hist",   # bin method
  "N_bins": 30,
  "D_min": 1e-9,
  "D_max": 1e-4,
  "diam_scale": "log"
}
```
For `frac_ccn` / `Nccn`:
```python
{"s_eval": np.linspace(0.01, 1.0, 50), "T": 298.15}
```

### Grid Helpers (Higher-Level)
Use these when you want structured multi‑panel output without manual subplot code.

1. `make_grid_popvars(rows, columns, var_cfg=None, ...)`:
   - `rows`: list of population config dicts (or prebuilt populations).
   - `columns`: list of variable names (each one axis column).
   - Builds each population once per row.

2. `make_grid_scenarios_models(scenarios, variables, model_cfg_builders, ...)`:
   - `scenarios`: list of scenario config dicts (merged / union of keys needed by each model builder).
   - `model_cfg_builders`: list of callables; each receives a scenario dict copy and returns a `ParticlePopulation` (e.g., PartMC then MAM4) to overlay lines per axis.
   - Each axis: one variable, multiple model populations (styled via linestyles/colors lists or defaults).

These helpers call `plot_lines` internally and apply axis formatting; you can still call `fig.suptitle`, `fig.tight_layout()`, etc., afterward.

---
## 3. Mapping Old Code to New Abstractions

| Legacy Element | Replacement with PyParticle |
|----------------|-----------------------------|
| `build_sizedist_df(outputs, ...)` building pandas DataFrame | (Not needed) Directly call `plot_lines("dNdlnD", (population,), var_cfg)`; the internal prep bins the distribution. |
| Seaborn `FacetGrid` loops for ridge bars | Use `make_grid_popvars` (single model) or `make_grid_scenarios_models` (multi-model overlay). |
| Manual color blending / outline logic | Optional: pass `colors=[...]`, `linestyles=[...]` to `plot_lines` or rely on Matplotlib defaults. Post‑refactor minimal styling is acceptable. |
| Ridge “bars” / filled polygons | Simplify to line plots only (explicit requirement). |
| `plot_ridge(df_partmc, dfB=df_mam4, ...)` comparing two models | Use `make_grid_scenarios_models` with two builder callables returning PartMC & MAM4 populations. |
| `get_Dlims` manual diameter bounds | Set `D_min`, `D_max` in `var_cfg` for uniform bin domains if cross‑scenario comparability needed. |

---
## 4. Concrete Refactor Steps

1. Remove / deprecate (or wrap as no‑op) functions: `build_sizedist_df`, `plot_ridge`, ridge‑style color utilities. Keep only if other code still imports them; otherwise, delete.
2. Add a lightweight adapter that, given raw ensemble output objects (`partmc_output`, `mam4_output`), returns PyParticle `ParticlePopulation` objects:
   - If those outputs ALREADY encapsulate or embed PyParticle populations, just extract them.
   - Else: construct a PyParticle config using existing directories and call `build_population` / `build_mam4`.
3. Assemble scenario configurations:
```python
scenario_cfgs = []
for sid in scenario_names:
    scenario_cfgs.append({
        # PartMC keys
        "type": "partmc",
        "partmc_dir": f"{partmc_root}/{sid}",
        "timestep": timestep,
        "repeat": repeat_num,
        # MAM4 keys (namespaced separately for builder extraction below)
        "output_filename": f"{mam4_root}/{sid}/mam_output.nc",
        # Additional mam4 required params if not stored in file (GSD, D_min, ...)
    })
```
4. Define model builders:
```python
from PyParticle.population import build_population
from PyParticle.population.factory.mam4 import build as build_mam4

def partmc_builder(cfg):
    part_keys = {"type","partmc_dir","timestep","repeat","species_modifications"}
    part_cfg = {k: v for k,v in cfg.items() if k in part_keys}
    return build_population(part_cfg)

def mam4_builder(cfg):
    mam_keys = {"type","output_filename","timestep","GSD","D_min","D_max","N_bins","p","T"}
    mam_cfg = {k: v for k,v in cfg.items() if k in mam_keys}
    mam_cfg.setdefault("type", "mam4")
    from PyParticle.population.factory.mam4 import build as build_mam4
    return build_mam4(mam_cfg)
```
5. Decide variables list (e.g., `variables = ["dNdlnD", "frac_ccn"]`).
6. Optionally build per‑variable config mapping:
```python
import numpy as np
var_cfg = {
  "dNdlnD": {"N_bins": 40, "D_min": 1e-9, "D_max": 2e-6, "wetsize": True},
  "frac_ccn": {"s_eval": np.linspace(5e-4, 0.02, 40)}
}
```
7. Create grid:
```python
from PyParticle.viz.grids import make_grid_scenarios_models
fig, axes = make_grid_scenarios_models(
    scenario_cfgs,
    variables,
    model_cfg_builders=[partmc_builder, mam4_builder],
    var_cfg=var_cfg,
    figsize=(4*len(variables), 3*len(scenario_cfgs))
)
fig.suptitle("PartMC vs MAM4 (line plots)")
fig.tight_layout()
fig.savefig("comparison.png", dpi=180)
```
8. For a single‑model (only PartMC) scenario × variable grid:
```python
from PyParticle.viz.grids import make_grid_popvars
partmc_rows = [
  {"type":"partmc","partmc_dir": f"{partmc_root}/{sid}", "timestep": timestep, "repeat": repeat_num}
  for sid in scenario_names
]
fig, axes = make_grid_popvars(partmc_rows, ["dNdlnD", "frac_ccn"], var_cfg=None, figsize=(8, 3*len(partmc_rows)))
fig.tight_layout()
fig.savefig("partmc_only.png", dpi=180)
```

---
## 5. Edge Cases & Guidance

1. Missing Files: Immediately raise explicit errors; do not substitute fake arrays.
2. Bin Consistency: If comparing multiple scenarios or models, set identical `N_bins`, `D_min`, `D_max` across all `dNdlnD` plots for consistent x‑domain.
3. Performance: Populations reused within the same figure should be built once (grid helpers already do this per row). Avoid re‑building inside per‑variable loops.
4. Styling: Legends are empty by default because lines lack labels (the current `plot_lines` returns only artists). If labeled distinctions (e.g., PartMC vs MAM4) are desired, you can post‑annotate:
```python
for ax in axes.ravel():
    for i, line in enumerate(ax.lines):
        line.set_label(["PartMC","MAM4"][i])
    ax.legend(frameon=False, fontsize=9)
```
5. CCN Fractions: Ensure the physical range of supersaturation matches legacy behavior; adapt `s_eval` accordingly.
6. Optical Adds (Optional Later): Provide `varname` among optical coeffs with a `var_cfg` containing `wvls` and `rh_grid` (wavelength in meters). Example:
```python
opt_cfg = {"wvls": [450e-9, 550e-9, 650e-9], "rh_grid": [0.0, 0.8, 0.9, 0.95], "vs_wvl": True}
plot_lines("b_scat", (population,), opt_cfg, ax=ax)
```

---
## 6. Deleting / Simplifying Legacy Functions

- `plot_ridge`: Remove entirely. Its functional objective (multi‑row scenario visualization) is superseded by grid helpers plus direct line plotting.
- `build_sizedist_df`: Remove; binning occurs internally in `prepare_dNdlnD` invoked by `plot_lines`.
- Color utilities (`get_row_colors`, `blend_colors`, etc.): Remove unless reused outside plotting. Start minimal; reintroduce only if required.

---
## 7. Minimal Refactored Example (Put in New Script)
```python
import json, numpy as np
from pathlib import Path
from PyParticle.viz.grids import make_grid_scenarios_models
from PyParticle.population import build_population
from PyParticle.population.factory.mam4 import build as build_mam4

# Scenario setup
scenario_ids = ["001","005","007"]
partmc_root = Path("/abs/path/partmc_runs")
mam4_root = Path("/abs/path/mam4_runs")
timestep = 0

scenario_cfgs = []
for sid in scenario_ids:
    scenario_cfgs.append({
        "type": "partmc",
        "partmc_dir": str(partmc_root / sid),
        "timestep": timestep,
        "repeat": 0,
        "output_filename": str(mam4_root / sid / "mam_output.nc"),
    })

def partmc_builder(cfg):
    keys = {"type","partmc_dir","timestep","repeat"}
    return build_population({k: cfg[k] for k in keys})

def mam4_builder(cfg):
    return build_mam4({"type":"mam4","output_filename": cfg["output_filename"], "timestep": cfg["timestep"]})

variables = ["dNdlnD", "frac_ccn"]
var_cfg = {
  "dNdlnD": {"N_bins": 40, "D_min": 1e-9, "D_max": 2e-6, "wetsize": True},
  "frac_ccn": {"s_eval": np.linspace(5e-4, 0.02, 40)}
}

fig, axes = make_grid_scenarios_models(
    scenario_cfgs, variables, [partmc_builder, mam4_builder], var_cfg=var_cfg,
    figsize=(8, 3*len(scenario_cfgs))
)
for ax in axes.ravel():
    if len(ax.lines) == 2:
        ax.lines[0].set_label("PartMC")
        ax.lines[1].set_label("MAM4")
        ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("partmc_mam4_comparison.png", dpi=180)
```

---
## 8. Acceptance Criteria
Refactored module should:
1. Produce line plots (no filled bars) for size distributions per scenario.
2. Support optional overlay of second model (MAM4) with distinguishable linestyle.
3. Remove pandas/seaborn dependencies from plotting layer (matplotlib only via PyParticle).
4. Work with real PartMC & MAM4 output; raise on missing files.
5. Allow pluggable variable configuration without editing core plotting logic.
6. Preserve ability to extend with CCN or optical variables using same grid scaffolding.

---
## 9. Non-Goals / Don’ts
- Do NOT recreate ridge/baseline offset aesthetics.
- Do NOT introduce mock/synthetic particle arrays if data absent.
- Do NOT modify upstream ensemble generation logic beyond what is necessary to pass correct config dicts to builders.
- Do NOT silently auto‑guess missing physics parameters—fail loudly instead.

---
## 10. Deliverables
Provide:
1. Updated plotting script(s) using PyParticle viz.
2. Removal of obsolete ridge utilities.
3. OPTIONAL: Small README snippet summarizing new usage (if repository includes docs).
4. Confirmation that seaborn/pandas imports are no longer required in plotting path.

End of prompt.

---
## 11. Additional Example Request (New Figure Matching Existing `viz_grid_partmc_and_mam4`)

Create a NEW example script (do not overwrite the existing one) that reproduces a PartMC vs MAM4 multi-scenario comparison figure (same visual semantics as `examples/viz_grid_partmc_and_mam4.py`). Name suggestion: `examples/viz_grid_partmc_and_mam4_alt.py`.

Requirements:
1. Use `make_grid_scenarios_models` exactly as in the original example (no custom subplot loops).
2. Allow CLI overrides for: `--config <path/to/partmc_mam4_example.json>`, `--out <outfile.png>`, `--variables dNdlnD frac_ccn`.
3. Provide an optional flag `--include-optics` that, when set, appends `b_scat` to variables AND supplies an optics-specific var_cfg override (e.g., `{"b_scat": {"wvls": [550e-9], "rh_grid": [0.0, 0.9], "vs_wvl": True}}`). If optics data / morphology prerequisites are absent, raise a clear error (do not fallback silently).
4. Add a small legend labeling model lines (PartMC vs MAM4) for each axis.
5. Surface tension warning still suppressed.
6. If any scenario directory or MAM4 file missing, exit with non-zero code after printing a concise diagnostic summary of all missing items.
7. Output file name default: `examples/out_grid_partmc_mam4_alt.png` unless `--out` specified.

Script Skeleton (adapt and enrich):
```python
#!/usr/bin/env python
import argparse, json, warnings
from pathlib import Path
import numpy as np
from PyParticle.viz.grids import make_grid_scenarios_models
from PyParticle.population import build_population
from PyParticle.population.factory.mam4 import build as build_mam4

def partmc_builder(cfg):
    keys = {"type","partmc_dir","timestep","repeat","species_modifications"}
    return build_population({k: v for k in cfg if k in keys})

def mam4_builder(cfg):
    mkeys = {"type","output_filename","timestep","GSD","D_min","D_max","N_bins","p","T"}
    mc = {k: cfg[k] for k in mkeys if k in cfg}
    mc.setdefault("type","mam4")
    return build_mam4(mc)

def build_var_cfg(variables, include_optics):
    vcfg = {
      "dNdlnD": {"N_bins": 40, "D_min": 1e-9, "D_max": 2e-6, "wetsize": True},
      "frac_ccn": {"s_eval": np.linspace(5e-4, 0.02, 40)},
      "Nccn": {"s_eval": np.linspace(5e-4, 0.02, 40)}
    }
    if include_optics and "b_scat" in variables:
        vcfg["b_scat"] = {"wvls": [550e-9], "rh_grid": [0.0], "vs_wvl": True}
    return {k: vcfg[k] for k in variables if k in vcfg}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--variables", nargs="*", default=["dNdlnD","frac_ccn"]) 
    ap.add_argument("--include-optics", action="store_true")
    args = ap.parse_args()
    cfg_all = json.load(open(args.config))
    scenario_ids = cfg_all.get("scenarios", [])
    if not scenario_ids:
        raise SystemExit("No scenarios listed in config JSON")
    partmc_root = Path(cfg_all.get("partmc_root",""))
    mam4_root = Path(cfg_all.get("mam4_root",""))
    timestep = cfg_all.get("timestep", 0)

    partmc_base = {"type":"partmc","timestep": timestep, "repeat": cfg_all.get("repeat",1),
                   "species_modifications": cfg_all.get("partmc_species_modifications",{})}
    mam4_defaults = dict(cfg_all.get("mam4_defaults", {}))
    mam4_defaults.setdefault("timestep", timestep)

    # Build scenarios (merged dict per scenario)
    scenario_cfgs = []
    missing = []
    for sid in scenario_ids:
        partmc_dir = partmc_root / sid
        mam4_file = mam4_root / sid / "mam_output.nc"
        if not partmc_dir.exists():
            missing.append(f"PartMC:{partmc_dir}")
        if not mam4_file.exists():
            missing.append(f"MAM4:{mam4_file}")
        scenario_cfgs.append({**partmc_base, "partmc_dir": str(partmc_dir), **mam4_defaults, "output_filename": str(mam4_file)})
    if missing:
        raise SystemExit("Missing required inputs:\n" + "\n".join(missing))

    # Variables (append b_scat if optics flag)
    variables = list(args.variables)
    if args.include_optics and "b_scat" not in variables:
        variables.append("b_scat")

    var_cfg = build_var_cfg(variables, args.include_optics)

    warnings.filterwarnings(
        "ignore", message="Surface tension not implemented; returning default",
        category=UserWarning, module="PyParticle.aerosol_particle")

    fig, axes = make_grid_scenarios_models(
        scenario_cfgs, variables, [partmc_builder, mam4_builder], var_cfg=var_cfg,
        figsize=(4*len(variables), 3*len(scenario_cfgs))
    )
    # Label model lines for each axis
    for ax in axes.ravel():
        if len(ax.lines) >= 2:
            ax.lines[0].set_label("PartMC")
            ax.lines[1].set_label("MAM4")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("PartMC vs MAM4 (Alt)")
    fig.tight_layout()
    out = args.out or (Path(args.config).parent / "out_grid_partmc_mam4_alt.png")
    fig.savefig(out, dpi=180)
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
```

Add this new script alongside the existing examples. It must not rely on seaborn or pandas.

