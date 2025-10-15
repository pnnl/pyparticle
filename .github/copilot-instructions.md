## PyParticle – Focused guide for AI coding agents

Goal: Be productive quickly when extending aerosol particle, population, optics, and viz functionality. Keep public APIs stable and follow existing registry + config patterns.

### Core architecture (src-layout under `src/PyParticle`)
1. Data layer: `datasets/species_data/*` consumed via species registry (`species/base.py`, `species/registry.py`). Override path with `PYPARTICLE_DATA_PATH`.
2. Domain models:
   - `aerosol_particle.py` (`Particle` dataclass: dry/wet diameters, kappa, masses)
   - `population/base.py` (`ParticlePopulation` container)
3. Builders (factory-discovery by module import):
   - Population: `population/builder.py` + dynamic discovery in `population/factory/registry.py` (expects each module to expose `build(config)`; module name becomes `type`).
   - Optics: `optics/builder.py` + `optics/factory/registry.py` (morphology modules expose `build(base_particle, config)` or use `@register`).
4. Optics abstractions: `optics/base.py` (`OpticalParticle`, `OpticalPopulation`) aggregated via `build_optical_population` (iterates base population ids, attaches optical particles).
5. Visualization subsystem: `viz/` split by concern (`layout.py`, `plotting.py`, `styling.py`, `formatting.py`, plus higher-level grid helpers in `viz/grids.py`). Plot helpers never set axis labels; formatting functions do that explicitly.
6. Public API surface: `__init__.py` re-exports (avoid breaking names used in examples/tests: `Particle`, `build_population`, `build_optical_population`, species registry helpers).

### Extension patterns (copy existing style)
Population type (add under `population/factory/`):
```
# new_type.py
def build(config):
    # config['type'] == 'new_type'
    ... return ParticlePopulation(...)
```
No manual registration needed; discovery imports every module in the folder and collects those with a `build` attr.

Optics morphology (add under `optics/factory/`):
```
from .registry import register
@register("my_morph")
def build(base_particle, config):
    ... return OpticalParticleSubclass(...)
```
OR supply a module-level `build` callable (will be picked up). Must accept `(base_particle, config)`.

Visualization grid helpers: pass either existing populations or config dicts; config dicts are auto-built via `build_population` before plotting. Always return fig/axes and leave labeling to formatting utilities.

### Config + units conventions
- Optics wavelength inputs in examples may use microns (`wvl_grid_um`) then converted to meters (`*1e-6`) before builder receives `wvl_grid` (meters only internally).
- Relative humidity grid key: `rh_grid`; default `[0.0]` if absent.
- Population config `type` must match filename of builder module (`binned_lognormals`, `monodisperse`, `partmc`, `mam4`, etc.).

### Testing & workflows
- Fast local / CI tests (unit + smoke): `pytest -q --maxfail=1 --disable-warnings` or targeted per `tests/README.md` (e.g. `pytest -q tests/unit`).
- Integration (optional heavy deps): `./tools/run_integration.sh` or `pytest -q tests/integration` after creating dev env (`environment-dev.yml`).
- Reference comparison harness (optional analytics regression):
```
pytest -q tests/run_all_comparisons.py --input examples/configs/ccn_single_na_cl.yml --compare both --output reports/reference_report.json
```

### Environment variables leveraged in tests/fixtures
- `PYPARTICLE_RUN_EXAMPLES=1` gate executing example scripts.
- `PYPARTICLE_FAST=1` (set internally for faster example behavior).
- `PYTEST_SEED=1337` deterministic RNG.
- `MPLBACKEND=Agg` enforced for headless plotting.
- `CUDA_VISIBLE_DEVICES=""` disable GPU usage.
- `PYTEST_ALLOW_NETWORK=1` only if network needed (default off).

### Adding / modifying species data
- Update `datasets/species_data/aero_data.dat`; ensure lookups via `get_species` still succeed.
- If relocating data, set `PYPARTICLE_DATA_PATH` in env during tests/CI.

### Common pitfalls (avoid)
- Forgetting `type` in config dicts (population/optics) -> ValueError in builders.
- Using microns inside optics builder; convert before calling.
- Mutating public exports in `__init__.py` without updating dependent tests/examples.
- Adding a new factory module without a `build` function or `@register` (it will be ignored).

### Strict data policy (important)
- Never fabricate or silently substitute mock populations, synthetic NetCDF content, or placeholder numeric arrays in examples or library code. If required external data (e.g., PartMC output directories, MAM4 NetCDF file, species data) is missing or an optional dependency (e.g., `netCDF4`) is not installed, raise a clear, explicit error (`FileNotFoundError` / `ModuleNotFoundError`) with remediation steps. Do NOT auto-generate stand‑in data.

### Lightweight smoke usage example
```
from PyParticle import build_population, build_optical_population
pop_cfg = {"type": "binned_lognormals", "modes": [...], "species": {...}}
pop = build_population(pop_cfg)
opt_cfg = {"type": "homogeneous", "rh_grid": [0.0, 0.5], "wvl_grid": [550e-9]}
opt_pop = build_optical_population(pop, opt_cfg)
```

### When writing code
- Maintain small, pure functions; mirror existing builder + registry style.
- Prefer reusing utilities (`utilities.py`) for lognormal or parsing logic instead of duplicating.
- New plotting functions should return artists and not set labels; let `formatting` handle presentation.

Feedback welcome: If a workflow or extension path is unclear (e.g. adding a fractal morphology implementation or new grid visualization), request an expansion and specify the gap.

### Conda environments (runtime guidance)
- Base development & non-PARTMC examples: create/activate `environment.yml` -> `conda env create -f environment.yml` (env name: `pyparticle`).
- PARTMC or MAM4 population examples (require `netCDF4` and possibly large NetCDF inputs): use `environment-partmc.yml` (env name: `pyparticle`).
- Always run PartMC/MAM4 examples with: `conda run -n pyparticle python examples/<script>.py` to ensure `netCDF4` is present.
- Do NOT silently fall back to another environment—raise if `netCDF4` import fails or required files absent.
 - Variant generation workflow: run `python tools/generate_env_variants.py --write` to (re)generate `environment-dev.yml` and `environment-partmc.yml` from the base `environment.yml` before creating/updating those envs.
 - Create variant envs after generation:
     * Dev: `conda env create -f environment-dev.yml` (name auto-suffixed `-dev`).
     * PartMC: `conda env create -f environment-partmc.yml` (name auto-suffixed `-partmc`).
 - If env files already exist and you modify base deps, regenerate variants and `conda env update -f environment-partmc.yml -n pyparticle`.

### Advanced: PARTMC & MAM4 population builders
PARTMC (`population/factory/partmc.py`)
- Required config keys: `type: "partmc"`, `partmc_dir` (directory containing `out/` NetCDF files), `timestep` (int), `repeat` (int).
- Optional: `n_particles` (subsample), `N_tot` (override total number conc), `species_modifications` (dict of per-species property overrides), `specdata_path`, `suppress_warning` (bool), `add_mixing_ratios` (attach gas mixing ratios).
- Behavior: Opens NetCDF, reads `aero_species`, `aero_particle_mass`, `aero_id`, and either `aero_num_conc` or derives concentration as `1/comp_vol`. Subsamples if `n_particles` specified. Each particle built via `make_particle_from_masses(...)` then inserted with scaled number concentration so selected subset preserves total (`N_tot`).
- Extension tips: If you add new fields (e.g. temperature-dependent properties) read variable inside the loop and stash on population (e.g. `population.extra[field] = arr`). Keep NetCDF import guarded by try/except for optional dependency.

MAM4 (`population/factory/mam4.py`)
- Required config keys (current implementation): `type: "mam4"`, `output_filename` (NetCDF), `timestep`, arrays: `GSD` (per-mode geometric std dev list), diameter bounds `D_min`, `D_max`, `N_bins`, thermodynamic state `p` (Pa), `T` (K). (Note: several have TODO/fixme comments; treat current behavior as provisional.)
- Internals: Reads modal number (`num_aer`), species masses (`so4_aer`, `soa_aer`), dry & wet geometric mean diameters (`dgn_a`, `dgn_awet`). Converts number to number concentration via dry-air density `rho_dry_air = MOLAR_MASS_DRY_AIR * p / (R*T)`. Filters modes with positive N. Computes per-mode water mass by difference of 3rd moment (wet minus dry) using `power_moments_from_lognormal(3, N, gmd, gsd)` as a surrogate for volume; multiplies by liquid water density. Constructs per-mode species fractions (SO4, OC, H2O) and hands an augmented config to the existing `binned_lognormals` builder (reusing population logic).
- Caveats: `species_modification` key (singular) used (typo vs modifications). `aero_spec_names` hard-coded to four modes of `[SO4, OC, H2O]` lists; adjust when expanding species list. `D_is_wet` flag selects whether to treat geometric mean diameters as wet; currently set True. If adding additional species, update both `aero_spec_names` and fraction assembly logic.

### Visualization internals (grids)
- Grid helpers (`viz/grids.py`) wrap base plotting primitives without side effects on global state. They always:
    1. Build populations from config dicts (injecting `timestep` for scenario/time grids).
    2. Call `plot_lines(varname, (population,), var_cfg, ax)` for each axis (multiple variables sequentially for scenario/timestep grids).
    3. Infer axis labels from the last `plot_lines` return (`labs`).
    4. Apply heuristics (log x-scale if var name suggests diameter: contains `dNdlnD`, `D`, or `diam`).
    5. Apply `format_axes` then `add_legend` per axis.
- `make_grid_popvars`: rows = populations/configs, columns = variable names (one var per axis). Optional `time` parameter passed into each config.
- `make_grid_scenarios_timesteps`: rows = scenario configs (no prebuilt populations), columns = timesteps, `variables` = list of varnames all plotted on each axis (multi-line). Title auto-set to `t=<timestep>`.
- `make_grid_mixed`: convenience wrapper to allow mixing prebuilt `ParticlePopulation` objects and config dict rows; otherwise same semantics as popvars.
- Sharing: `sharex_columns` and `sharey_rows` join axes after plotting (calls `get_shared_x_axes().join(...)`). Disable if axes have heterogeneous domains.
- Styling: Spine removal for top/right by default; toggle with `hide_spines=False`.
- Extending: New grid pattern? Implement analogous helper that (a) creates layout via `make_grid`, (b) builds populations lazily per row/column, (c) calls `plot_lines` consistently, and (d) defers all labeling/legend logic to formatting utilities.

### Practical examples
PARTMC sample config stub:
```
partmc_cfg = {
    "type": "partmc",
    "partmc_dir": "./cases/urban_case",  # contains out/*.nc
    "timestep": 3600,  # seconds index in file naming convention
    "repeat": 0,
    "n_particles": 5000,
    "species_modifications": {"SO4": {"density": 1800.0}}
}
pop = build_population(partmc_cfg)
```

MAM4 sample config sketch (simplified):
```
mam4_cfg = {
    "type": "mam4",
    "output_filename": "mam4_output.nc",
    "timestep": 3,
    "GSD": [1.6, 1.7, 1.8, 2.0],
    "D_min": 1e-9, "D_max": 1e-5, "N_bins": 40,
    "p": 90000.0, "T": 285.0
}
pop_mam4 = build_population(mam4_cfg)
```

Grid usage (scenario vs time):
```
from PyParticle.viz.grids import make_grid_scenarios_timesteps
scenarios = [partmc_cfg, mam4_cfg]
timesteps = [0, 3600, 7200]
variables = ["dNdlnD", "mass_fraction"]
fig, axes = make_grid_scenarios_timesteps(scenarios, timesteps, variables)
```
