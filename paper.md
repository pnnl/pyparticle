---
title: "PyParticle: modular tools to build aerosol particle populations and compute optics"
tags:
  - Python
  - aerosols
  - atmospheric science
  - aerosol-cloud interactions
  - aerosol-radiation interactions

authors:
  - name: Laura Fierce
    orcid: 0000-0000-0000-0000
    affiliation: 1
  - name: Payton Beeler
    orcid: 0000-0003-4759-1461
    affiliation: 1
affiliations:
  - name: Pacific Northwest National Laboratory
    index: 1
date: 2025-10-02
bibliography: paper.bib
---

# Summary

**PyParticle** is a lightweight Python library for describing and analyzing aerosol particle populations. The package includes modular builders for aerosol species, particle populations, and particle morphologies, which interface with existing models for derived aerosol properties, such as cloud condensation nuclei (CCN) activity, ice nucleation potential (INP), and optical properties. Design emphasizes a **builder/registry pattern** so new aerosol species, population types, and morphologies are added by dropping small modules into `factory/` folders—without modifying the core API.

The core components include:
- **AerosolSpecies**, **AerosolParticle**, **ParticlePopulation** classes that provide a standardized representation of aerosol particles from diverse data sources.
- **Species builder** that supplies physical properties (e.g., density, refractive index) for aerosol species, with optional per-species overrides.
- **Population builders** for parametric and model-derived populations (e.g., *binned lognormal*, *monodisperse*, and loaders for *MAM4* and *PartMC* outputs).
- **Optical particle builders** that compute wavelength- and RH-dependent optical properties for different particle morphologies (e.g., *homogeneous*, *core–shell*) using existing libraries.
- **Freezing particle builders** that estimate INP-relevant metrics from particle composition and size (see *Design & architecture*).
- An **analysis module** that calculates particle- and population-level variables from PyParticle populations.
- A **viz** package to generate figures from PyParticle populations.

Example scripts demonstrate (i) optical properties for lognormal mixtures, (ii) comparisons of CCN activity between MAM4- and PartMC-derived populations, and (iii) freezing-oriented calculations on common temperature/RH grids.

# Statement of need

The physical properties of aerosols must be well quantified for a variety of atmospheric, air quality, and industrial applications. A wide range of tools have been developed to simulate and observe aerosol particle populations, producing varied aerosol data that is often difficult to compare directly. **PyParticle** provides a standardized description of aerosol particle populations and facilitates evaluation of derived aerosol properties.

**Leveraging existing models.** PyParticle is designed to interoperate with established aerosol-property and optical models rather than reimplementing them. For optical validations and reference calculations the package can call external packages such as PyMieScatt [@PyMieScatt]. For hygroscopicity and CCN-relevant calculations it follows the kappa-Köhler framework [@Petters2007], treating kappa as a per-species property that can be supplied by the species registry or overridden at runtime. Model loaders (e.g., MAM4, PartMC) convert model outputs to the PyParticle internal representation so downstream analyses (CCN, optics, freezing) can use the same utilities. Where third-party packages are optional (e.g., `netCDF4` for NetCDF I/O or PyMieScatt for reference curves) PyParticle raises explicit errors with clear remediation so analyses are deterministic and reproducible.

**Modular structure.** The codebase follows a strict builder/registry pattern so new capabilities are added by dropping a single module into a `factory/` folder. Population builders (`population/factory/`), optics morphologies (`optics/factory/`), freezing morphologies (`freezing/factory/`), and species providers (`species/`) expose a small, well-documented `build(...)` function (or use a decorator-based registry). At runtime, discovery maps the config `type` string to the appropriate builder. This keeps the public API small while enabling experiment-specific extensions without changing core code.

**Implication for practice.** The same downstream computations (e.g., CCN spectra, optical coefficients, freezing propensity) can be run on a MAM4 snapshot, a PartMC particle file, or a synthetic lognormal mixture with identical configuration. Because species properties and morphologies are provided through modular factories, sensitivity studies (e.g., refractive indices, mixing rules, ice nucleation rate, or kappa values) become simple configuration changes rather than code forks. This encourages transparent, process-level benchmarking across diverse datasets.

# Software description

## Design & architecture

The repository is organized around clear extension points:

- **`species/`** — The species registry provides canonical physical properties (e.g., density [kg m⁻³], kappa [–], molar mass [kg mol⁻¹], surface tension [N m⁻¹]) and a file/registry fallback. Public helpers include `register_species(...)`, `get_species(name, **mods)`, `list_species()`, and `retrieve_one_species(...)`. Resolution order is (1) the environment override `PYPARTICLE_DATA_PATH/species_data/aero_data.dat`, (2) packaged data in `PyParticle/datasets/species_data/aero_data.dat`, then (3) a user-specified `specdata_path`. Per-species overrides (e.g., `{"SO4": {"kappa": 0.6}}`) apply at load time.

```python
from PyParticle.species.registry import get_species
so4 = get_species("SO4", kappa=0.6)
````

* **`aerosol_particle`** — Defines the `Particle` class and helpers to build particles from species names and masses/diameters. A `Particle` stores per-species masses, dry/wet diameters, effective kappa, and basic metadata. Helpers provide kappa-Köhler growth and CCN activity [@Petters2007]. By default, CCN is treated with the homogeneous-sphere assumption and water surface tension.

```python
from PyParticle.aerosol_particle import make_particle
p = make_particle(D=100e-9, aero_spec_names=["SO4"], aero_spec_frac=[1.0], D_is_wet=True)
print(p.get_Ddry(), p.get_tkappa())
```

* **`population/`** — Exposes `build_population(config)` and a discovery system mapping `config["type"]` to a module in `population/factory/`. The `binned_lognormals` builder requires `"N"`, `"GMD"` (m), `"GSD"`, `"aero_spec_names"`, `"aero_spec_fracs"`, and binning parameters. It expands lognormal modes into discrete bins, builds per-bin `Particle`s, and assigns number concentration.

```python
from PyParticle.population import build_population
pop = build_population({"type": "binned_lognormals", "N": [1e7], "GMD": [100e-9],
                        "GSD": [1.6], "aero_spec_names": [["SO4"]],
                        "aero_spec_fracs": [[1.0]], "N_bins": 120})
```

* **`optics/`** — `build_optical_population(pop, config)` attaches per-particle optical morphologies over `wvl_grid` (m) and `rh_grid` (default `[0.0]`). Morphologies (`homogeneous`, `core_shell`) compute scattering/absorption/extinction cross-sections and asymmetry parameter `g`. The resulting `OpticalPopulation` aggregates to population-level coefficients (m⁻¹).

```python
from PyParticle.optics import build_optical_population
opt_pop = build_optical_population(pop, {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]})
print(opt_pop.get_optical_coeff("b_scat", rh=0.0, wvl=550e-9))
```

* **`freezing/`** — Contains routines for assessing ice nucleation potential (INP). Uses particle composition and surface area to calculate freezing proxies and exposes a builder pattern so new parameterizations can be added. Accepts `Particle` or `ParticlePopulation` inputs and returns particle-level and population metrics (e.g., total ice nucleation rate, activated fraction vs. time).

```python
from PyParticle.population import build_population
from PyParticle.freezing import build_freezing_population
pop = build_population({"type": "binned_lognormals", "N": [1e7], "GMD": [100e-9],
                        "GSD": [1.6], "aero_spec_names": [["SO4"]],
                        "aero_spec_fracs": [[1.0]], "N_bins": 120})
freezing_pop = build_freezing_population(pop, {"morphology": "homogeneous", "T_grid": [-60, -50, -40, -30], "T_units": "C"})
print(freezing_pop.get_nucleation_rate(T=-30))
```

* **`analysis/`** — Provides utilities for size distributions (`dN/dlnD`), moments, mass/volume fractions, hygroscopic growth factors, and CCN spectra. Returns NumPy arrays or lightweight dataclasses for plotting and statistics.

* **`viz/`** — Provides plotter builders, style management, and grid helpers for consistent visualization of population outputs.

## Design & architecture

The repository is organized around clear extension points; below are the authoritative behaviors and public helpers so manuscript text (and example code) can be written precisely.

- `species/` — Runtime registry + file lookup. Public helpers: `register_species(spec)`, `get_species(name, **modifications)`, `list_species()`, and `retrieve_one_species(name, specdata_path=None, spec_modifications={})`. Lookup resolution order is: (1) `PYPARTICLE_DATA_PATH/species_data/aero_data.dat` if the env var is set, (2) packaged resource `PyParticle/datasets/species_data/aero_data.dat`, (3) passed `specdata_path` or the package `data_path/'species_data'`. The file parser expects whitespace-separated columns (name, density, ions_in_solution, molar_mass, kappa, ...). Per-spec overrides (e.g., `{'SO4': {'kappa': 0.6}}`) are applied at retrieval time. Missing species raises `ValueError("Species data for '<name>' not found in data file.")`.

```python
from PyParticle.species.registry import get_species
so4 = get_species("SO4", kappa=0.6)
```

- `aerosol_particle` — Particle construction and single‑particle helpers. The module exposes `make_particle(D, aero_spec_names, aero_spec_frac, species_modifications={}, D_is_wet=True, specdata_path=...)` and `make_particle_from_masses(aero_spec_names, spec_masses, ...)` that return a `Particle` instance. `Particle` stores an ordered sequence of `AerosolSpecies` and a NumPy array of per‑species masses (kg). Important methods: `get_Ddry()`, `get_Dwet(RH, T)`, `get_tkappa()`, `get_critical_supersaturation(T)`, `get_vol_tot()`, and `get_trho()`. Kappa‑Köhler growth is implemented via `compute_Dwet(...)` and critical supersaturation via `get_critical_supersaturation(...)`. Default physical constants: surface tension = 0.072 N m⁻¹, water density = 1000 kg m⁻³. Implementation note: the current `Particle` is a regular Python class and contains a vestigial `__post_init__` method (from a prior dataclass refactor); the public construction helpers are the supported API.

```python
from PyParticle.aerosol_particle import make_particle
p = make_particle(D=100e-9, aero_spec_names=["SO4"], aero_spec_frac=[1.0], D_is_wet=True)
print(p.get_Ddry(), p.get_tkappa())
```

- `population/` — Builder discovery and population containers. Public convenience: `build_population(config)` which constructs a `PopulationBuilder(config)` and calls its `build()` method. `PopulationBuilder` looks up `config['type']` in `population/factory/` via a discovery routine that imports modules and uses any top‑level `build` callable. Example builder `binned_lognormals` requires per‑mode lists: `N`, `GMD` (m), `GSD`, `aero_spec_names`, `aero_spec_fracs` and `N_bins` (or an integer to be applied to all modes). Optional keys: `N_sigmas` (default 5), `D_min`, `D_max` (computed if absent), `species_modifications`, `surface_tension`, `D_is_wet`, and `specdata_path`. Behavior: the builder forms a master species list for the population, expands each lognormal mode into discrete diameter bins, builds per‑bin `Particle`s with `make_particle(...)` and inserts them into a `ParticlePopulation` with per‑bin number concentrations computed from the lognormal PDF. Unknown `type` raises `ValueError("Unknown population type: <type>")`.

```python
from PyParticle.population import build_population
pop = build_population({
  "type": "binned_lognormals",
  "N": [1e7], "GMD": [100e-9], "GSD": [1.6],
  "aero_spec_names": [["SO4"]], "aero_spec_fracs": [[1.0]],
  "N_bins": 120
})
```

- `optics/` — Optics builders and morphologies. `build_optical_population(base_population, config)` accepts `wvl_grid` (meters) and `rh_grid` (default `[0.0]`). It attaches wavelength‑aware refractive indices to species (via `refractive_index.build_refractive_index`), constructs an `OpticalPopulation(base_population, rh_grid, wvl_grid)`, then builds per‑particle morphology instances via the morphology registry and copies per‑particle `Cabs`, `Csca`, `Cext`, and `g` arrays into the population. Species-level `species_modifications` are taken from `config` (preferred) or from `base_population.species_modifications`. Morphologies implement `compute_optics()` to populate per‑particle cross‑section cubes (shape = N_rh × N_wvl); `OpticalPopulation.get_optical_coeff(optics_type, rh=None, wvl=None)` aggregates to coefficients (m⁻¹) using `sum_i C_i * N_i` and returns scalars or grids depending on the selection.

```python
from PyParticle.optics import build_optical_population
opt_pop = build_optical_population(pop, {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]})
print(opt_pop.get_optical_coeff("b_scat", rh=0.0, wvl=550e-9))
```

- `freezing/` — Freezing particle builders and morphologies. `build_freezing_population(base_population, config)` accepts `T_grid` (Celcius or Kelvin), `T_units` (default `K`), and `apecies_modifications` (default `{}`). It attaches temperature‑aware ice nucleation rate to species (via `base.retrieve_Jhet_val`), constructs an `FreezingPopulation(base_population)`, then builds per‑particle morphology instances via the morphology registry and copies per‑particle `Jhet` and `INSA` arrays into the population. Species-level `species_modifications` are taken from `config` (preferred) or from `base_population.species_modifications`. Morphologies implement `compute_Jhet()` to populate per‑particle Jhet.

```python
from PyParticle.population import build_population
from PyParticle.freezing import build_freezing_population
pop = build_population({"type": "binned_lognormals", "N": [1e7], "GMD": [100e-9],
                        "GSD": [1.6], "aero_spec_names": [["SO4"]],
                        "aero_spec_fracs": [[1.0]], "N_bins": 120})
freezing_pop = build_freezing_population(pop, {"morphology": "homogeneous", "T_grid": [-60, -50, -40, -30], "T_units": "C"})
print(freezing_pop.get_nucleation_rate(T=-30))
```

*Implementation notes.* The codebase uses SI units internally (meters for diameters/wavelengths) and defaults `rh_grid` to `[0.0]`. Optional dependencies such as `netCDF4` or `PyMieScatt` are imported only where needed; in their absence the code raises `ModuleNotFoundError` with an actionable message rather than silently substituting mock data.


# Acknowledgements

This work benefited from discussions in the aerosol modeling community about process-level benchmarking and hierarchical evaluation. Development of PyParticle predates its use within other applications and was supported under prior project funding; it is linked here to enable those use cases alongside broader applications. We thank contributors and users who provided feedback on APIs, testing, and example design.

# References

