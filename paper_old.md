<!-- ---
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
    orcid: 0000-0000-0000-0000
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

**Modular structure.** The codebase follows a strict builder/registry pattern so new capabilities are added by dropping a single module into a `factory/` folder. Population builders (`population/factory/`), optics morphologies (`optics/factory/`), and species providers (`species/`) expose a small, well-documented `build(...)` function (or use a decorator-based registry). At runtime, discovery maps the config `type` string to the appropriate builder. This keeps the public API small while enabling experiment-specific extensions without changing core code.

**Implication for practice.** The same downstream computations (e.g., CCN spectra, optical coefficients, freezing propensity) can be run on a MAM4 snapshot, a PartMC particle file, or a synthetic lognormal mixture with identical configuration. Because species properties and morphologies are provided through modular factories, sensitivity studies (e.g., refractive indices, mixing rules, or kappa values) become simple configuration changes rather than code forks. This encourages transparent, process-level benchmarking across diverse datasets.

# Software description

## Design & architecture

The repository is organized around clear extension points:

- **`species/`**: species registry and data loading (default properties; users can supply `species_modifications` to override, e.g., `n_550`, `k_550`, `alpha_n`, `alpha_k`).
- **`aerosol_particle`**: defines the `Particle` dataclass and helpers. `Particle` is the single-particle representation used to construct particle populations, defined by a list of [Species and per-species masses]. 
Helpers provide calcualtiosn for basic phyiscal properties, water uptake, and CCN activivity. We note that water uptake and CCN calculations are currently implemented on the base particle using the kappa-Köhler framework, wherein particles are represented as homogeneous spheres with the surface tension of water; a separate module for more complex treatements of water uptake will be included in a future release. 
- **`population/`**: population containers and `build_population(config)`. Modules in `population/factory/` (e.g., `binned_lognormals.py`, `mam4.py`, `partmc.py`) implement builders discoverable by name.
- **`optics/`**: `build_optical_population(pop, config)` attaches per-particle optics using morphology modules in `optics/factory/` (e.g., `homogeneous`, `core_shell`) over `wvl_grid` and `rh_grid`.

- **`freezing/`**: contains routines for assessing ice nucleation potential (INP). Implementations use per-particle composition and surface area metrics to calculate freezing proxies (e.g., insoluble surface area or solute-limited immersion behavior) and expose a builder pattern so new parameterizations (classical nucleation theory, empirical INP schemes, mechanistic approaches) can be added. The code accepts `Particle` or `ParticlePopulation` inputs and returns particle-level metrics and population statistics (e.g., activated fraction vs. temperature) with explicit temperature/RH controls.


- **`analysis/`**: utilities for population and particle diagnostics such as size distributions (`dN/dlnD`), moments, mass/volume fractions, hygroscopic growth factors, and CCN spectra using kappa-Köhler. Functions accept `ParticlePopulation` objects or configs (which viz helpers can auto-build) and return compact NumPy arrays or small dataclasses for plotting and reports.

- **`viz/`**: plotter builder (`build_plotter("state_line", cfg)`), style manager, and grid helpers for consistent figures.


**Conventions.** Internal units prioritize SI (e.g., wavelength in meters). RH grids default to `[0.0]`. Population `config["type"]` must match a module filename in the relevant `factory/`.
 -->


Good call. In JOSS papers, you usually **don’t need a long API reference** — JOSS is more about (i) what the software does, (ii) why it matters, and (iii) a short sketch of how it works. Detailed APIs belong in your docs or demos.

So here’s a **merged full `paper.md`** with your text, the filled-out module descriptions I just wrote, and the *Key APIs* section removed (since the illustrative examples already cover usage). This keeps the paper lean and well within JOSS norms.

---

````markdown
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
    orcid: 0000-0000-0000-0000
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

**Modular structure.** The codebase follows a strict builder/registry pattern so new capabilities are added by dropping a single module into a `factory/` folder. Population builders (`population/factory/`), optics morphologies (`optics/factory/`), and species providers (`species/`) expose a small, well-documented `build(...)` function (or use a decorator-based registry). At runtime, discovery maps the config `type` string to the appropriate builder. This keeps the public API small while enabling experiment-specific extensions without changing core code.

**Implication for practice.** The same downstream computations (e.g., CCN spectra, optical coefficients, freezing propensity) can be run on a MAM4 snapshot, a PartMC particle file, or a synthetic lognormal mixture with identical configuration. Because species properties and morphologies are provided through modular factories, sensitivity studies (e.g., refractive indices, mixing rules, or kappa values) become simple configuration changes rather than code forks. This encourages transparent, process-level benchmarking across diverse datasets.

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

* **`freezing/`** — Contains routines for assessing ice nucleation potential (INP). Uses particle composition and surface area to calculate freezing proxies and exposes a builder pattern so new parameterizations can be added. Accepts `Particle` or `ParticlePopulation` inputs and returns particle-level and population metrics (e.g., activated fraction vs. T).



* **`analysis/`** — Provides utilities for size distributions (`dN/dlnD`), moments, mass/volume fractions, hygroscopic growth factors, and CCN spectra. Returns NumPy arrays or lightweight dataclasses for plotting and statistics.

* **`viz/`** — Provides plotter builders, style management, and grid helpers for consistent visualization of population outputs.


# Acknowledgements

This work benefited from discussions in the aerosol modeling community about process-level benchmarking and hierarchical evaluation. Development of PyParticle predates its use within other applications and was supported under prior project funding; it is linked here to enable those use cases alongside broader applications. We thank contributors and users who provided feedback on APIs, testing, and example design.

# References

```

---


<!-- ---
## Key APIs (sketch)

```python
from PyParticle.population import build_population
pop = build_population({
  "type": "binned_lognormals",
  "N": [1e7], "GMD": [50e-9], "GSD": [1.6],
  "aero_spec_names": [["SO4"]], "aero_spec_fracs": [[1.0]],
  "N_bins": 200, "D_min": 1e-9, "D_max": 1e-5,
})

from PyParticle.optics import build_optical_population
opt = build_optical_population(pop, {"type": "homogeneous",
                                     "wvl_grid": [550e-9],
                                     "rh_grid": [0.0]})



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
     orcid: 0000-0000-0000-0000
     affiliation: 1
affiliations:
  - name: Pacific Northwest National Laboratory
    index: 1
  # - name: Second Institution
  #   index: 2
date: 2025-10-02
bibliography: paper.bib
---

# Summary

**PyParticle** is a lightweight Python library for describing and analyzing aerosol particle populations. The package includes modular builders for aerosol species, particle populations, and particle morphologies, which interface with existing models for derived aerosol properties, such as cloud condensation nuclei (CCN) activity, ice nucleation potential (INP), and optical properties. Design emphasizes a **builder/registry pattern** so new aerosol species, population types, and morphologies are added by dropping small modules into `factory/` folders—without modifying the core API. 

The core components include:
- **AerosolSpecies**, **AerosolParticle**, **ParticlePopulation** classes that provide a strandardized representation of aerosol particles from diverse data sources.
- **Species builder** that supplies physical properties (e.g., density, refractive index) for aerosol species, with optional per-species overrides.
- **Population builders** for parametric and model-derived populations (e.g., *binned lognormal*, *monodisperse*, and loaders for *MAM4* and *PartMC* outputs).
- **Optical particle builders** that compute wavelength- and RH-dependent optical properties for different particle morphologies (e.g., *homogeneous*, *core–shell*) using existing libraries. 
- **Freezing particle builders** that ... xx
- An **analysis module** that calculates particle- and population-level variables from PyParticle popoulations.
- A **viz** package to generate figures from the PyParticle populations.

Example scripts demonstrate (i) optical properties for lognormal mixtures, (ii) comparisons of CCN activity between MAM4- and PartMC-derived populations, and (iii) xx Freezing 

# Statement of need
The physical properites of aerosols must be well quantified for a variety of atmopsheric, air quality, and industrial applications. A wide range of tools have been developed to simulate and observed aerosol particle populations, producing varied aerosol data that is often difficult to compare. **PyParticle** provides a standardized description of aerosol particle populations and facilitates evaluation of drived aerosol properties. 

[Add paragraph on how it is designed to leverage existing models (e.g., PyMieScatt, kappa-Kohler, )]

[Add paragraph on modular structure]

[Add paragraph on what this all means -- easier to compare diverse aerosol observations, easier to analyze, easy to add/change/perform sensitivity tests]


# Software description

## Design & architecture

The repository is organized around clear extension points:
- **`aerosol_particle`**: [fill out]
- **`species/`**: species registry and data loading (default properties; user can supply `species_modifications` to override, e.g., `n_550`, `k_550`, `alpha_n`, `alpha_k`).
- **`population/`**: population containers and `build_population(config)`. Modules in `population/factory/` (e.g., `binned_lognormals.py`, `mam4.py`, `partmc.py`) implement builders discoverable by name.
- **`optics/`**: `build_optical_population(pop, config)` attaches per-particle optics using morphology modules in `optics/factory/` (e.g., `homogeneous`, `core_shell`) over `wvl_grid` and `rh_grid`.
- **`freezing/`**:  [fill out] 
- **`analysis/`**: [fill out]
- **`viz/`**: plotter builder (`build_plotter("state_line", cfg)`), style manager, and grid helpers for consistent figures.
[Note: CCN calculations are currently performed on the base particle; can we add that somewhere? Maybe after analysis before feezing or before optics? Or just a note at the end?]

**Conventions.** Internal units prioritize SI (e.g., wavelength in meters). RH grids default to `[0.0]`. Population `config["type"]` must match a module filename in `factory/`.

## Key APIs (sketch)

```python
from PyParticle.population import build_population
pop = build_population({
  "type": "binned_lognormals",
  "N": [1e7], "GMD": [50e-9], "GSD": [1.6],
  "aero_spec_names": [["SO4"]], "aero_spec_fracs": [[1.0]],
  "N_bins": 200, "D_min": 1e-9, "D_max": 1e-5,
})

from PyParticle.optics import build_optical_population
opt = build_optical_population(pop, {"type": "homogeneous",
                                     "wvl_grid": [550e-9],
                                     "rh_grid": [0.0]})
For visualization:

python
Copy code
from PyParticle.viz.builder import build_plotter
cfg = {"varname": "b_scat", "var_cfg": {"wvl_grid": [550e-9], "rh_grid": [0.0]}}
plotter = build_plotter("state_line", cfg)
# plotter.plot(pop, ax, label="my population")
Illustrative examples
The repository includes runnable examples mirroring common analyses:

Optics for lognormal mixtures — builds several binned lognormal populations at different GMDs, computes b_scat vs. wavelength, and (optionally) overlays a PyMieScatt-based reference curve for validation [@PyMieScatt].

MAM4 vs. PartMC — loads a timestep from each model, compares size distributions (dN/dlnD) and b_scat@550 nm under consistent assumptions. Requires NetCDF outputs from each model [@MAM4; @PartMC].

These examples also demonstrate species property overrides and deterministic styling via the viz helpers.

Validation
Numerical agreement: For simple, monodisperse or single-mode lognormal cases with fixed refractive index, PyParticle reproduces PyMieScatt scattering/absorption to within small relative error (tunable through bin count and wavelength grid).*

Unit sanity: All internally reported optical coefficients are in m⁻¹; helper scripts that use PyMieScatt convert from Mm⁻¹ where necessary.

Reproducibility: Example scripts are non-interactive, use fixed random seeds (if any), and write plots/CSVs deterministically.

*Exact tolerances depend on bin resolution, diameter range, and refractive-index interpolation; recommended defaults are provided in the examples.

Availability & installation
PyParticle is pure Python with common scientific dependencies (e.g., NumPy, SciPy, Matplotlib; NetCDF libraries when using model loaders). A conda environment file and pytest test slice are provided for quick checks. Optional PyMieScatt is used only for reference comparisons.

bash
Copy code
# create environment (example)
conda env create -f environment.yml
conda activate pyparticle
pytest -q
Limitations & scope
The species registry provides pragmatic defaults; users should document any overrides for scientific studies.

Morphology support currently targets common atmospheric assumptions (e.g., homogeneous, core–shell). Extending to complex aggregates is possible via the same factory interface.

MAM4/PartMC loaders require access to model outputs (not shipped with the package).

Acknowledgements
This work benefited from discussions in the aerosol modeling community about process-level benchmarking and hierarchical evaluation. Development of PyParticle predates its use within AMBRS and was supported under other project funding; it is linked here to enable AMBRS use cases alongside broader applications. We thank contributors and users who provided feedback on APIs, testing, and example design. -->