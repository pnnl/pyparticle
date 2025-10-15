---
title: "PyParticle: modular tool to build and analyze aerosol particle populations"
tags:
  - Python
  - aerosols
  - atmospheric science
  - aerosol-cloud interactions
  - aerosol-radiation interactions
authors:
  - name: Laura Fierce
    orcid: "0000-0002-8682-1453"
    affiliation: 1
  - name: Payton Beeler
    orcid: "0000-0003-4759-1461"
    affiliation: 1
affiliations:
  - name: Pacific Northwest National Laboratory
    index: 1
date: 2025-10-02

bibliography: /Users/fier887/Library/CloudStorage/OneDrive-PNNL/Code/PyParticle/paper.bib 
---

# Summary

**PyParticle** is a lightweight Python library for describing and analyzing aerosol particle populations. The package includes modular builders for aerosol species, particle populations, and particle morphologies, which interface with existing models for derived aerosol properties, such as cloud condensation nuclei (CCN) activity, ice-nucleating particle (INP) potential, and optical properties. Its **builder/registry** design allows new aerosol species, population types, and morphologies to be added by adding small modules to the appropriate `factory/` folders without modifying code API. This modular, reproducible framework facilitates process-level investigations, sensitivity analyses, and intercomparison studies across diverse model and observational datasets. 

The core components include:
- **AerosolSpecies**, **AerosolParticle**, **ParticlePopulation** classes that provide a standardized representation of aerosol particles from diverse data sources.
- **Species builder** that supplies physical properties (e.g., density, hygroscopicity parameter, refractive index) for aerosol species, with optional per-species overrides.
- **Population builders** for parametric and model-derived populations (e.g., *binned lognormal*, *monodisperse*, and loaders for *MAM4* and *PartMC* simulation output).
- **Optical particle builders** that compute wavelength- and RH-dependent optical properties for different particle morphologies (e.g., *homogeneous*, *core–shell*) using existing libraries.
- **Freezing particle builders** that estimate INP-relevant metrics from particle composition and size.
- An **analysis module** that calculates particle- and population-level variables from PyParticle populations.
- A **viz** package to generate figures from PyParticle populations.

Example scripts demonstrate (i) optical properties for lognormal mixtures, (ii) comparisons of CCN activity between MAM4- and PartMC-derived populations, and (iii) freezing-oriented calculations on common temperature/RH grids.

# Statement of need
The physical properties of aerosols must be well quantified for a variety of atmospheric, air quality, and industrial applications. A wide range of tools have been developed to simulate [e.g., @Riemer2009_JGR_PartMC; @Liu2016_GMD_MAM4; @Bauer2008_ACP_MATRIX; @Zaveri2008_JGR_MOSAIC] and observe [e.g., @Jayne2000_AST_AMS; @DeCarlo2006_AnalChem_HRToFAMS; @Knutson1975_JAS_DMA; @Wang1990_AST_SEMS; @Schwarz2006_JGR_SP2; @Schwarz2008_JGR_SP2; @Zelenyuk2015_JASMS_miniSPLAT] aerosol particle populations, producing varied aerosol data that is often difficult to compare directly. **PyParticle** provides a standardized description of aerosol particle populations and facilitates evaluation of derived aerosol properties.

**Leveraging existing models.** PyParticle is designed to interoperate with established aerosol-property models rather than reimplementing them. The current implementation focuses on aerosol effects relevant for the atmosphere, with many functions adapted from previous studies of CCN activity [@Fierce2013_JGR; @Fierce2017_JGR; @Fierce2024_JAS], optical properties [@Fierce2016_NatComm; @Fierce2020_PNAS], and mixing state approximations [@Fierce2015_ACP; @Fierce2024_JAS; @Zheng2022_ACP_mixing_state]. For optical calculations, PyParticle can call external packages such as PyMieScatt [@PyMieScatt2018_JQSRT]. For hygroscopicity and CCN-relevant calculations it follows the kappa-Köhler framework [@Petters2007_ACP], treating kappa as a per-species property that can be supplied by the species registry or overridden at runtime. Immersion freezing is represented using the stochastic immersion freezing model [@KnopfAlpert2013_Faraday; @AlpertKnopf2016_ACP]. Model loaders (e.g., MAM4, PartMC) convert model outputs to the PyParticle internal representation so downstream analyses (CCN, optics, freezing) can use the same utilities. Where third-party packages are optional (e.g., `netCDF4` for NetCDF I/O or PyMieScatt for reference curves), PyParticle raises explicit errors with clear remediation so analyses are deterministic and reproducible.

**Modular structure.** The codebase follows a strict builder/registry pattern so new capabilities are added by dropping a single module into a `factory/` folder. Modules for particle populations (`population/factory/`), optics morphologies (`optics/factory/`), freezing models (`freezing/factory/`), and aerosol species (`species/`) expose a small, well-documented `build(...)` function (or use a decorator-based registry). At runtime, discovery maps the config `type` string to the appropriate builder. This keeps the public API small while enabling application-specific extensions without changing core code.

**Implication for practice.** The same downstream computations (e.g., CCN spectra, optical coefficients, freezing propensity) can be applied to many descriptions of aerosol particle populations. The current implementation includes modules for representing populations from PartMC, MAM4, or parameterized distributions, each defined with identical configurations. Because species properties and morphologies are provided through modular factories, sensitivity studies (e.g., refractive indices, mixing rules, or kappa values) become simple configuration changes rather than code modifications. This encourages transparent, process-level benchmarking across diverse datasets.

# Software description

## Design & architecture

The repository is organized around clear extension points:

- **`species/`** — The species registry provides canonical physical properties (e.g., density [kg m⁻³], kappa [–], molar mass [kg mol⁻¹], surface tension [N m⁻¹]) and a file/registry fallback. Public helpers include `register_species(...)`, `get_species(name, **mods)`, `list_species()`, and `retrieve_one_species(...)`. Resolution order is (1) the environment override `PYPARTICLE_DATA_PATH/species_data/aero_data.dat`, (2) packaged data in `PyParticle/datasets/species_data/aero_data.dat`, then (3) a user-specified `specdata_path`. Per-species overrides (e.g., `{"SO4": {"kappa": 0.6}}`) apply at load time.

```python
from PyParticle.species.registry import get_species
so4 = get_species("SO4", kappa=0.6)
```

* **`aerosol_particle`** — Defines the `Particle` class and helpers to build particles from species names and masses/diameters. A `Particle` stores per-species masses, dry/wet diameters, effective kappa, and basic metadata. Helpers provide kappa-Köhler growth and CCN activity [@Petters2007_ACP]. By default, CCN is treated with the homogeneous-sphere assumption and water surface tension. An example:

```python
from PyParticle.aerosol_particle import make_particle
p = make_particle(D=100e-9, aero_spec_names=["SO4"], aero_spec_frac=[1.0], D_is_wet=True)
print(p.get_Ddry(), p.get_tkappa())
```

* **`population/`** — Exposes `build_population(config)` and a discovery system mapping `config["type"]` to a module in `population/factory/`. The config settings are specific to the population type. For example, the `binned_lognormals` builder requires `"N"`, `"GMD"` (m), `"GSD"`, `"aero_spec_names"`, `"aero_spec_fracs"`, and binning parameters. The `partmc` builder, on the other hand, requires specification of the directory where simulation data is located and the simulation timestep. Each population-builder produces a standardized particle population, defined by a list of constituent `aerosol_species`, an array of `species_masses` that provide the mass [kg] of each species in each particle, an array of `num_conc` that describe the number concentration [m⁻3] associated with each computational particle, and a list of ids for each particle. An individual `Particle` is added to or extracted from the `ParticlePopulation` using the attached `add_particle` and `get_particle` functions, respectively. 

```python
from PyParticle.population import build_population
pop = build_population({"type": "binned_lognormals", "N": [1e7], "GMD": [100e-9],
                        "GSD": [1.6], "aero_spec_names": [["SO4"]],
                        "aero_spec_fracs": [[1.0]], "N_bins": 120})
```

* **`optics/`** — `build_optical_population(pop, config)` attaches per-particle optical morphologies over `wvl_grid` (m) and `rh_grid` (default `[0.0]`). Morphologies (`homogeneous`, `core_shell`) compute per-particle scattering, absorption, and extinction cross-sections and asymmetry parameter `g`. The resulting `OpticalPopulation` aggregates to population-level coefficients (m⁻¹).

```python
from PyParticle.optics import build_optical_population
opt_pop = build_optical_population(pop, {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]})
print(opt_pop.get_optical_coeff("b_scat", rh=0.0, wvl=550e-9))
```

* **`freezing/`** — Contains routines for assessing the potential for immersion freezing. The freezing module uses particle composition and surface area to calculate freezing proxies and exposes a builder pattern so new parameterizations can be added. Accepts `Particle` or `ParticlePopulation` inputs and returns particle-level and population metrics (e.g., activated fraction vs. T).

```python
# Placeholder example (update when API is finalized):
# from PyParticle.freezing import build_freezing_population
# frz = build_freezing_population(pop, {"type": "stochastic"})
```

* **`analysis/`** — Provides utilities for size distributions (`dN/dlnD`), moments, mass/volume fractions, hygroscopic growth factors, and CCN spectra. Returns NumPy arrays or lightweight dataclasses for plotting and statistics.

* **`viz/`** — Provides plotter builders, style management, and grid helpers for consistent visualization of population outputs.

*Implementation notes.* The codebase uses SI units internally (meters for diameters/wavelengths) and defaults `rh_grid` to `[0.0]`. Optional dependencies such as `netCDF4` or `PyMieScatt` are imported only where needed; in their absence the code raises `ModuleNotFoundError` with an actionable message rather than silently substituting mock data.

# Acknowledgements
The PyParticle package was originally developed under the Integrated Cloud, Land-surface, and Aerosol System Study (ICLASS), a Science Focus Area of the U.S. Department of Energy's Atmospheric System Research program at Pacific Northwest National Laboratory (PNNL). Optics modules and links to the PartMC and MAM4 modules was supported by PNNL's Laboratory Direct Research and Development program. PNNL is a multi-program national laboratory operated for the U.S. Department of Energy by Battelle Memorial Institute under Contract No. DE-AC05-76RL01830.

# References
<!-- Manual list included for Markdown-only preview. Safe to delete when using Pandoc/JOSS. -->

- Alpert, P. A., & Knopf, D. A. (2016). Analysis of isothermal and cooling-rate-dependent immersion freezing by a unifying stochastic ice nucleation model. *Atmospheric Chemistry and Physics*, 16, 2083–2107. https://doi.org/10.5194/acp-16-2083-2016
- Bauer, S. E., Wright, D. L., Koch, D., Lewis, E. R., McGraw, R., Chang, L.-S., Schwartz, S. E., & Ruedy, R. (2008). MATRIX: an aerosol microphysical module for global atmospheric models. *Atmospheric Chemistry and Physics*, 8, 6003–6035. https://doi.org/10.5194/acp-8-6003-2008
- DeCarlo, P. F., Kimmel, J. R., Trimborn, A., Northway, M. J., Jayne, J. T., Aiken, A. C., Gonin, M., Fuhrer, K., Horvath, T., Docherty, K. S., Worsnop, D. R., & Jimenez, J. L. (2006). Field-Deployable, High-Resolution, Time-of-Flight Aerosol Mass Spectrometer. *Analytical Chemistry*, 78(24), 8281–8289. https://doi.org/10.1021/ac061249n
- Fierce, L., & McGraw, R. L. (2017). Multivariate quadrature for representing cloud condensation nuclei activity of aerosol populations. *Journal of Geophysical Research: Atmospheres*, 122(18), 9867–9878. https://doi.org/10.1002/2016JD026335
- Fierce, L., Onasch, T. B., Cappa, C. D., Mazzoleni, C., China, S., Bhandari, J., Davidovits, P., Fischer, D. A., Helgestad, T. J., Lambe, A. T., Sedlacek, A. J., Smith, G. D., & Wolff, L. (2020). Radiative absorption enhancements by black carbon controlled by particle-to-particle heterogeneity in composition. *Proceedings of the National Academy of Sciences*, 117(10), 5196–5203. https://doi.org/10.1073/pnas.1919723117
- Fierce, L., Riemer, N., & Bond, T. C. (2013). When is cloud condensation nuclei activity sensitive to particle mixing state? *Journal of Geophysical Research: Atmospheres*, 118(24), 13476–13488. https://doi.org/10.1002/2013JD020608
- Fierce, L., Riemer, N., Bond, T. C., Bauer, S. E., Mena, F., & West, M. (2015). Explaining variance in black carbon’s aging timescale and its implications for absorption enhancement: a particle-resolved modeling study. *Atmospheric Chemistry and Physics*, 15, 3173–3191. https://doi.org/10.5194/acp-15-3173-2015
- Fierce, L., Yao, Y., Easter, R. C., Ma, P.-L., Sun, J., Wan, H., & Zhang, K. (2024). Quantifying structural errors in cloud condensation nuclei activity from reduced representation of aerosol size distributions. *Journal of Aerosol Science*, 181, 106388. https://doi.org/10.1016/j.jaerosci.2024.106388
- Fierce, L., Bond, T. C., Bauer, S. E., Mena, F., & Riemer, N. (2016). Black carbon absorption at the global scale is affected by particle-scale diversity in composition. *Nature Communications*, 7, 12361. https://doi.org/10.1038/ncomms12361
- Jayne, J. T., Leard, D. C., Zhang, X., Davidovits, P., Smith, K. A., Kolb, C. E., & Worsnop, D. R. (2000). Development of an Aerosol Mass Spectrometer for Size and Composition Analysis of Submicron Particles. *Aerosol Science and Technology*, 33(1–2), 49–70. https://doi.org/10.1080/027868200410840
- Knopf, D. A., & Alpert, P. A. (2013). A water activity based model of heterogeneous ice nucleation kinetics for freezing of water and aqueous solution droplets. *Faraday Discussions*, 165, 513–534. https://doi.org/10.1039/C3FD00035D
- Knutson, E. O., & Whitby, K. T. (1975). Aerosol classification by electric mobility: apparatus, theory, and applications. *Journal of Aerosol Science*, 6(6), 443–451. https://doi.org/10.1016/0021-8502(75)90060-9
- Liu, X., Ma, P.-L., Wang, H., Tilmes, S., Singh, B., Easter, R. C., Ghan, S. J., & Rasch, P. J. (2016). Description and evaluation of a new four-mode version of the Modal Aerosol Module (MAM4) within version 5.3 of the Community Atmosphere Model. *Geoscientific Model Development*, 9, 505–522. https://doi.org/10.5194/gmd-9-505-2016
- McGraw, R. (1997). Description of aerosol dynamics by the quadrature method of moments. *Aerosol Science and Technology*, 27(2), 255–265. https://doi.org/10.1080/02786829708965471
- Petters, M. D., & Kreidenweis, S. M. (2007). A single parameter representation of hygroscopic growth and cloud condensation nucleus activity. *Atmospheric Chemistry and Physics*, 7(8), 1961–1971. https://doi.org/10.5194/acp-7-1961-2007
- PyMieScatt: Sumlin, B. J., Heinson, W. R., & Chakrabarty, R. K. (2018). Retrieving the aerosol complex refractive index using PyMieScatt: A Mie computational package with visualization capabilities. *Journal of Quantitative Spectroscopy and Radiative Transfer*, 205, 127–134. https://doi.org/10.1016/j.jqsrt.2017.10.012
- Riemer, N., West, M., Zaveri, R. A., & Easter, R. C. (2009). Simulating the evolution of soot mixing state with a particle-resolved aerosol model. *Journal of Geophysical Research: Atmospheres*, 114, D09202. https://doi.org/10.1029/2008JD011073
- Schwarz, J. P., Gao, R.-S., Fahey, D. W., Thomson, D. S., Watts, L. A., Wilson, J. C., Reeves, J. M., Darbeheshti, M., Baumgardner, D. G., Kok, G., Chung, S.-H., Schulz, M., Hendricks, J., Lauer, A., Kärcher, B., Slowik, J. G., Rosenlof, K. H., Thompson, T. L., Langford, A. O., Loewenstein, M., & Aikin, K. C. (2006). Single-particle measurements of midlatitude black carbon and light-scattering aerosols from the boundary layer to the lower stratosphere. *Journal of Geophysical Research: Atmospheres*, 111, D16207. https://doi.org/10.1029/2006JD007076
- Schwarz, J. P., Spackman, J. R., Fahey, D. W., Gao, R.-S., Lohmann, U., Stier, P., Watts, L. A., Thomson, D. S., Lack, D. A., Pfister, L., Mahoney, M. J., Baumgardner, D. G., Wilson, J. C., & Reeves, J. M. (2008). Coatings and their enhancement of black carbon light absorption in the tropical atmosphere. *Journal of Geophysical Research: Atmospheres*, 113, D03203. https://doi.org/10.1029/2007JD009042
- Tilmes, S., Mills, M. J., Zhu, Y., Bardeen, C. G., Vitt, F., Yu, P., Fillmore, D., Liu, X., Toon, O. B., & Deshler, T. (2023). Description and performance of a sectional aerosol microphysical model in the Community Earth System Model (CESM2). *Geoscientific Model Development*, 16, 6087–6117. https://doi.org/10.5194/gmd-16-6087-2023
- Wang, S.-C., & Flagan, R. C. (1990). Scanning electrical mobility spectrometer. *Aerosol Science and Technology*, 13(2), 230–240. https://doi.org/10.1080/02786829008959441
- Yao, Y., Curtis, J. H., Ching, J., Zheng, Z., & Riemer, N. (2022). Quantifying the effects of mixing state on aerosol optical properties. *Atmospheric Chemistry and Physics*, 22, 9265–9282. https://doi.org/10.5194/acp-22-9265-2022
- Zaveri, R. A., Easter, R. C., Fast, J. D., & Peters, L. K. (2008). Model for Simulating Aerosol Interactions and Chemistry (MOSAIC). *Journal of Geophysical Research: Atmospheres*, 113, D13204. https://doi.org/10.1029/2007JD008782
- Zelenyuk, A., Imre, D., Wilson, J. M., Zhang, Z., Wang, J., & Mueller, K. (2015). Airborne Single Particle Mass Spectrometers (SPLAT II & miniSPLAT) and new software for data visualization and analysis in a geo-spatial context. *Journal of the American Society for Mass Spectrometry*, 26(2), 257–270. https://doi.org/10.1007/s13361-014-1043-4