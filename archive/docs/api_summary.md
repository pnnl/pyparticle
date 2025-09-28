# API Summary

← Back to Index

This page documents the core public entrypoints and brief examples to get started programmatically.

Core entrypoints

- build_population(config) — `PyParticle.population.builder.build_population`

  Signature: build_population(config: dict) -> ParticlePopulation

  Example:

  ```python
  from PyParticle.population.builder import build_population
  pop = build_population(pop_cfg)
  ```

- build_optical_population(base_population, config) — `PyParticle.optics.builder.build_optical_population`

  Signature: build_optical_population(base_population: ParticlePopulation, config: dict) -> OpticalPopulation

  Example:

  ```python
  from PyParticle.optics.builder import build_optical_population
  opt_pop = build_optical_population(pop, {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]})
  ```

- make_particle(D, species_list, fractions, species_modifications={}, D_is_wet=True) — `PyParticle.aerosol_particle.make_particle`

  Signature: make_particle(D: float, aero_spec_names: Sequence[str] or Sequence[AerosolSpecies], aero_spec_frac: Sequence[float], ...) -> Particle

  Example:

  ```python
  from PyParticle.aerosol_particle import make_particle
  p = make_particle(100e-9, ["SO4"], [1.0], species_modifications={"SO4": {"n_550":1.45}})
  ```

- get_species(name, **overrides) — `PyParticle.species.registry.get_species`

  Signature: get_species(name: str, **overrides) -> AerosolSpecies

  Example:

  ```python
  from PyParticle.species.registry import get_species
  so4 = get_species("SO4", n_550=1.45, k_550=0.0)
  ```

- OpticalPopulation.get_optical_coeff(optics_type, rh=None, wvl=None)

  Signature: get_optical_coeff(optics_type: str, rh: float|None, wvl: float|None) -> np.ndarray|float

  Example:

  ```python
  bsc = opt_pop.get_optical_coeff("b_scat", rh=0.0)
  ```

Glossary (short)

- Particle: small container for a set of `AerosolSpecies` and per-spec masses for a single particle.
- ParticlePopulation: container of many Particles, their number concentrations, and population-level metadata (including `species_modifications`).
- OpticalParticle: per-particle optical wrapper (holds RH/λ grids and cross-section cubes).
- OpticalPopulation: aggregated per-particle cross-sections into population coefficients.
