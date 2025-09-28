# Populations

← Back to Index

What is a population?

A `ParticlePopulation` is a container representing a set of particle types (each defined by a list of `AerosolSpecies`) together with per-particle masses and number concentrations. Populations are constructed from configuration dictionaries by the population builder and are the primary input to optics construction.

Configuration reference — `binned_lognormals`

This is the most commonly used population type. Key config fields:

| Key | Type | Default | Notes |
|---|---:|---|---|
| type | str | required | "binned_lognormals" |
| N | list[float] | required | total number concentration per mode (units: see N_units) |
| N_units | str | "m-3" | number concentration units (m-3 or cm-3) |
| GMD | list[float] | required | geometric mean diameter (meters unless GMD_units set) |
| GMD_units | str | "m" | 'm' or 'nm' |
| GSD | list[float] | required | geometric standard deviation(s) |
| aero_spec_names | list[list[str]] | required | species per mode (e.g. [["SO4"], ["BC","OC"]]) |
| aero_spec_fracs | list[list[float]] | required | mass fractions per mode aligned with `aero_spec_names` |
| N_bins | int | 100 | number of bins per mode used to discretize the distribution |
| D_min, D_max | float | computed | optional cutoffs (meters) |
| species_modifications | dict | {} | population-level per-spec overrides; stored on returned population |

Two examples

- Single-species example (copy/paste):

```python
from PyParticle.population.builder import build_population

cfg = {
    "type": "binned_lognormals",
    "N": [1e8], "GMD": [100e-9], "GSD": [1.6],
    "aero_spec_names": [["SO4"]], "aero_spec_fracs": [[1.0]],
    "N_bins": 50,
    "species_modifications": {"SO4": {"density": 1770, "n_550": 1.45}}
}
pop = build_population(cfg)
```

- Multi-species example (two-spec mode):

```python
cfg = {
    "type": "binned_lognormals",
    "N": [1e8], "GMD": [150e-9], "GSD": [1.6],
    "aero_spec_names": [["BC","SO4"]], "aero_spec_fracs": [[0.2, 0.8]],
    "N_bins": 60,
    "species_modifications": {"SO4": {"n_550":1.45}, "BC": {"kappa":0.45}}
}
pop = build_population(cfg)
```

Where `species_modifications` lives

Population builders record `species_modifications` on the returned `ParticlePopulation` object (field name `species_modifications`). This is intentional: optics construction may need to attach wavelength-aware refractive indices once per species using those overrides. If an optics config omits `species_modifications`, the optics builder falls back to `base_population.species_modifications`.

Practical tips

- Keep diameter units consistent (use meters for all internal API calls). Convert only when interfacing with third-party libs that require other units (e.g., PyMieScatt uses nm).
- If adding new species, update the datasets under `src/PyParticle/datasets/species_data/` and make sure tests set `PYPARTICLE_DATA_PATH` appropriately.
