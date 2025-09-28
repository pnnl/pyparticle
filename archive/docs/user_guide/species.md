# Species

← Back to Index

Overview

Species data (molecular weight, density, refractive-index base parameters) are provided in `src/PyParticle/datasets/species_data/`. The species registry (`PyParticle.species.registry`) exposes `get_species(name, **overrides)` which returns an `AerosolSpecies` object. Use `overrides` (the same keys used in `species_modifications`) to alter density, refractive-index parameters, or other per-spec properties.

Typical override keys

- `density` (kg/m³)
- `n_550` (real refractive index at 550 nm)
- `k_550` (imag refractive index at 550 nm)
- `alpha_n`, `alpha_k` (spectral slopes)
- `kappa` (hygroscopicity)

Example — using `get_species` directly

```python
from PyParticle.species.registry import get_species
so4 = get_species("SO4", density=1770, n_550=1.45, k_550=0.0)
```

Setting `PYPARTICLE_DATA_PATH`

By default the package reads data shipped under `src/PyParticle/datasets/species_data`. For tests or development you can override where species data are loaded from by setting the environment variable `PYPARTICLE_DATA_PATH` before running Python or pytest:

```bash
export PYPARTICLE_DATA_PATH=/path/to/local/species_data
pytest -q
```

Notes

- Prefer setting per-spec overrides at the population level via `species_modifications` (a mapping passed in population configs) — population builders will pass these into `get_species` when constructing the `ParticlePopulation`.
- If you add a new species file or modify the data format, add a unit test that verifies `get_species` still returns a valid `AerosolSpecies` object and that key attributes (density, name) match expected values.
