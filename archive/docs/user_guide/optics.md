# Optics

← Back to Index

Overview

Optics are constructed from a base `ParticlePopulation` and an optics configuration. The builder (`PyParticle.optics.builder.build_optical_population`) first ensures each species has a wavelength-aware refractive-index attached (using population-level overrides when provided), then constructs per-particle `OpticalParticle` objects (morphology-specific) and aggregates them into an `OpticalPopulation`.

Required keys for optics config

- `type` — morphology type (e.g., "homogeneous", "core_shell")
- `wvl_grid` — list/array of wavelengths in **meters** (internal convention)
- `rh_grid` — list/array of relative humidities (0.0–1.0 or 0–100 depending on usage)

Morphology discovery

Morphology factories live in `src/PyParticle/optics/factory/`. The optics builder discovers registered morphologies and uses the `type` key to pick the right factory. This allows adding new morphologies without touching the core builder.

Simple examples

- Single wavelength homogeneous sphere (quick):

```python
from PyParticle.population.builder import build_population
from PyParticle.optics.builder import build_optical_population

pop = build_population(pop_cfg)
opt_cfg = {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]}
opt_pop = build_optical_population(pop, opt_cfg)
print(opt_pop.get_optical_coeff("b_scat", rh=0.0))
```

- Multi-wavelength retrieval:

```python
opt_cfg = {"type": "homogeneous", "wvl_grid": list(np.linspace(450e-9, 800e-9, 6)), "rh_grid": [0.0]}
opt_pop = build_optical_population(pop, opt_cfg)
b_scat = opt_pop.get_optical_coeff("b_scat", rh=0.0)
print(b_scat.shape)  # one value per wavelength
```

Units & conversions

- Inside the library, wavelengths are meters. If you use third-party tools (e.g., PyMieScatt) that require nanometers, convert explicitly and document the conversion (see `examples/helpers/pymiescatt_comparison.py`).

Performance note

- The optics builder tries to attach wavelength-aware refractive indices once per species (using the known `wvl_grid`) to avoid per-particle recomputation. When adding morphologies, avoid re-computing RIs unnecessarily.
