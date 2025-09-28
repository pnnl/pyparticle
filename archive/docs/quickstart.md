# Quickstart

← Back to Index

This Quickstart shows the minimal steps to install PyParticle, build a small population, attach optics, and query a scattering coefficient. It is written for users who want to run examples locally. Follow the `environment.yml` to create the conda environment.

Installation (conda)

1. Create the environment (runs in a few minutes):

```bash
conda env create -f environment.yml -n pyparticle
conda activate pyparticle
```

Hello World — build a population and compute b_scat

The following short example builds a single-mode binned lognormal population, constructs homogeneous-sphere optics at a single wavelength, and queries the scattering coefficient b_scat.

```python
from PyParticle.population.builder import build_population
from PyParticle.optics.builder import build_optical_population

pop_cfg = {
    "type": "binned_lognormals",
    "GMD": [100e-9], "GSD": [1.6], "N": [1e8],
    "aero_spec_names": [["SO4"]], "aero_spec_fracs": [[1.0]],
    "N_bins": 50,
    "species_modifications": {"SO4": {"n_550": 1.45, "k_550": 0.0}}
}
pop = build_population(pop_cfg)

opt_cfg = {"type": "homogeneous", "wvl_grid": [550e-9], "rh_grid": [0.0]}
opt_pop = build_optical_population(pop, opt_cfg)
print(opt_pop.get_optical_coeff("b_scat", rh=0.0))
```

Notes

- Units: wavelengths are expressed in meters inside PyParticle (e.g. 550 nm == 550e-9 m).
- `species_modifications` is a population-level mapping of per-species overrides (density, refractive index parameters). Population builders record these on the returned `ParticlePopulation` so optics can reuse them.

Run a smoke test

With the conda env active, run a minimal test:

```bash
pytest -q tests/unit/test_population_smoke_skeleton.py
```

Troubleshooting (common issues)

- SciPy / PyMieScatt missing: make sure you created and activated the `pyparticle` or `pyparticle-partmc` environment depending on tests you run.
- Missing species data: set `PYPARTICLE_DATA_PATH` to the local `src/PyParticle/datasets` path if loading fails.
- Numerical mismatch with PyMieScatt: small differences can arise from binning/integration choices; see `examples/helpers/pymiescatt_comparison.py` for reproduction guidance.

This Quickstart intentionally keeps examples compact; for deeper usage, see the User Guide pages in `docs/user_guide/`.
