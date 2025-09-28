# Factories & Extending PyParticle

← Back to Index

This guide explains how the package discovers and loads population factories and optics morphologies, and provides templates to add new ones.

Discovery patterns

- Population factories: drop a module into `src/PyParticle/population/factory/` that exposes a `build(config)` callable. The population builder uses `pkgutil` to import modules in that folder and picks modules exposing a `build` function.

- Optics morphologies: drop a module into `src/PyParticle/optics/factory/` and register a callable with the optics registry. Use the provided `@register("name")` decorator so the morph is discoverable.

Minimal population factory template (`my_new_type.py`)

```python
from ..base import ParticlePopulation
from PyParticle import make_particle
from PyParticle.species.registry import get_species

def build(config):
    # Read config and populate species_modifications
    species_modifications = config.get('species_modifications', {})

    # Example: build a list of unique species names from config
    pop_species_names = ['SO4']
    pop_species = tuple(get_species(n, **species_modifications.get(n, {})) for n in pop_species_names)

    pop = ParticlePopulation(species=pop_species, spec_masses=[], num_concs=[], ids=[], species_modifications=species_modifications)

    # create particles (example single particle)
    p = make_particle(100e-9, pop_species, [1.0], species_modifications=species_modifications)
    pop.set_particle(p, 0, 1e8)
    return pop
```

Minimal optics morphology template (`my_morph.py`)

```python
from .registry import register
from PyParticle.optics.base import OpticalParticle

@register('my_morph')
def build(base_particle, config):
    class MyOpticalParticle(OpticalParticle):
        def compute_optics(self):
            # Fill self.Cabs, self.Csca, self.Cext, self.g arrays
            # Use self.wvl_grid and self.rh_grid to match shapes
            pass
    return MyOpticalParticle(base_particle, config)
```

Checklist for adding tests

1. Add a unit test in `tests/unit/` that imports your factory module and calls `build()` with a minimal config.
2. Assert the returned `ParticlePopulation` has expected attributes (non-empty `ids`, `species`, `num_concs`).
3. For optics, build an `OpticalPopulation` and call `get_optical_coeff('b_scat', rh=0.0)` to check shapes.
4. Mark integration tests that require optional deps with markers (see `tests/conftest.py` for examples).

Running tests locally

```bash
conda activate pyparticle
pytest -q tests/unit/test_your_factory.py
```
