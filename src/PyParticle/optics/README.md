# PyParticle Optics Submodule

This submodule implements modular, extensible optical models for particle populations.

## Adding a New Morphology

1. **Create a new file** in `PyParticle/optics/`, e.g. `my_model.py`.
2. **Subclass `OpticalParticle`** from `base.py` and implement:
   - `compute_optics()`
   - `get_cross_sections()`
   - `get_refractive_indices()`
   - `get_cross_section(optics_type, rh_idx=None, wvl_idx=None)`
3. **Register your class** in `MORPHOLOGY_REGISTRY` in `factory.py`.
4. **Use** via `create_optical_particle("my_model", ...)` or through the population manager.

## Example Usage

```python
from PyParticle.optics import create_optical_particle

optical_particle = create_optical_particle(
    morphology="core-shell",
    species=species,
    masses=masses,
    rh_grid=rh_grid,
    wvl_grid=wvl_grid,
    temp=293.15,
    specdata_path=specdata_path,
    species_modifications=species_modifications
)
optical_particle.compute_optics()
cross_sections = optical_particle.get_cross_sections()
```

## Maintaining and Extending

- **Centralize mapping** of string options (like `optics_type`) in `utils.py` and per-model as needed.
- **Document new models** in this README.
- **Raise exceptions** for unsupported options rather than printing warnings.

## DRY Cross-section Lookups

All models and populations use centralized dictionary-based cross-section lookups,
so new optical property types can be added by updating the mapping in `utils.py`
or in a model's `OPTICS_TYPE_MAP`.

## Testing

Add or update unit tests to verify that all models produce correct results for their optical properties.
