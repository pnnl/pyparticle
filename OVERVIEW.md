# PyParticle — Overview (editable)

This file is a concise, editable overview of the PyParticle repository meant for developers
and future automated agents. Edit freely and use as a single-source summary.

## Short description
PyParticle provides data structures and builders for aerosol particles and particle populations, flexible 
packages for computing derived properties relevant for atmospheric, engineering, and human health, and
helpers for visualizing aerosol particles and populations. The package is organized so new population types, 
particle morphologies, and analysios code are discovered automatically via factory registries (add a module 
under the appropriate `factory/` folder with a `build` callable and use the provided `@register` decorator).

## Key modules and responsibilities
- `src/PyParticle/__init__.py`: public exports, and dataset path resolution (honors `PYPARTICLE_DATA_PATH`).
- `src/PyParticle/aerosol_particle.py`: `Particle` dataclass, `make_particle`, `make_particle_from_masses`, diameter/kappa/Kohler helpers.
- `src/PyParticle/species/`: `AerosolSpecies` dataclass and `registry.py` offering `get_species`, `register_species`, and `retrieve_one_species` (file-based fallback uses `datasets/species_data/aero_data.dat`).
- `src/PyParticle/population/`: `ParticlePopulation` base class, `build_population(config)` wrapper, and `population/factory` implementations (`binned_lognormals`, `monodisperse`, `partmc`, `mam4`).
- `src/PyParticle/optics/`: `OpticalParticle` interface, `OpticalPopulation` aggregator, `builder.py`, `refractive_index.py`, and `optics/factory` modules (morphologies like `homogeneous`, `core_shell`).
- `src/PyParticle/viz/`: plotting helpers.
- `datasets/species_data/aero_data.dat`: canonical species definitions used by `retrieve_one_species`.

## Public API (most used)
- Particle constructors: `Particle`, `make_particle`, `make_particle_from_masses`.
- Builders: `build_population(config)`, `build_optical_population(base_population, config)`.
- Species helpers: `get_species(name, **mods)`, `register_species(...)`.
- Viz helpers: `viz.grids.make_grid_*` functions returning `(fig, ax)`.

## Extension points and how to add features
- Add a new population type:
  - Create `src/PyParticle/population/factory/<new_type>.py` that exposes `build(config)`.
  - `build_population` discovers it automatically; the module name becomes `config['type']`.
- Add a new optics morphology:
  - Create `src/PyParticle/optics/factory/<morph>.py` implementing `build(base_particle, config)` or use `@register("name")` in the module.
- Add new species:
  - Edit `datasets/species_data/aero_data.dat` and run tests, or register at runtime with `register_species`.

## Conventions & notes
- Wavelengths: examples sometimes use microns; internals expect SI (meters). Builders typically convert if needed.
- `PYPARTICLE_DATA_PATH` environment variable overrides bundled `datasets/`.
- Discovery registries tolerate import errors (they skip broken modules), so removal of a module may not produce runtime errors until the module is used.

## Known smells / places to tidy
- Presence of archived/legacy modules (e.g. `analysis_archive.py`, `tests_archive/`) and generated metadata (`pyparticle.egg-info`, `__pycache__`) — candidates for archiving/cleanup.
- Several modules contain commented blocks and `# fixme` notes; consider cleaning or moving TODOs to issue tracker.

## Quick developer checklist
1. Run unit tests: `pytest -q --maxfail=1 --disable-warnings`.
2. Add small unit tests when modifying public behavior.
3. Use `build_population` / `build_optical_population` for example configs — they enforce `type` and other conventions.

---

Edit this file in-place to refine wording or add links to design docs.
