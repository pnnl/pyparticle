"""PyParticle package

Lightweight package providing aerosol particle, species, population, and
optics builders. This module exposes the high-level API used by examples
and tests: particle constructors, species registry, population builders,
and optics builders.

Exports (selected):
- Particle, make_particle, make_particle_from_masses
- AerosolSpecies and registry helpers (get_species, register_species, ...)
- build_population, build_optical_particle, build_optical_population

See package submodules for implementation details.
"""
import os
import numpy as np
from pathlib import Path


def _get_data_path():
    """Resolve the package datasets path.

    Resolution order:
    1. Environment override via PYPARTICLE_DATA_PATH (absolute path to datasets dir).
    2. Package-local datasets directory (based on this file's location).

    This is robust when PyParticle is imported from other packages: it will
    locate the datasets bundled with this package rather than using the
    current working directory.
    """
    env = os.getenv("PYPARTICLE_DATA_PATH")
    if env:
        return Path(env)

    # Default: prefer datasets directory next to this package's source files
    pkg_root = Path(__file__).resolve().parent
    candidate = pkg_root / "datasets"
    if candidate.exists():
        return candidate

    # Otherwise, look for a top-level 'datasets' directory in ancestor folders
    for parent in pkg_root.parents:
        cand = parent / "datasets"
        if cand.exists():
            return cand

    # Final fallback: use CWD/datasets to preserve backwards compatibility
    return Path.cwd() / "datasets"


# Exported path for modules that expect a pathlib-like object.
data_path = _get_data_path()

# Public helpers
from .utilities import get_number

from .aerosol_particle import Particle, make_particle, make_particle_from_masses

# Updated imports for new species/registry structure
from .species.base import AerosolSpecies
from .species.registry import (
    get_species,
    register_species,
    list_species,
    extend_species,
    retrieve_one_species,
)

from .population.base import ParticlePopulation
from .population import build_population

from .optics.builder import build_optical_particle, build_optical_population
