from importlib.resources import files, as_file
import pyparticle as _pkg
from pathlib import Path
import os

def get_data_path() -> Path:
    # Highest priority: explicit override
    if (p := os.environ.get("PYPARTICLE_DATA_PATH")):
        return Path(p).expanduser()

    # Packaged datasets inside the installed wheel
    ds = files(_pkg).joinpath("datasets")
    if ds.is_dir():
        # If callers need a filesystem path (e.g., for C libs), materialize it:
        with as_file(ds) as pth:
            return Path(pth)

    # Last resort for source checkouts
    from pathlib import Path as _P
    cand = _P(__file__).resolve().parent / "datasets"
    return cand

data_path = get_data_path()
__all__ = ["data_path", "get_data_path"]

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