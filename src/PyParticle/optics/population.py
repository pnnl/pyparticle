import numpy as np
# Use the builder API to construct optical particles
from .builder import build_optical_particle

class OpticalPopulation:
    """
    Manages a population of optical particles, possibly of mixed morphologies.
    """

    def __init__(self, rh_grid, wvl_grid):
        self.rh_grid = rh_grid
        self.wvl_grid = wvl_grid
        self.particles = []
        self.num_concs = []
        self.ids = []

    def add_particle(self, particle, num_conc, morphology="core-shell", **kwargs):
        config = {
            "type": morphology,
            "rh_grid": list(self.rh_grid),
            "wvl_grid": list(self.wvl_grid),
            "temp": getattr(particle, "temp", 293.15),
            "specdata_path": kwargs.get("specdata_path", None),
            "species_modifications": kwargs.get("species_modifications", None),
        }
        optical_particle = build_optical_particle(particle, config)
        optical_particle.compute_optics()
        self.particles.append(optical_particle)
        self.num_concs.append(num_conc)
        self.ids.append(getattr(particle, "id", len(self.ids)))

    def get_optical_coeff(self, optics_type, rh=None, wvl=None):
        """
        Compute the total optical property for the population by summing over all particles and concentrations.
        """
        idx_rh = idx_wvl = None
        if rh is not None and wvl is not None:
            idx_rh = list(self.rh_grid).index(rh)
            idx_wvl = list(self.wvl_grid).index(wvl)
        total = None
        for i, particle in enumerate(self.particles):
            arr = particle.get_cross_section(optics_type, idx_rh, idx_wvl)
            if total is None:
                total = np.zeros_like(arr)
            total += arr * self.num_concs[i]
        return total

