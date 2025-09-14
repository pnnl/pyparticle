import numpy as np
from .base import OpticalParticle
# from .utils import OPTICS_TYPE_MAP
from .factory.registry import register


class FractalAggregateParticle(OpticalParticle):
    """
    Fractal aggregate morphology (stub example).
    """
    # OPTICS_TYPE_MAP = {
    #     "total_abs": "Cabs",
    #     "total_scat": "Csca",
    #     "total_ext": "Cext",
    # }
    
    # fixme: does this contain all of the parameters needed for the BNN?
    def __init__(self, species, masses, rh_grid, wvl_grid, temp, specdata_path, species_modifications):
        self.species = species
        self.masses = masses
        self.rh_grid = rh_grid
        self.wvl_grid = wvl_grid
        self.temp = temp
        self.specdata_path = specdata_path
        self.species_modifications = species_modifications

        N_rh = len(rh_grid)
        N_wvl = len(wvl_grid)
        self.Cabs = np.zeros((N_rh, N_wvl))  # placeholder
        self.Csca = np.zeros((N_rh, N_wvl))
        self.Cext = np.zeros((N_rh, N_wvl))
        self.g = np.zeros((N_rh, N_wvl))
        self.ri_effs = self.zeros((N_rh, N_wvl))
        
        # fixme: make these optional
        # self.Cabs_bc = np.zeros((N_rh, N_wvl))
        # self.Csca_bc = np.zeros((N_rh, N_wvl))
        # self.Cext_bc = np.zeros((N_rh, N_wvl))

        # self.Cabs_clear = np.zeros((N_rh, N_wvl))
        # self.Csca_clear = np.zeros((N_rh, N_wvl))
        # self.Cext_clear = np.zeros((N_rh, N_wvl))
        
        # self.ri_eff = np.ones(N_wvl) * 1.7  # placeholder

    def compute_optics(self):
        """
        Placeholder for fractal aggregate optics.
        """
        self.Cabs[:] = 0.0
        self.Csca[:] = 0.0
        self.Cext[:] = 0.0
        self.g[:] = 0.0

    def get_cross_sections(self):
        return {k: getattr(self, v) for k, v in self.OPTICS_TYPE_MAP.items() if hasattr(self, v)}

    def get_refractive_indices(self):
        return {
            "ri_eff": self.ri_eff
        }

    def get_cross_section(self, optics_type, rh_idx=None, wvl_idx=None):
        array_name = self.OPTICS_TYPE_MAP.get(optics_type)
        if array_name is None:
            raise ValueError(f"Unknown optics_type: {optics_type}")
        arr = getattr(self, array_name)
        if rh_idx is not None and wvl_idx is not None:
            return arr[rh_idx, wvl_idx]
        return arr
    
# fixme: build one particle?
@register("fractal")
def build(base_particle, *args, **kwargs):
    return FractalAggregateParticle(base_particle, *args, **kwargs)

