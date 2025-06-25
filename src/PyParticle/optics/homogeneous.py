import numpy as np
from .base import OpticalParticle
from .utils import OPTICS_TYPE_MAP

#from PyMieScatt import MieQ

class HomogeneousParticle(OpticalParticle):
    """
    Homogeneous sphere morphology optical particle model.
    """
    OPTICS_TYPE_MAP = {
        "total_abs": "Cabs",
        "total_scat": "Csca",
        "total_ext": "Cext",
    }

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
        self.Cabs = np.zeros((N_rh, N_wvl))
        self.Csca = np.zeros((N_rh, N_wvl))
        self.Cext = np.zeros((N_rh, N_wvl))
        self.g = np.zeros((N_rh, N_wvl))
        self.ri = np.ones(N_wvl) * 1.54  # placeholder

    def compute_optics(self):
        """
        Compute optical properties using Mie theory for homogeneous spheres.
        """
        
        # import MieQ only if needed
        try:
            from PyMieScatt import  MieQ
        except ImportError:
            raise ImportError(
                "PyMieScatt is required for 'PyParticle.optics.homogeneous.compute_optics'. "
                "Please install it with 'pip install PyMieScatt'."
            )
        
        diameter_nm = 150.0  # placeholder value
        for rr, _ in enumerate(self.rh_grid):
            crossect = np.pi/4. * (diameter_nm * 1e-9) ** 2
            for ww, wavelength_m in enumerate(self.wvl_grid):
                wavelength_nm = wavelength_m * 1e9
                m = self.ri[ww]
                output_dict = MieQ(
                    m, wavelength_nm, diameter_nm, asDict=True, asCrossSection=False)
                self.Cext[rr, ww] = output_dict['Qext'] * crossect
                self.Csca[rr, ww] = output_dict['Qsca'] * crossect
                self.Cabs[rr, ww] = output_dict['Qabs'] * crossect
                self.g[rr, ww] = output_dict['g']

    def get_cross_sections(self):
        return {k: getattr(self, v) for k, v in self.OPTICS_TYPE_MAP.items() if hasattr(self, v)}

    def get_refractive_indices(self):
        return {
            "ri": self.ri
        }

    def get_cross_section(self, optics_type, rh_idx=None, wvl_idx=None):
        array_name = self.OPTICS_TYPE_MAP.get(optics_type)
        if array_name is None:
            raise ValueError(f"Unknown optics_type: {optics_type}")
        arr = getattr(self, array_name)
        if rh_idx is not None and wvl_idx is not None:
            return arr[rh_idx, wvl_idx]
        return arr
