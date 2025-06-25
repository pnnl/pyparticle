import numpy as np
from .base import OpticalParticle
from .utils import OPTICS_TYPE_MAP
# from PyMieScatt.CoreShell import MieQCoreShell

class CoreShellParticle(OpticalParticle):
    """
    Core-shell morphology optical particle model.
    """
    OPTICS_TYPE_MAP = OPTICS_TYPE_MAP.copy()

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
        # Initialize all needed arrays
        self.Cabs = np.zeros((N_rh, N_wvl))
        self.Csca = np.zeros((N_rh, N_wvl))
        self.Cext = np.zeros((N_rh, N_wvl))
        self.g = np.zeros((N_rh, N_wvl))

        self.Cabs_bc = np.zeros((N_rh, N_wvl))
        self.Csca_bc = np.zeros((N_rh, N_wvl))
        self.Cext_bc = np.zeros((N_rh, N_wvl))

        self.Cabs_clear = np.zeros((N_rh, N_wvl))
        self.Csca_clear = np.zeros((N_rh, N_wvl))
        self.Cext_clear = np.zeros((N_rh, N_wvl))

        self.core_ris = np.ones(N_wvl) * 1.5  # placeholder
        self.shell_ris = np.ones(N_wvl) * 1.6  # placeholder

    def compute_optics(self):
        """
        Compute optical properties using Mie theory for core-shell particles.
        """
        
        # import MieQCoreShell only if needed
        try:
            from PyMieScatt import  MieQCoreShell
        except ImportError:
            raise ImportError(
                "PyMieScatt is required for 'PyParticle.optics.core_shell.compute_optics'. "
                "Please install it with 'pip install PyMieScatt'."
            )
        
        dCore_nm = 100.0  # placeholder
        dShell_nm = 200.0  # placeholder
        for rr, _ in enumerate(self.rh_grid):
            total_crossect = np.pi/4. * (dShell_nm * 1e-9) ** 2
            for ww, wavelength_m in enumerate(self.wvl_grid):
                wavelength_nm = wavelength_m * 1e9
                mCore = self.core_ris[ww]
                mShell = self.shell_ris[ww]
                output_dict = MieQCoreShell(
                    mCore, mShell, wavelength_nm, dCore_nm, dShell_nm, asDict=True, asCrossSection=False)
                self.Cext[rr, ww] = output_dict['Qext'] * total_crossect
                self.Csca[rr, ww] = output_dict['Qsca'] * total_crossect
                self.Cabs[rr, ww] = output_dict['Qabs'] * total_crossect
                self.g[rr, ww] = output_dict['g']
                # For demonstration, use same values for the "bc" and "clear" arrays
                self.Cext_bc[rr, ww] = self.Cext[rr, ww]
                self.Csca_bc[rr, ww] = self.Csca[rr, ww]
                self.Cabs_bc[rr, ww] = self.Cabs[rr, ww]
                self.Cext_clear[rr, ww] = self.Cext[rr, ww]
                self.Csca_clear[rr, ww] = self.Csca[rr, ww]
                self.Cabs_clear[rr, ww] = self.Cabs[rr, ww]

    def get_cross_sections(self):
        # Return all available cross-section arrays
        return {k: getattr(self, v) for k, v in self.OPTICS_TYPE_MAP.items() if hasattr(self, v)}

    def get_refractive_indices(self):
        return {
            "core_ris": self.core_ris,
            "shell_ris": self.shell_ris
        }

    def get_cross_section(self, optics_type, rh_idx=None, wvl_idx=None):
        array_name = self.OPTICS_TYPE_MAP.get(optics_type)
        if array_name is None:
            raise ValueError(f"Unknown optics_type: {optics_type}")
        arr = getattr(self, array_name)
        if rh_idx is not None and wvl_idx is not None:
            return arr[rh_idx, wvl_idx]
        return arr
