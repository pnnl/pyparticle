import numpy as np
from ..base import OpticalParticle
from .registry import register
from ...aerosol_particle import Particle
import copy 
@register("core_shell")
class CoreShellParticle(OpticalParticle):
    """
    Core-shell morphology with RH and wavelength dependence.
    Uses PyMieScatt if available; otherwise a size-parameter-based fallback.
    """
    def __init__(self, base_particle, config):
        super().__init__(base_particle=base_particle, config=config)
        self._add_effective_ris()
    
    # def __init__(self, base_particle, config):
    #     # fixme: can all of this be moved to the base optical particle?
    #     super().__init__(base_particle.species, base_particle.masses)

    #     # Grids
    #     self.rh_grid = np.asarray(config.get("rh_grid", [0.0]), dtype=float)
    #     self.wvl_grid = np.asarray(config.get("wvl_grid", [550e-9]), dtype=float)  # meters
    #     self.temp = float(config.get("temp", 293.15))

    #     # Options
    #     self.specdata_path = config.get("specdata_path", None)
    #     self.species_modifications = config.get("species_modifications", {})
        
    #     N_rh = len(self.rh_grid)
    #     N_wvl = len(self.wvl_grid)
        
    #     self.Cabs = np.zeros((N_rh, N_wvl))
    #     self.Csca = np.zeros((N_rh, N_wvl))
    #     self.Cext = np.zeros((N_rh, N_wvl))
    #     self.g    = np.zeros((N_rh, N_wvl))

    #     self.spec_ris=[]
    #     for ii, spec in enumerate(self.species):
    #         self.spec_ris.append(RI_fun(spec, self.wvl_grid, temp=self.temp, specdata_path=self.specdata_path, species_modifications=self.species_modifications))        
        
    # def _clone_particle(self):
    #     return Particle(species=self.base_particle.species, masses=self.base_particle.masses.copy())
    
    def compute_optics(self):

        # todo: leave options for other optical core-shell optical models
        # Try PyMieScatt first
        try:
            from PyMieScatt import MieQCoreShell
            use_pymie = True
        except ImportError:
            use_pymie = False

        for rr, rh in enumerate(self.rh_grid):
            D_shell_m = self.get_Dwet(RH=rh, T=self.temp, sigma_sa=self.get_surface_tension())

            r_m = 0.5 * D_shell_m
            area = np.pi * r_m * r_m

            if use_pymie:
                D_shell_nm = D_shell_m * 1e9
                D_core_nm = self.get_Dcore() * 1e9
                #D_core_nm = max(1.0, self.core_frac * D_shell_nm)
                for ww, lam_m in enumerate(self.wvl_grid):
                    lam_nm = float(lam_m * 1e9)
                    mCore = complex(self.core_ris[ww])
                    mShell = complex(self.shell_ris[rr, ww])
                    out = MieQCoreShell(
                        mCore, mShell, lam_nm, D_core_nm, D_shell_nm,
                        asDict=True, asCrossSection=False
                    )
                    self.Cext[rr, ww] = out["Qext"] * area
                    self.Csca[rr, ww] = out["Qsca"] * area
                    self.Cabs[rr, ww] = out["Qabs"] * area
                    self.g[rr, ww]    = out["g"]
            else:
                raise ImportError("PyMieScatt is required for core-shell optics calculations.")
                # # Fallback: toy Mie behavior varying with size parameter x
                # for ww, lam_m in enumerate(self.wvl_grid):
                #     lam = float(lam_m)
                #     x = 2.0 * np.pi * r_m / lam
                #     Qext = 2.0 * (1.0 - np.exp(-x / 2.0))       # increases with x, saturates ~2
                #     Cext = Qext * area
                #     omega0 = self.single_scatter_albedo
                #     Csca = omega0 * Cext
                #     Cabs = (1.0 - omega0) * Cext
                #     g = 0.2 + 0.7 * (1.0 - np.exp(-x / 10.0))   # grows with x, < 1
                #     g = float(np.clip(g, 0.0, 0.95))

                #     self.Cext[rr, ww] = Cext
                #     self.Csca[rr, ww] = Csca
                #     self.Cabs[rr, ww] = Cabs
                #     self.g[rr, ww]    = g
    def get_cross_sections(self):
        return {
            "Cabs": self.Cabs,
            "Csca": self.Csca,
            "Cext": self.Cext,
            "g": self.g,
        }
    
    # def get_refractive_indices(self):
    #     return {"ri": self.ri}
    
    
    def _add_effective_ris(self):
        N_wvl = len(self.wvl_grid)
        vks = self.get_vks() 
        
        real_ris = np.zeros(N_wvl)
        imag_ris = np.zeros(N_wvl)
        for ii in self.idx_core():
            real_ris += self.species[ii].refractive_index.real_ri_fun(self.wvl_grid) * vks[ii] / self.core_vol
            imag_ris += self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid) * vks[ii] / self.core_vol
        core_ris = copy.deepcopy(real_ris) + copy.deepcopy(imag_ris) * 1j
        
        real_ris = np.zeros(N_wvl)
        imag_ris = np.zeros(N_wvl)
        for ii in self.idx_dry_shell():
            real_ris += self.species[ii].refractive_index.real_ri_fun(self.wvl_grid) * vks[ii] / self.shell_dry_vol
            imag_ris += self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid) * vks[ii] / self.shell_dry_vol
        dry_shell_ris = copy.deepcopy(real_ris) + copy.deepcopy(imag_ris) * 1j
        
        ii = self.idx_h2o()
        real_ris = self.species[ii].refractive_index.real_ri_fun(self.wvl_grid)
        imag_ris = self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid)
        h2o_ris = copy.deepcopy(real_ris) + copy.deepcopy(imag_ris) * 1j
        
        self.core_ris = np.array([complex(np.real(one_core_ri),np.imag(one_core_ri)) for one_core_ri in core_ris])# np.array(core_ris, dtype=complex)
        self.dry_shell_ris = np.array([complex(np.real(one_shell_ri),np.imag(one_shell_ri)) for one_shell_ri in dry_shell_ris])# np.array(dry_shell_ris, dtype=complex)
        self.h2o_ris = np.array([complex(np.real(one_h2o_ri),np.imag(one_h2o_ri)) for one_h2o_ri in h2o_ris])# np.array(h2o_ris, dtype=complex)
    
    def get_shell_ri(self,rr,ww):
        shell_ri = (
            (self.h2o_ris[ww] * self.h2o_vols[rr]
            + self.dry_shell_ris[ww] * self.shell_dry_vol)
            / (self.h2o_vols[rr] + self.shell_dry_vol)
        )
        
        return shell_ri
    
    def get_core_diam(self):
        return ((6./np.pi) * self.core_vol) ** (1./3.)
    
    def get_cross_section(self, optics_type, rh_idx=None, wvl_idx=None):
        key = str(optics_type).lower()
        if key in ("b_abs", "absorption", "abs"):
            arr = self.Cabs
        elif key in ("b_scat", "scattering", "scat"):
            arr = self.Csca
        elif key in ("b_ext", "extinction", "ext"):
            arr = self.Cext
        elif key in ("g", "asymmetry"):
            arr = self.g
        else:
            raise ValueError(f"Unknown optics_type: {optics_type}")
        if rh_idx is not None and wvl_idx is not None:
            return arr[rh_idx, wvl_idx]
        return arr

def build(base_particle, config):
    return CoreShellParticle(base_particle, config)