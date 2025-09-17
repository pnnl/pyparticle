import numpy as np
from ..base import OpticalParticle
from .registry import register
from ...aerosol_particle import Particle

@register("core_shell")
class CoreShellParticle(OpticalParticle):
    """
    Core-shell morphology with RH and wavelength dependence.
    Uses PyMieScatt if available; otherwise a size-parameter-based fallback.
    """

    def __init__(self, base_particle, config):
        super().__init__(base_particle.species, base_particle.masses)
        self.base_particle = base_particle

        # Grids
        self.rh_grid = np.asarray(config.get("rh_grid", [0.0]), dtype=float)
        self.wvl_grid = np.asarray(config.get("wvl_grid", [550e-9]), dtype=float)  # meters
        self.temp = float(config.get("temp", 293.15))

        # Options
        self.specdata_path = config.get("specdata_path", None)
        self.species_modifications = config.get("species_modifications", {})
        self.core_frac = float(config.get("core_frac", 0.6))  # D_core = core_frac * D_shell
        self.single_scatter_albedo = float(config.get("single_scatter_albedo", 0.9))
        self.fallback_kappa = float(config.get("fallback_kappa", 0.3))
        # Diameter unit for base particle getters: 'm' or 'um'
        self.diameter_units = str(config.get("diameter_units", "um")).lower()

        N_rh = len(self.rh_grid)
        N_wvl = len(self.wvl_grid)
        self.Cabs = np.zeros((N_rh, N_wvl))
        self.Csca = np.zeros((N_rh, N_wvl))
        self.Cext = np.zeros((N_rh, N_wvl))
        self.g    = np.zeros((N_rh, N_wvl))

        # Placeholder refractive indices (replace with your effective RI model)
        self.core_ris  = np.ones(N_wvl) * 1.5
        self.shell_ris = np.ones((N_rh, N_wvl)) * 1.6

    def _clone_particle(self):
        return Particle(self.base_particle.species, self.base_particle.masses.copy())

    def _to_meters(self, D):
        # Convert diameter D from configured units to meters
        if self.diameter_units.startswith("um"):
            return float(D) * 1e-6
        return float(D)

    def _Dwet_at_rh(self, rh):
        """
        Wet diameter in meters at RH in [0,1].
        """
        # Try model-native water equilibrium
        try:
            p = self._clone_particle()
            p._equilibrate_h2o(S=float(rh), T=self.temp)  # adjust if your API expects percent
            D = p.get_Dwet()
            if np.isfinite(D) and D > 0:
                return self._to_meters(D)
        except Exception:
            pass

        # Fallback: simple kappa growth (volume-based => diameter^(1))
        rh_safe = float(np.clip(rh, 0.0, 0.99))
        try:
            Ddry = self._to_meters(self.base_particle.get_Ddry())
        except Exception:
            Ddry = 200e-9
        GF = (1.0 + self.fallback_kappa * rh_safe / max(1e-6, (1.0 - rh_safe))) ** (1.0 / 3.0)
        return float(Ddry) * GF

    def compute_optics(self):
        # Try PyMieScatt first
        try:
            from PyMieScatt import MieQCoreShell
            use_pymie = True
        except ImportError:
            use_pymie = False

        for rr, rh in enumerate(self.rh_grid):
            D_shell_m = self._Dwet_at_rh(rh)   # varies with RH
            if not np.isfinite(D_shell_m) or D_shell_m <= 0.0:
                continue

            r_m = 0.5 * D_shell_m
            area = np.pi * r_m * r_m

            if use_pymie:
                D_shell_nm = D_shell_m * 1e9
                D_core_nm = max(1.0, self.core_frac * D_shell_nm)
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
                # Fallback: toy Mie behavior varying with size parameter x
                for ww, lam_m in enumerate(self.wvl_grid):
                    lam = float(lam_m)
                    x = 2.0 * np.pi * r_m / lam
                    Qext = 2.0 * (1.0 - np.exp(-x / 2.0))       # increases with x, saturates ~2
                    Cext = Qext * area
                    omega0 = self.single_scatter_albedo
                    Csca = omega0 * Cext
                    Cabs = (1.0 - omega0) * Cext
                    g = 0.2 + 0.7 * (1.0 - np.exp(-x / 10.0))   # grows with x, < 1
                    g = float(np.clip(g, 0.0, 0.95))

                    self.Cext[rr, ww] = Cext
                    self.Csca[rr, ww] = Csca
                    self.Cabs[rr, ww] = Cabs
                    self.g[rr, ww]    = g
    def get_cross_sections(self):
        return {
            "Cabs": self.Cabs,
            "Csca": self.Csca,
            "Cext": self.Cext,
            "g": self.g,
        }
    
    def get_refractive_indices(self):
        return {"ri": self.ri}
    
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