import numpy as np
from ..base import OpticalParticle
from .registry import register
from ...aerosol_particle import Particle


@register("homogeneous")
class HomogeneousParticle(OpticalParticle):
    """
    Homogeneous sphere morphology optical particle model with RH and wavelength dependence.

    Constructor expects (base_particle, config) to align with the factory builder.

    Config options:
      - rh_grid: array-like of RH in [0,1]
      - wvl_grid: array-like of wavelengths in meters
      - temp: temperature [K], used if your particle supports water equilibrium
      - diameter_units: 'um' or 'm' for base_particle diameter getters (default 'um')
      - single_scatter_albedo: fallback SSA when PyMieScatt is unavailable (default 0.9)
      - fallback_kappa: kappa used in a simple growth model if equilibrium not available (default 0.3)
      - n_550, alpha_n: optional top-level real RI at 550 nm and spectral slope
      - k_550, alpha_k: optional top-level imaginary RI at 550 nm and spectral slope
      - specdata_path, species_modifications: passed through for future effective-RI usage
    """

    def __init__(self, base_particle, config):
        # Initialize as a Particle using the base particle's composition
        super().__init__(base_particle.species, base_particle.masses)

        self.base_particle = base_particle

        # Grids and settings
        self.rh_grid = np.asarray(config.get("rh_grid", [0.0]), dtype=float)
        self.wvl_grid = np.asarray(config.get("wvl_grid", [550e-9]), dtype=float)  # meters
        self.temp = float(config.get("temp", 293.15))
        self.diameter_units = str(config.get("diameter_units", "um")).lower()

        # Options and placeholders
        self.specdata_path = config.get("specdata_path", None)
        self.species_modifications = config.get("species_modifications", {})

        # Fallback optical behavior when PyMieScatt isn't available
        self.single_scatter_albedo = float(config.get("single_scatter_albedo", 0.9))
        self.fallback_kappa = float(config.get("fallback_kappa", 0.3))

        # Optional simple spectral parameterization for n and k
        self.n_550 = float(config.get("n_550", 1.54))
        self.alpha_n = float(config.get("alpha_n", 0.0))
        self.k_550 = float(config.get("k_550", 0.0))
        self.alpha_k = float(config.get("alpha_k", 0.0))

        # Allocate outputs
        N_rh = len(self.rh_grid)
        N_wvl = len(self.wvl_grid)
        self.Cabs = np.zeros((N_rh, N_wvl))
        self.Csca = np.zeros((N_rh, N_wvl))
        self.Cext = np.zeros((N_rh, N_wvl))
        self.g    = np.zeros((N_rh, N_wvl))

        # Refractive index array per wavelength (placeholder spectral model)
        self.ri = self._build_ri_spectrum(self.wvl_grid)

    def _build_ri_spectrum(self, wvl_m):
        """
        Simple spectral model for complex refractive index:
          n(λ) = n_550 * (λ / 0.55e-6)^alpha_n
          k(λ) = k_550 * (λ / 0.55e-6)^alpha_k
        """
        lam0 = 0.55e-6
        scale = (np.asarray(wvl_m, dtype=float) / lam0).astype(float)
        n = self.n_550 * np.power(scale, self.alpha_n)
        k = self.k_550 * np.power(scale, self.alpha_k)
        return n + 1j * k

    def _clone_particle(self):
        # Fresh particle with same species & masses to avoid mutating shared base_particle
        return Particle(self.base_particle.species, self.base_particle.masses.copy())

    def _to_meters(self, D):
        """Convert diameter from configured units to meters."""
        if self.diameter_units.startswith("um"):
            return float(D) * 1e-6
        return float(D)

    def _Dwet_at_rh(self, rh):
        """
        Return wet diameter (meters) at RH in [0,1].
        Tries particle's water equilibrium; falls back to simple kappa growth.
        """
        # Try model-native water equilibrium
        try:
            p = self._clone_particle()
            p._equilibrate_h2o(S=float(rh), T=self.temp)  # adjust if API expects percent
            D = p.get_Dwet()
            if np.isfinite(D) and D > 0:
                return self._to_meters(D)
        except Exception:
            pass

        # Fallback: kappa-Köhler-like growth factor -> diameter scales with GF^(1/3)
        rh_safe = float(np.clip(rh, 0.0, 0.99))
        try:
            Ddry = self._to_meters(self.base_particle.get_Ddry())
        except Exception:
            Ddry = 200e-9
        GF = (1.0 + self.fallback_kappa * rh_safe / max(1e-6, (1.0 - rh_safe))) ** (1.0 / 3.0)
        return float(Ddry) * GF

    def compute_optics(self):
        """
        Compute cross-sections and asymmetry parameter per (RH, wavelength).
        Prefer PyMieScatt if available; otherwise use a size-parameter-based fallback.
        """
        # Attempt to use PyMieScatt if available
        try:
            from PyMieScatt import MieQ
            use_pymie = True
        except ImportError:
            use_pymie = False

        for rr, rh in enumerate(self.rh_grid):
            D_m = self._Dwet_at_rh(rh)  # varies with RH
            if not np.isfinite(D_m) or D_m <= 0.0:
                continue

            r_m = 0.5 * D_m
            area = np.pi * r_m * r_m  # geometric cross-section

            if use_pymie:
                D_nm = D_m * 1e9
                for ww, lam_m in enumerate(self.wvl_grid):
                    lam_nm = float(lam_m * 1e9)
                    m = complex(self.ri[ww])
                    out = MieQ(m, lam_nm, D_nm, asDict=True, asCrossSection=False)
                    # Convert efficiencies to absolute cross sections via geometric area
                    self.Cext[rr, ww] = out["Qext"] * area
                    self.Csca[rr, ww] = out["Qsca"] * area
                    self.Cabs[rr, ww] = out["Qabs"] * area
                    self.g[rr, ww]    = out["g"]
            else:
                # Fallback: toy Mie-like behavior varying with size parameter x = 2πr/λ
                for ww, lam_m in enumerate(self.wvl_grid):
                    lam = float(lam_m)
                    x = 2.0 * np.pi * r_m / lam

                    # Extinction efficiency rises toward ~2 as x increases
                    Qext = 2.0 * (1.0 - np.exp(-x / 2.0))
                    Cext = Qext * area

                    # Partition using fallback SSA
                    omega0 = self.single_scatter_albedo
                    Csca = omega0 * Cext
                    Cabs = (1.0 - omega0) * Cext

                    # Asymmetry parameter grows with x, capped < 1
                    g = 0.2 + 0.7 * (1.0 - np.exp(-x / 10.0))
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
    """Optional fallback factory callable for discovery."""
    return HomogeneousParticle(base_particle, config)