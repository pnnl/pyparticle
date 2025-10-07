# optics/factory/homogeneous.py
import numpy as np
import math

from .registry import register
from ..base import OpticalParticle
from ..refractive_index import build_refractive_index

try:
    from PyParticle._patch import patch_pymiescatt
    patch_pymiescatt()
    from PyMieScatt import MieQ
except Exception as e:
    MieQ = None
    _PMS_ERR = e


@register("homogeneous")
class HomogeneousParticle(OpticalParticle):
    """
    Homogeneous sphere morphology optical particle model with RH and wavelength dependence.

    Constructor expects (base_particle, config) to align with the factory builder.

    Optional config (read by OpticalParticle or here):
      - rh_grid, wvl_grid, temp (K), specdata_path, species_modifications
      - single_scatter_albedo (fallback SSA when PyMieScatt is unavailable; default: 0.9)
    """

    def __init__(self, base_particle, config):
        super().__init__(base_particle, config)

        # Refractive indices are attached at the population level by the
        # optics builder; the base class's _attach_refractive_indices is
        # guarded and will no-op if the species already have wavelength-aware
        # RIs. Keep the call to the base preparation intact.

        # User-tunable fallback SSA (only used if PyMieScatt is missing)
        self.single_scatter_albedo = float(config.get("single_scatter_albedo", 0.9))

        # Precompute geometry & per-wavelength dry/water RIs
        self._prepare_geometry_and_ris()

        # Do the optics
        self.compute_optics()

    def _prepare_geometry_and_ris(self):
        """
        Precompute:
          - dry particle volume
          - water volumes vs RH (from Dwet - Ddry)
          - wavelength-dependent RIs for dry mix and water
        """
        # Total dry volume (all non-water solid/liquid species "dry")
        self.dry_vol = float(self.get_vol_dry())

        # Water volumes vs RH from Dwet(RH) and Ddry
        Ddry = float(self.get_Ddry())
        vol_dry_geom = (math.pi / 6.0) * (Ddry ** 3)

        self.h2o_vols = np.zeros(len(self.rh_grid))
        for rr, rh in enumerate(self.rh_grid):
            Dw = float(self.get_Dwet(RH=float(rh), T=self.temp))
            self.h2o_vols[rr] = (math.pi / 6.0) * (Dw ** 3 - Ddry ** 3) if rh > 0.0 else 0.0

        # Per-wavelength complex RIs
        Nw = len(self.wvl_grid)
        self.dry_ris = np.zeros(Nw, dtype=complex)
        self.h2o_ris = np.zeros(Nw, dtype=complex)

        # Species partitioning (use Particle-provided helpers if available)
        vks = self.get_vks()  # species "dry" volumes for partitioning
        h2o_idx = self.idx_h2o()

        # Build dry mixture RI by volume-weighted averaging of all non-water species
        dry_indices = [ii for ii in range(len(self.species)) if ii != h2o_idx]

        # Guard against degenerate dry volume
        vol_norm = self.dry_vol if self.dry_vol > 0.0 else 1.0

        for ww in range(Nw):
            # Dry effective RI
            if len(dry_indices) > 0 and self.dry_vol > 0.0:
                n_dry = 0.0
                k_dry = 0.0
                for ii in dry_indices:
                    f = float(vks[ii] / vol_norm)
                    n_dry += self.species[ii].refractive_index.real_ri_fun(self.wvl_grid[ww]) * f
                    k_dry += self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid[ww]) * f
                self.dry_ris[ww] = complex(n_dry, k_dry)
            else:
                self.dry_ris[ww] = complex(1.0, 0.0)

            # Water RI
            n_w = self.species[h2o_idx].refractive_index.real_ri_fun(self.wvl_grid[ww])
            k_w = self.species[h2o_idx].refractive_index.imag_ri_fun(self.wvl_grid[ww])
            self.h2o_ris[ww] = complex(n_w, k_w)

        # Sanity check: geometric dry volume from Ddry should match sum(vks) reasonably
        # (not enforcing; just computed above as vol_dry_geom for possible debugging)
        _ = vol_dry_geom  # kept for parity with CoreShell; not currently used

    def _mixture_ri(self, rr: int, ww: int) -> complex:
        """
        Volume-weighted homogeneous mixture of dry material and water at a given RH and wavelength.
        """
        v_h2o = self.h2o_vols[rr]
        v_dry = self.dry_vol
        if (v_h2o + v_dry) <= 0.0:
            return complex(1.0, 0.0)
        return (self.h2o_ris[ww] * v_h2o + self.dry_ris[ww] * v_dry) / (v_h2o + v_dry)

    def compute_optics(self):
        """
        Compute cross-sections and asymmetry parameter per (RH, wavelength).
        Prefer PyMieScatt if available; otherwise use a size-parameter-based fallback.
        """
        use_pymie = MieQ is not None

        for rr, rh in enumerate(self.rh_grid):
            D_m = float(self.get_Dwet(RH=float(rh), T=self.temp, sigma_sa=self.get_surface_tension()))
            r_m = 0.5 * D_m
            area = math.pi * r_m * r_m  # geometric cross-section

            if use_pymie:
                D_nm = D_m * 1e9
                for ww, lam_m in enumerate(self.wvl_grid):
                    lam_nm = float(lam_m * 1e9)
                    m = complex(self._mixture_ri(rr, ww))
                    out = MieQ(m, lam_nm, D_nm, asDict=True, asCrossSection=False)
                    # Convert efficiencies to absolute cross sections via geometric area
                    self.Cext[rr, ww] = out["Qext"] * area
                    self.Csca[rr, ww] = out["Qsca"] * area
                    self.Cabs[rr, ww] = out["Qabs"] * area
                    self.g[rr, ww]    = out["g"]
            else:
                # Fallback: simple Mie-like behavior vs size parameter x = 2πr/λ
                for ww, lam_m in enumerate(self.wvl_grid):
                    lam = float(lam_m)
                    x = 2.0 * math.pi * r_m / lam

                    # Extinction efficiency rises toward ~2 as x increases
                    Qext = 2.0 * (1.0 - math.exp(-x / 2.0))
                    Cext = Qext * area

                    # Partition using fallback SSA
                    omega0 = self.single_scatter_albedo
                    Csca = omega0 * Cext
                    Cabs = (1.0 - omega0) * Cext

                    # Asymmetry parameter grows with x, capped < 1
                    g = 0.2 + 0.7 * (1.0 - math.exp(-x / 10.0))
                    g = float(np.clip(g, 0.0, 0.95))

                    self.Cext[rr, ww] = Cext
                    self.Csca[rr, ww] = Csca
                    self.Cabs[rr, ww] = Cabs
                    self.g[rr, ww]    = g

    # Convenience getters (unchanged)
    def get_cross_sections(self):
        return {
            "Cabs": self.Cabs,
            "Csca": self.Csca,
            "Cext": self.Cext,
            "g": self.g,
        }

    def get_refractive_indices(self):
        return {"dry_ri": self.dry_ris, "h2o_ri": self.h2o_ris}

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
