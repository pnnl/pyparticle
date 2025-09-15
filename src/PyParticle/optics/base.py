"""Optical particle and population base classes.

Defines abstract `OpticalParticle` interface and `OpticalPopulation` which
aggregates per-particle cross-sections into population-level optical
coefficients (extinction, scattering, absorption, asymmetry).
"""

from abc import abstractmethod
import numpy as np

# Adjust imports to your tree
from ..aerosol_particle import Particle
from ..population.base import ParticlePopulation


class OpticalParticle(Particle):
    """
    Base class for all optical particle morphologies.
    """

    @abstractmethod
    def compute_optics(self):
        """Compute per-particle optical cross-sections across RH and wavelength grids.

        Implementations should populate attributes like `Cabs`, `Csca`, `Cext`, and `g`.
        """
    
    @abstractmethod
    def get_cross_sections(self):
        pass

    @abstractmethod
    def get_refractive_indices(self):
        pass
    
    @abstractmethod
    def get_cross_section(self, optics_type, rh_idx=None, wvl_idx=None):
        pass


class OpticalPopulation(ParticlePopulation):
    """
    Manages a population of optical particles, possibly of mixed morphologies.
    Holds cross-section cubes per particle and provides population-aggregated optics.
    """

    def __init__(self, base_population, rh_grid, wvl_grid):
        # Initialize ParticlePopulation state
        super().__init__(
            species=base_population.species,
            spec_masses=np.array(base_population.spec_masses, copy=True),
            num_concs=np.array(base_population.num_concs, copy=True),
            ids=list(base_population.ids).copy(),
        )

        self.rh_grid = np.asarray(rh_grid, dtype=float)
        self.wvl_grid = np.asarray(wvl_grid, dtype=float)
        # Prepare storage for per-particle cross-section cubes
        N_part = len(self.ids)
        N_rh = len(self.rh_grid)
        N_wvl = len(self.wvl_grid)
        self.Cabs = np.zeros((N_part, N_rh, N_wvl), dtype=float)
        self.Csca = np.zeros((N_part, N_rh, N_wvl), dtype=float)
        self.Cext = np.zeros((N_part, N_rh, N_wvl), dtype=float)
        self.g    = np.zeros((N_part, N_rh, N_wvl), dtype=float)
    def add_optical_particle(self, optical_particle, part_id, **kwargs):
        optical_particle.compute_optics()
        idx = self.find_particle(part_id)
        if idx >= len(self.ids) or self.ids[idx] != part_id:
            raise ValueError(f"part_id {part_id} not found in OpticalPopulation ids.")
        self.Cabs[idx, :, :] = optical_particle.Cabs
        self.Csca[idx, :, :] = optical_particle.Csca
        self.Cext[idx, :, :] = optical_particle.Cext
        self.g[idx, :, :]    = optical_particle.g

    def _select_indices(self, rh, wvl):
        """
        Convert RH / wavelength values to indices using tolerant float matching.
        Returns (rh_idx, wvl_idx) where each is either an int or slice(None).
        This avoids NumPy advanced indexing pitfalls with range/list indexers.
        """
        # RH
        if rh is None:
            rh_idx = slice(None)
        else:
            rh_arr = np.asarray(self.rh_grid, dtype=float)
            hit = np.where(np.isclose(rh_arr, float(rh), rtol=0, atol=1e-12))[0]
            if len(hit) == 0:
                raise ValueError(f"Requested rh={rh} not found in rh_grid {list(self.rh_grid)}")
            rh_idx = int(hit[0])

        # Wavelength
        if wvl is None:
            wvl_idx = slice(None)
        else:
            wvl_arr = np.asarray(self.wvl_grid, dtype=float)
            hit = np.where(np.isclose(wvl_arr, float(wvl), rtol=0, atol=1e-18))[0]
            if len(hit) == 0:
                raise ValueError(f"Requested wvl={wvl} not found in wvl_grid {list(self.wvl_grid)}")
            wvl_idx = int(hit[0])

        return rh_idx, wvl_idx

    def _safe_index_2d(self, arr2d, i, j):
        """
        Robust 2D indexing that handles:
          - both slice(None): return full array
          - one int, one slice: return a 1D array
          - both ints: return scalar
          - if given sequences (list/ndarray) for both, use np.ix_ to get the grid
        """
        # Normalize Python range -> slice
        if isinstance(i, range):
            i = slice(i.start, i.stop, i.step)
        if isinstance(j, range):
            j = slice(j.start, j.stop, j.step)

        seq_types = (list, tuple, np.ndarray)

        if isinstance(i, seq_types) and isinstance(j, seq_types):
            i_idx = np.asarray(i, dtype=int)
            j_idx = np.asarray(j, dtype=int)
            return arr2d[np.ix_(i_idx, j_idx)]
        elif isinstance(i, seq_types):
            i_idx = np.asarray(i, dtype=int)
            return arr2d[i_idx, j]
        elif isinstance(j, seq_types):
            j_idx = np.asarray(j, dtype=int)
            return arr2d[i, j_idx]
        else:
            return arr2d[i, j]

    def get_optical_coeff(self, optics_type, rh=None, wvl=None):
        """
        Compute the population-level optical property.

        optics_type:
          - 'b_abs','absorption','abs'     -> sum_i Cabs_i * N_i
          - 'b_scat','scattering','scat'   -> sum_i Csca_i * N_i
          - 'b_ext','extinction','ext'     -> sum_i Cext_i * N_i
          - 'g','asymmetry'                -> scattering-weighted mean:
                                             sum_i (g_i * Csca_i * N_i) / sum_i (Csca_i * N_i)

        rh, wvl:
          - None means return values across the full grid dimension(s).
          - If both provided, return a scalar.
        """
        rh_idx, wvl_idx = self._select_indices(rh, wvl)

        key = str(optics_type).lower()
        w = self.num_concs.reshape(-1, 1, 1)  # weight by number concentration

        if key in ('b_abs', 'absorption', 'abs'):
            total = np.sum(self.Cabs * w, axis=0)
        elif key in ('b_scat', 'scattering', 'scat'):
            total = np.sum(self.Csca * w, axis=0)
        elif key in ('b_ext', 'extinction', 'ext'):
            total = np.sum(self.Cext * w, axis=0)
        elif key in ('g', 'asymmetry'):
            num = np.sum(self.g * self.Csca * w, axis=0)
            den = np.sum(self.Csca * w, axis=0)
            with np.errstate(invalid='ignore', divide='ignore'):
                total = np.where(den > 0, num / den, 0.0)
        else:
            raise ValueError(f"optics_type = {optics_type} not implemented.")

        out = self._safe_index_2d(total, rh_idx, wvl_idx)
        if np.ndim(out) == 0:
            return float(out)
        return out