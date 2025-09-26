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

from .refractive_index import add_RI_to_spec
from .. import data_path
import copy

class OpticalParticle(Particle):
    """
    Base class for all optical particle morphologies.
    """

    def __init__(self, base_particle, config):
        # fixme: can all of this be moved to the base optical particle?
        super().__init__(species=base_particle.species, masses=base_particle.masses)
        
        # Grids
        self.rh_grid = np.asarray(config.get("rh_grid", [0.0]), dtype=float)
        self.wvl_grid = np.asarray(config.get("wvl_grid", [550e-9]), dtype=float)  # meters
        self.temp = float(config.get("temp", 293.15))

        # Options
        self.specdata_path = config.get("specdata_path", data_path / 'species_data')
        self.species_modifications = config.get("species_modifications", {})
        
        N_rh = len(self.rh_grid)
        N_wvl = len(self.wvl_grid)
        
        self.Cabs = np.zeros((N_rh, N_wvl))
        self.Csca = np.zeros((N_rh, N_wvl))
        self.Cext = np.zeros((N_rh, N_wvl))
        self.g    = np.zeros((N_rh, N_wvl))
        
        self._add_spec_RIs(specdata_path=self.specdata_path,
                          species_modifications=self.species_modifications)
        # self.spec_ris=[]
        # for spec in self.species:
        #     self.spec_ris.append(RI_fun(spec, self.wvl_grid, temp=self.temp, specdata_path=self.specdata_path, species_modifications=self.species_modifications))        
    
    def _add_spec_RIs(
            self,specdata_path=data_path / 'species_data',
            species_modifications={}):
            # return_lookup=False,return_params=False):
        old_specs = self.species
        print('old_specs', old_specs)
        wvls = self.wvl_grid
        new_specs = []
        for old_spec in old_specs:
            print('old_spec', old_spec)
            if old_spec.name in species_modifications.keys():
                spec_modifications = species_modifications[old_spec.name]
            elif 'SOA' in species_modifications.keys() and old_spec.name in ['MSA','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']:
                spec_modifications = species_modifications['SOA']
            else:
                spec_modifications = {}
            
            
            new_spec = add_RI_to_spec(
                old_spec,wvls=self.wvl_grid,
                specdata_path=specdata_path,
                species_modifications=species_modifications)
            new_specs.append(copy.deepcopy(new_spec))
        self.species = new_specs
        

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
    Aggregate per-particle optical cross-sections (C*: m²) onto a common RH×λ grid.
    Aggregation to coefficients (m⁻¹) is done by summing C* × number_concentration (m⁻³).
    """

    def __init__(self, base_population, rh_grid, wvl_grid):
        self.base_population = base_population
        self.species = base_population.species
        self.spec_masses = base_population.spec_masses.copy()
        self.num_concs = base_population.num_concs.copy()
        self.ids = list(base_population.ids)

        self.rh_grid = np.asarray(rh_grid, dtype=float)
        self.wvl_grid = np.asarray(wvl_grid, dtype=float)

        nP = len(self.ids)
        nR = len(self.rh_grid)
        nW = len(self.wvl_grid)

        # main cubes: per particle
        self.Cabs  = np.zeros((nP, nR, nW))
        self.Csca  = np.zeros((nP, nR, nW))
        self.Cext  = np.zeros((nP, nR, nW))
        self.g     = np.zeros((nP, nR, nW))

        # optional variant cubes (filled by morphologies that provide them)
        self.Cabs_bc  = None; self.Csca_bc  = None; self.Cext_bc  = None; self.g_bc  = None
        self.Cabs_clear=None; self.Csca_clear=None; self.Cext_clear=None; self.g_clear=None

        # lazily computed
        self.tkappas = None
        self.shell_tkappas = None

    def _find_index(self, part_id):
        try:
            return self.ids.index(part_id)
        except ValueError:
            raise ValueError(f"Particle id {part_id} not found in population ids")

    def add_optical_particle(self, optical_particle, part_id):
        """
        Copy computed cross-sections from a per-particle 'optical_particle'
        into the population arrays. Ensures compute_optics() is called once.
        """
        # Ensure arrays are computed
        if getattr(optical_particle, "Cext", None) is None or optical_particle.Cext.size == 0:
            optical_particle.compute_optics()

        i = self._find_index(part_id)

        # Basic cubes (shape (nR, nW))
        self.Cabs[i, :, :] = optical_particle.Cabs
        self.Csca[i, :, :] = optical_particle.Csca
        self.Cext[i, :, :] = optical_particle.Cext
        self.g[i,   :, :]  = optical_particle.g

        # Variants (create on first use)
        for name in ("Cabs_bc","Csca_bc","Cext_bc","g_bc",
                     "Cabs_clear","Csca_clear","Cext_clear","g_clear"):
            if hasattr(optical_particle, name) and getattr(optical_particle, name) is not None:
                if getattr(self, name) is None:
                    shape = (len(self.ids), len(self.rh_grid), len(self.wvl_grid))
                    setattr(self, name, np.zeros(shape))
                getattr(self, name)[i, :, :] = getattr(optical_particle, name)

    # --- aggregation helpers ---

    def _select_indices(self, rh, wvl):
        """Return (rh_idx, wvl_idx) or (slice(None), slice(None)) if None."""
        if rh is None:
            rh_idx = slice(None)
        else:
            arr = np.asarray(self.rh_grid)
            hits = np.where(np.isclose(arr, rh))[0]
            if len(hits) == 0:
                raise ValueError(f"RH {rh} not found in rh_grid {self.rh_grid}")
            rh_idx = int(hits[0])

        if wvl is None:
            wvl_idx = slice(None)
        else:
            arr = np.asarray(self.wvl_grid)
            hits = np.where(np.isclose(arr, wvl))[0]
            if len(hits) == 0:
                raise ValueError(f"Wavelength {wvl} not found in wvl_grid {self.wvl_grid}")
            wvl_idx = int(hits[0])

        return rh_idx, wvl_idx

    def get_optical_coeff(self, optics_type: str, rh=None, wvl=None, bconly: bool=False):
        """
        Aggregate to m⁻¹. optics_type in:
          'total_abs','pure_bc_abs','clear_abs',
          'total_scat','pure_bc_scat','clear_scat'
        """
        key = str(optics_type).lower().strip()
        rh_idx, wvl_idx = self._select_indices(rh, wvl)

        # choose source cube(s)
        def pick(name_main, name_variant=None):
            if name_variant:
                cube = getattr(self, name_variant)
                if cube is None:
                    raise ValueError(f"{name_variant} not available for this morphology.")
                return cube
            return getattr(self, name_main)

        if   key == "total_abs":    cube = pick("Cabs")
        elif key == "pure_bc_abs":  cube = pick("Cabs", "Cabs_bc")
        elif key == "clear_abs":    cube = pick("Cabs", "Cabs_clear")
        elif key == "total_scat":   cube = pick("Csca")
        elif key == "pure_bc_scat": cube = pick("Csca", "Csca_bc")
        elif key == "clear_scat":   cube = pick("Csca", "Csca_clear")
        else:
            raise ValueError(f"Unknown optics_type: {optics_type}")

        # filter particles if bconly
        part_sel = slice(None)
        if bconly:
            # BC index by name
            idx_bc = [i for i, s in enumerate(self.species) if getattr(s, "name", "").upper() == "BC"]
            if not idx_bc:
                # if no BC species exists, result is zero
                return 0.0
            bc_col = self.spec_masses[:, idx_bc[0]]
            part_mask = (bc_col > 0.0)
            part_sel = np.where(part_mask)[0]
            if part_sel.size == 0:
                return 0.0

        # Sum over particles: (m²) * (m⁻³) -> m⁻¹
        num = self.num_concs.reshape(-1, 1, 1)  # (nP,1,1)
        summed = np.sum(cube[part_sel, :, :] * num[part_sel, :, :], axis=0)

        out = summed[rh_idx, wvl_idx]
        return float(out) if np.ndim(out) == 0 else out

    # --- kappa summaries for parity with original ---

    def compute_effective_kappas(self):
        self.tkappas = np.zeros(len(self.ids))
        self.shell_tkappas = np.zeros(len(self.ids))
        for ii, pid in enumerate(self.ids):
            p = self.base_population.get_particle(pid)
            self.tkappas[ii] = float(p.get_tkappa())
            self.shell_tkappas[ii] = float(p.get_shell_tkappa())