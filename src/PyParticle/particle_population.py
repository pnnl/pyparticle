#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

from dataclasses import dataclass
from typing import Tuple
# from typing import Optional
from warnings import warn
# from scipy.constants import R
# import scipy.optimize as optimize

from . import Particle
from . import AerosolSpecies
import numpy as np

@dataclass
class ParticlePopulation:
    """ParticlePopulation: the definition of a population of particles
    in terms of the number concentrations of different particles """
    
    species: Tuple[AerosolSpecies, ...] # shape = N_species
    spec_masses: np.array # shape = (N_particles, N_species)
    num_concs: np.array # shape = N_particles
    ids: Tuple[int, ...] # shape = N_particles
    
    def find_particle(self, part_id):
        if part_id in self.ids:
            idx, = np.where([one_id == part_id for one_id in self.ids])
            if len(idx)>1:
                ValueError('part_id is listed more than once in self.ids')
            else:
                idx = idx[0]
        else:
            idx = len(self.ids)
        return idx
    
    def get_particle(self, part_id):
        if part_id in self.ids:
            idx_particle = self.find_particle(part_id)
            return Particle(self.species, self.spec_masses[idx_particle,:])
        else:
            raise ValueError(str(part_id) + ' not in ids')
            
    def set_particle(self, particle, part_id, num_conc, suppress_warning=False):
        part_id = int(part_id)
        if part_id not in self.ids:
            if not suppress_warning:
                warn('part_id not in self.ids, adding ' + str(part_id))
            self.add_particle(particle, part_id, num_conc)
        else:
            self.species = particle.species
            idx = self.find_particle(part_id)
            self.spec_masses[idx,:] = particle.masses
            self.num_concs[idx] = num_conc
            self.ids[idx] = part_id
    
    def add_particle(self, particle, part_id, num_conc):
        if len(self.ids) == 0:
            self.species = particle.species
            self.spec_masses = np.zeros([1,len(particle.species)])
            self.spec_masses[0,:] = particle.masses
            self.num_concs = np.hstack([num_conc])
            self.ids = [part_id]
        else:
            self.spec_masses = np.vstack([self.spec_masses, particle.masses.reshape(1,-1)])
            self.num_concs = np.hstack([self.num_concs, num_conc])
            self.ids.append(part_id)
            
    def get_effective_radius(self):
        rs = []
        for part_id in self.ids:
            particle = self.get_particle(part_id)
            rs.append(particle.get_Dwet()/2.)
        rs = np.asarray(rs)
        Ns = self.num_concs
        return np.sum(rs*Ns)/np.sum(Ns)
    
    def get_Ntot(self):
        return np.sum(self.num_concs)