#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple
# from typing import Optional
from warnings import warn
from scipy.constants import R
import scipy.optimize as optimize

from . import Particle
from . import AerosolSpecies

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
    
    def get_particle(self, part_id, idx_h2o=-1):
        if part_id in self.ids:
            # print([one_id == part_id for one_id in self.ids])
            idx_particle = self.find_particle(part_id)
            print(idx_particle,self.spec_masses.shape)
            return Particle(self.species, self.spec_masses[idx_particle,:], idx_h2o=idx_h2o)
        else:
            raise ValueError(str(part_id) + ' not in ids')
            
    def set_particle(self, particle, part_id, num_conc):
        if part_id not in self.ids:
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
            self.spec_masses = np.zeros([1,len(self.species)])
            self.spec_masses[0,:] = particle.masses
            self.num_concs = np.array([num_conc])
            self.ids = [part_id]
        else:
            self.spec_masses = np.hstack([self.spec_masses, particle.masses])
            self.num_concs = np.hstack([self.num_concs, num_conc])
            self.ids.append(part_id)