#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to create a monodisperse population

@author: Laura
"""

from . import ParticlePopulation
from . import make_particle
import numpy as np


def build(
        population_settings,
        specdata_path='../datasets/aerosol/species_data/', 
        surface_tension=0.072):
    D = population_settings['D']
    aero_spec_names = population_settings['aero_spec_names']
    aero_spec_fracs = population_settings['aero_spec_fracs']
    
    if 'H2O' in aero_spec_names or 'h2o' in aero_spec_names:
        if aero_spec_names[-1].upper() != 'h2o':
            print('error! H2O must be last')
        # else:
        #     aero_spec_names = aero_spec_names[:-1]
        #     aero_spec_fracs = aero_spec_fracs[:-1]
    else:
        aero_spec_names.append('H2O')
        aero_spec_fracs = np.hstack([aero_spec_fracs,0.])
        
        # idx_h2o = np.where([spec_name.lower() == 'h2o' for spec_name in aero_spec_names])
        
    particle = make_particle(
        D, aero_spec_names, aero_spec_fracs,
        specdata_path=specdata_path, 
        surface_tension=surface_tension)
    
    monodisperse_population = ParticlePopulation(species=particle.species,spec_masses=[],num_concs=[],ids=[])
    # print(getattr(monodisperse_population))
    monodisperse_population.set_particle(particle, population_settings['part_id'], population_settings['num_conc'])
    return monodisperse_population
