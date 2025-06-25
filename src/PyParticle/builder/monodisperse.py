#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to create a monodisperse population

@author: Laura Fierce
"""

from . import ParticlePopulation
from . import make_particle
from . import data_path
import numpy as np
from importlib import reload

def build(
        population_settings, *,
        species_modifications={},
        n_particles=1, specdata_path = data_path / 'species_data'):
    
    if specdata_path == None:
        specdata_path = data_path / 'species_data'
    
    D = population_settings['D']
    aero_spec_names = population_settings['aero_spec_names']
    aero_spec_fracs = population_settings['aero_spec_fracs']
    
    particle = make_particle(
        D, aero_spec_names, aero_spec_fracs,
        species_modifications=species_modifications, 
        specdata_path=specdata_path)
    
    monodisperse_population = ParticlePopulation(species=particle.species,spec_masses=[],num_concs=[],ids=[])
    # print(getattr(monodisperse_population))
    print(population_settings)
    if 'ids' in population_settings.keys():
        part_ids = population_settings['ids']
    else:
        part_ids = range(n_particles)
    
    for ii in part_ids:
        monodisperse_population.set_particle(particle, ii, population_settings['num_conc']/n_particles)
    
    return monodisperse_population
