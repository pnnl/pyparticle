#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration script for using PyParticle to compute particle properties 

@author: Laura Fierce
"""
from PyParticle._patch import patch_pymiescatt; patch_pymiescatt(), del patch_pymiescatt

import numpy as np
import PyParticle
from importlib import reload
reload(np)

rh_grid = np.hstack([0.,0.99])
# wvl_grid = np.hstack([350e-9,550e-9,750e-9])
wvl_grid = np.hstack([550e-9])

# from dataclasses import asdict
import json 
import csv

# D = 100e-9
# aero_spec_names = ['BC','SO4','OC','H2O']
# aero_spec_fracs = np.hstack([0.7,0.15,0.05,0.1])
# particle = PyParticle.make_particle(D, aero_spec_names, aero_spec_fracs)


# cs_particle = PyParticle.make_optical_particle(particle, rh_grid, wvl_grid)

# print(cs_particle)

# population_settings = {
#     'D':100e-9,
#     'num_conc':10e9,
#     'aero_spec_names':['BC','OC','SO4','H2O'],
#     'aero_spec_fracs':np.array([0.1,0.2,0.7,0.])}
# particle_population = PyParticle.monodisperse.build(
#     population_settings, n_sd=3, specdata_path = PyParticle.data_path + 'species_data/')

# optical_population = PyParticle.make_optical_population(particle_population, rh_grid, wvl_grid)
# species_modifications = {
#     'BC':{'kappa':1.2,'k_550':0.1,'alpha_k':0}}
from pyprojroot import here
datadir = here() / 'datasets' / 'partmc'
species_modifications = {
    'SOA':{'k_550':1e-3,'alpha_k':0}}
population_settings = {
    'partmc_dir': datadir, 
    'timestep':13, 'repeat':1}
particle_population = PyParticle.builder.partmc.build(
    population_settings,n_particles=None,species_modifications=species_modifications)
optical_population = PyParticle.make_optical_population(
    particle_population, rh_grid, wvl_grid,species_modifications=species_modifications)


from dataclasses import asdict
optical_pop_dict = asdict(optical_population)
optical_pop_dict['population_settings'] = population_settings
optical_pop_dict['species_modifications'] = species_modifications

output_file = 'sample.pickle'
import pickle
pickle.dump(optical_pop_dict, open(output_file, 'wb') )
