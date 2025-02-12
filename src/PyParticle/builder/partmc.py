#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: Laura Fierce
"""

from . import ParticlePopulation
from . import make_particle_from_masses
from . import data_path
import numpy as np
import os
from netCDF4 import Dataset
<<<<<<< HEAD:src/PyParticle/builder/partmc.py
<<<<<<< HEAD:src/PyParticle/builder/partmc.py
from pathlib import Path

=======
# from importlib import reload
# reload(np)
>>>>>>> 7cbaa2e (lognormal builder with fixed bug):PyParticle/builder/partmc.py
=======
# from importlib import reload
# reload(np)
>>>>>>> main:PyParticle/builder/partmc.py


def build(
        population_settings, *,
        n_particles = None,
        N_tot = None,
        species_modifications = {},
        specdata_path = None,
        suppress_warning=True):
    
    partmc_dir = Path(population_settings['partmc_dir'])
    timestep = population_settings['timestep']
    repeat = population_settings['repeat']
    partmc_filepath = get_ncfile(partmc_dir / 'out', timestep, repeat)
    if specdata_path == None:
        specdata_path = partmc_dir
    
    currnc = Dataset(partmc_filepath)
    aero_spec_names = currnc.variables['aero_species'].names.split(',')
    spec_masses = np.array(currnc.variables['aero_particle_mass'][:])
    part_ids = np.array([one_id for one_id in currnc.variables['aero_id'][:]],dtype=int)
    
    if 'aero_num_conc' in currnc.variables.keys():
        num_concs = currnc.variables['aero_num_conc'][:]
    else:
        num_concs = 1./currnc.variables['aero_comp_vol'][:]
    
    if N_tot == None:
        N_tot = np.sum(num_concs)
    
    if n_particles == None:
        idx = np.arange(len(part_ids))
    elif n_particles<=len(part_ids):
        idx = np.random.choice(np.arange(len(part_ids)),size=n_particles,replace=False)
    else:
        raise IndexError('n_particles>len(part_ids)')
    
    partmc_population = ParticlePopulation(species=[],spec_masses=[],num_concs=[],ids=[])
    for ii in idx:
        particle = make_particle_from_masses(
            aero_spec_names, 
            spec_masses[:,ii],
            specdata_path= data_path / 'species_data',
            species_modifications=species_modifications)
        partmc_population.set_particle(
            particle, part_ids[ii], num_concs[ii]*N_tot/np.sum(num_concs[idx]), suppress_warning=suppress_warning)
    return partmc_population

def get_ncfile(partmc_output_dir, timestep, repeat):
    print(partmc_output_dir)
    for root, dirs, files in os.walk(partmc_output_dir):
        f = files[0]
    if f.startswith('urban_plume_wc_'):
        preface_string = 'urban_plume_wc_' #''.join([c for idx,c in enumerate(f) if idx<f.find('0')])
    elif f.startswith('urban_plume_'):
        preface_string = 'urban_plume_'
    else:
        preface_string = 'YOU_NEED_TO_CHANGE_preface_string_'
    ncfile = partmc_output_dir / (preface_string + str(int(repeat)).zfill(4) + '_' + str(int(timestep)).zfill(8) + '.nc')
    return ncfile