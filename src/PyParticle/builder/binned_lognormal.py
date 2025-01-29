#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to create
@author: Laura Fierce
"""
from . import ParticlePopulation
from . import make_particle
from scipy.stats import norm
import numpy as np
from . import data_path


from .. import data_path
def build(
<<<<<<< HEAD:src/PyParticle/builder/binned_lognormal.py
        population_settings, *,
        specdata_path=data_path / 'species_data',
        species_modifications={},
=======
        population_settings,
        species_modifications={},
        specdata_path=data_path + 'species_data/', 
>>>>>>> 7cbaa2e (lognormal builder with fixed bug):PyParticle/builder/binned_lognormal.py
        surface_tension=0.072, D_is_wet=False):
    D_min = population_settings['D_min']
    D_max = population_settings['D_max']
    N_bins = population_settings['N_bins']
    D_mids = np.logspace(np.log10(D_min),np.log10(D_max),num=N_bins)
    bin_width = D_mids[1] - D_mids[0]
    
    Ntot = population_settings['Ntot']
    log10_GMD = np.log10(population_settings['GMD'])
    log10_GSD = np.log10(population_settings['GSD'])
    
    pdf_wrt_logD = norm(loc=log10_GMD, scale=log10_GSD)
    N_per_bins = pdf_wrt_logD.pdf(np.log10(D_mids))*bin_width
    N_per_bins = Ntot*N_per_bins/np.sum(N_per_bins)
    
    # assume same mass fraction across the population
    aero_spec_names = population_settings['aero_spec_names']
    aero_spec_fracs = population_settings['aero_spec_fracs']
    assert(len(aero_spec_fracs) == len(aero_spec_names))
    
    
    
    lognormal_population = ParticlePopulation(
        species=aero_spec_names,spec_masses=[],num_concs=[],ids=[])
    for dd,(D,N_per_bin) in enumerate(zip(D_mids,N_per_bins)):
        particle = make_particle(
            D,
            aero_spec_names.copy(),  # w/o cpy the object gets modded
            aero_spec_fracs.copy(),  # which invalidates assumptions. (assertions in code)
            specdata_path=specdata_path, 
            species_modifications=species_modifications, 
            D_is_wet=D_is_wet)
        part_id = dd
        lognormal_population.set_particle(
            particle, part_id, N_per_bin)
    
    return lognormal_population
