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

def build(
        population_settings,
        specdata_path='../datasets/aerosol/species_data/', 
        surface_tension=0.072):
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
    
    if 'H2O' in aero_spec_names or 'h2o' in aero_spec_names:
        if aero_spec_names[-1].upper() != 'h2o':
            print('error! H2O must be last')
    else:
        aero_spec_names.append('H2O')
        aero_spec_fracs = np.hstack([aero_spec_fracs,0.])
    
    lognormal_population = ParticlePopulation(
        species=aero_spec_names,spec_masses=[],num_concs=[],ids=[])
    for dd,(D,N_per_bin) in enumerate(zip(D_mids,N_per_bins)):
        particle = make_particle(
            D, aero_spec_names, aero_spec_fracs,
            specdata_path=specdata_path, 
            surface_tension=surface_tension)
        part_id = dd
        lognormal_population.set_particle(
            particle, part_id, N_per_bin)
    
    return lognormal_population
