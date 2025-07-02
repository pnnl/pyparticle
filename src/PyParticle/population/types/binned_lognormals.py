#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a binned lognormal population
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from PyParticle import make_particle
from PyParticle.species.registry import get_species
from scipy.stats import norm
import numpy as np

def build(
        population_settings, *,
        specdata_path=None,
        species_modifications={},
        surface_tension=0.072, D_is_wet=False):
    # Get AerosolSpecies objects from names + modifications
    aero_spec_names = population_settings['aero_spec_names']
    # Support species_modifications as a dict: {name: {field: value, ...}}
    species_list = tuple(
        get_species(name, **species_modifications.get(name, {}))
        for name in aero_spec_names
    )
    
    lognormals_population = ParticlePopulation(
        species=species_list, spec_masses=[], num_concs=[], ids=[])
    D_min = population_settings['D_min']
    D_max = population_settings['D_max']
    N_bins = population_settings['N_bins']
    for (Ntot, GMD, GSD, aero_spec_fracs) in zip(
            population_settings['N'],
            population_settings['GMD'], 
            population_settings['GSD'],
            population_settings['aero_spec_fracs']):
        D_mids = np.logspace(np.log10(D_min), np.log10(D_max), num=N_bins)
        bin_width = D_mids[1] - D_mids[0]
        pdf_wrt_logD = norm(loc=np.log10(GMD), scale=np.log10(GSD))
        N_per_bins = pdf_wrt_logD.pdf(np.log10(D_mids)) * bin_width
        N_per_bins = Ntot * N_per_bins / np.sum(N_per_bins)
        for dd, (D, N_per_bin) in enumerate(zip(D_mids, N_per_bins)):
            particle = make_particle(
                D,
                species_list,
                aero_spec_fracs.copy(),
                species_modifications=species_modifications,
                D_is_wet=D_is_wet)
            part_id = dd
            lognormals_population.set_particle(
                particle, part_id, N_per_bin)
    return lognormals_population