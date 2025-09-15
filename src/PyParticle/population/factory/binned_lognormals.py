#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a binned lognormal population
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from ..utils import expand_compounds_for_population
from PyParticle import make_particle
from PyParticle.species.registry import get_species
from .registry import register
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

@register("binned_lognormals")
def build(config):
    # fixme: make this +/- certain number of sigmas (rather than min/max diams)
    # D_min = float(config['D_min'])
    # D_max = float(config['D_max'])
        
    N_list = config['N']
    GMD_list = config['GMD']
    GSD_list = config['GSD']
    
    N_bins_list = config['N_bins']
    if type(config['N_bins']) is list:
        N_bins_list = config.get('N_bins')
    else:
        N_bins_val = config.get('N_bins',100)
        N_bins_list = [N_bins_val]*len(GMD_list)
    
    N_sigmas = config.get('N_sigmas',5) # used to set bin ranges for each mode
    D_min = config.get('D_min',min(GMD_list)-max(GSD_list)*N_sigmas)
    D_max = config.get('D_max',max(GMD_list)+max(GSD_list)*N_sigmas)
    
    aero_spec_names_list = config['aero_spec_names']
    aero_spec_fracs_list = config['aero_spec_fracs']
    # Support compound-like species names (e.g., NaCl, (NH4)2SO4)
    aero_spec_names_list, aero_spec_fracs_list = expand_compounds_for_population(
        aero_spec_names_list, aero_spec_fracs_list
    )
    species_modifications = config.get('species_modifications', {})
    surface_tension = config.get('surface_tension', 0.072)
    D_is_wet = config.get('D_is_wet', False)
    specdata_path = config.get('specdata_path', None)
    
    print(D_is_wet)
    # Build master species list for the *population*, preserving order
    pop_species_names = []
    for mode_names in aero_spec_names_list:
        for name in mode_names:
            if name not in pop_species_names:
                pop_species_names.append(name)
    # Build species objects
    pop_species_list = tuple(
        get_species(spec_name, **species_modifications.get(spec_name, {}))
        for spec_name in pop_species_names
    )

    # Create the population object with the right species list
    lognormals_population = ParticlePopulation(
        species=pop_species_list, spec_masses=[], num_concs=[], ids=[]
    )

    
    part_id = 0
    for mode_idx, (Ntot, GMD, GSD, mode_spec_names, mode_spec_fracs,N_bins) in enumerate(
            zip(N_list, GMD_list, GSD_list, aero_spec_names_list, aero_spec_fracs_list,N_bins_list)):
        D_min = np.exp(np.log(GMD) - N_sigmas/2. * np.log(GSD))
        D_max = np.exp(np.log(GMD) + N_sigmas/2. * np.log(GSD))
        D_mids = np.logspace(np.log10(D_min), np.log10(D_max), num=N_bins)
        bin_width = np.log10(D_mids[1]) - np.log10(D_mids[0])
        
        # Map this mode's fractions to the full population species list
        # For each species in pop_species_names, use the fraction from this mode, or 0 if not present
        mode_spec_name_to_frac = dict(zip(mode_spec_names, mode_spec_fracs))
        pop_aligned_fracs = [mode_spec_name_to_frac.get(n, 0.0) for n in pop_species_names]
        pdf_wrt_logD = norm(loc=np.log10(GMD), scale=np.log10(GSD))
        N_per_bins = pdf_wrt_logD.pdf(np.log10(D_mids)) * bin_width
        N_per_bins = float(Ntot) * N_per_bins / np.sum(N_per_bins)
        
        for dd, (D, N_per_bin) in enumerate(zip(D_mids, N_per_bins)):
            particle = make_particle(
                D,
                pop_species_list,
                pop_aligned_fracs.copy(),
                species_modifications=species_modifications,
                D_is_wet=D_is_wet)
            part_id += 1
            lognormals_population.set_particle(
                particle, part_id, N_per_bin)
    return lognormals_population