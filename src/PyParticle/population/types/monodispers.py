#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a monodisperse population
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from PyParticle import make_particle
from PyParticle.species.registry import get_species
import numpy as np

def build(
        population_settings, *,
        specdata_path=None,
        species_modifications={},
        D_is_wet=False):
    aero_spec_names = population_settings['aero_spec_names']
    species_list = tuple(
        get_species(name, **species_modifications.get(name, {}))
        for name in aero_spec_names
    )
    N = population_settings['N']
    D = population_settings['D']
    aero_spec_fracs = population_settings['aero_spec_fracs']

    monodisp_population = ParticlePopulation(
        species=species_list, spec_masses=[], num_concs=[], ids=[])
    for i in range(len(N)):
        particle = make_particle(
            D[i],
            species_list,
            aero_spec_fracs[i].copy(),
            species_modifications=species_modifications,
            D_is_wet=D_is_wet)
        part_id = i
        monodisp_population.set_particle(
            particle, part_id, N[i])
    return monodisp_population