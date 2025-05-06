#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import netCDF4
from . import ParticlePopulation
from . import make_particle
from scipy.stats import norm
import numpy as np
from . import data_path

from . import binned_lognormals

def build(
        population_settings, *,
        specdata_path=data_path / 'species_data',
        species_modifications={},
        surface_tension=0.072, D_is_wet=False, 
        #fixme: put rhos in "constants.py" and double-check values
        rho_h2o=1000., rho_so4=1840., rho_soa=1000., rho_else=1800.,#just guessing?
        ):
    output_filename = population_settings['output_filename']
    timestep = population_settings['timestep']
    
    # set these in population_settings
    # population_settings['D_min']
    # population_settings['D_max']
    # population_settings['N_bins']
    # population_settings['GSD']
    
    gmd_wet = netCDF4.Dataset(output_filename)['dgn_awet'][:,timestep]
    gmd_dry = netCDF4.Dataset(output_filename)['dgn_a'][:,timestep]
    
    if D_is_wet:
        population_settings['GMD'] = gmd_wet
    else:
        population_settings['GMD'] = gmd_dry
    # fixme: assume same for wet and dry?
    gsd = population_settings['GSD'] 
    
    # fixme: right now, species data is tied to PartMC defaults; make more flexible
    population_settings['aero_spec_names'] = ['so4','ARO1','BC','h2o']
    so4_aer = netCDF4.Dataset(output_filename)['so4_aer'][:,timestep]
    soa_aer = netCDF4.Dataset(output_filename)['soa_aer'][:,timestep]
    
    # fixme: double-check units
    num_aer = netCDF4.Dataset(output_filename)['num_aer'][:,timestep]
    
    population_settings['N'] = num_aer
    tot_vol = lambda N, gmd, gsd: np.pi/6.*N*np.exp(3*np.log(gmd) + 3**2*np.log(gsd)**2/2.)
    
    # fixme: output more species
    vol_else = tot_vol(num_aer, gmd_dry, gsd) - so4_aer/rho_so4 - soa_aer/rho_soa
    else_aer = vol_else*rho_else
    else_aer[else_aer<0.] = 0.
    
    vol_h2o = tot_vol(num_aer, gmd_wet, gsd) - tot_vol(num_aer, gmd_dry, gsd)
    h2o_aer = vol_h2o*rho_h2o
    
    wetmass_aer = so4_aer + soa_aer + else_aer + h2o_aer
    drymass_aer = so4_aer + soa_aer + else_aer
    population_settings['aero_spec_fracs'] = np.vstack([
        so4_aer/wetmass_aer, soa_aer/wetmass_aer, else_aer/wetmass_aer, h2o_aer/wetmass_aer])
    mam4_population = binned_lognormals.build(
        population_settings,
        specdata_path=specdata_path,
        species_modifications=species_modifications,
        surface_tension=surface_tension, D_is_wet=D_is_wet)
    
    return mam4_population