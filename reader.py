#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import json
import PyParticle
from scipy.interpolate import interp1d
import numpy as np
from importlib import reload
reload(np)

def read_optical_population(output_filename,morphology='core-shell'):
    
    pop_dict = json.load(open(output_filename,'r'))
    if morphology == 'core-shell':
        optical_population = PyParticle.CoreShellPopulation(
            species=populate_aero_specs(pop_dict['species']), 
            spec_masses=np.array(pop_dict['spec_masses']), 
            num_concs=np.array(pop_dict['num_concs']), 
            ids=np.array(pop_dict['ids'],dtype=int),
            rh_grid=np.array(pop_dict['rh_grid']),
            wvl_grid=np.array(pop_dict['wvl_grid']), 
            temp=pop_dict['temp'], 
            core_vols=np.array(pop_dict['core_vols']),
            shell_dry_vols=np.array(pop_dict['shell_dry_vols']),
            h2o_vols=np.array(pop_dict['h2o_vols']), 
            core_ris=np.array(pop_dict['core_ris_real']) + 1j*np.array(pop_dict['core_ris_imag']),
            dry_shell_ris=np.array(pop_dict['dry_shell_ris_real']) + 1j*np.array(pop_dict['dry_shell_ris_imag']),
            h2o_ris=np.array(pop_dict['h2o_ris_real']) + 1j*np.array(pop_dict['h2o_ris_imag']),
            Cexts=np.array(pop_dict['Cexts']),
            Cscas=np.array(pop_dict['Cscas']),
            Cabss=np.array(pop_dict['Cabss']),
            gs=np.array(pop_dict['gs']),
            Cprs=np.array(pop_dict['Cprs']),
            Cbacks=np.array(pop_dict['Cbacks']),
            Cexts_bc=np.array(pop_dict['Cexts_bc']),
            Cscas_bc=np.array(pop_dict['Cscas_bc']),
            Cabss_bc=np.array(pop_dict['Cabss_bc']),
            gs_bc=np.array(pop_dict['gs_bc']),
            Cprs_bc=np.array(pop_dict['Cprs_bc']),
            Cbacks_bc=np.array(pop_dict['Cbacks_bc']),
            Cexts_clear=np.array(pop_dict['Cexts_clear']),
            Cscas_clear=np.array(pop_dict['Cscas_clear']),
            Cabss_clear=np.array(pop_dict['Cabss_clear']),
            gs_clear=np.array(pop_dict['gs_clear']),
            Cprs_clear=np.array(pop_dict['Cprs_clear']),
            Cbacks_clear=np.array(pop_dict['Cbacks_clear']))
    else:
        print('error: morphology must be \'core-shell\'')
    return optical_population
    
def populate_aero_specs(specs_dict):
    species = []
    for spec_name in specs_dict.keys():
        one_spec = PyParticle.AerosolSpecies(
            spec_name, 
            specs_dict[spec_name]['density'], 
            specs_dict[spec_name]['kappa'],
            specs_dict[spec_name]['molar_mass'],
            specs_dict[spec_name]['surface_tension'])
        one_spec.refractive_index=populate_RI(
            specs_dict[spec_name]['refractive_index'])
        species.append(one_spec)
    
    return species
    
def populate_RI(ri_dict):
    wvls = ri_dict['wvls']
    real_ris = ri_dict['real_ris']
    imag_ris = ri_dict['imag_ris']
    RI_params = ri_dict['RI_params']
    if RI_params == None:
        real_ri_fun = lambda wvl: PyParticle.RI_fun(
            wvl, ri_dict['RI_params']['n_550'], ri_dict['RI_params']['alpha_n'])
        imag_ri_fun = lambda wvl: PyParticle.RI_fun(
            wvl, ri_dict['RI_params']['k_550'], ri_dict['RI_params']['alpha_k'])
    else:
        real_ri_fun = lambda wvl: interp1d(wvls, real_ris)(wvl)
        imag_ri_fun = lambda wvl: interp1d(wvls, imag_ris)(wvl)
    
    return PyParticle.RefractiveIndex(
        real_ri_fun, imag_ri_fun, wvls, real_ris, imag_ris, RI_params)
    
    
    