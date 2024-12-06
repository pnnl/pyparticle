#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""
import numpy as np
import os
import PyParticle
from importlib import reload
reload(np)
from dataclasses import asdict
import json
import netCDF4

def run_ppe(
        N_samples,
        ensemble_dir,
        processed_output_dir,
        all_run_nums,
        all_timesteps,
        run_weights='uniform',
        timestep_weights='uniform',
        runid_digits=4,
        wvl_grid=np.linspace(350e-9,950e-9,7),
        rh_grid = 1.-np.logspace(0,-2,13),
        specs_to_vary=['SOA','SO4','NH4','NO3'],
        n_550_range = [1.3,1.8],
        n_550_scale = 'lin',
        n_alpha_range = [0.,0.],
        n_alpha_scale = 'lin',
        k_550_range = [0,0.2],
        k_550_scale = 'lin',
        k_alpha_range = [0.,0.],
        k_alpha_scale = 'lin',
        kappa_range=[1e-3,1.2],
        kappa_scale='log',
        repeat_nums=[1],
        test=False):
    
    if not os.path.exists(processed_output_dir):
        os.mkdir(processed_output_dir)
    
    if type(run_weights) == type('uniform'):
        if run_weights == 'uniform':
            run_weights = np.ones(len(all_run_nums))/len(all_run_nums)
        else:
            print('not defined for run_weights = \'' + run_weights + '\'')
    
    if type(timestep_weights) == type('uniform'):    
        if timestep_weights == 'uniform':
            timestep_weights = np.ones(len(all_timesteps))/len(all_timesteps)
        else:
            print('not defined for timestep_weights = \'' + timestep_weights + '\'')
    
    run_nums = sample_choices(N_samples, all_run_nums, run_weights)
    timesteps = sample_choices(N_samples, all_timesteps, timestep_weights)
    # partmc_dirs = sample_partmc_runs(N_samples, ensemble_dir, all_run_nums, run_weights=run_weights, runid_digits=runid_digits)
    # timesteps = sample_timesteps(N_samples, all_timesteps, timestep_weights=timestep_weights)
    
    spec_params = {}
    for spec in specs_to_vary:
        kappa_samples = sample_parameter_uniform(N_samples, kappa_range, param_scale=kappa_scale)
        k_550_samples = sample_parameter_uniform(N_samples, k_550_range, param_scale=k_550_scale)
        k_alpha_samples = sample_parameter_uniform(N_samples, k_alpha_range, param_scale=k_alpha_scale)
        n_550_samples = sample_parameter_uniform(N_samples, n_550_range, param_scale=n_550_scale)
        n_alpha_samples = sample_parameter_uniform(N_samples, n_alpha_range, param_scale=n_alpha_scale)
        spec_params[spec] = {
            'kappa':kappa_samples,
            'k_550':k_550_samples, 'alpha_k':k_alpha_samples, 
            'n_550':n_550_samples, 'alpha_n':n_alpha_samples}
    
    all_species_modifications = []
    all_population_settings = []
    for ii,(run_num,timestep) in enumerate(zip(run_nums,timesteps)):
        partmc_dir = ensemble_dir + str(run_num).zfill(runid_digits) + '/' # partmc_dirs[ii]
        
        species_modifications = {}
        for spec in specs_to_vary:
            species_modifications[spec] = {}
            for param in species_modifications[spec].keys():
                species_modifications[spec][param] = spec_params[spec][param][ii]
        if test:
            species_modifications['NO3'] = {}
            species_modifications['NO3']['n_550'] = 1.5
            species_modifications['NO3']['k_550'] = 0.
            species_modifications['SO4'] = {}
            species_modifications['SO4']['n_550'] = 1.5
            species_modifications['SO4']['k_550'] = 0.
            species_modifications['NH4'] = {}
            species_modifications['NH4']['n_550'] = 1.5
            species_modifications['NH4']['k_550'] = 0.
            species_modifications['SOA'] = {}
            species_modifications['SOA']['n_550'] = 1.45
            species_modifications['SOA']['k_550'] = 0.
            species_modifications['BC'] = {}
            species_modifications['BC']['n_550'] = 1.82
            species_modifications['BC']['k_550'] = 0.74
            species_modifications['OC'] = {}
            species_modifications['OC']['n_550'] = 1.45
            species_modifications['OC']['k_550'] = 0.
            species_modifications['H2O'] = {}
            species_modifications['H2O']['n_550'] = 1.33
            species_modifications['H2O']['k_550'] = 0.
        
        all_species_modifications.append(species_modifications)
        
        for repeat_num in repeat_nums:
            population_settings = {
                'partmc_dir':partmc_dir, 
                'timestep':int(timestep), 'repeat':int(repeat_num), 'test':test}
            all_population_settings.append(population_settings)
            
            if test:
                ncfile = PyParticle.builder.partmc.get_ncfile(population_settings['partmc_dir'] + 'out/', population_settings['timestep'], population_settings['repeat'])
                currnc = netCDF4.Dataset(ncfile)
                rh_grid = np.array([currnc.variables['relative_humidity'][:].data])
            
            particle_population = PyParticle.builder.partmc.build(
                population_settings,n_particles=None,species_modifications=species_modifications)
            optical_population = PyParticle.make_optical_population(
                particle_population, rh_grid, wvl_grid,species_modifications=species_modifications)
            
            optical_pop_dict = make_population_dictionary(optical_population)
            optical_pop_dict = separate_ris(optical_pop_dict)
            optical_pop_dict = arrays_to_lists(optical_pop_dict)
            
            if test:
                optical_pop_dict['test'] = {}
                optical_pop_dict['test']['Cabss'] = np.array(currnc.variables['aero_absorb_cross_sect'][:].data)
                optical_pop_dict['test']['Cscas'] = np.array(currnc.variables['aero_scatter_cross_sect'][:].data)
                optical_pop_dict['test']['core_ris_real'] = np.array(currnc.variables['aero_refract_core_real'][:].data)
                optical_pop_dict['test']['core_ris_imag'] = np.array(currnc.variables['aero_refract_core_imag'][:].data)
                optical_pop_dict['test']['shell_ris_real'] = np.array(currnc.variables['aero_refract_shell_real'][:].data)
                optical_pop_dict['test']['shell_ris_imag'] = np.array(currnc.variables['aero_refract_shell_imag'][:].data)
                optical_pop_dict['test']['core_vols'] = np.array(currnc.variables['aero_core_vol'][:].data)
                
                optical_pop_dict['test'] = arrays_to_lists(optical_pop_dict['test'])
                
            
            optical_pop_dict['population_settings'] = population_settings
            optical_pop_dict['species_modifications'] = species_modifications
            
            output_filename = get_output_filename(
                processed_output_dir, run_num, timestep, 
                repeat_num=repeat_num, runid_digits=runid_digits)
            
            output_filename = 'sample.json'
            with open(output_filename, "w") as outfile: 
                json.dump(optical_pop_dict, outfile)
        print('completed sample ii = ' + str(ii))
    
    return all_population_settings, all_species_modifications

def get_output_filename(processed_output_dir, run_num, timestep, repeat_num=1, runid_digits=4, file_format='json'):
    output_filename = ( 
        processed_output_dir + str(run_num).zfill(runid_digits) + '_' 
        + str(timestep).zfill(6) + '_' + str(repeat_num).zfill(2) 
        + '.' + file_format)
    
    return output_filename

def make_specs_dictionary(species,default_wvls=np.linspace(2e-7,7e-6,94)):
    aero_specs_dict = {}
    for one_spec in species:
        if type(one_spec.refractive_index.wvls) == type(None):
            wvls = default_wvls
        else:
            wvls = one_spec.refractive_index.wvls
        
        aero_specs_dict[one_spec.name] = asdict(one_spec)
        aero_specs_dict[one_spec.name]['refractive_index'] = asdict(one_spec.refractive_index)
        
        aero_specs_dict[one_spec.name]['refractive_index']['wvls'] = wvls
        if type(aero_specs_dict[one_spec.name]['refractive_index']['RI_params']) == type(None):
            aero_specs_dict[one_spec.name]['refractive_index']['real_ris'] = (
                aero_specs_dict[one_spec.name]['refractive_index']['real_ri_fun'](wvls))
            aero_specs_dict[one_spec.name]['refractive_index']['imag_ris'] = (
                aero_specs_dict[one_spec.name]['refractive_index']['imag_ri_fun'](wvls))
        else:
            RI_params = aero_specs_dict[one_spec.name]['refractive_index']['RI_params']
            n_550 = RI_params['n_550']
            alpha_n = RI_params['alpha_n']
            aero_specs_dict[one_spec.name]['refractive_index']['real_ris'] = (
                PyParticle.RI_fun(wvls, n_550, alpha_n))
            k_550 = RI_params['k_550']
            alpha_k = RI_params['alpha_k']
            aero_specs_dict[one_spec.name]['refractive_index']['imag_ris'] = (
                PyParticle.RI_fun(wvls, k_550, alpha_k))
        
        del aero_specs_dict[one_spec.name]['refractive_index']['real_ri_fun']
        del aero_specs_dict[one_spec.name]['refractive_index']['imag_ri_fun']
            
        aero_specs_dict[one_spec.name] = arrays_to_lists(aero_specs_dict[one_spec.name])
        aero_specs_dict[one_spec.name]['refractive_index'] = arrays_to_lists(
            aero_specs_dict[one_spec.name]['refractive_index'])
    
    return aero_specs_dict
    
def arrays_to_lists(dictionary):
    for one_key in dictionary.keys():
        if type(dictionary[one_key]) == type(np.zeros(2)):
            dictionary[one_key] = dictionary[one_key].tolist()
    return dictionary

def make_population_dictionary(optical_population):
    optical_pop_dict = {}
    
    optical_pop_dict['species'] = make_specs_dictionary(optical_population.species)
    optical_pop_dict['spec_masses'] = optical_population.spec_masses
    optical_pop_dict['num_concs'] = optical_population.num_concs
    optical_pop_dict['ids'] = optical_population.ids
    
    optical_pop_dict['rh_grid'] = optical_population.rh_grid
    optical_pop_dict['wvl_grid'] = optical_population.wvl_grid
    
    optical_pop_dict['temp'] = optical_population.temp
    
    optical_pop_dict['core_vols'] = optical_population.core_vols
    optical_pop_dict['shell_dry_vols'] = optical_population.shell_dry_vols
    optical_pop_dict['h2o_vols'] = optical_population.h2o_vols
    
    optical_pop_dict['core_ris'] = optical_population.core_ris
    optical_pop_dict['dry_shell_ris'] = optical_population.dry_shell_ris
    optical_pop_dict['h2o_ris'] = optical_population.h2o_ris
    
    optical_pop_dict['Cexts'] = optical_population.Cexts
    optical_pop_dict['Cscas'] = optical_population.Cscas
    optical_pop_dict['Cabss'] = optical_population.Cabss
    optical_pop_dict['gs'] = optical_population.gs
    optical_pop_dict['Cprs'] = optical_population.Cprs
    optical_pop_dict['Cbacks'] = optical_population.Cbacks
    
    optical_pop_dict['Cexts'] = optical_population.Cexts
    optical_pop_dict['Cscas'] = optical_population.Cscas
    optical_pop_dict['Cabss'] = optical_population.Cabss
    optical_pop_dict['gs'] = optical_population.gs
    optical_pop_dict['Cprs'] = optical_population.Cprs
    optical_pop_dict['Cbacks'] = optical_population.Cbacks
    
    optical_pop_dict['Cexts_bc'] = optical_population.Cexts_bc
    optical_pop_dict['Cscas_bc'] = optical_population.Cscas_bc
    optical_pop_dict['Cabss_bc'] = optical_population.Cabss_bc
    optical_pop_dict['gs_bc'] = optical_population.gs_bc
    optical_pop_dict['Cprs_bc'] = optical_population.Cprs_bc
    optical_pop_dict['Cbacks_bc'] = optical_population.Cbacks_bc

    optical_pop_dict['Cexts_nobc'] = optical_population.Cexts_nobc
    optical_pop_dict['Cscas_nobc'] = optical_population.Cscas_nobc
    optical_pop_dict['Cabss_nobc'] = optical_population.Cabss_nobc
    optical_pop_dict['gs_nobc'] = optical_population.gs_nobc
    optical_pop_dict['Cprs_nobc'] = optical_population.Cprs_nobc
    optical_pop_dict['Cbacks_nobc'] = optical_population.Cbacks_nobc
    
    return optical_pop_dict

    
def separate_ris(optical_pop_dict):
    optical_pop_dict['core_ris_real'] = np.real(optical_pop_dict['core_ris'])
    optical_pop_dict['core_ris_imag'] = np.imag(optical_pop_dict['core_ris'])
    
    optical_pop_dict['dry_shell_ris_real'] = np.real(optical_pop_dict['dry_shell_ris'])
    optical_pop_dict['dry_shell_ris_imag'] = np.imag(optical_pop_dict['dry_shell_ris'])
    
    optical_pop_dict['h2o_ris_real'] = np.real(optical_pop_dict['h2o_ris'])
    optical_pop_dict['h2o_ris_imag'] = np.imag(optical_pop_dict['h2o_ris'])
    
    del optical_pop_dict['core_ris']
    del optical_pop_dict['dry_shell_ris']
    del optical_pop_dict['h2o_ris']
    
    return optical_pop_dict
    
    
def sample_parameter_uniform(N_samples, param_range, param_scale='lin'):
    if param_scale == 'lin':
        param_samples = np.random.uniform(low=param_range[0],high=param_range[1],size=N_samples)
    elif param_scale == 'log':
        param_samples = 10.**np.random.uniform(low=np.log10(param_range[0]),high=np.log10(param_range[1]),size=N_samples)
    return param_samples

def sample_choices(N_samples, list_of_choices, choice_weights):
    samples = np.random.choice(list_of_choices,N_samples,replace=True,p=choice_weights)
    return samples
