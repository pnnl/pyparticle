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
import subprocess

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
            
            optical_pop_dict = PyParticle.make_population_dictionary(optical_population)
            optical_pop_dict = PyParticle.separate_ris(optical_pop_dict)
            optical_pop_dict = PyParticle.arrays_to_lists(optical_pop_dict)
            
            if test:
                optical_pop_dict['test'] = {}
                optical_pop_dict['test']['Cabss'] = np.array(currnc.variables['aero_absorb_cross_sect'][:].data)
                optical_pop_dict['test']['Cscas'] = np.array(currnc.variables['aero_scatter_cross_sect'][:].data)
                optical_pop_dict['test']['core_ris_real'] = np.array(currnc.variables['aero_refract_core_real'][:].data)
                optical_pop_dict['test']['core_ris_imag'] = np.array(currnc.variables['aero_refract_core_imag'][:].data)
                optical_pop_dict['test']['shell_ris_real'] = np.array(currnc.variables['aero_refract_shell_real'][:].data)
                optical_pop_dict['test']['shell_ris_imag'] = np.array(currnc.variables['aero_refract_shell_imag'][:].data)
                optical_pop_dict['test']['core_vols'] = np.array(currnc.variables['aero_core_vol'][:].data)
                
                optical_pop_dict['test'] = PyParticle.arrays_to_lists(optical_pop_dict['test'])
                
            
            optical_pop_dict['population_settings'] = population_settings
            optical_pop_dict['species_modifications'] = species_modifications
            
            output_filename = PyParticle.get_output_filename(
                processed_output_dir, run_num, timestep, 
                repeat_num=repeat_num, runid_digits=runid_digits)
            
            with open(output_filename, "w") as outfile: 
                json.dump(optical_pop_dict, outfile)
        print('completed sample ii = ' + str(ii))
    
    return all_population_settings, all_species_modifications

def run_ppe__splitToNodes(
        N_samples,
        ensemble_dir,
        processed_output_dir,
        run_file_dir,
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
    
    if test:
        print('Warning: test = True doesn\'t do anything')
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
    sample_padding = PyParticle.get_sample_padding(N_samples)
    sample_num = 0
    for ii,(run_num,timestep) in enumerate(zip(run_nums,timesteps)):
        
        partmc_dir = ensemble_dir + str(run_num).zfill(runid_digits) + '/' # partmc_dirs[ii]
        
        species_modifications = {}
        for spec in specs_to_vary:
            species_modifications[spec] = {}
            for param in spec_params[spec].keys():
                species_modifications[spec][param] = spec_params[spec][param][ii]
        
        all_species_modifications.append(species_modifications)
        
        for repeat_num in repeat_nums:
            sample_id_str = str(sample_num).zfill(sample_padding)
            
            population_settings = {
                'partmc_dir':partmc_dir, 
                'timestep':int(timestep), 'repeat':int(repeat_num), 'test':test}
            
            run_one(
                    sample_id_str, population_settings, species_modifications, 
                    rh_grid, wvl_grid, processed_output_dir, run_file_dir,
                    runid_digits=runid_digits)
            
            sample_num += 1
            
        print(
            'submitted sample' + sample_id_str + ', ii = ' + str(ii), 
            ', timestep = ' + str(timestep) + ', run_num = ' + str(run_num))
    
    return all_population_settings, all_species_modifications

def run_one(
        sample_id_str, population_settings, species_modifications, 
        rh_grid, wvl_grid, processed_output_dir, run_file_dir,
        runid_digits=4):
    
    
    partmc_dir = population_settings['partmc_dir']
    
    python_runfile_list= [
        'import PyParticle',
        'import json',
        'import numpy as np',
        'from importlib import reload',
        'reload(np)',
        'partmc_dir = ' + '\'' + partmc_dir + '\'',
        'run_num = ' + str(int(
            partmc_dir[(len(partmc_dir)-(runid_digits+1)):(len(partmc_dir)-1)])),
        'timestep = ' + str(population_settings['timestep']),
        'repeat_num = ' + str(population_settings['repeat']),
        'particle_population = PyParticle.builder.partmc.build(' + str(population_settings) + ',n_particles=None, species_modifications=' + str(species_modifications) + ')',
        'optical_population = PyParticle.make_optical_population(particle_population, ' + get_array_str(rh_grid) + ', '  + get_array_str(wvl_grid) + ', species_modifications=' + str(species_modifications) + ')',
        'optical_pop_dict = PyParticle.make_population_dictionary(optical_population)',
        'optical_pop_dict = PyParticle.separate_ris(optical_pop_dict)',
        'optical_pop_dict = PyParticle.arrays_to_lists(optical_pop_dict)',
        'optical_pop_dict[\'population_settings\'] = ' + str(population_settings),
        'optical_pop_dict[\'species_modifications\'] = ' + str(species_modifications),
        # 'output_filename =  \'sample.json\'', # 
        'output_filename = PyParticle.get_output_filename(\'' + processed_output_dir + '\', \'' + sample_id_str + '\', run_num, timestep, repeat_num=repeat_num, runid_digits=' + str(runid_digits) + ')', 
        'with open(output_filename, "w") as outfile:',
        '    json.dump(optical_pop_dict, outfile)'
        ]
    
    python_run_filename = run_file_dir + sample_id_str + '.py'
    with open(python_run_filename, "w") as file:
        for oneline in python_runfile_list:
            file.write(oneline + "\n")
    
    sh_runfile_list = [
        '#!/bin/tcsh',
        '#SBATCH -A sooty2',
        '#SBATCH -p shared',
        '#SBATCH -t 00:40:00',
        '#SBATCH -N 1',
        '#SBATCH -o ' + run_file_dir + sample_id_str + '.out',
        '#SBATCH -e ' + run_file_dir + sample_id_str + '.err',
        '#SBATCH -J ' + sample_id_str,
        '',
        'python ' + run_file_dir + sample_id_str + '.py'
        ]
    
    sh_runfilename = run_file_dir + sample_id_str + '.sh'
    with open(sh_runfilename, "w") as file:
        for oneline in sh_runfile_list:
            file.write(oneline + "\n")
    
    try:
        result = subprocess.run(
            ["sbatch", sh_runfilename],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        print(f"Job submitted successfully. Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while submitting the job:\n{e.stderr}")
    

def get_array_str(numpy_array):
    array_str = 'np.array(['
    for array_val in numpy_array:
        array_str += str(array_val) + ','
    array_str = array_str[:-1]
    array_str += '])'
    return array_str

def sample_parameter_uniform(N_samples, param_range, param_scale='lin'):
    if param_scale == 'lin':
        param_samples = np.random.uniform(low=param_range[0],high=param_range[1],size=N_samples)
    elif param_scale == 'log':
        param_samples = 10.**np.random.uniform(low=np.log10(param_range[0]),high=np.log10(param_range[1]),size=N_samples)
    return param_samples

def sample_choices(N_samples, list_of_choices, choice_weights):
    samples = np.random.choice(list_of_choices,N_samples,replace=True,p=choice_weights)
    return samples
