#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import PyParticle
import numpy as np
from importlib import reload
reload(np)
from reader import read_optical_population
import os

import distutils
from pymc3 import Model, glm, sample
from scipy.stats import norm

def train_Edry_onewvl(
        training_filenames, wvl, morphology='core-shell',
        num_tune=1000, num_samples=3000):
    num_spinup = int(num_tune*1.5)
    data = {'Rbc':[],'kap':[],'ri_real':[],'ri_imag':[],'Edry_minus1':[],'Edry_to_wet_minus1':[]}
    optical_population = read_optical_population(training_filenames[0],morphology=morphology)
    rh_grid = optical_population.rh_grid
    Edry_to_wet_minus1 = np.zeros([len(training_filenames),len(rh_grid)])
    for ii,output_filename in enumerate(training_filenames):
        optical_population = read_optical_population(output_filename,morphology=morphology)
        data['Rbc'].append(get_population_variable(optical_population, 'Rbc', wvl=wvl, rh=0.))
        data['kap'].append(get_population_variable(optical_population, 'shell_tkappa', wvl=wvl, rh=0.))
        data['ri_real'].append(get_population_variable(optical_population, 'real_shell_ri', wvl=wvl, rh=0.))
        data['ri_imag'].append(get_population_variable(optical_population, 'imag_shell_ri', wvl=wvl, rh=0.))
        Edry = get_population_variable(optical_population, 'Eabs_clear', wvl=wvl, rh=0.)
        data['Edry_minus1'].append(Edry-1.)
    
    for varname in data.keys():
        data[varname] = np.array(data[varname])
    
    data['Edry'] = np.array(data['Edry_minus1']) + 1.
    
# =============================================================================
#     Edry -- clear
# =============================================================================
    with Model() as model:
        glm.GLM.from_formula('Edry_minus1 ~ np.log(1 / (1 + Rbc)) + np.log(1 / (1 + Rbc)):ri_real + 0', data)
        trace = sample(num_samples,tune=num_tune,cores=1)
    
    model_params = {}
    for varname in trace.varnames:
        model_params[varname] = np.mean(trace[varname][num_spinup:])
        model_params[varname + '_sd'] = np.std(trace[varname][num_spinup:])
    model_params['sd'] = np.mean(trace['sd'][num_spinup:])
    
    Edry_fun = (
        lambda Rbc, ri_real: model_params['np.log(1 / (1 + Rbc))']*np.log(1/(1+Rbc)) 
        + model_params['np.log(1 / (1 + Rbc)):ri_real']*np.log(1/(1+Rbc))*ri_real + 1.)
    Edry_sd = model_params['sd']
    
    return Edry_fun, Edry_sd, model_params, data
    

    
def train_Eabs_models_onewvl(
        training_filenames, wvl, morphology='core-shell',
        num_tune=1000, num_samples=3000):
    num_spinup = int(num_tune*1.5)
    
# =============================================================================
#     load all data
# =============================================================================
    data = {'Rbc':[],'kap':[],'ri_real':[],'ri_imag':[],'Edry_minus1':[],'Edry_to_wet_minus1':[]}
    optical_population = read_optical_population(training_filenames[0],morphology=morphology)
    rh_grid = optical_population.rh_grid
    Edry_to_wet_minus1 = np.zeros([len(training_filenames),len(rh_grid)])
    for ii,output_filename in enumerate(training_filenames):
        optical_population = read_optical_population(output_filename,morphology=morphology)
        data['Rbc'].append(get_population_variable(optical_population, 'Rbc', wvl=wvl, rh=0.))
        data['kap'].append(get_population_variable(optical_population, 'shell_tkappa', wvl=wvl, rh=0.))
        data['ri_real'].append(get_population_variable(optical_population, 'real_shell_ri', wvl=wvl, rh=0.))
        data['ri_imag'].append(get_population_variable(optical_population, 'imag_shell_ri', wvl=wvl, rh=0.))
        Edry = get_population_variable(optical_population, 'Eabs_clear', wvl=wvl, rh=0.)
        data['Edry_minus1'].append(Edry-1.)
        
        for rr, rh in enumerate(rh_grid):
            Eabs_wet = get_population_variable(optical_population, 'Eabs_clear', wvl=wvl, rh=rh)
            Edry_to_wet_minus1[ii,rr] = Eabs_wet/Edry - 1.
            
    
    data['Edry_to_wet_minus1'] = Edry_to_wet_minus1
    data['rh_grid'] = rh_grid
    
    for varname in data.keys():
        data[varname] = np.array(data[varname])
    
    data['Edry'] = np.array(data['Edry_minus1']) + 1.
    data['Edry_to_wet'] = np.array(data['Edry_to_wet_minus1']) + 1.
    
    
# =============================================================================
#     Edry -- clear
# =============================================================================
    with Model() as model:
        glm.GLM.from_formula('Edry_minus1 ~ np.log(1 / (1 + Rbc)) + np.log(1 / (1 + Rbc)):ri_real + 0', data)
        trace = sample(num_samples,tune=num_tune,cores=1)
    
    model_params = {}
    for varname in trace.varnames:
        model_params[varname] = np.mean(trace[varname][num_spinup:])
        model_params[varname + '_sd'] = np.std(trace[varname][num_spinup:])
    model_params['sd'] = np.mean(trace['sd'][num_spinup:])
    
    Edry_fun = (
        lambda Rbc, ri_real: model_params['np.log(1 / (1 + Rbc))']*np.log(1/(1+Rbc)) 
        + model_params['np.log(1 / (1 + Rbc)):ri_real']*np.log(1/(1+Rbc))*ri_real + 1.)
    Edry_sd = model_params['sd']
    

# =============================================================================
#     Ewet/Edry -- clear
# =============================================================================
    model_params_allrh = []
    for rr,rh in enumerate(rh_grid):
        with Model() as model:
            glm.GLM.from_formula('Edry_to_wet_minus1 ~ np.log(1 / (1 + Rbc)) + np.log(1 / (1 + Rbc)):np.log(kap) + np.log(kap) + ri_real',data)
            trace = sample(num_samples,tune=num_tune,cores=1)
        
        model_params = {}
        for varname in trace.varnames:
            model_params[varname] = np.mean(trace[varname][num_spinup:])
            model_params[varname + '_sd'] = np.std(trace[varname][num_spinup:])
        model_params['sd'] = np.mean(trace['sd'][num_spinup:])
        
        Edry_to_wet_fun = (
            lambda Rbc, kap, ri_real: model_params['np.log(1 / (1 + Rbc))']*np.log(1/(1+Rbc)) 
            + model_params['np.log(1 / (1 + Rbc)):np.log(kap)']*np.log(1/(1+Rbc))*np.log(kap) 
            + model_params['kap']*np.log(kap) + model_params['ri_real']*ri_real + 1.)
        Edry_to_wet_sd = model_params['sd']
        model_params_allrh.append(model_params)

# =============================================================================
#     Ewet -- combine Edry and Ewet/Edry
# =============================================================================
    Ewet_fun = lambda Rbc, kap, ri_real: ( 
        Edry_fun(Rbc, ri_real)*Edry_to_wet_fun(Rbc, kap, ri_real))
    
    


# def train_Eabs_dry_to_wet_model_onewvl(
#         training_filenames, wvl, morphology='core-shell',
#         num_tune=1000, num_samples=3000):
#     num_spinup = int(num_tune*1.5)
    
#     data = {'Rbc':[],'ri_real':[], 'Edry_minus1':[]}
#     for output_filename in training_filenames:
#         optical_population = read_optical_population(output_filename,morphology=morphology)
#         data['Rbc'].append(get_population_variable(optical_population, 'Rbc', wvl=wvl, rh=0.))
#         data['ri_real'].append(get_population_variable(optical_population, 'real_shell_ri', wvl=wvl, rh=0.) - 1.)
#         # data['ri_imag'].append(get_population_variable(optical_population, 'imag_shell_ri', wvl=wvl, rh=0.) - 1.)
#         data['Edry_minus1'].append(get_population_variable(optical_population, 'Eabs_clear', wvl=wvl, rh=0.)-1.)
    
#     data['Rbc'] = np.array(data['Rbc'])
#     data['ri_real'] = np.array(data['ri_real'])
#     data['Edry_minus1'] = np.array(data['Edry_minus1'])
#     data['Edry'] = np.array(data['Edry_minus1']) + 1.
    
#     # model = Model('Edry ~ np.log(1/(1+Rbc)) + np.log(1/(1+Rbc)):ri_real + 0', data)
#     # results = model.fit(draws=1000)
    
#     with Model() as model:
#         glm.GLM.from_formula('Edry_to_wet ~ np.log(1/(Rbc+1)) + np.log(1/(Rbc+1)):np.log(kap) + np.log(kap) + real_ri',data)
#         trace = sample(num_samples,tune=num_tune,cores=1)
    
#     model_params = {}
#     for varname in trace.varnames:
#         model_params[varname] = np.mean(trace[varname][num_spinup:])
#         model_params[varname + '_sd'] = np.std(trace[varname][num_spinup:])
#     model_params['sd'] = np.mean(trace['sd'][num_spinup:])
    
#     Edry_fun_withRI = (
#         lambda Rbc, ri_real: model_params['np.log(1 / (1 + Rbc))']*np.log(1/(1+Rbc)) 
#         + model_params['np.log(1 / (1 + Rbc)):ri_real']*np.log(1/(1+Rbc))*ri_real + 1.)
    



    # norm.pdf(row['Edry'], loc=mean_prediction, scale=sigma)
    
    # m1_dry_withRI = np.mean(trace['np.log(1 / (1 + Rbc))'][2000:])
    # m2_dry_withRI = np.mean(trace['np.log(1 / (1 + Rbc)):ri_real'][2000:])
    # Edry_fun_withRI = lambda Rbc, ri_real: m1_dry_withRI*np.log(1/(1+Rbc)) + m2_dry_withRI*np.log(1/(1+Rbc))*(ri_real-1.) + 1.
    # Edry_sd_withRI = np.mean(trace['sd'][2000:])
    # return model_params

# import bambi as bmb
# import pandas as pd

# # Define the probabilistic model
# def train_Eabs_dry_model_onewvl(
#         training_filenames, wvl, morphology='core-shell',
#         num_tune=1000, num_samples=3000):

#     # Load data from training files
#     data = {'Rbc': [], 'ri_real': [], 'Edry': []}
#     for output_filename in training_filenames:
#         optical_population = read_optical_population(output_filename, morphology=morphology)
#         data['Rbc'].append(get_population_variable(optical_population, 'Rbc', wvl=wvl, rh=0.))
#         data['ri_real'].append(get_population_variable(optical_population, 'real_shell_ri', wvl=wvl, rh=0.) - 1.)
#         data['Edry'].append(get_population_variable(optical_population, 'Eabs_clear', wvl=wvl, rh=0.))

#     # Convert data to pandas DataFrame for modeling
#     df = pd.DataFrame({
#         'Rbc': np.array(data['Rbc']),
#         'ri_real': np.array(data['ri_real']),
#         'Edry': np.array(data['Edry'])
#     })

#     # Build the probabilistic model using Bambi
#     formula = 'Edry ~ np.log(1 / (1 + Rbc)) + np.log(1 / (1 + Rbc)):ri_real'
#     model = bmb.Model(formula, df)

#     # Fit the model
#     trace = model.fit(tune=num_tune, draws=num_samples, cores=1)
    
#     # Extract posterior parameter estimates
#     model_params = {}
#     for param in trace.posterior:
#         mean_value = trace.posterior[param].mean().item()
#         sd_value = trace.posterior[param].std().item()
#         model_params[param] = mean_value
#         model_params[f'{param}_sd'] = sd_value

#     # Add log10(sigma) metrics if present
#     if 'sigma' in trace.posterior:
#         model_params['logsd'] = np.log10(trace.posterior['sigma'].mean().item())
#         model_params['logsd_sd'] = np.log10(trace.posterior['sigma'].std().item())

#     return model_params, trace

# def predict_pdf(trace, df):
#     """
#     Predict the probability density function (PDF) for each Edry value using the posterior samples.
#     """
#     import arviz as az

#     # Generate predictions from the posterior predictive
#     posterior_predictive = az.from_pymc3(trace)
#     predicted_values = posterior_predictive.posterior_predictive['Edry_obs'].values

#     # Calculate the PDF for each Edry value
#     pdf_results = []
#     for i, observed_value in enumerate(df['Edry']):
#         pdf = np.mean(predicted_values[:, i] == observed_value)
#         pdf_results.append(pdf)

#     return pdf_results 

def get_population_variable(optical_population, varname, wvl=350e-9, rh=0.):
    if varname == 'rh' or varname == 'RH':
        idx_rh, idx_wvl = optical_population.get_grid_indices(rh,wvl)
        vardat = optical_population.rh_grid[idx_rh]
    elif varname == 'Rbc':
        idx_shell, = np.where(
            [one_idx not in [optical_population.idx_bc(),optical_population.idx_h2o()]
             for one_idx in np.arange(optical_population.spec_masses.shape[1])])
        mass_shell = np.sum(np.sum(optical_population.spec_masses[:,idx_shell],axis=1)*optical_population.num_concs)
        mass_bc = np.sum(optical_population.spec_masses[:,optical_population.idx_bc()]*optical_population.num_concs)
        vardat = mass_shell/mass_bc
    elif varname == 'Rbc_vol':
        vardat = optical_population.shell_dry_vols/optical_population.core_vols
    elif varname == 'imag_shell_ri':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='imag', rh=rh, wvl=wvl)
    elif varname == 'real_shell_ri':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='real', rh=rh, wvl=wvl)
    elif varname == 'shell_tkappa':
        if optical_population.shell_tkappas == None:
            optical_population._add_effective_kappas()
        vardat = (
            np.sum(optical_population.shell_tkappas*optical_population.shell_dry_vols*optical_population.num_concs)/
            np.sum(optical_population.shell_dry_vols*optical_population.num_concs))
    elif varname == 'tkappa':
        if optical_population.tkappas == None:
            optical_population._add_effective_kappas()
        vardat = (
            np.sum(optical_population.tkappas*(optical_population.shell_dry_vols+optical_population.core_vols)*optical_population.num_concs)/
            np.sum((optical_population.shell_dry_vols+optical_population.core_vols)*optical_population.num_concs))
    elif varname == 'Eabs':
        babs_total = optical_population.get_optical_coeff(optics_type='total_abs',rh=float(rh),wvl=float(wvl))
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl))
        vardat = babs_total/babs_pure_bc
    elif varname == 'Eabs_clear':
        babs_clear = optical_population.get_optical_coeff(optics_type='clear_abs',rh=float(rh),wvl=float(wvl))
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl))
        vardat = babs_clear/babs_pure_bc
    elif varname == 'bc_mass_conc':
        vardat = np.sum(optical_population.spec_masses[:,optical_population.idx_bc()]*optical_population.num_concs)
    elif varname == 'MAC_bc':
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl))
        mass_bc = np.sum(optical_population.spec_masses[:,optical_population.idx_bc()]*optical_population.num_concs)
        vardat = babs_pure_bc/mass_bc
        
    return vardat

def get_bcfree(optical_population,varname='MAC',rh=0.,wvl=550e-9):
    idx_rh, idx_wvl = optical_population.get_grid_indices(rh, wvl)
    idx_bcfree, = np.where(optical_population.spec_masses[:,optical_population.idx_bc()]==0.)
    
    if varname == 'mass_conc':
        rho_h2o = optical_population.get_particle(optical_population.ids[0]).get_rho_w()
        dry_masses = np.sum(optical_population.spec_masses[:,:-1],axis=1)
        masses = dry_masses + optical_population.h2o_vols[:,idx_rh]*rho_h2o
        vardat = np.sum(masses[idx_bcfree]*optical_population.num_concs[idx_bcfree])
    elif varname == 'b_abs':
        crossects = optical_population.Cabss[:,idx_rh,idx_wvl]
        vardat = np.sum(crossects[idx_bcfree]*optical_population.num_concs[idx_bcfree])
    elif varname == 'MAC':
        b_abs = optical_population.get_bcfree(varname='b_abs',rh=rh,wvl=wvl)
        mass_conc = optical_population.get_bcfree(varname='mass_conc',rh=rh,wvl=wvl)
        vardat = b_abs/mass_conc
    elif varname == 'imag_ri':
        vardat = np.imag( 
            optical_population.dry_shell_ris[idx_bcfree,idx_wvl]*optical_population.shell_dry_vols[idx_bcfree] + 
            optical_population.h2o_ri[idx_wvl]*optical_population.h2o_vols[idx_bcfree,idx_rh])/(
                optical_population.shell_dry_vols[idx_bcfree] + optical_population.h2o_vols[idx_bcfree,idx_rh])
    elif varname == 'real_ri':
        vardat = np.real( 
            optical_population.dry_shell_ris[idx_bcfree,idx_wvl]*optical_population.shell_dry_vols[idx_bcfree] + 
            optical_population.h2o_ri[idx_wvl]*optical_population.h2o_vols[idx_bcfree,idx_rh])/(
                optical_population.shell_dry_vols[idx_bcfree] + optical_population.h2o_vols[idx_bcfree,idx_rh])
    elif varname == 'kappa':
        vardat = (
            np.sum(optical_population.tkappas[idx_bcfree]*optical_population.shell_dry_vols[idx_bcfree]*optical_population.num_concs[idx_bcfree])/
            np.sum(optical_population.shell_dry_vols[idx_bcfree]*optical_population.num_concs[idx_bcfree]))
    
    return vardat
        
def split_testing_training(
        processed_output_dir, frac_testing=0.1, frac_validation=0.,
        runid_digits=4, timestep_digits=6, repeat_digits=2, file_format='json'):
    sample_ids, run_nums, timesteps, repeat_nums = unravel_files(processed_output_dir)
    N_samples = int(len(run_nums))
    unique_run_nums = np.unique(run_nums)
    
    testing_run_nums = np.random.choice(unique_run_nums, int(len(unique_run_nums)*frac_testing))
    
    unique_run_nums_leftover = np.array([run_num for run_num in unique_run_nums if run_num not in testing_run_nums])
    validation_run_nums = np.random.choice(unique_run_nums, int(len(unique_run_nums_leftover)*frac_validation))
    
    training_run_nums = np.array([run_num for run_num in unique_run_nums_leftover if run_num not in validation_run_nums])
    
    idx_testing, = np.where([run_num in testing_run_nums for run_num in run_nums])
    idx_validation, = np.where([run_num in validation_run_nums for run_num in run_nums])
    idx_training, = np.where([run_num in training_run_nums for run_num in run_nums])
    
    
    if len(idx_validation)>0 and frac_validation == 0.:
        print('error: validation set should be empty if frac_validation == 0')
    
    if not (np.sort(np.concatenate((idx_testing, idx_validation, idx_training))) == np.arange(0, N_samples)).all():
        print('error: testing/training runs not right')
    
    if not (np.sort(run_nums[idx_testing]) == np.sort(testing_run_nums)).all():
        print('error: testing runs not right')
    
    if not (np.sort(run_nums[idx_validation]) == np.sort(validation_run_nums)).all():
        print('error: validation runs not right')
    
    if not (np.sort(run_nums[idx_training]) == np.sort(training_run_nums)).all():
        print('error: training runs not right')
    
    sample_padding = PyParticle.get_sample_padding(N_samples)
    
    testing_filenames = []
    for ii in idx_testing:
        sample_id_str = str(sample_ids[ii]).zfill(sample_padding)
        testing_filenames.append(PyParticle.get_output_filename(
            processed_output_dir, sample_id_str, run_nums[ii], timesteps[ii], repeat_num=repeat_nums[ii], 
            runid_digits=runid_digits, timestep_digits=timestep_digits, repeat_digits=repeat_digits, file_format=file_format))
    
    validation_filenames = []
    for ii in idx_validation:
        sample_id_str = str(sample_ids[ii]).zfill(sample_padding)
        validation_filenames.append(PyParticle.get_output_filename(
            processed_output_dir, sample_id_str, run_nums[ii], timesteps[ii], 
            repeat_num=repeat_nums[ii], runid_digits=runid_digits, file_format=file_format))
    
    
    training_filenames = []
    for ii in idx_training:
        sample_id_str = str(sample_ids[ii]).zfill(sample_padding)
        training_filenames.append(PyParticle.get_output_filename(
            processed_output_dir, sample_id_str, run_nums[ii], timesteps[ii], 
            repeat_num=repeat_nums[ii], runid_digits=runid_digits, file_format=file_format))
    
    return training_filenames, testing_filenames, validation_filenames
    
def unravel_files(processed_output_dir, file_format='json'):
    all_files = [onefile[:-(len(file_format)+1)] for onefile in os.listdir(processed_output_dir) if onefile.endswith(file_format)]
    split_strings = [onefile.split('_') for onefile in all_files]
    
    sample_ids = np.zeros(len(split_strings),dtype=int)
    run_ids = np.zeros(len(split_strings),dtype=int)
    timesteps = np.zeros(len(split_strings),dtype=int)
    repeat_nums = np.zeros(len(split_strings),dtype=int)
    
    for ii,one_split in enumerate(split_strings):
        sample_ids[ii] = int(one_split[0])
        run_ids[ii] = int(one_split[1])
        timesteps[ii] = int(one_split[2])
        repeat_nums[ii] = int(one_split[3])
    
    return sample_ids, run_ids, timesteps, repeat_nums