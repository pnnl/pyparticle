#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 11:44:11 2024

@author: Laura Fierce
"""
import numpy as np
import pandas as pd
import os
import PyParticle
from reader import read_optical_population
 
def build_testing_and_training_ensembles(
        ensemble_dir, 
        # frac_testing=0.1, frac_validation=0.,
        # sample_padding=5,runid_digits=4, timestep_digits=6, repeat_digits=2, file_format='json',
        varnames = [
            'rh','wvl','Rbc','shell_tkappa','dry_shell_real_ri','dry_shell_imag_ri',
            'Edry','Eclear','Edry_minus1','Eclear_over_dry', 'Eclear_over_dry_minus1'],
        morphology='core-shell',wvls=[550e-9],rhs=[0]):
    
    for prefix in ['training','testing','validation']:
        build_ensemble_dataframe(
                ensemble_dir,prefix,
                varnames=varnames,morphology=morphology,wvls=wvls,rhs=rhs)        
        
        
def initialize_ensemble(
        processed_output_dir, ensemble_dir,
        frac_testing=0.1, frac_validation=0.,
        sample_padding=5, runid_digits=4, timestep_digits=6, repeat_digits=2, file_format='json',
        varnames = [
            'rh','wvl','Rbc','kap','dry_shell_real_ri','dry_shell_imag_ri',
            'Edry','Eclear','Edry_minus1','Eclear_over_dry', 'Eclear_over_dry_minus1']):
    
    output_filenames_sets = split_testing_training(
            processed_output_dir, frac_testing=frac_testing, frac_validation=frac_validation,
            sample_padding=sample_padding,runid_digits=runid_digits, 
            timestep_digits=timestep_digits, repeat_digits=repeat_digits, file_format=file_format)
    
    
    for (prefix,output_filenames) in zip(['training','testing','validation'],output_filenames_sets):
        # path_df, path_filenames, path_completed_filenames = get_ensemble_paths(ensemble_dir, prefix)
        df = initialize_dataframe(varnames)
        save_ensemble(ensemble_dir, prefix, df, output_filenames, [], file_format=file_format)
    
def read_ensemble(ensemble_dir, prefix, file_format='json'):
    path_df, path_filenames, path_completed_filenames = get_ensemble_paths(ensemble_dir, prefix)
    
    if file_format == 'json':
        df = pd.read_json(path_df)
    elif file_format == 'csv':
        df = pd.read_csv(path_df)
    else:
        print('file_format = \'' + file_format + '\' is invalid; must be json or csv')
    
    with open(path_filenames, "r") as file:
        filenames = file.read().splitlines()  # Removes newline characters
    
    with open(path_completed_filenames, "r") as file:
        completed_filenames = file.read().splitlines()  # Removes newline characters
    
    return df, filenames, completed_filenames
    
def save_ensemble(
        ensemble_dir, prefix,
        df, output_filenames, completed_filenames, file_format='json'):
    
    path_df, path_filenames, path_completed_filenames = get_ensemble_paths(ensemble_dir, prefix)
    
    if file_format == 'json':
        df.to_json(path_df)
    elif file_format == 'csv':
        df.to_csv(path_df)
    else:
        print('file_format = \'' + file_format + '\' is invalid; must be json or csv')
    
    with open(path_filenames,'w') as f:
        for oneline in output_filenames:
            f.write(oneline + '\n')
    
    with open(path_completed_filenames,'w') as f:
        for oneline in completed_filenames:
            f.write(oneline + '\n')
    
def get_ensemble_paths(ensemble_dir, prefix):
    path_df = ensemble_dir + prefix + '_df.txt'
    path_filenames = ensemble_dir + prefix + '_filenames.txt'
    path_completed_filenames = ensemble_dir + prefix + '_completed_filenames.txt'
    return path_df, path_filenames, path_completed_filenames

def build_ensemble_dataframe(
        ensemble_dir,prefix,
        varnames = [
            'rh','wvl','Rbc','kap','dry_shell_real_ri','dry_shell_imag_ri',
            'Edry','Eclear','Edry_minus1','Eclear_over_dry', 'Eclear_over_dry_minus1'],
        morphology='core-shell',wvls=[550e-9],rhs=[0], file_format='json'):
    
    # path_df, path_filenames, path_completed_filenames = get_ensemble_paths(ensemble_dir, prefix)
    # df_training = initialize_dataframe(varnames)
    # df_orig, output_filenames, completed_filenames = read_ensemble(ensemble_dir, prefix, file_format=file_format)
    # found_one = False
    # ii = 0
    # while not found_one:
    #     print(ii,output_filenames[ii])
    #     try:
    #         optical_population = read_optical_population(output_filenames[ii],morphology=morphology)
    #         found_one = True
    #     except:
    #         ii += 1
    df, output_filenames, completed_filenames = read_ensemble(ensemble_dir, prefix, file_format=file_format)
    
# num_incorrect = 0
# for ii,output_filename in enumerate(output_filenames):
#     try:
#         optical_population = read_optical_population(output_filename,morphology=morphology)
#     except:
#         num_incorrect += 1
#         print(output_filename)
# print(num_incorrect, '/', len(output_filenames))
    if len(output_filenames)>0:
        found_one = False
        ii = 0
        while not found_one:
            print(ii,output_filenames[ii])
            try:
                optical_population = read_optical_population(output_filenames[ii],morphology=morphology)
                found_one = True
            except:
                ii += 1
        # optical_population = read_optical_population(output_filenames[0],morphology=morphology)
        
        if rhs == 'all':
            rhs = optical_population.rh_grid
        if wvls == 'all':
            wvls = optical_population.wvl_grid
        
        
        # Edry_to_wet_minus1 = np.zeros([len(output_filenames),len(rh_grid)])
        for ii,output_filename in enumerate(output_filenames):
            
            if output_filename not in completed_filenames:
                # df_orig, output_filenames, completed_filenames = read_ensemble(ensemble_dir, prefix)
                
                df = append_ensemble_dataframe(
                    df,output_filename,wvls,rhs,varnames,morphology=morphology)
                
                completed_filenames.append(output_filename)
                save_ensemble(ensemble_dir, prefix, df, output_filenames, completed_filenames)
                
                print(ii + 1,'/',len(output_filenames))
    # return df, completed_output_filenames

def initialize_dataframe(varnames):
    data = {}
    for varname in varnames:
        data[varname] = []
    df = pd.DataFrame(data)
    return df
    
def append_ensemble_dataframe(
        df_orig,output_filename,wvls,rhs,varnames,morphology='core-shell'):
    try:
        data = dataframe_to_dict(df_orig)
        optical_population = read_optical_population(output_filename,morphology=morphology)
        for rh in rhs:
            for wvl in wvls:
                for varname in varnames:
                    vals_list = list(data[varname])
                    vals_list.append(get_population_variable(optical_population, varname, wvl=wvl, rh=rh))
                    data[varname] = np.array(vals_list)
        df = pd.DataFrame(data)
    except:
        df = df_orig
        print(output_filename,'didn\'t work')
    
    return df


def dataframe_to_dict(df):
    data_dict = {}
    for varname in df.keys():
        data_dict[varname] = df[varname].values
    return data_dict


def get_population_variable(optical_population, varname, wvl=350e-9, rh=0.):
    idx_rh, idx_wvl = optical_population.get_grid_indices(rh,wvl)
    if varname == 'rh' or varname == 'RH':
        # vardat = optical_population.rh_grid[idx_rh]
        vardat = rh
    elif varname == 'wvl':
        vardat = wvl
    elif varname == 'Rbc':
        idx_bc = optical_population.idx_bc()
        idx_bc_containing, = np.where(optical_population.spec_masses[:,idx_bc]>0)
        idx_shell, = np.where(
            [one_idx not in [idx_bc]
             for one_idx in np.arange(optical_population.spec_masses.shape[1])])
        masses_shell = np.sum(np.vstack([optical_population.spec_masses[ii,idx_shell] for ii in idx_bc_containing]),axis=1)
        mass_shell = np.sum(masses_shell*optical_population.num_concs[idx_bc_containing])
        mass_bc = np.sum(optical_population.spec_masses[idx_bc_containing,optical_population.idx_bc()]*optical_population.num_concs[idx_bc_containing])
        vardat = mass_shell/mass_bc
    elif varname == 'Rbc_vol':
        idx_bc = optical_population.idx_bc()
        idx_bc_containing, = np.where(optical_population.spec_masses[:,idx_bc]>0)
        vol_shell = optical_population.num_concs[idx_bc_containing]*(optical_population.shell_dry_vols[idx_bc_containing]+optical_population.h2o_vols[idx_bc_containing,idx_rh])
        vol_core = optical_population.num_concs[idx_bc_containing]*optical_population.core_vols[idx_bc_containing]
        
        vardat = vol_shell/vol_core
    elif varname == 'Rbc_dry':
        idx_bc = optical_population.idx_bc()
        idx_bc_containing, = np.where(optical_population.spec_masses[:,idx_bc]>0)
        
        idx_shell, = np.where(
            [one_idx not in [optical_population.idx_bc(),optical_population.idx_h2o()]
             for one_idx in np.arange(optical_population.spec_masses.shape[1])])
        masses_shell = np.sum(np.vstack([optical_population.spec_masses[ii,idx_shell] for ii in idx_bc_containing]),axis=1)
        mass_shell = np.sum(masses_shell*optical_population.num_concs[idx_bc_containing])
        mass_bc = np.sum(optical_population.spec_masses[idx_bc_containing,optical_population.idx_bc()]*optical_population.num_concs[idx_bc_containing])
        vardat = mass_shell/mass_bc
    elif varname == 'Rbc_vol_dry':
        vardat = optical_population.shell_dry_vols/optical_population.core_vols
        
    elif varname == 'Rbc_dry_std':
        idx_bc = optical_population.idx_bc()
        idx_bc_containing, = np.where(optical_population.spec_masses[:,idx_bc]>0)
        vol_shell = optical_population.num_concs[idx_bc_containing]*(optical_population.shell_dry_vols[idx_bc_containing]+optical_population.h2o_vols[idx_bc_containing,idx_rh])
        vol_core = optical_population.num_concs[idx_bc_containing]*optical_population.core_vols[idx_bc_containing]
        
        vardat = vol_shell/vol_core
        
    elif varname == 'shell_imag_ri':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='imag', rh=rh, wvl=wvl, bconly=False)
    elif varname == 'shell_real_ri':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='real', rh=rh, wvl=wvl, bconly=False)
    elif varname == 'shell_imag_ri_dry':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='imag', rh=0., wvl=wvl, bconly=False)
    elif varname == 'shell_real_ri_dry':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='real', rh=0., wvl=wvl, bconly=False)
    elif varname == 'shell_imag_ri_bc':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='imag', rh=rh, wvl=wvl, bconly=True)
    elif varname == 'shell_real_ri_bc':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='real', rh=rh, wvl=wvl, bconly=True)
    elif varname == 'shell_imag_ri_dry_bc':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='imag', rh=0., wvl=wvl, bconly=True)
    elif varname == 'shell_real_ri_dry_bc':
        vardat = optical_population.get_average_ri(
            morph_component='shell', ri_component='real', rh=0., wvl=wvl, bconly=True)
    elif varname == 'shell_imag_ri_bcfree':
        vardat = get_bcfree(optical_population,varname='imag_ri',rh=rh,wvl=wvl)
    elif varname == 'shell_real_ri_bcfree':
        vardat = get_bcfree(optical_population,varname='real_ri',rh=rh,wvl=wvl)
    elif varname == 'shell_imag_ri_dry_bcfree':
        vardat = get_bcfree(optical_population,varname='imag_ri',rh=0.,wvl=wvl)
    elif varname == 'shell_real_ri_dry_bcfree':
        vardat = get_bcfree(optical_population,varname='real_ri',rh=0.,wvl=wvl)
    elif varname == 'shell_tkappa':
        if not isinstance(optical_population.shell_tkappas, np.ndarray):
            optical_population._add_effective_kappas()
        vardat = (
            np.sum(optical_population.shell_tkappas*optical_population.shell_dry_vols*optical_population.num_concs)/
            np.sum(optical_population.shell_dry_vols*optical_population.num_concs))
    elif varname == 'tkappa':
        if not isinstance(optical_population.tkappas, np.ndarray):
            optical_population._add_effective_kappas()
        vardat = (
            np.sum(optical_population.tkappas*(optical_population.shell_dry_vols+optical_population.core_vols)*optical_population.num_concs)/
            np.sum((optical_population.shell_dry_vols+optical_population.core_vols)*optical_population.num_concs))
    elif varname == 'shell_tkappa':
        if not isinstance(optical_population.shell_tkappas, np.ndarray):
            optical_population._add_effective_kappas()
        vardat = (
            np.sum(optical_population.shell_tkappas*optical_population.shell_dry_vols*optical_population.num_concs)/
            np.sum(optical_population.shell_dry_vols*optical_population.num_concs))
    elif varname == 'tkappa_bc':
        if not isinstance(optical_population.tkappas, np.ndarray):
            optical_population._add_effective_kappas()
        
        idx_bc = optical_population.idx_bc()
        idx, = np.where(optical_population.spec_masses[:,idx_bc]>0)
        
        vardat = (
            np.sum(optical_population.tkappas[idx]*(optical_population.shell_dry_vols[idx]+optical_population.core_vols[idx])*optical_population.num_concs[idx])/
            np.sum((optical_population.shell_dry_vols[idx]+optical_population.core_vols[idx])*optical_population.num_concs[idx]))
    elif varname == 'shell_tkappa_bc':
        if not isinstance(optical_population.shell_tkappas, np.ndarray):
            optical_population._add_effective_kappas()
        
        idx_bc = optical_population.idx_bc()
        idx, = np.where(optical_population.spec_masses[:,idx_bc]>0)
                
        vardat = (
            np.sum(optical_population.shell_tkappas[idx]*optical_population.shell_dry_vols[idx]*optical_population.num_concs[idx])/
            np.sum(optical_population.shell_dry_vols[idx]*optical_population.num_concs[idx]))
    elif varname == 'tkappa_bcfree':
        if not isinstance(optical_population.tkappas, np.ndarray):
            optical_population._add_effective_kappas()
        
        idx_bc = optical_population.idx_bc()
        idx, = np.where(optical_population.spec_masses[:,idx_bc]==0.)
        
        vardat = (
            np.sum(optical_population.tkappas[idx]*(optical_population.shell_dry_vols[idx]+optical_population.core_vols[idx])*optical_population.num_concs[idx])/
            np.sum((optical_population.shell_dry_vols[idx]+optical_population.core_vols[idx])*optical_population.num_concs[idx]))
    elif varname == 'shell_tkappa_bcfree':
        if not isinstance(optical_population.shell_tkappas, np.ndarray):
            optical_population._add_effective_kappas()
            
        idx_bc = optical_population.idx_bc()
        idx, = np.where(optical_population.spec_masses[:,idx_bc]==0.)
        vardat = (
            np.sum(optical_population.shell_tkappas[idx]*optical_population.shell_dry_vols[idx]*optical_population.num_concs[idx])/
            np.sum(optical_population.shell_dry_vols[idx]*optical_population.num_concs[idx]))
    elif varname == 'Eabs':
        babs_total = optical_population.get_optical_coeff(optics_type='total_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        vardat = babs_total/babs_pure_bc
    elif varname == 'Eclear':
        babs_total = optical_population.get_optical_coeff(optics_type='clear_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        vardat = babs_total/babs_pure_bc
    elif varname == 'Eclear_dry':
        babs_total = optical_population.get_optical_coeff(optics_type='clear_abs',rh=0.,wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=0.,wvl=float(wvl),bconly=True)
        vardat = babs_total/babs_pure_bc
    elif varname == 'Eclear_dry_minus1':
        babs_total = optical_population.get_optical_coeff(optics_type='clear_abs',rh=0.,wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=0.,wvl=float(wvl),bconly=True)
        vardat = babs_total/babs_pure_bc - 1.
    elif varname == 'Edry':
        babs_total = optical_population.get_optical_coeff(optics_type='total_abs',rh=0.,wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=0.,wvl=float(wvl),bconly=True)
        vardat = babs_total/babs_pure_bc
    elif varname == 'Edry_minus1':
        babs_total = optical_population.get_optical_coeff(optics_type='total_abs',rh=0.,wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=0.,wvl=float(wvl),bconly=True)
        vardat = babs_total/babs_pure_bc - 1.
    elif varname == 'Eabs_clear':
        babs_clear = optical_population.get_optical_coeff(optics_type='clear_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        vardat = babs_clear/babs_pure_bc
    elif varname == 'Eclear_over_dry':
        babs_total = optical_population.get_optical_coeff(optics_type='clear_abs',rh=0.,wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=0.,wvl=float(wvl),bconly=True)
        Edry = babs_total/babs_pure_bc
        
        babs_clear = optical_population.get_optical_coeff(optics_type='clear_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        Eclear = babs_clear/babs_pure_bc
        
        vardat = Eclear/Edry
    elif varname == 'Eclear_over_dry_minus1':
        babs_total = optical_population.get_optical_coeff(optics_type='clear_abs',rh=0.,wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=0.,wvl=float(wvl),bconly=True)
        Edry = babs_total/babs_pure_bc
        
        babs_clear = optical_population.get_optical_coeff(optics_type='clear_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        Eclear = babs_clear/babs_pure_bc
        
        vardat = Eclear/Edry - 1.
    elif varname == 'bc_mass_conc':
        vardat = np.sum(optical_population.spec_masses[:,optical_population.idx_bc()]*optical_population.num_concs)
    elif varname == 'MAC_bc':
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        mass_bc = np.sum(optical_population.spec_masses[:,optical_population.idx_bc()]*optical_population.num_concs[optical_population.idx_bc()])
        vardat = babs_pure_bc/mass_bc
    elif varname == 'MAC_bcfree':
        vardat = get_bcfree(optical_population,varname='MAC',rh=rh,wvl=wvl)
    elif varname == 'MAC_bcfree_dry':
        vardat = get_bcfree(optical_population,varname='MAC',rh=0.,wvl=wvl)
    elif varname == 'Eabs_unclear_dry':
        babs_unclear = optical_population.get_optical_coeff(optics_type='total_abs',rh=0.,wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=0.,wvl=float(wvl),bconly=True)
        vardat = babs_unclear/babs_pure_bc
    elif varname == 'Eabs_unclear':
        babs_unclear = optical_population.get_optical_coeff(optics_type='total_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        babs_pure_bc = optical_population.get_optical_coeff(optics_type='pure_bc_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        vardat = babs_unclear/babs_pure_bc
    elif varname == 'Eabs_unclear_wet_over_dry':
        babs_unclear_wet = optical_population.get_optical_coeff(optics_type='total_abs',rh=float(rh),wvl=float(wvl),bconly=True)
        babs_unclear_dry = optical_population.get_optical_coeff(optics_type='total_abs',rh=0.,wvl=float(wvl),bconly=True)
        vardat = babs_unclear_wet/babs_unclear_dry
    
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
        crossects = optical_population.Cabss[:,idx_rh,idx_wvl]
        b_abs = np.sum(crossects[idx_bcfree]*optical_population.num_concs[idx_bcfree])
        
        rho_h2o = optical_population.get_particle(optical_population.ids[0]).get_rho_w()
        dry_masses = np.sum(optical_population.spec_masses[:,:-1],axis=1)
        masses = dry_masses + optical_population.h2o_vols[:,idx_rh]*rho_h2o
        
        mass_conc = np.sum(masses[idx_bcfree]*optical_population.num_concs[idx_bcfree])
        vardat = b_abs/mass_conc
    elif varname == 'imag_ri':
        vardat = np.sum(np.imag( 
            optical_population.dry_shell_ris[idx_bcfree,idx_wvl]*optical_population.shell_dry_vols[idx_bcfree] + 
            optical_population.h2o_ris[idx_bcfree,idx_wvl]*optical_population.h2o_vols[idx_bcfree,idx_rh]))/np.sum(
                optical_population.shell_dry_vols[idx_bcfree] + optical_population.h2o_vols[idx_bcfree,idx_rh])
    elif varname == 'real_ri':
        vardat = np.sum(np.real( 
            optical_population.dry_shell_ris[idx_bcfree,idx_wvl]*optical_population.shell_dry_vols[idx_bcfree] + 
            optical_population.h2o_ris[idx_bcfree,idx_wvl]*optical_population.h2o_vols[idx_bcfree,idx_rh]))/np.sum(
                optical_population.shell_dry_vols[idx_bcfree] + optical_population.h2o_vols[idx_bcfree,idx_rh])
        # vardat = np.real( 
        #     optical_population.dry_shell_ris[idx_bcfree,idx_wvl]*optical_population.shell_dry_vols[idx_bcfree] + 
        #     optical_population.h2o_ris[idx_bcfree,idx_wvl]*optical_population.h2o_vols[idx_bcfree,idx_rh])/(
        #         optical_population.shell_dry_vols[idx_bcfree] + optical_population.h2o_vols[idx_bcfree,idx_rh])
    elif varname == 'kappa' or varname == 'kap':
        vardat = (
            np.sum(optical_population.tkappas[idx_bcfree]*optical_population.shell_dry_vols[idx_bcfree]*optical_population.num_concs[idx_bcfree])/
            np.sum(optical_population.shell_dry_vols[idx_bcfree]*optical_population.num_concs[idx_bcfree]))
    
    return vardat

def split_testing_training(
        processed_output_dir, frac_testing=0.1, frac_validation=0.,
        sample_padding=5,runid_digits=4, timestep_digits=6, repeat_digits=2, file_format='json'):
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
    
    
    if not (np.array([one_run in np.sort(run_nums[idx_testing]) for one_run in np.sort(testing_run_nums)]).all() and 
        np.array([one_run in np.sort(testing_run_nums) for one_run in np.sort(run_nums[idx_testing])]).all()):
        print('error: testing runs not right')
    
    if not (np.array([one_run in np.sort(run_nums[idx_validation]) for one_run in np.sort(validation_run_nums)]).all() and 
        np.array([one_run in np.sort(validation_run_nums) for one_run in np.sort(run_nums[idx_validation])]).all()):
    # not (np.sort(run_nums[idx_validation]) == np.sort(validation_run_nums)).all():
        print('error: validation runs not right')
    
    # if not (np.sort(run_nums[idx_training]) == np.sort(training_run_nums)).all():
    if not (np.array([one_run in np.sort(run_nums[idx_training]) for one_run in np.sort(training_run_nums)]).all() and 
        np.array([one_run in np.sort(training_run_nums) for one_run in np.sort(run_nums[idx_training])]).all()):
        print('error: training runs not right')
    
    # sample_padding = PyParticle.get_sample_padding(N_samples)
    
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
