#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import PyParticle
import numpy as np
from reader import read_optical_population
import os

import distutils
from pymc import Model, sample
import bambi
from scipy.stats import norm
import pandas as pd

from helper_store_ensemble import read_ensemble


def train_models(ensemble_dir):
    df_training, filenames, completed_filenames = read_ensemble(ensemble_dir,'training')
    df_testing, filenames, completed_filenames = read_ensemble(ensemble_dir,'testing')
    
    
def train_Eabs_clear_dry_onewvl(df_training_allwvl, num_tune=1000, num_samples=3000,wvl=550e-9):
    num_spinup = int(num_tune*1.5)
    
    # with Model() as model2:
    #     glm.GLM.from_formula('Eclear_dry_minus1 ~ np.log(1 / (1 + Rbc_dry)) + 0', df_training)
    #     trace2 = sample(num_samples,tune=num_tune,cores=1)
    # with Model() as model2:
        
    
    df_training_allwvl['Eclear_dry_minus1'] = df_training_allwvl['Eclear_dry'] - 1.
    
    idx_wvl,=np.where(df_training_allwvl['wvl']==550e-9)
    df_training = df_training_allwvl.iloc[idx_wvl]
    
    # model2 = bambi.Model('Eclear_dry_minus1 ~ np.log(1 / (1 + Rbc_dry)) + 0', data=df_training)
    # trace2 = model2.fit(draws=num_samples, tune=num_tune)
    # model_params2 = {}
    
    # # trace2 = sample(num_samples,tune=num_tune,cores=1)
    
    # model_params2 = {}
    # for varname in model2.components['mu'].terms.keys():
    #     model_params2[varname] = np.mean(trace2.posterior[varname][:,num_spinup:].values.ravel())
    #     model_params2[varname + '_sd'] = np.std(trace2.posterior[varname][:,num_spinup:].values.ravel())
    # model_params2['sigma'] = np.mean(trace2.posterior['sigma'][:,num_spinup:].values.ravel())
    
    # # Edry_fun2 = (
    # #     lambda Rbc: model_params['np.log(1 / (1 + Rbc_dry))']*np.log(1/(1+Rbc)) + 1.)
    # Edry_fun2 = (
    #     lambda Rbc: model_params2['np.log(1 / 1 + Rbc_dry)']*np.log(1/(1+Rbc)) + 1.)
    # Edry_sd2 = model_params2['sigma']
    
    
    # model3 = bambi.Model('Eclear_dry_minus1 ~ np.log(1 + Rbc_dry) + 0', data=df_training)
    # trace3 = model3.fit(draws=num_samples, tune=num_tune)
    # model_params3 = {}
    
    # # trace2 = sample(num_samples,tune=num_tune,cores=1)
    
    # model_params3 = {}
    # for varname in model3.components['mu'].terms.keys():
    #     model_params3[varname] = np.mean(trace3.posterior[varname][:,num_spinup:].values.ravel())
    #     model_params3[varname + '_sd'] = np.std(trace3.posterior[varname][:,num_spinup:].values.ravel())
    # model_params3['sigma'] = np.mean(trace3.posterior['sigma'][:,num_spinup:].values.ravel())
    
    # # Edry_fun2 = (
    # #     lambda Rbc: model_params['np.log(1 / (1 + Rbc_dry))']*np.log(1/(1+Rbc)) + 1.)
    # Edry_fun3 = (
    #     lambda Rbc: model_params3['np.log(1 + Rbc_dry)']*np.log((1+Rbc)) + 1.)
    # Edry_sd3 = model_params3['sigma']
    
    
    # model2 = bambi.Model(
    #     'Eclear_dry_minus1 ~ np.log(1 / (1 + Rbc_dry)) + np.log(1 / (1 + Rbc_dry)):shell_real_ri_dry_bc + 0', df_training)
    # results = model2.fit(draws=num_tune)
    
    # with Model() as model:
    #     glm.GLM.from_formula('Eclear_dry_minus1 ~ np.log(1 / (1 + Rbc_dry)) + np.log(1 / (1 + Rbc_dry)):shell_real_ri_dry_bc + 0', df_training)
    #     trace = sample(num_samples,tune=num_tune,cores=1)
    model = bambi.Model('Eclear_dry_minus1 ~ ' + 
                        # 'np.log(1 + Rbc_dry) ' + 
                        '+ np.log(1 + Rbc_dry):shell_real_ri_dry_bc ' + 
                        '+ shell_real_ri_dry_bc ' + 
                        '+ 0', data=df_training)
    trace = model.fit(draws=num_samples, tune=num_tune,cores=1)
    # trace2 = model2.fit(draws=num_samples, tune=num_tune)
    
    model_params = {}
    for varname in model.components['mu'].terms.keys():
        model_params[varname] = np.mean(trace.posterior[varname][:,num_spinup:].values.ravel())
        model_params[varname + '_sd'] = np.std(trace.posterior[varname][:,num_spinup:].values.ravel())
    model_params['sigma'] = np.mean(trace.posterior['sigma'][:,num_spinup:].values.ravel())
    
    
    # model_params = {}
    # for varname in trace.varnames:
    #     model_params[varname] = np.mean(trace[varname][num_spinup:])
    #     model_params[varname + '_sd'] = np.std(trace[varname][num_spinup:])
    # model_params['sd'] = np.mean(trace['sd'][num_spinup:])
    
    Edry_fun = (
        lambda Rbc, ri_real: 
            # model_params['np.log(1 + Rbc_dry)']*np.log(1+Rbc) 
            + model_params['np.log(1 + Rbc_dry):shell_real_ri_dry_bc']*np.log(1+Rbc)*ri_real 
            + model_params['shell_real_ri_dry_bc']*ri_real 
            + 1.)
    Edry_sd = model_params['sigma']

    