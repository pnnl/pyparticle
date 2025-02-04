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
from pymc3 import Model, glm, sample
import bambi
from scipy.stats import norm
import pandas as pd

from helper_store_ensemble import read_ensemble


def train_models(ensemble_dir):
    df_training, filenames, completed_filenames = read_ensemble(ensemble_dir,'training')
    df_testing, filenames, completed_filenames = read_ensemble(ensemble_dir,'testing')
    
    
def train_Eabs_clear_dry_onewvl(df_training, num_tune=1000, num_samples=3000):
    num_spinup = int(num_tune*1.5)
    
    with Model() as model2:
        glm.GLM.from_formula('Eclear_dry_minus1 ~ np.log(1 / (1 + Rbc_dry)) + 0', df_training)
        trace2 = sample(num_samples,tune=num_tune,cores=1)
    
    model_params2 = {}
    for varname in trace2.varnames:
        model_params2[varname] = np.mean(trace2[varname][num_spinup:])
        model_params2[varname + '_sd'] = np.std(trace2[varname][num_spinup:])
    model_params2['sd'] = np.mean(trace2['sd'][num_spinup:])
    
    # Edry_fun2 = (
    #     lambda Rbc: model_params['np.log(1 / (1 + Rbc_dry))']*np.log(1/(1+Rbc)) + 1.)
    Edry_fun2 = (
        lambda Rbc: model_params2['np.log(1 / (1 + Rbc_dry))']*np.log(1/(1+Rbc)) + 1.)
    Edry_sd2 = model_params2['sd']
    
    # model2 = bambi.Model(
    #     'Eclear_dry_minus1 ~ np.log(1 / (1 + Rbc_dry)) + np.log(1 / (1 + Rbc_dry)):shell_real_ri_dry_bc + 0', df_training)
    # results = model2.fit(draws=num_tune)
    
    with Model() as model:
        glm.GLM.from_formula('Eclear_dry_minus1 ~ np.log(1 / (1 + Rbc_dry)) + np.log(1 / (1 + Rbc_dry)):shell_real_ri_dry_bc + 0', df_training)
        trace = sample(num_samples,tune=num_tune,cores=1)
    
    model_params = {}
    for varname in trace.varnames:
        model_params[varname] = np.mean(trace[varname][num_spinup:])
        model_params[varname + '_sd'] = np.std(trace[varname][num_spinup:])
    model_params['sd'] = np.mean(trace['sd'][num_spinup:])
    
    Edry_fun = (
        lambda Rbc, ri_real: model_params['np.log(1 / (1 + Rbc_dry))']*np.log(1/(1+Rbc)) 
        + model_params['np.log(1 / (1 + Rbc_dry)):shell_real_ri_dry_bc']*np.log(1/(1+Rbc))*ri_real + 1.)
    Edry_sd = model_params['sd']

    