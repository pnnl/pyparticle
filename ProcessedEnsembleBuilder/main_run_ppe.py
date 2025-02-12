#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import numpy as np
import ppe

# processed_output_dir = '/Users/fier887/Downloads/processed_output/library_18_abs2/'
processed_output_dir = '/pic/projects/sooty2/fierce/processed_output/library_18_abs2/'

# ensemble_dir = '/Users/fier887/Downloads/box_simulations3/library_18_abs2/'
ensemble_dir = '/pic/projects/sooty2/fierce/partmc_output/library_18_abs2/'

all_run_nums = range(3,101)
run_weights = np.ones(len(all_run_nums))/len(all_run_nums)
all_timesteps = range(2,48)
timestep_weights = np.zeros(len(all_timesteps))
timestep_weights[:12] = np.ones(len(timestep_weights[:12]))
timestep_weights[12:] = 0.1*np.ones(len(timestep_weights[12:]))
timestep_weights = timestep_weights/np.sum(timestep_weights)

N_samples = 100

runid_digits=4
# wvl_grid=np.linspace(350e-9,950e-9,7)
# rh_grid = 1.-np.logspace(0,-2,13)
rh_grid = np.hstack([0.])
wvl_grid = np.hstack([550e-9])

specs_to_vary=['SOA','SO4','NH4','NO3']
n_550_range = [1.3,1.8]
n_550_scale = 'lin'
n_alpha_range = [0.,0.]
n_alpha_scale = 'lin'
k_550_range = [0,0.2]
k_550_scale = 'lin'
k_alpha_range = [0.,0.]
k_alpha_scale = 'lin'
kappa_range=[1e-3,1.2]
kappa_scale='log'
repeat_nums=[1]

ppe.run_ppe(
    N_samples,
    ensemble_dir,
    processed_output_dir,
    all_run_nums,
    all_timesteps,
    run_weights=run_weights,
    timestep_weights=timestep_weights,
    runid_digits=runid_digits,
    wvl_grid=wvl_grid,
    rh_grid=rh_grid,
    specs_to_vary=specs_to_vary,
    n_550_range=n_550_range,
    n_550_scale=n_550_scale,
    n_alpha_range=n_alpha_range,
    n_alpha_scale=n_alpha_scale,
    k_550_range=k_550_range,
    k_550_scale=k_550_scale,
    k_alpha_range=k_alpha_range,
    k_alpha_scale=k_alpha_scale,
    kappa_range=kappa_range,
    kappa_scale=kappa_scale,
    repeat_nums=repeat_nums,
    test=True)