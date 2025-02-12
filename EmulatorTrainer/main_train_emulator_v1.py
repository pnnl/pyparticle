#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:32:38 2024

@author: Laura Fierce
"""


from helper_train_emulator import split_testing_training
from untitled0 import train_Edry_onewvl

# processed_output_dir = '/pic/projects/sooty2/fierce/processed_output/library_18_abs2/'
processed_output_dir = '/Users/fier887/Downloads/processed_output/library_18_abs2/'
training_filenames, testing_filenames, validation_filenames = split_testing_training(
        processed_output_dir, frac_testing=0.1, frac_validation=0.,
        runid_digits=4, timestep_digits=6, repeat_digits=2, file_format='json')

wvl = 550e-9
Edry_fun, Edry_sd, model_params, data = train_Edry_onewvl(
        training_filenames, wvl, morphology='core-shell',
        num_tune=1000, num_samples=3000)

