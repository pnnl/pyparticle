#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

from helper_train_emulator import read_ensemble

ensemble_dir = '/Users/fier887/Downloads/processed_ensembles/library_18_abs2/000007/'
df_training, filenames, completed_filenames = read_ensemble(ensemble_dir,'training')
df_testing, filenames, completed_filenames = read_ensemble(ensemble_dir,'testing')

num_tune=1000
num_samples=3000
# train_Eabs_clear_dry_onewvl(df_training, num_tune=num_tune, num_samples=num_samples)