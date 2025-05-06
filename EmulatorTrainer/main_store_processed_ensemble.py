#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

import os
import numpy as np
from helper_store_ensemble import initialize_ensemble, build_testing_and_training_ensembles

build_new = True
rhs = 'all'#[0] #np.sort(1.-np.logspace(-2,0,5))
wvls = 'all'
# processed_output_dir = '/pic/projects/sooty2/fierce/processed_output/library_18_abs2/'
# ensemble_over_dir =  '/pic/projects/sooty2/fierce/processed_ensembles/library_18_abs2/'

processed_output_dir = '/qfs/projects/ambrs/PyParticle_populations/library_18_abs2/'
ensemble_over_dir = '/qfs/projects/ambrs/processed_ensembles/library_18_abs2/'

# processed_output_dir = '/Users/fier887/Downloads/processed_output/library_18_abs2/'
# ensemble_over_dir = '/Users/fier887/Downloads/processed_ensembles/library_18_abs2/'

# processed_output_dir = '/Users/fier887/Downloads/processed_output/library_18_abs2_small/'
# ensemble_over_dir = '/Users/fier887/Downloads/processed_ensembles/library_18_abs2_small/'
if not os.path.exists(ensemble_over_dir):
    os.mkdir(ensemble_over_dir)

ensemble_dir_ids = [onedir for onedir in os.listdir(ensemble_over_dir) if not onedir.startswith('.')]

if len(ensemble_dir_ids) == 0:
    ensemble_num = 0
    dir_padding = 6
    build_new = True
else:
    dir_padding = len(ensemble_dir_ids[0])
    ensemble_num = int(max(np.array([int(ensemble_dir_id) for ensemble_dir_id in ensemble_dir_ids])))

if build_new:
    ensemble_num += 1
    ensemble_num = int(ensemble_num)

ensemble_dir = ensemble_over_dir + str(ensemble_num).zfill(dir_padding) + '/'
if not os.path.exists(ensemble_dir):
    os.mkdir(ensemble_dir)

# varnames = [
#     'rh','wvl','Rbc','Rbc_dry','shell_tkappa_bc','tkappa_bc','tkappa_bcfree','tkappa',
#     'shell_real_ri_dry','shell_imag_ri_dry','shell_real_ri','shell_imag_ri',
#     'shell_real_ri_dry_bc','shell_imag_ri_dry_bc','shell_real_ri_bc','shell_imag_ri_bc',
#     'shell_real_ri_dry_bcfree','shell_imag_ri_dry_bcfree','shell_real_ri_bcfree','shell_imag_ri_bcfree',
#     'Eclear_dry','Eclear_dry_minus1','Edry_minus1','Eclear_over_dry', 'Eclear_over_dry_minus1','Edry','Eclear',
#     'MAC_bc','MAC_bcfree','MAC_bcfree_dry','Eabs_unclear_dry',
#     'Eabs_unclear','Eabs_unclear_wet_over_dry']

varnames = [
    'rh','wvl','Rbc','Rbc_dry','Rbc_vol','Rbc_dry_vol','Rbc_vol','Rbc_dry_vol',
    # fixme: add these
    'Rbc_dry_sigma', # BC-weighted standard deviation
    'corr_bcmass_Rbc_dry', # correlation between Rbc and BC mass
    'GMD_bc', # GMD of BC-containing particles
    'GSD_bc', # GSD of BC-containing particles
    'shell_tkappa_bc','tkappa_bc','tkappa_bcfree',
    'shell_real_ri_dry_bc','shell_imag_ri_dry_bc','shell_real_ri_bc','shell_imag_ri_bc',
    'shell_real_ri_dry_bcfree','shell_imag_ri_dry_bcfree','shell_real_ri_bcfree','shell_imag_ri_bcfree',
    'Eclear_dry','Edry','Eclear','Eabs_unclear',
    'MAC_bc','MAC_bcfree','MAC_bcfree_dry',
    # fixme: add these
    'Eclear_dry_uniform','Edry_uniform','Eclear_uniform','Eabs_unclear_uniform',
    ]

frac_testing=0.1
frac_validation=0.
sample_padding=5
runid_digits=4
timestep_digits=6
repeat_digits=2
file_format='json'

if build_new:
    initialize_ensemble(
            processed_output_dir, ensemble_dir,
            frac_testing=frac_testing, frac_validation=frac_validation,
            sample_padding=sample_padding,runid_digits=runid_digits, 
            timestep_digits=timestep_digits, repeat_digits=repeat_digits, file_format=file_format,
            varnames=varnames)

build_testing_and_training_ensembles(
        ensemble_dir, 
        varnames = varnames,
        morphology='core-shell',wvls=wvls,rhs=rhs)