#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a population from a PARTMC NetCDF file
@author: Laura Fierce
"""

from ..base import ParticlePopulation
from PyParticle import make_particle_from_masses
from PyParticle.species.registry import get_species

import numpy as np
import os
from pathlib import Path
from .registry import register
from ...constants import MOLAR_MASS_DRY_AIR, R, DENSITY_LIQUID_WATER
from ...utilities import power_moments_from_lognormal

try:
    import netCDF4
    _HAS_NETCDF4 = True
except ModuleNotFoundError:
    netCDF4 = None
    _HAS_NETCDF4 = False


if _HAS_NETCDF4:
    @register("mam4")
    def build(config):
        output_filename = Path(config['output_filename'])
        timestep = config['timestep']
        # fixme: use config.get() to set defatuls
        GSDs = config['GSD']
        # fixme: make this +/- certain number of sigmas (rather than min/max diams)
        D_min = config['D_min']
        D_max = config['D_max']
        N_bins = config['N_bins']
        species_modifications = config.get('species_modification',{}) # modify species properties during post-processing

        currnc = netCDF4.Dataset(output_filename)

        num_aer = currnc['num_aer']
        so4_aer = currnc['so4_aer']
        soa_aer = currnc['soa_aer']
        dgn_a = currnc['dgn_a']
        dgn_awet = currnc['dgn_awet']

        from .binned_lognormals import build as build_binned_lognormals

        p = config['p']
        T = config['T']

        lognormals_cfg = config

        # fixme: make this right
        rho_dry_air = MOLAR_MASS_DRY_AIR * p / (R * T)
        Ns = num_aer[:,timestep] * rho_dry_air # N/m^3
        idx, = np.where(Ns>0)
        Ns = Ns[idx]
        GMDs = dgn_a[idx,timestep]
        GMDs_wet = dgn_awet[idx,timestep]

        mass_so4 = so4_aer[idx,timestep] * rho_dry_air
        mass_soa = soa_aer[idx,timestep] * rho_dry_air
        # fixme: not actually sure what assumptions go into this for MAM4, assuming same GSD to compute mass H2O
        mass_h2o = DENSITY_LIQUID_WATER * np.pi/6 * np.array(
            [
                (power_moments_from_lognormal(3,N,gmd_wet,gsd) 
                 - power_moments_from_lognormal(3, N, gmd, gsd)) 
                for (N, gmd, gmd_wet, gsd) in zip(Ns, GMDs, GMDs_wet, GSDs)])

        # fixme: need to remove a mode if N==0
        lognormals_cfg['N'] = Ns
        lognormals_cfg['D_is_wet'] = True
        if lognormals_cfg['D_is_wet'] :
            lognormals_cfg['GMD'] = GMDs_wet #dgn_a[:,timestep]
        else:
            lognormals_cfg['GMD'] = GMDs #dgn_a[:,timestep]

    # debug: GMDs computed above
        # fixme: limited species for now
        lognormals_cfg['aero_spec_names'] = [['SO4','OC','H2O'],['SO4','OC','H2O'],['SO4','OC','H2O'],['SO4','OC','H2O']]
        # fixme: mam4 species not totally aligned
        #   lognormals_cfg['aero_spec_names'] = [['SO4','OC','MSA','BC','OIN','Na','H2O'],['SO4','SOA','NCL','H2O'],['SO4','OC','H2O'],['SO4','OC','H2O']]


        aero_spec_fracs = []
        for (m_so4,m_soa,m_h2o) in zip(mass_so4,mass_soa,mass_h2o):
            # m_tot = m_so4 + m_soa + m_h2o
            spec_frac = np.array([m_so4,m_soa,m_h2o])
            spec_frac /= np.sum(spec_frac)
            spec_frac[np.isnan(spec_frac)] = 0.
            aero_spec_fracs.append(spec_frac)

        lognormals_cfg['aero_spec_fracs'] = aero_spec_fracs # np.array([mass_so4/mass_tot,mass_soa/mass_tot,mass_h2o/mass_tot]).transpose()

        mam4_population = build_binned_lognormals(lognormals_cfg)
        return mam4_population

else:
    def build(config):
        raise ModuleNotFoundError(
            "The MAM4 population factory requires the 'netCDF4' package, "
            "which is not installed. Install it with:\n"
            "  conda install -c conda-forge netCDF4\n"
            "or\n"
            "  pip install netCDF4"
        )


def get_ncfile(partmc_output_dir, timestep, repeat):
    for root, dirs, files in os.walk(partmc_output_dir):
        f = files[0]
    if f.startswith('urban_plume_wc_'):
        preface_string = 'urban_plume_wc_'
    elif f.startswith('urban_plume_'):
        preface_string = 'urban_plume_'
    else:
        try:
            idx = partmc_output_dir[(partmc_output_dir.find('/')+1):].find('/')
            prefix_str = partmc_output_dir[(partmc_output_dir.find('/')+1):][:idx] + '_'
        except:
            try:
                preface_string, repeat2, timestep2 = f.split('_')
                preface_string += '_'
            except:
                preface_string = 'YOU_NEED_TO_CHANGE_preface_string_'
    ncfile = partmc_output_dir / (preface_string + str(int(repeat)).zfill(4) + '_' + str(int(timestep)).zfill(8) + '.nc')
    return ncfile

    # aero_spec_names = currnc.variables['aero_species'].names.split(',')
    # Get AerosolSpecies objects with modifications if any
    # species_list = tuple(
    #     get_species(name, **species_modifications.get(name, {}))
    #     for name in aero_spec_names
    # )
    
    # spec_masses = np.array(currnc.variables['aero_particle_mass'][:])
    # part_ids = np.array([one_id for one_id in currnc.variables['aero_id'][:]], dtype=int)

    # if 'aero_num_conc' in currnc.variables.keys():
    #     num_concs = currnc.variables['aero_num_conc'][:]
    # else:
    #     num_concs = 1. / currnc.variables['aero_comp_vol'][:]
    
    # if N_tot is None:
    #     N_tot = np.sum(num_concs)

    # if n_particles is None:
    #     idx = np.arange(len(part_ids))
    # elif n_particles <= len(part_ids):
    #     idx = np.random.choice(np.arange(len(part_ids)), size=n_particles, replace=False)
    # else:
    #     raise IndexError('n_particles > len(part_ids)')

    # partmc_population = ParticlePopulation(species=species_list, spec_masses=[], num_concs=[], ids=[])
    # for ii in idx:
    #     particle = make_particle_from_masses(
    #         aero_spec_names, 
    #         spec_masses[:, ii],
    #         species_modifications=species_modifications,
    #     )
    #     partmc_population.set_particle(
    #         particle, part_ids[ii], num_concs[ii] * N_tot / np.sum(num_concs[idx]), suppress_warning=suppress_warning
    #     )
    
    # if add_mixing_ratios:
    #     gas_mixing_ratios = np.array(currnc.variables['gas_mixing_ratio'][:])
    #     partmc_population.gas_mixing_ratios = gas_mixing_ratios
    # return currnc


def get_ncfile(partmc_output_dir, timestep, repeat):
    for root, dirs, files in os.walk(partmc_output_dir):
        f = files[0]
    if f.startswith('urban_plume_wc_'):
        preface_string = 'urban_plume_wc_'
    elif f.startswith('urban_plume_'):
        preface_string = 'urban_plume_'
    else:
        try:
            idx = partmc_output_dir[(partmc_output_dir.find('/')+1):].find('/')
            prefix_str = partmc_output_dir[(partmc_output_dir.find('/')+1):][:idx] + '_'
        except:
            try:
                preface_string, repeat2, timestep2 = f.split('_')
                preface_string += '_'
            except:
                preface_string = 'YOU_NEED_TO_CHANGE_preface_string_'
    ncfile = partmc_output_dir / (preface_string + str(int(repeat)).zfill(4) + '_' + str(int(timestep)).zfill(8) + '.nc')
    return ncfile