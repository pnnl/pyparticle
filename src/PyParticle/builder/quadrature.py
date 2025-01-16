#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Functions to create a monodisperse population

@author: Laura Fierce
"""
from . import retrieve_one_species
from . import ParticlePopulation
from . import make_particle
from . import data_path
import numpy as np
from scipy.optimize import fsolve


def build(
        population_settings, specdata_path = data_path / 'species_data'):
    
    if specdata_path == None:
        specdata_path = data_path / 'species_data'
    log10gmds = np.log10(population_settings['GMDs_dry'])
    log10gsds = np.log10(population_settings['GSDs_dry'])
    Ns = population_settings['Ns']
    
    aero_spec_names = population_settings['aero_spec_names']        
    aero_spec_fracs = population_settings['aero_spec_fracs']
    N_quads = population_settings['N_quads']
        
    drydias_q, nums_q, modes_q = construct_quadrature(Ns,log10gmds,log10gsds,N_quads)
    aero_species = []
    for spec_name in aero_spec_names:
        aero_species.append(retrieve_one_species(spec_name, specdata_path=specdata_path))
    
    spec_fracs_q = np.zeros([0,len(aero_species)])
    
    ids = []
    for mode_num in modes_q:
        print(aero_spec_fracs[mode_num,:].shape,spec_fracs_q.shape)
        spec_fracs_q = np.vstack([spec_fracs_q,aero_spec_fracs[mode_num,:]])
        for ii in range(N_quads[mode_num]):
            ids.append((mode_num+1)*10 + ii+1)
    
    quadrature_population = ParticlePopulation(species=aero_species,spec_masses=[],num_concs=[],ids=[])
    for qq in range(len(drydias_q)):
        print(ids[qq])
        particle = make_particle(
            drydias_q[qq], aero_spec_names, aero_spec_fracs[qq,:], 
            specdata_path= data_path / 'species_data', D_is_wet=False)
        quadrature_population.set_particle(particle, ids[qq], nums_q[qq])
    return quadrature_population


def construct_quadrature(Ns,log10gmds,log10gsds,N_quads,return_weights=True):
    nums_q = np.array([])
    drydias_q = np.array([])
    modes_q = []
    for jj,(N,mu,sig) in enumerate(zip(Ns,log10gmds,log10gsds)):
        x_q,w_q = np.polynomial.hermite.hermgauss(N_quads[jj])
        for q,(x,w) in enumerate(zip(x_q,w_q)):
            nums_q = np.append(nums_q,N*w/np.sqrt(np.pi))
            drydias_q = np.append(drydias_q,10**(mu+np.sqrt(2)*sig*x))
            modes_q.append(jj)
    if return_weights:
        weights_q = nums_q/sum(nums_q)
        return drydias_q, weights_q, modes_q
    else:
        return drydias_q, nums_q, modes_q
    
    
def get_drydias(
        wetdias_q,weights_q,modes_q,volfrac_aero,rho_h2o=1000.,
        equil_assumption='uniform',rho_aero=1000.,tkappa=0.65,
        T0=310.15,S0=1.):
    if equil_assumption == 'uniform':
        massfrac_q = np.ones(len(wetdias_q))*volfrac_aero
        drydias_q = (massfrac_q*rho_h2o)**(1/3)*wetdias_q/((massfrac_q*rho_h2o)**(1/3) + ((1-massfrac_q)*rho_aero)**(1/3))
    elif equil_assumption == 'equil':
        drydias_q = np.zeros(wetdias_q.shape)
        for qq in range(len(drydias_q)):
            drydias_q[qq] = get_dry_diameter__equil(wetdias_q[qq],tkappa,rho_aero,T0,S0)
    elif equil_assumption == 'lb_equil':
        drydias_q = np.zeros(wetdias_q.shape)
        idx_small, = np.where([this_mode=='l' or this_mode=='b' for this_mode in modes_q])
        dryvol_total = sum(np.pi/6*wetdias_q**3*volfrac_aero)
        for ii in idx_small:
            drydias_q[ii] = get_dry_diameter__equil(wetdias_q[ii],tkappa,rho_aero,T0,S0)
        dryvol_in_small = sum(np.pi/6*drydias_q[idx_small]**3)
        idx_big, = np.where([this_mode=='o' for this_mode in modes_q])
        if dryvol_in_small<=dryvol_total:
            drydias_q[idx_big] = (wetdias_q[idx_big]**3*(dryvol_total - dryvol_in_small)/(np.pi/6*wetdias_q[idx_big]**3))**(1/3)
    else:
        print('only coded for \'assumption == uniform\', \'assumption == equil\', \'assumption == lb_equil\' right now')
    return drydias_q

def get_dry_diameter__equil(D_wet,tkappa_aero,density_aero,Tv,S_env_initial):
    zero_this = lambda Dd: get_equilibrium_diameter(Dd,tkappa_aero,Tv,S_env_initial)[0] - D_wet
    D_dry = fsolve(zero_this,D_wet/2.)[0]
    return D_dry

def get_equilibrium_diameter(Dd,tkappa,Tv,S):
    # computes the equilibrium diameter for a droplet having dry diameter (Dd), hygroscopicity parameter (tkappa), and at an environmental supersaturation ration (S)
    zero_this = lambda D: get_droplet_saturation_ratio(D,Dd,tkappa,Tv) - S
    D = fsolve(zero_this,Dd)
    D = fsolve(zero_this,D)
    return D
