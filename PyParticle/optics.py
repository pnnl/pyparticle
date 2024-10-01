#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for connecting the ParticlePopulation package to optics models

@author: Laura Fierce
"""

import sys
import numpy as np
from scipy import interpolate

# if float(sys.version[:np.where([ii == '.' for ii in sys.version])[0][1]])<3.10:
#     from PyMieScatt import MieQ
#     from PyMieScatt import MieQCoreShell
# else:
#     from . import Py3Wrapper
#     MieQ = Py3Wrapper('PyMieScatt','MieQ')
#     delegate = Py3Wrapper('PyMieScatt','CoreShell')
#     print(delegate)
#     # delegate = py3bridge.Py3Wrapper('spam', 'listdir')
#     # CoreShell = delegate('.')
from PyMieScat.CoreShell import MieQCoreShell
from . import get_number
from dataclasses import dataclass
# from population import Particle
from . import Particle
import numpy as np
from typing import Callable
from typing import Optional
from warnings import warn
from . import data_path


def make_optical_particle(
        particle,rh_grid,wvl_grid,temp=293.15,
        compute_optics=True,
        specdata_path=data_path + 'species_data/',
        return_lookup=False,return_params=False):
    cs_particle = CoreShellParticle(
        particle=particle,rh_grid=rh_grid,wvl_grid=wvl_grid,temp=temp)
    cs_particle._add_spec_RIs(
        specdata_path=specdata_path,return_lookup=return_lookup,return_params=return_params)
    cs_particle._add_params()
    
    if compute_optics:
        cs_particle._add_optics()
    return cs_particle
@dataclass
class CoreShellParticle:
    """CoreShellParticle: the definition of a core-shell ``optical particle" """
    particle: Particle
    rh_grid: np.array # shape = (N_rh,)
    wvl_grid: np.array # shape = (N_wvl,)
    
    temp: float = 293.15
    
    
    # computed from particle
    core_vol: float = None
    shell_dry_vol: float  = None
    h2o_vol: np.array = None # shape = (N_rh)
    
    # computed from particle and refractive index data for each species
    core_ris: np.array = None # shape = (N_wvl,)
    dry_shell_ris: np.array  = None# shape = (N_wvl,)
    h2o_ris = np.array = None # shape = N_wvl,
    
    # the following are parameters computed by PyMieScatt
    Cext: np.array  = None# shape = (N_rh,N_wvl)
    Csca: np.array  = None# shape = (N_rh,N_wvl)
    Cabs: np.array  = None# shape = (N_rh,N_wvl)
    g: np.array  = None# shape = (N_rh,N_wvl)
    Cpr: np.array  = None# shape = (N_rh,N_wvl)
    Cback: np.array  = None# shape = (N_rh,N_wvl)
    Cratio: np.array = None # shape = (N_rh,N_wvl)
    
    def _add_spec_RIs(
            self,specdata_path=data_path + 'species_data/',
            return_lookup=False,return_params=False):
        old_specs = self.particle.species
        wvls = self.wvl_grid
        new_specs = []
        for old_spec in old_specs:
            new_spec = _add_spec_RI(
                old_spec,wvls,
                specdata_path=specdata_path,
                return_lookup=return_lookup,return_params=return_params)
            new_specs.append(new_spec)
        self.particle.species = new_specs
        
    def _add_params(self):
        particle = self.particle
        vks = particle.get_vks() 
        idx_core = particle.idx_core()
        idx_dry_shell = particle.idx_dry_shell()
        # idx_h2o = particle.idx_h2o()
        
        self.core_vol = np.sum(vks[idx_core])
        
        self.dry_shell_vol = np.sum(vks[idx_dry_shell])
        
        self._add_h2o_vols()
        self._add_effective_ris()
        
    def _add_h2o_vols(
            self,
            sigma_h2o=0.072, rho_h2o=1000., MW_h2o=18e-3):
        T = self.temp
        
        particle = self.particle
        idx_h2o = particle.idx_h2o()
        sigma_h2o = particle.species[idx_h2o].surface_tension
        rho_h2o = particle.species[idx_h2o].density
        MW_h2o = particle.species[idx_h2o].molar_mass
        
        h2o_vols = np.zeros(len(self.rh_grid))
        for rr,rh in enumerate(self.rh_grid):
            h2o_vols[rr] = np.pi/6.*particle.get_Dwet(
                    RH=rh,T=T, sigma_h2o=sigma_h2o, 
                    rho_h2o=rho_h2o, MW_h2o=MW_h2o)**3.
        self.h2o_vols = h2o_vols
            
    def _add_effective_ris(self):
        particle = self.particle
        vks = particle.get_vks() 
        core_ris = 0.
        for ii in particle.idx_core():
            print(particle.species[ii].refractive_index.real_ri_fun(550e-9))
            real_ris = particle.species[ii].refractive_index.real_ri_fun(self.wvl_grid)
            imag_ris = particle.species[ii].refractive_index.imag_ri_fun(self.wvl_grid)
            core_ris += (real_ris + imag_ris*1j) * vks[ii] / self.core_vol
        
        dry_shell_ris = 0.
        for ii in particle.idx_dry_shell():
            real_ris = particle.species[ii].refractive_index.real_ri_fun(self.wvl_grid)
            imag_ris = particle.species[ii].refractive_index.imag_ri_fun(self.wvl_grid)
            dry_shell_ris += (real_ris + imag_ris*1j) * vks[ii] / self.dry_shell_vol
        
        ii = particle.idx_h2o()
        real_ris = particle.species[ii].refractive_index.real_ri_fun(self.wvl_grid)
        imag_ris = particle.species[ii].refractive_index.imag_ri_fun(self.wvl_grid)
        h2o_ris = (real_ris + imag_ris*1j)
        
        self.core_ris = core_ris
        self.dry_shell_ris = dry_shell_ris
        self.h2o_ris = h2o_ris
    
    def get_shell_ri(self,rr,ww):
        shell_ri = (
            (self.h2o_ris[ww] * self.h2o_vols[rr]
            + self.dry_shell_ris[ww] * self.dry_shell_vol) 
            / (self.h2o_vols[rr] + self.dry_shell_vol)
        )
        return shell_ri
    
    def get_core_diam(self):
        return ((6./np.pi) * self.core_vol) ** (1./3.)
    
    def get_dry_diam(self):
        return ((6./np.pi) * (self.core_vol + self.dry_shell_vol)) ** (1./3.)

    def get_wet_diam(self,rr):
        return ((6./np.pi) * (self.core_vol + self.dry_shell_vol + self.h2o_vols[rr])) ** (1./3.)
    
    def get_wet_diams(self):
        return ((6./np.pi) * (self.core_vol + self.dry_shell_vol + self.h2o_vols)) ** (1./3.)
    
    def _add_optics(self):
        N_rh = len(self.rh_grid)
        N_wvl = len(self.wvl_grid)
        self.Cext = np.zeros([N_rh,N_wvl])
        self.Csca = np.zeros([N_rh,N_wvl])
        self.Cabs = np.zeros([N_rh,N_wvl])
        self.g = np.zeros([N_rh,N_wvl])
        self.Cpr = np.zeros([N_rh,N_wvl])
        self.Cback = np.zeros([N_rh,N_wvl])
        self.Cratio = np.zeros([N_rh,N_wvl])
        
        
        # # computed from particle
        # core_vol: float = None
        # shell_dry_vol: float  = None
        # h2o_vol: np.array = None # shape = (N_rh)
        
        # # computed from particle and refractive index data for each species
        # core_ris: np.array = None # shape = (N_wvl,)
        # dry_shell_ris: np.array  = None# shape = (N_wvl,)
        # h2o_ris = np.array = None # shape = N_wvl,
        
        dCore_nm = 1e9 * self.get_core_diam()
        for rr in range(N_rh):
            dShell_nm = 1e9 * self.get_wet_diam(rr)
            for ww,wavelength_m in enumerate(self.wvl_grid):
                wavelength_nm = wavelength_m * 1e9
                mCore = self.core_ris[ww]
                mShell = self.get_shell_ri(rr,ww)
                
                print(mCore,mShell,dCore_nm,dShell_nm, wavelength_nm)
                output_dict = MieQCoreShell(
                    mCore, mShell, wavelength_nm, dCore_nm, dShell_nm, 
                    asDict=True, asCrossSection=True)
                self.Cext[rr,ww] = output_dict['Cext']
                self.Csca[rr,ww] = output_dict['Csca']
                self.Cabs[rr,ww] = output_dict['Cabs']
                self.g[rr,ww] = output_dict['g']
                self.Cpr[rr,ww] = output_dict['Cpr']
                self.Cback[rr,ww] = output_dict['Cback']
                self.Cratio[rr,ww] = output_dict['Cratio']
                
@dataclass
class RefractiveIndex:
    """RefractiveIndex: the definition of wavelength-dependent refractive index.
    Applies to an individual aerosol component or for a mixture of components """
    
    # conventiona is to always use real_ri_fun and imag_ri_fun if they are defined
    # (i.e., wvls, real_ris, and imag_ris are ignored)
    real_ri_fun: Callable[[float],float]
    imag_ri_fun: Callable[[float],float]
    
    # if real_ris and imag_ris are defined, wvls must also be defined
    wvls: Optional[np.array] = None
    real_ris: Optional[np.array] = None
    imag_ris: Optional[np.array] = None
    
    RI_params: Optional[dict] = None

def _add_spec_RI(
        aero_spec,wvls,real_ri_fun=None,imag_ri_fun=None,
        specdata_path='../../datasets/aerosol/species_data/',return_lookup=False,return_params=False):
    
    spec_name = aero_spec.name.upper()
    if real_ri_fun == None:
            if spec_name == 'H2O':
                ri_h2o_filename = specdata_path + 'ri_water.csv'
                wavelength_list = []
                n_list = []
                k_list = []
                with open(ri_h2o_filename) as data_file:
                    for line in data_file:
                        if not 'Wavelength' in line:
                            split_output = line.split(',')
                            wavelength_list.append(1e-6*get_number(split_output[0]))
                            n_list.append(get_number(split_output[3]))
                            k_list.append(get_number(split_output[4]))
                    wvls = np.hstack(wavelength_list)
                    real_ris = np.hstack(n_list)
                    imag_ris = np.hstack(k_list)
                
                real_ri_fun = interpolate.interp1d(wvls, real_ris)# lambda wvl: interpolate.interp1d(wvls, real_ris)(wvl)
                imag_ri_fun = interpolate.interp1d(wvls, imag_ris) # lambda wvl: interpolate.interp1d(wvls, imag_ris)(wvl)#np.interp(wvl, wvls, imag_ris)        
            else:
                RI_params = get_RI_params(spec_name)
                val_550 = RI_params['n_550']; val_alpha = RI_params['alpha_n']
                real_ri_fun = lambda wvl: val_550*(wvl/550e-9)**(val_alpha)
                
                val_550 = RI_params['k_550']; val_alpha = RI_params['alpha_k']
                imag_ri_fun = lambda wvl: val_550*(wvl/550e-9)**(val_alpha)
                # real_ri_fun = lambda wvl: RI_params['n_550']*(wvl/(550e-9))**(-RI_params['alpha_n'])
                # imag_ri_fun = lambda wvl: RI_params['k_550']*(wvl/(550e-9))**(-RI_params['alpha_k'])
            
    if not return_lookup:
        real_ris = None
        imag_ris = None
    
    if not return_params:
        RI_params = None
            
    aero_spec.refractive_index = RefractiveIndex(
        real_ri_fun = real_ri_fun,
        imag_ri_fun = imag_ri_fun,
        wvls = wvls,
        real_ris = real_ris,
        imag_ris = imag_ris,
        RI_params = RI_params)
            
            print(aero_spec.refractive_index.real_ri_fun(550e-9))
    # aero_spec.refractive_index = spec_RI
    return aero_spec

def get_RI_params(name):
    if name.upper() in ['SO4','NH4','NO3','NA','CL','MSA']:
        k_550 = 0.
        n_550 = 1.55
        alpha_n = 0.044 
        alpha_k = 0.
        
        # based on wavelength dependence of NaNO3 (only inorganic salt at RH=0%)
        # data from here: https://eodg.atm.ox.ac.uk/ARIA/data?Salts/Sodium_Nitrate/10%25_(Cotterell_et_al._2017)/NaNO3_10_Cotterell_2017.ri
        # underlying data:
        #   Reference: Cotterell, M.I., Willoughby, R.E., Bzdek, B.R., Orr-Ewing, A.J. and Reid, J.P., A Complete Parameterization of the Relative Humidity and Wavelength Dependence of the Refractive Index of Hygroscopic Inorganic Aerosol Particles.
        #   DOI: 10.5194/acp-17-9837-2017
    elif name.upper() == 'BC':
        k_550 = 0.74
        n_550 = 1.82
        alpha_n = 0.
        alpha_k = 0.
    elif name.upper() == 'OIN':
        k_550 = 0.006
        n_550 = 1.68
        alpha_n = 0.
        alpha_k = 0.
    else: # organics
        k_550 = 0.
        n_550 = 1.45
        alpha_n = 0.
        alpha_k = 0.
        
        warn(name + ' is not in the list, assuming its an arbitrary organic')
    
    RI_param_dict = {'n_550':n_550, 'k_550':k_550, 'alpha_n':alpha_n, 'alpha_k':alpha_k}
    return RI_param_dict

    
