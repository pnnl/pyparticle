#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Laura Fierce
"""

from .aerosol_species import AerosolSpecies
from .aerosol_species import retrieve_one_species
from . import data_path
from dataclasses import dataclass
from typing import Tuple
# from typing import Optional
import numpy as np
from scipy.constants import R
import scipy.optimize as opt

from importlib import reload
reload(np)

@dataclass
class Particle:
    """Particle: the definition of an individual aerosol particle
    in terms of the amounts of different constituent species """
    species: Tuple[AerosolSpecies, ...]
    masses: Tuple[float, ...]

    def __post_init__(self):
        assert(len(self.species) == len(self.masses) )
    
    def idx_h2o(self):
        return np.where([
            spec.name.upper() == 'H2O' for spec in self.species])[0][0]
    
    def idx_dry(self):
        idx_all = np.arange(len(self.species))        
        idx_h2o = self.idx_h2o()
        
        if idx_h2o == -1:
            idx_not_h2o = idx_all[:-1]
        elif idx_h2o >= 0:
            idx_not_h2o = np.hstack([idx for idx in idx_all if idx != idx_h2o])
        else:    
            idx_not_h2o = np.hstack([idx_all[:idx_h2o],idx_all[idx_h2o:][1:]])
        return idx_not_h2o
        
    def idx_core(self,core_specs=['BC']):
        return np.where([
            spec.name in core_specs for spec in self.species])[0]
    
    def idx_dry_shell(self,core_specs=['BC']):
        return np.where([
            spec.name not in core_specs + ['H2O'] for spec in self.species])[0]
    
    def idx_spec(self, spec_name):
        return np.where([
            spec.name in spec_name for spec in self.species])
        
    def get_spec_rhos(self):
        spec_rhos = np.hstack([one_spec.density for one_spec in self.species])
        return spec_rhos
    
    def get_spec_kappas(self):
        spec_kappas = np.hstack([one_spec.kappa for one_spec in self.species])
        return spec_kappas
    
    def get_spec_MWs(self):
        spec_MWs = np.hstack([one_spec.molar_mass for one_spec in self.species])
        return spec_MWs
    
    def get_mass_dry(self):
        mks = self.masses
        mass_dry = np.sum(mks[self.idx_dry()])
        return mass_dry

    def get_mass_tot(self):
        mks = self.masses
        mass_tot = np.sum(mks)
        return mass_tot
    
    def get_spec_mass(self,spec_name):
        idx = self.idx_spec(spec_name)
        mks = self.masses
        return mks[idx]

    def get_spec_vol(self,spec_name):
        idx = self.idx_spec(spec_name)
        vks = self.get_vks()
        return vks[idx]
    
    def get_spec_moles(self,spec_name):
        idx = self.idx_spec(spec_name)
        moles = self.get_moles()
        spec_moles = moles[idx][0]
        return spec_moles
    
    def get_spec_rho(self,spec_name):
        idx = self.idx_spec(spec_name)
        rhos = self.get_spec_rhos()
        return rhos[idx[0]]
            
    def get_rho_h2o(self):
        return self.species[self.idx_h2o].density
    
    def get_mass_h2o(self):
        return self.masses[self.idx_h2o()]

    def get_moles(self):
        mks = self.masses
        MWs = self.get_spec_MWs()
        return mks/MWs
        
    def get_vks(self):
        mks = self.masses
        rhos = self.get_spec_rhos()
        return mks/rhos

    def get_vol_tot(self):
        vks = self.get_vks()
        vol_tot = np.sum(vks)
        return vol_tot
        
    def get_vol_dry(self):
        vks = self.get_vks()
        vol_dry = np.sum(vks[self.idx_dry()])
        return vol_dry

    def get_vol_core(self):
        vks = self.get_vks()
        vol_core = np.sum(vks[self.idx_core()])
        return vol_core
        
    def get_vol_dry_shell(self):
        vks = self.get_vks()
        vol_dry_shell = np.sum(vks[self.idx_dry_shell()])
        return vol_dry_shell
    
    def get_Dwet(
            self,RH=None,T=None,
            sigma_h2o=0.072, rho_h2o=1000., MW_h2o=18e-3):
        if RH==None:
            vol_wet = self.get_vol_tot()
            Dwet = (vol_wet*6./np.pi)**(1./3.)
        else:
            Dwet = compute_Dwet(
                self.get_Ddry(), self.get_tkappa(), RH, T, 
                sigma_h2o=sigma_h2o, rho_h2o=rho_h2o, MW_h2o=MW_h2o)
        
        return Dwet
    
    # def get_Dwet_from_RH(self,RH):
    #     pass # need to fix this; maybe add Kohler as a separate package? 
    def get_Ddry(self):
        vol_dry = self.get_vol_dry()
        Ddry = (vol_dry*6./np.pi)**(1./3.)
        return Ddry
    
    def get_Dcore(self):
        vol_core = self.get_vol_core()
        Dcore = (vol_core*6./np.pi)**(1./3.)
        return Dcore
    
    def get_rho_w(self):
        return 1000. # kg/m^3 -- todo: fix this later
    # def get_rho_w(self):
    #     idx_h2o, = np.where([one_spec.name.upper()=='H2O' for one_spec in self.species])
    #     print(idx_h2o)
    #     rho_w = float(self.species[idx_h2o].density)
    #     return rho_w
    
    def get_tkappa(self):
        # compute effective kappa
        vks = self.get_vks()
        spec_kappas = self.get_spec_kappas()
        idx_not_h2o, = np.where([one_spec.name.upper()!='H2O' for one_spec in self.species])
        tkappa = np.sum(vks[idx_not_h2o]*spec_kappas[idx_not_h2o])/np.sum(vks[idx_not_h2o])
        return tkappa
    
    def get_shell_tkappa(self):
        # compute effective kappa
        vks = self.get_vks()
        spec_kappas = self.get_spec_kappas()
        idx_not_h2o_or_bc, = np.where([
            one_spec.name.upper()!='H2O' and one_spec.name.upper()!='BC' for one_spec in self.species])
        shell_tkappa = np.sum(vks[idx_not_h2o_or_bc]*spec_kappas[idx_not_h2o_or_bc])/np.sum(vks[idx_not_h2o_or_bc])
        return shell_tkappa
    
    def get_trho(self):
        # compute effective density
        mks = self.masses
        vks = self.get_vks()
        trho = np.sum(mks)/np.sum(vks)
        
        return trho
    
    def get_critical_supersaturation(self, return_D_crit=False):
        idx_h2o, = np.where([AeroSpec.name.upper()=='H2O' for AeroSpec in self.AeroSpecs])
        Ddry=self.Ddry
        tkappa=self.tkappa
        T=self.T
        sigma_h2o=self.surface_tension
        rho_h2o=self.AeroSpecs[idx_h2o].density
        MW_h2o=self.AeroSpecs[idx_h2o].molar_mass
        
        A = 4.*sigma_h2o*MW_h2o/(R*T*rho_h2o);
        
        if tkappa>0.2 and not return_D_crit:
            s_critical = (np.exp((4.*A**3./(27.*Ddry**3.*tkappa))**(0.5))-1.)*100.
        else:
            f = lambda x: compute_Sc_funsixdeg(x,A,tkappa,Ddry)
            soln = opt.root(f,Ddry*10);
            x = soln.x[0]
            D_critical = x
            s_critical = (((x**3.0-Ddry**3.0)/(x**3-Ddry**3*(1.0-tkappa))*np.exp(A/x)) - 1.)*100.
        
        if return_D_crit:
            return s_critical,D_critical
        else:
            return s_critical
        
def make_particle(
        D, aero_spec_names, aero_spec_frac, 
        specdata_path= data_path / 'species_data',
        species_modifications={}, 
        D_is_wet=True):
    if not 'H2O' in aero_spec_names and not 'h2o' in aero_spec_names:
        aero_spec_names.append('H2O')
        aero_spec_frac = np.hstack([aero_spec_frac, np.array([0.])])
    assert(len(aero_spec_frac) == len(aero_spec_names))
    
    AeroSpecs = []
    for name in aero_spec_names:
        if name in species_modifications.keys():
            spec_modifications = species_modifications[name]
        elif 'SOA' in species_modifications.keys() and name in ['MSA','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']:
            spec_modifications = species_modifications['SOA']
        else:
            spec_modifications = {}
        AeroSpecs.append(retrieve_one_species(name, specdata_path=specdata_path, spec_modifications=spec_modifications))
    
    assert(len(aero_spec_frac) == len(AeroSpecs))
    if D_is_wet:# or 'H2O' not in aero_spec_names or 'h2o' not in aero_spec_names:
        vol = np.pi/6.*D**3.
        mass = effective_density(aero_spec_frac,AeroSpecs)*vol
        spec_masses = mass*aero_spec_frac
    else:
        dryvol = np.pi/6.*D**3.
        drymass = effective_density(aero_spec_frac[:-1],AeroSpecs[:-1])*dryvol
        massfrac_h2o = aero_spec_frac[-1]
        mass_h2o = massfrac_h2o * drymass/(1. - massfrac_h2o)
        spec_masses = np.hstack([drymass*aero_spec_frac[:-1], mass_h2o])
    
    return Particle(species=AeroSpecs,masses=spec_masses)

def make_particle_from_masses(
        aero_spec_names, spec_masses,
        specdata_path= data_path / 'species_data',
        species_modifications = {}):
    AeroSpecs = []
    for name in aero_spec_names:
        if name in species_modifications.keys():
            spec_modifications = species_modifications[name]
        else:
            spec_modifications = {}
        AeroSpecs.append(retrieve_one_species(name, specdata_path=specdata_path, spec_modifications=spec_modifications))
    return Particle(species=AeroSpecs,masses=spec_masses)
    

def compute_Sc_funsixdeg(diam,A,tkappa,dry_diam):
    c6=1.0;
    c4=-(3.0*(dry_diam**3)*tkappa/A); 
    c3=-(2.0-tkappa)*(dry_diam**3); 
    c0=(dry_diam**6.0)*(1.0-tkappa);
    
    z = c6*(diam**6.0) + c4*(diam**4.0) + c3*(diam**3.0) + c0;
    return z

def compute_Dwet(Ddry, kappa, RH, T, sigma_h2o=0.072, rho_h2o=1000., MW_h2o=18e-3):
    if RH>0. and kappa>0.:
        A = 4*sigma_h2o*MW_h2o/(R*T*rho_h2o)
        zero_this = lambda gf: RH/np.exp(A/(Ddry*gf))-(gf**3.-1.)/(gf**3.-(1.-kappa))
        return Ddry*opt.brentq(zero_this,1.,10000000.)
    else:
        return Ddry
    
def effective_density(aero_spec_fracs,AeroSpecs):
    _ = [aero_spec_fracs[kk]/AeroSpecs[kk].density for kk in range(len(AeroSpecs))]
    return 1./np.sum(_)
