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
from typing import Optional
import numpy as np
from scipy.constants import R
import scipy.optimize as opt


@dataclass
class Particle:
    """Particle: the definition of an individual aerosol particle
    in terms of the amounts of different constituent species """
    species: Tuple[AerosolSpecies, ...]
    masses: Tuple[float, ...]
    # idx_h2o: Optional[int] = -1
    
    def idx_h2o(self):
        return np.where([
            spec.name.upper() != 'H2O' for spec in self.species])[0][0]
        
        
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
    
    def get_spec_rhos(self):
        spec_rhos = np.hstack([one_spec.density for one_spec in self.species])
        return spec_rhos
    
    def get_spec_kappas(self):
        spec_kappas = np.hstack([one_spec.kappa for one_spec in self.species])
        return spec_kappas
    
    def get_mass_dry(self):
        mks = self.masses
        mass_dry = np.sum(mks[self.idx_dry()])
        return mass_dry

    def get_mass_tot(self):
        mks = self.masses
        mass_tot = np.sum(mks)
        return mass_tot
        
    def get_rho_h2o(self):
        return self.species[self.idx_h2o].density
    
    def get_mass_h2o(self):
        return self.masses[self.idx_h2o]
        
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
    
    def get_trho(self):
        # compute effective density
        mks = self.masses
        vks = self.get_vks()
        trho = np.sum(mks)/np.sum(vks)
        
        # # alternative:
        # spec_rhos = self.get_spec_rhos()
        # trho = np.sum(vks*spec_rhos)/np.sum(vks)
        return trho
    
    def get_critical_supersaturation(self, return_D_crit=False):
        idx_h2o, = np.where([AeroSpec.name.upper()=='H2O' for AeroSpec in self.AeroSpecs])
        Ddry=self.Ddry
        tkappa=self.tkappa
        T=self.T
        sigma_h2o=self.surface_tension
        rho_h2o=self.AeroSpecs[idx_h2o].density
        MW_h2o=self.AeroSpecs[idx_h2o].molecular_weight
        
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
        D, aero_spec_names, aero_spec_fracs, 
        specdata_path= data_path + 'species_data/', 
        surface_tension=0.072):
    AeroSpecs = []
    for name in aero_spec_names:
        AeroSpecs.append(retrieve_one_species(name, specdata_path=specdata_path, surface_tension=surface_tension))
    vol = np.pi/6.*D**3.
    mass = effective_density(aero_spec_fracs,AeroSpecs)*vol
    # spec_masses = np.array([mass*spec_frac for spec_frac in aero_spec_fracs]) 
    spec_masses = mass*aero_spec_fracs

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
    return 1./np.sum([aero_spec_fracs[kk]/AeroSpecs[kk].density for kk in range(len(AeroSpecs))])
