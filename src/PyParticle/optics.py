#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for connecting the ParticlePopulation package to optics models

@author: Laura Fierce
"""
from ._patch import patch_pymiescatt; patch_pymiescatt(); del patch_pymiescatt
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
from PyMieScatt.CoreShell import MieQCoreShell, MieQ
from . import get_number
from dataclasses import dataclass
# from dataclasses_json import dataclass_json
from . import Particle
from . import ParticlePopulation
from . import AerosolSpecies
import numpy as np
from typing import Callable
from typing import Optional
from typing import Tuple
from warnings import warn
from . import data_path

import copy

        
def make_optical_particle(
        particle,rh_grid,wvl_grid,
        morphology='core-shell',compute_optics=True,temp=293.15,
        specdata_path=data_path / 'species_data',
        species_modifications={}):
    
    if morphology == 'core-shell':
        optical_particle = CoreShellParticle(
            particle.species,
            particle.masses,
            rh_grid=rh_grid,wvl_grid=wvl_grid,temp=temp)
        optical_particle._add_spec_RIs(
            specdata_path=specdata_path,species_modifications=species_modifications)
        
        optical_particle._add_params()
    else:
        print('only coded for morphology = \'core-shell\'')
    
    if compute_optics:
        optical_particle._add_optics()
        
    return optical_particle

def make_optical_population(
        particle_population, rh_grid, wvl_grid,
        morphology='core-shell',compute_optics=True,temp=293.15,
        species_modifications={},
        specdata_path=data_path / 'species_data',
        suppress_warning=True):
        # return_lookup=False,return_params=False):
    optical_population = CoreShellPopulation(
        species = particle_population.species,
        spec_masses = particle_population.spec_masses,
        num_concs = particle_population.num_concs,
        ids = particle_population.ids,
        rh_grid = np.array(rh_grid),
        wvl_grid = np.array(wvl_grid)
        )
    
    for (part_id,num_conc) in zip(particle_population.ids,particle_population.num_concs):
        particle = particle_population.get_particle(part_id)
        optical_particle = make_optical_particle(
                particle, np.array(rh_grid), np.array(wvl_grid),
                morphology=morphology, compute_optics=compute_optics,
                temp=temp, specdata_path=specdata_path,
                species_modifications=species_modifications)
                # return_lookup=return_lookup,return_params=return_params)
        optical_population.set_particle(
            optical_particle, part_id, num_conc,suppress_warning=suppress_warning)
    
    optical_population._add_effective_kappas()
    
    return optical_population


# todo: need to rethink this
# @dataclass_json
@dataclass
class CoreShellPopulation(ParticlePopulation):
    # species: Tuple[AerosolSpecies, ...] # shape = N_species
    # spec_masses: np.array # shape = (N_particles, N_species)
    # num_concs: np.array # shape = N_particles
    # ids: Tuple[int, ...] # shape = N_particles
    
    rh_grid: np.array# = None
    wvl_grid: np.array #= None
    
    temp: float = 293.15
    
    tkappas: np.array = None
    shell_tkappas: np.array = None
    
    # computed from particle
    core_vols: np.array = None
    shell_dry_vols: np.array  = None
    h2o_vols: np.array = None # shape = (N_rh)
    
    # computed from particle and refractive index data for each species
    core_ris: np.array = None # shape = (N_wvl,)
    dry_shell_ris: np.array  = None# shape = (N_wvl,)
    h2o_ris: np.array = None # shape = N_wvl,
    
    # the following are parameters computed by PyMieScatt
    Cexts: np.array  = None# shape = (N_rh,N_wvl)
    Cscas: np.array  = None# shape = (N_rh,N_wvl)
    Cabss: np.array  = None# shape = (N_rh,N_wvl)
    gs: np.array  = None# shape = (N_rh,N_wvl)
    Cprs: np.array  = None# shape = (N_rh,N_wvl)
    Cbacks: np.array  = None# shape = (N_rh,N_wvl)
    
    Cexts_bc: np.array  = None# shape = (N_rh,N_wvl)
    Cscas_bc: np.array  = None# shape = (N_rh,N_wvl)
    Cabss_bc: np.array  = None# shape = (N_rh,N_wvl)
    gs_bc: np.array  = None# shape = (N_rh,N_wvl)
    Cprs_bc: np.array  = None# shape = (N_rh,N_wvl)
    Cbacks_bc: np.array  = None# shape = (N_rh,N_wvl)
    
    Cexts_clear: np.array  = None# shape = (N_rh,N_wvl)
    Cscas_clear: np.array  = None# shape = (N_rh,N_wvl)
    Cabss_clear: np.array  = None# shape = (N_rh,N_wvl)
    gs_clear: np.array  = None# shape = (N_rh,N_wvl)
    Cprs_clear: np.array  = None# shape = (N_rh,N_wvl)
    Cbacks_clear: np.array  = None# shape = (N_rh,N_wvl)
    
    def find_particle(self, part_id):
        if part_id in self.ids:
            idx, = np.where([one_id == part_id for one_id in self.ids])
            if len(idx)>1:
                ValueError('part_id is listed more than once in self.ids')
            else:
                idx = idx[0]
        else:
            idx = len(self.ids)
        return idx
    
    def set_particle(self, optical_particle, part_id, num_conc,suppress_warning=False):    
        idx = self.find_particle(part_id)
        if not isinstance(self.core_vols, np.ndarray):
        # type(self.core_vols) == type(None):
            if not suppress_warning:
                warn('empty population, adding ' + str(part_id))
            self.add_particle(optical_particle, part_id, num_conc)
        elif len(self.core_vols)<(idx+1):
            if not suppress_warning:
                warn('part_id not in self.ids, adding ' + str(part_id))
            self.add_particle(optical_particle, part_id, num_conc)
        else:
            self.species = optical_particle.species
            
            self.spec_masses[idx,:] = optical_particle.masses
            self.num_concs[idx] = num_conc
            self.ids[idx] = part_id
            
            self.core_vols[idx] = optical_particle.core_vol
            self.shell_dry_vols[idx] = optical_particle.shell_dry_vol
            self.h2o_vols[idx] = optical_particle.h2o_vol
            
            self.core_ris[idx,:] = optical_particle.core_ris
            self.dry_shell_ris[idx,:] = optical_particle.dry_shell_ris
            self.h2o_ris[idx,:] = optical_particle.h2o_ris
            
            self.Cexts[idx,:,:] = optical_particle.Cext
            self.Cscas[idx,:,:] = optical_particle.Csca
            self.Cabss[idx,:,:] = optical_particle.Cabs
            self.gs[idx,:,:] = optical_particle.g
            self.Cprs[idx,:,:] = optical_particle.Cpr
            self.Cbacks[idx,:,:] = optical_particle.Cback            
            
            self.Cexts_bc[idx,:,:] = optical_particle.Cext_bc
            self.Cscas_bc[idx,:,:] = optical_particle.Csca_bc
            self.Cabss_bc[idx,:,:] = optical_particle.Cabs_bc
            self.gs_bc[idx,:,:] = optical_particle.g_bc
            self.Cprs_bc[idx,:,:] = optical_particle.Cpr_bc
            self.Cbacks_bc[idx,:,:] = optical_particle.Cback_bc
            
            self.Cexts_clear[idx,:,:] = optical_particle.Cext_clear
            self.Cscas_clear[idx,:,:] = optical_particle.Csca_clear
            self.Cabss_clear[idx,:,:] = optical_particle.Cabs_clear
            self.gs_clear[idx,:,:] = optical_particle.g_clear
            self.Cprs_clear[idx,:,:] = optical_particle.Cpr_clear
            self.Cbacks_clear[idx,:,:] = optical_particle.Cback_clear
    
    def add_particle(self, optical_particle, part_id, num_conc):
        N_rh = len(optical_particle.rh_grid)
        N_wvl = len(optical_particle.wvl_grid)
        
        if len(self.ids) == 0 or type(self.core_vols) == type(None):
            self.spec_masses = np.zeros([1,len(optical_particle.species)])
            self.spec_masses[0,:] = optical_particle.masses
            self.num_concs = np.hstack([num_conc])
            self.ids = [part_id]
            
            self.rh_grid = optical_particle.rh_grid
            self.wvl_grid = optical_particle.wvl_grid
            
            self.temp = optical_particle.temp
            
            self.core_ris = np.zeros([1, N_wvl],dtype=complex)
            self.core_ris[0,:] = optical_particle.core_ris
            self.dry_shell_ris = np.zeros([1, N_wvl],dtype=complex)
            self.dry_shell_ris[0,:] = optical_particle.dry_shell_ris
            self.h2o_ris = np.zeros([1, N_wvl],dtype=complex)
            self.h2o_ris[0,:] = optical_particle.h2o_ris
            
            self.core_vols = np.hstack([optical_particle.core_vol])
            self.shell_dry_vols = np.hstack([optical_particle.shell_dry_vol])
            self.h2o_vols = np.zeros([1, N_rh])
            self.h2o_vols[0,:] = optical_particle.h2o_vols
            
            self.Cexts = np.zeros([1,N_rh,N_wvl])
            self.Cscas = np.zeros([1,N_rh,N_wvl])
            self.Cabss = np.zeros([1,N_rh,N_wvl])
            self.gs = np.zeros([1,N_rh,N_wvl])
            self.Cprs = np.zeros([1,N_rh,N_wvl])
            self.Cbacks = np.zeros([1,N_rh,N_wvl])

            self.Cexts[0,:,:] = optical_particle.Cext
            self.Cscas[0,:,:] = optical_particle.Csca
            self.Cabss[0,:,:] = optical_particle.Cabs
            self.gs[0,:,:] = optical_particle.g
            self.Cprs[0,:,:] = optical_particle.Cpr
            self.Cbacks[0,:,:] = optical_particle.Cback
            
            self.Cexts_bc = np.zeros([1,N_rh,N_wvl])
            self.Cscas_bc = np.zeros([1,N_rh,N_wvl])
            self.Cabss_bc = np.zeros([1,N_rh,N_wvl])
            self.gs_bc = np.zeros([1,N_rh,N_wvl])
            self.Cprs_bc = np.zeros([1,N_rh,N_wvl])
            self.Cbacks_bc = np.zeros([1,N_rh,N_wvl])

            self.Cexts_bc[0,:,:] = optical_particle.Cext_bc
            self.Cscas_bc[0,:,:] = optical_particle.Csca_bc
            self.Cabss_bc[0,:,:] = optical_particle.Cabs_bc
            self.gs_bc[0,:,:] = optical_particle.g_bc
            self.Cprs_bc[0,:,:] = optical_particle.Cpr_bc
            self.Cbacks_bc[0,:,:] = optical_particle.Cback_bc
            
            self.Cexts_clear = np.zeros([1,N_rh,N_wvl])
            self.Cscas_clear = np.zeros([1,N_rh,N_wvl])
            self.Cabss_clear = np.zeros([1,N_rh,N_wvl])
            self.gs_clear = np.zeros([1,N_rh,N_wvl])
            self.Cprs_clear = np.zeros([1,N_rh,N_wvl])
            self.Cbacks_clear = np.zeros([1,N_rh,N_wvl])

            self.Cexts_clear[0,:,:] = optical_particle.Cext_clear
            self.Cscas_clear[0,:,:] = optical_particle.Csca_clear
            self.Cabss_clear[0,:,:] = optical_particle.Cabs_clear
            self.gs_clear[0,:,:] = optical_particle.g_clear
            self.Cprs_clear[0,:,:] = optical_particle.Cpr_clear
            self.Cbacks_clear[0,:,:] = optical_particle.Cback_clear
            
        else:
            self.spec_masses = np.vstack([self.spec_masses, optical_particle.masses.reshape(1,-1)])
            self.num_concs = np.hstack([self.num_concs, num_conc])
            self.ids.append(part_id)
                        
            self.core_vols = np.hstack([self.core_vols, optical_particle.core_vol])
            self.shell_dry_vols = np.hstack([self.shell_dry_vols, optical_particle.shell_dry_vol])
            self.h2o_vols = np.vstack([self.h2o_vols, optical_particle.h2o_vols.reshape(1,N_rh)])
            
            self.core_ris = np.vstack([self.core_ris, optical_particle.core_ris.reshape(1,N_wvl)])
            self.dry_shell_ris = np.vstack([self.dry_shell_ris, optical_particle.dry_shell_ris.reshape(1,N_wvl)])
            self.h2o_ris = np.vstack([self.h2o_ris, optical_particle.h2o_ris.reshape(1,N_wvl)])
            
            self.Cexts = np.vstack([self.Cexts, optical_particle.Cext.reshape(1,N_rh,N_wvl)])
            self.Cscas = np.vstack([self.Cscas, optical_particle.Csca.reshape(1,N_rh,N_wvl)])
            self.Cabss = np.vstack([self.Cabss, optical_particle.Cabs.reshape(1,N_rh,N_wvl)])
            self.gs = np.vstack([self.gs, optical_particle.g.reshape(1,N_rh,N_wvl)])
            self.Cprs = np.vstack([self.Cprs, optical_particle.Cpr.reshape(1,N_rh,N_wvl)])
            self.Cbacks = np.vstack([self.Cbacks, optical_particle.Cback.reshape(1,N_rh,N_wvl)])
            
            self.Cexts_bc = np.vstack([self.Cexts_bc, optical_particle.Cext_bc.reshape(1,N_rh,N_wvl)])
            self.Cscas_bc = np.vstack([self.Cscas_bc, optical_particle.Csca_bc.reshape(1,N_rh,N_wvl)])
            self.Cabss_bc = np.vstack([self.Cabss_bc, optical_particle.Cabs_bc.reshape(1,N_rh,N_wvl)])
            self.gs_bc = np.vstack([self.gs_bc, optical_particle.g_bc.reshape(1,N_rh,N_wvl)])
            self.Cprs_bc = np.vstack([self.Cprs_bc, optical_particle.Cpr_bc.reshape(1,N_rh,N_wvl)])
            self.Cbacks_bc = np.vstack([self.Cbacks_bc, optical_particle.Cback_bc.reshape(1,N_rh,N_wvl)])
            
            self.Cexts_clear = np.vstack([self.Cexts_clear, optical_particle.Cext_clear.reshape(1,N_rh,N_wvl)])
            self.Cscas_clear = np.vstack([self.Cscas_clear, optical_particle.Csca_clear.reshape(1,N_rh,N_wvl)])
            self.Cabss_clear = np.vstack([self.Cabss_clear, optical_particle.Cabs_clear.reshape(1,N_rh,N_wvl)])
            self.gs_clear = np.vstack([self.gs_clear, optical_particle.g_clear.reshape(1,N_rh,N_wvl)])
            self.Cprs_clear = np.vstack([self.Cprs_clear, optical_particle.Cpr_clear.reshape(1,N_rh,N_wvl)])
            self.Cbacks_clear = np.vstack([self.Cbacks_clear, optical_particle.Cback_clear.reshape(1,N_rh,N_wvl)])
    
    def idx_bc(self):
        idx, = np.where([spec.name == 'BC' for spec in self.species]); idx = idx[0]
        return idx
    
    def idx_h2o(self):
        idx, = np.where([spec.name == 'H2O' for spec in self.species]); idx = idx[0]
        return idx
    
    def get_particle(self, part_id, morphology='core-shell'):
        if part_id in self.ids:
            idx_particle = self.find_particle(part_id)
            return Particle(self.species, self.spec_masses[idx_particle,:])
        else:
            raise ValueError(str(part_id) + ' not in ids')
    
    def get_optical_coeff(self,optics_type='total_abs',rh='all',wvl='all',bconly=False):
        if rh == 'all' and wvl == 'all':
            N_rh = len(self.rh_grid)
            N_wvl = len(self.wvl_grid)
            b_optics = np.zeros([N_rh,N_wvl])
            for rr in range(N_rh):
                for ww in range(N_wvl):
                    if optics_type == 'total_abs':
                        crossects = self.Cabss[:,rr,ww]
                    elif optics_type == 'pure_bc_abs':
                        crossects = self.Cabss_bc[:,rr,ww]
                    elif optics_type == 'clear_abs':
                        crossects = self.Cabss_clear[:,rr,ww]
                    elif optics_type == 'total_scat':
                        crossects = self.Cscas[:,rr,ww]
                    elif optics_type == 'pure_bc_scat':
                        crossects = self.Cscas_bc[:,rr,ww]
                    elif optics_type == 'clear_scat':
                        crossects = self.Cscas_clear[:,rr,ww]      
                    else:
                        print('optics_type =', optics_type, 'not included')
                    b_optics[rr,ww] = np.sum(crossects*self.num_concs)
        elif type(rh) in [float,np.float64] and type(wvl) in [float,np.float64]:
            idx_rh, idx_wvl = self.get_grid_indices(rh, wvl)
            if optics_type == 'total_abs':
                crossects = self.Cabss[:,idx_rh,idx_wvl]
            elif optics_type == 'pure_bc_abs':
                crossects = self.Cabss_bc[:,idx_rh,idx_wvl]
            elif optics_type == 'clear_abs':
                crossects = self.Cabss_clear[:,idx_rh,idx_wvl]
            elif optics_type == 'total_scat':
                crossects = self.Cscas[:,idx_rh,idx_wvl]
            elif optics_type == 'pure_bc_scat':
                crossects = self.Cscas_bc[:,idx_rh,idx_wvl]
            elif optics_type == 'clear_scat':
                crossects = self.Cscas_clear[:,idx_rh,idx_wvl]
            else:
                print('optics_type =', optics_type, 'not included')
            
            if bconly:
                idx_bc_containing = np.where(self.spec_masses[:,self.idx_bc()]>0.)
                b_optics = np.sum(crossects[idx_bc_containing]*self.num_concs[idx_bc_containing])
            else:
                b_optics = np.sum(crossects*self.num_concs)
        # else:
        #     print('not yet coded for rh = ' + str(rh) + ' and wvl = ' + str(wvl))
        
        return b_optics
    
    # def get_bscat(self,optics_type='total'):
    #     N_rh = len(self.rh_grid)
    #     N_wvl = len(self.wvl_grid)
    #     b_scat = np.zeros([N_rh,N_wvl])
    #     for rr in range(N_rh):
    #         for ww in range(N_wvl):
    #             if optics_type == 'total':
    #                 crossects = self.Cscas[:,rr,ww]
    #             elif optics_type == 'pure_bc':
    #                 crossects = self.Cscas_bc[:,rr,ww]
    #             elif optics_type == 'no_bc':
    #                 crossects = self.Cscas_clear[:,rr,ww]                    
    #             else:
    #                 print('optics_type =', optics_type, 'not included')
    #             b_scat[rr,ww] = np.sum(crossects*self.num_concs)
    #     return b_scat
    
    
    def _add_effective_kappas(self):
        self.tkappas = np.zeros(len(self.ids))
        self.shell_tkappas = np.zeros(len(self.ids))
        
        for ii,part_id in enumerate(self.ids):
            self.tkappas[ii] = self.get_particle(part_id).get_tkappa()
            self.shell_tkappas[ii] = self.get_particle(part_id).get_shell_tkappa()
    
    def get_grid_indices(self, rh, wvl):
        if rh not in self.rh_grid:
            print('rh must be in ' + str(self.rh_grid))
        if wvl not in self.wvl_grid:
            print('wvl must be in ' + str(self.wvl_grid))
        
        idx_rh, = np.where(self.rh_grid==rh); idx_rh = idx_rh[0]
        idx_wvl, = np.where(self.wvl_grid==wvl); idx_wvl = idx_wvl[0]
        
        return idx_rh, idx_wvl
    
    def get_average_ri(
            self, morph_component='dry_shell',ri_component='total',
            rh=0., wvl=550e-9, bconly=False):
        
        idx_rh, idx_wvl = self.get_grid_indices(rh, wvl)
        # if morph_component == 'dry_shell':
        #     vols = self.shell_dry_vols
        #     ris = self.dry_shell_ris[:,idx_wvl]
        # el
        if morph_component == 'shell':
            dry_shell_vols = self.shell_dry_vols
            dry_shell_ris = self.dry_shell_ris[:,idx_wvl]
            h2o_vols = self.h2o_vols[:,idx_rh]
            h2o_ris = self.h2o_ris[:,idx_wvl]
            vols = dry_shell_vols + h2o_vols
            ris = (dry_shell_vols*dry_shell_ris + h2o_vols*h2o_ris)/vols
        elif morph_component == 'core':
            vols = self.core_vols
            ris = self.core_ris[:,idx_wvl]
        elif morph_component == 'total':
            dry_shell_vols = self.shell_dry_vols
            dry_shell_ris = self.dry_shell_ris[:,idx_wvl]
            h2o_vols = self.h2o_vols[:,idx_rh]
            h2o_ris = self.h2o_ris[:,idx_wvl]
            core_vols = self.core_vols
            core_ris = self.core_ris[:,idx_wvl]
            vols = core_vols + dry_shell_vols + h2o_vols
            ris = (core_vols*core_ris + dry_shell_vols*dry_shell_ris + h2o_vols*h2o_ris)/vols
        else:
            print('morph_component must be \'total\', \'core\', \'shell\' or \'dry_shell\'')
        
        if bconly:
            idx = np.where(self.spec_masses[:,self.idx_bc()]>0.)
        else:
            idx = np.arange(0,len(ris))
            
        if ri_component == 'total':
            avg_ri = np.sum(vols[idx]*ris[idx])/np.sum(vols[idx])
        elif ri_component == 'real':
            avg_ri = np.real(np.sum(vols[idx]*ris[idx])/np.sum(vols[idx]))
        elif ri_component == 'imag':
            avg_ri = np.imag(np.sum(vols[idx]*ris[idx])/np.sum(vols[idx]))
        else:
            print('ri_component must be \'total\', \'real\', or \'imag\'')
        
        return avg_ri

@dataclass
class CoreShellParticle(Particle):
    """CoreShellParticle: the definition of a core-shell ``optical particle" """
    # particle: Particle
    
    rh_grid: np.array # shape = (N_rh,)
    wvl_grid: np.array # shape = (N_wvl,)
    
    temp: float = 293.369563
    
    # computed from particle
    core_vol: float = None
    shell_dry_vol: float  = None
    h2o_vols: np.array = None # shape = (N_rh)
    
    # computed from particle and refractive index data for each species
    core_ris: np.array = None # shape = (N_wvl,)
    dry_shell_ris: np.array  = None# shape = (N_wvl,)
    h2o_ris: np.array = None # shape = N_wvl,
    
    # the following are parameters computed by PyMieScatt
    Cext: np.array  = None# shape = (N_rh,N_wvl)
    Csca: np.array  = None# shape = (N_rh,N_wvl)
    Cabs: np.array  = None# shape = (N_rh,N_wvl)
    g: np.array  = None# shape = (N_rh,N_wvl)
    Cpr: np.array  = None# shape = (N_rh,N_wvl)
    Cback: np.array  = None# shape = (N_rh,N_wvl)
    Cratio: np.array = None # shape = (N_rh,N_wvl)
    
    Cext_bc: np.array  = None# shape = (N_rh,N_wvl)
    Csca_bc: np.array  = None# shape = (N_rh,N_wvl)
    Cabs_bc: np.array  = None# shape = (N_rh,N_wvl)
    g_bc: np.array  = None# shape = (N_rh,N_wvl)
    Cpr_bc: np.array  = None# shape = (N_rh,N_wvl)
    Cback_bc: np.array  = None# shape = (N_rh,N_wvl)
    Cratio_bc: np.array = None # shape = (N_rh,N_wvl)
    
    Cext_clear: np.array  = None# shape = (N_rh,N_wvl)
    Csca_clear: np.array  = None# shape = (N_rh,N_wvl)
    Cabs_clear: np.array  = None# shape = (N_rh,N_wvl)
    g_clear: np.array  = None# shape = (N_rh,N_wvl)
    Cpr_clear: np.array  = None# shape = (N_rh,N_wvl)
    Cback_clear: np.array  = None# shape = (N_rh,N_wvl)
    Cratio_clear: np.array = None # shape = (N_rh,N_wvl)
    
    def _add_spec_RIs(
            self,specdata_path=data_path / 'species_data',
            species_modifications={}):
            # return_lookup=False,return_params=False):
        old_specs = self.species
        wvls = self.wvl_grid
        new_specs = []
        for old_spec in old_specs:
            if old_spec.name in species_modifications.keys():
                spec_modifications = species_modifications[old_spec.name]
            elif 'SOA' in species_modifications.keys() and old_spec.name in ['MSA','ARO1','ARO2','ALK1','OLE1','API1','API2','LIM1','LIM2']:
                spec_modifications = species_modifications['SOA']
            else:
                spec_modifications = {}
            
            new_spec = _add_spec_RI(
                old_spec,#wvls,
                specdata_path=specdata_path,
                spec_modifications=spec_modifications)
                # return_lookup=return_lookup,return_params=return_params)
            new_specs.append(copy.deepcopy(new_spec))
        self.species = new_specs
        # print(self.species)
        
    def _add_params(self):
        vks = self.get_vks()
        idx_core = self.idx_core()
        idx_dry_shell = self.idx_dry_shell()
        
        self.core_vol = np.sum(vks[idx_core])
        
        self.shell_dry_vol = np.sum(vks[idx_dry_shell])
        
        self._add_h2o_vols()
        self._add_effective_ris()
        
    def _add_h2o_vols(
            self,
            sigma_h2o=0.072, rho_h2o=1000., MW_h2o=18e-3):
        T = self.temp
        
        # particle = self.particle
        idx_h2o = self.idx_h2o()
        sigma_h2o = self.species[idx_h2o].surface_tension
        rho_h2o = self.species[idx_h2o].density
        MW_h2o = self.species[idx_h2o].molar_mass
        
        h2o_vols = np.zeros(len(self.rh_grid))
        dry_vols = np.pi/6.*self.get_dry_diam()**3
        for rr,rh in enumerate(self.rh_grid):
            wet_vols = np.pi/6.*self.get_Dwet(
                    RH=rh,T=T, sigma_h2o=sigma_h2o, 
                    rho_h2o=rho_h2o, MW_h2o=MW_h2o)**3.
            if rh>0.:
                h2o_vols[rr] = wet_vols - dry_vols
            else:
                h2o_vols[rr] = 0.
        self.h2o_vols = h2o_vols
    
    def _add_effective_ris(self):
        N_wvl = len(self.wvl_grid)
        vks = self.get_vks() 
        
        real_ris = np.zeros(N_wvl)
        imag_ris = np.zeros(N_wvl)
        for ii in self.idx_core():
            real_ris += self.species[ii].refractive_index.real_ri_fun(self.wvl_grid) * vks[ii] / self.core_vol
            imag_ris += self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid) * vks[ii] / self.core_vol
        core_ris = copy.deepcopy(real_ris) + copy.deepcopy(imag_ris) * 1j
        
        real_ris = np.zeros(N_wvl)
        imag_ris = np.zeros(N_wvl)
        for ii in self.idx_dry_shell():
            real_ris += self.species[ii].refractive_index.real_ri_fun(self.wvl_grid) * vks[ii] / self.shell_dry_vol
            imag_ris += self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid) * vks[ii] / self.shell_dry_vol
        dry_shell_ris = copy.deepcopy(real_ris) + copy.deepcopy(imag_ris) * 1j
        
        ii = self.idx_h2o()
        real_ris = self.species[ii].refractive_index.real_ri_fun(self.wvl_grid)
        imag_ris = self.species[ii].refractive_index.imag_ri_fun(self.wvl_grid)
        h2o_ris = copy.deepcopy(real_ris) + copy.deepcopy(imag_ris) * 1j
        
        self.core_ris = np.array([complex(np.real(one_core_ri),np.imag(one_core_ri)) for one_core_ri in core_ris])# np.array(core_ris, dtype=complex)
        self.dry_shell_ris = np.array([complex(np.real(one_shell_ri),np.imag(one_shell_ri)) for one_shell_ri in dry_shell_ris])# np.array(dry_shell_ris, dtype=complex)
        self.h2o_ris = np.array([complex(np.real(one_h2o_ri),np.imag(one_h2o_ri)) for one_h2o_ri in h2o_ris])# np.array(h2o_ris, dtype=complex)
    
    def get_shell_ri(self,rr,ww):
        shell_ri = (
            (self.h2o_ris[ww] * self.h2o_vols[rr]
            + self.dry_shell_ris[ww] * self.shell_dry_vol)
            / (self.h2o_vols[rr] + self.shell_dry_vol)
        )
        
        return shell_ri
    
    def get_core_diam(self):
        return ((6./np.pi) * self.core_vol) ** (1./3.)
    
    def get_dry_diam(self):
        return ((6./np.pi) * (self.core_vol + self.shell_dry_vol)) ** (1./3.)

    def get_wet_diam(self,rr):
        return ((6./np.pi) * (self.core_vol + self.shell_dry_vol + self.h2o_vols[rr])) ** (1./3.)
    
    # def get_wet_diams(self):
    #     return ((6./np.pi) * (self.core_vol + self.shell_dry_vol + self.h2o_vols)) ** (1./3.)
    
    def get_total_crossect(self,rr):
        return np.pi/4.*self.get_wet_diam(rr)**2

    def get_core_crossect(self):
        return np.pi/4.*self.get_core_diam()**2
    
    def _add_optics(self):
        N_rh = len(self.rh_grid)
        N_wvl = len(self.wvl_grid)
        self.Cext = np.zeros([N_rh,N_wvl])
        self.Csca = np.zeros([N_rh,N_wvl])
        self.Cabs = np.zeros([N_rh,N_wvl])
        self.g = np.zeros([N_rh,N_wvl])
        self.Cpr = np.zeros([N_rh,N_wvl])
        self.Cback = np.zeros([N_rh,N_wvl])
        
        self.Cext_bc = np.zeros([N_rh,N_wvl])
        self.Csca_bc = np.zeros([N_rh,N_wvl])
        self.Cabs_bc = np.zeros([N_rh,N_wvl])
        self.g_bc = np.zeros([N_rh,N_wvl])
        self.Cpr_bc = np.zeros([N_rh,N_wvl])
        self.Cback_bc = np.zeros([N_rh,N_wvl])
        
        self.Cext_clear = np.zeros([N_rh,N_wvl])
        self.Csca_clear = np.zeros([N_rh,N_wvl])
        self.Cabs_clear = np.zeros([N_rh,N_wvl])
        self.g_clear = np.zeros([N_rh,N_wvl])
        self.Cpr_clear = np.zeros([N_rh,N_wvl])
        self.Cback_clear = np.zeros([N_rh,N_wvl])
        dCore_nm = 1e9 * self.get_core_diam()
        for rr in range(N_rh):
            dShell_nm = 1e9 * self.get_wet_diam(rr)
            
            core_crossect = self.get_core_crossect()
            total_crossect = self.get_total_crossect(rr)
            
            for ww,wavelength_m in enumerate(self.wvl_grid):
                wavelength_nm = wavelength_m * 1e9
                mCore = self.core_ris[ww]
                mShell = self.get_shell_ri(rr,ww)
                output_dict = MieQCoreShell(
                    mCore, mShell, wavelength_nm, dCore_nm, dShell_nm, 
                    asDict=True, asCrossSection=False)
                self.Cext[rr,ww] = output_dict['Qext']*total_crossect
                self.Csca[rr,ww] = output_dict['Qsca']*total_crossect
                self.Cabs[rr,ww] = output_dict['Qabs']*total_crossect
                self.g[rr,ww] = output_dict['g']
                self.Cpr[rr,ww] = output_dict['Qpr']*total_crossect
                self.Cback[rr,ww] = output_dict['Qback']*total_crossect
                
                mShell_clear = np.real(self.get_shell_ri(rr,ww))
                output_dict_clear = MieQCoreShell(
                    mCore, mShell_clear, wavelength_nm, dCore_nm, dShell_nm, 
                    asDict=True, asCrossSection=False)
                self.Cext_clear[rr,ww] = output_dict_clear['Qext']*total_crossect
                self.Csca_clear[rr,ww] = output_dict_clear['Qsca']*total_crossect
                self.Cabs_clear[rr,ww] = output_dict_clear['Qabs']*total_crossect
                self.g_clear[rr,ww] = output_dict_clear['g']
                self.Cpr_clear[rr,ww] = output_dict_clear['Qpr']*total_crossect
                self.Cback_clear[rr,ww] = output_dict_clear['Qback']*total_crossect
                
                if not np.isclose(dCore_nm, 0.):
                    output_dict_bc = MieQ(
                        mCore, wavelength_nm, dCore_nm, 
                        asDict=True, asCrossSection=False)
                    self.Cext_bc[rr,ww] = output_dict_bc['Qext']*core_crossect
                    self.Csca_bc[rr,ww] = output_dict_bc['Qsca']*core_crossect
                    self.Cabs_bc[rr,ww] = output_dict_bc['Qabs']*core_crossect
                    self.g_bc[rr,ww] = output_dict_bc['g']
                    self.Cpr_bc[rr,ww] = output_dict_bc['Qpr']*core_crossect
                    self.Cback_bc[rr,ww] = output_dict_bc['Qback']*core_crossect
                    
                    # output_dict_clear_shell = MieQ(
                    #     mShell, wavelength_nm, dShell_nm, 
                    #     asDict=True, asCrossSection=False)
                    # output_dict_clear_core = MieQ(
                    #     mShell, wavelength_nm, dCore_nm, 
                    #     asDict=True, asCrossSection=False)
                    
                #     self.Cext_clear[rr,ww] = output_dict_clear_shell['Qext']*total_crossect - output_dict_clear_core['Qext']*core_crossect
                #     self.Csca_nobc[rr,ww] = output_dict_nobc_shell['Qsca']*total_crossect - output_dict_nobc_core['Qsca']*core_crossect
                #     if not np.isclose(dCore_nm, 0.):
                #         self.Cabs_nobc[rr,ww] = output_dict_nobc_shell['Qabs']*total_crossect - output_dict_nobc_core['Qabs']*core_crossect
                #     self.g_nobc_core[rr,ww] = output_dict_nobc_core['g']
                #     self.g_nobc_shell[rr,ww] = output_dict_nobc_shell['g']
                #     self.Cpr_nobc[rr,ww] = output_dict_nobc_shell['Qpr']*total_crossect - output_dict_nobc_core['Qpr']*core_crossect
                #     self.Cback_nobc[rr,ww] = output_dict_nobc_shell['Qback']*total_crossect - output_dict_nobc_core['Qback']*core_crossect
                # else:
                #     self.Cext_nobc[rr,ww] = self.Cext[rr,ww]
                #     self.Csca_nobc[rr,ww] = self.Csca[rr,ww]
                #     self.Cabs_nobc[rr,ww] = self.Cabs[rr,ww]
                #     self.g_nobc_core[rr,ww] = self.g[rr,ww]
                #     self.g_nobc_shell[rr,ww] = self.g[rr,ww]
                #     self.Cpr_nobc[rr,ww] = self.Cpr[rr,ww]
                #     self.Cback_nobc[rr,ww] = self.Cback[rr,ww]

# FIXME: expand aerosol species to include RIs
    
@dataclass
class RefractiveIndex:
    """RefractiveIndex: the definition of wavelength-dependent refractive index.
    Applies to an individual aerosol component or for a mixture of components """
    
    # convention is to always use real_ri_fun and imag_ri_fun if they are defined
    # (i.e., wvls, real_ris, and imag_ris are ignored)
    real_ri_fun: Callable[[float],float]
    imag_ri_fun: Callable[[float],float]
    
    # if real_ris and imag_ris are defined, wvls must also be defined
    wvls: Optional[np.array] = None
    real_ris: Optional[np.array] = None
    imag_ris: Optional[np.array] = None
    
    RI_params: Optional[dict] = None
    
def _add_spec_RI(
        aero_spec,wvls=None,
        specdata_path='../../datasets/aerosol/species_data/',
        spec_modifications={}):
    
    spec_name = aero_spec.name.upper()
    
    if spec_name != 'H2O':
        RI_params = get_RI_params(spec_name)
        if 'n_550' in spec_modifications.keys():
            RI_params['n_550'] = spec_modifications['n_550']
        
        if 'alpha_n' in spec_modifications.keys():
            RI_params['alpha_n'] = spec_modifications['alpha_n']
        
        real_ri_fun = lambda wvl: RI_fun(wvl, RI_params['n_550'], RI_params['alpha_n'])
        
        if 'k_550' in spec_modifications.keys():
            RI_params['k_550'] = spec_modifications['k_550']
        
        if 'alpha_k' in spec_modifications.keys():
            RI_params['alpha_k'] = spec_modifications['alpha_k']
        
        imag_ri_fun = lambda wvl: RI_fun(wvl, RI_params['k_550'], RI_params['alpha_k'])
        
        if type(wvls) == type(None):
            real_ris = None
            imag_ris = None
        else:
            real_ris = real_ri_fun(wvls)
            imag_ris = imag_ri_fun(wvls)
    else:
        ri_h2o_filename = specdata_path / 'ri_water.csv'
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
        RI_params = None
    
    aero_spec.refractive_index = RefractiveIndex(
        wvls = wvls,
        real_ris = real_ris,
        imag_ris = imag_ris,
        real_ri_fun = real_ri_fun,
        imag_ri_fun = imag_ri_fun,
        RI_params = RI_params)
    
    return aero_spec

def RI_fun(wvl, val_550, val_alpha):
    return val_550*(wvl/550e-9)**val_alpha

def get_RI_params(name):
    if name.upper() in ['SO4','NH4','NO3','NA','CL','MSA','CO3']:
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
        
        #warn(name + ' is not in the list, assuming its an arbitrary organic')
    
    RI_param_dict = {'n_550':n_550, 'k_550':k_550, 'alpha_n':alpha_n, 'alpha_k':alpha_k}
    
    return RI_param_dict

    