#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class and functions for aerosol species

@author: Laura Fierce
"""
# import os
# import numpy as np
from dataclasses import dataclass
# from typing import Tuple
from typing import Optional
# from warnings import warn
# from scipy.constants import R
# import scipy.optimize as opt
from . import data_path 


@dataclass#(frozen=True)
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
    specific parameters (no state information)"""
    name: str          # name of the species
    density: float
    kappa: float
    molar_mass: float
    surface_tension: Optional[float] = 0.072
    
    def _populate_defaults(
            self,specdata_path= data_path / 'species_data'):
        aero_datafile = specdata_path / 'aero_data.dat'
        name = self.name
        with open(aero_datafile) as data_file:
            for line in data_file:
                if line.upper().startswith(name.upper()):
                    name_in_file,density,ions_in_solution,molecular_weight,kappa = line.split()
        self.density = float(density)
        self.molecular_weight = float(molecular_weight.replace('d','e'))
        self.kappa = float(kappa)
    
def retrieve_one_species(name, specdata_path=data_path / 'species_data', spec_modifications={}):
    aero_datafile = specdata_path / 'aero_data.dat'
    with open(aero_datafile) as data_file:
        for line in data_file:
            if line.upper().startswith(name.upper()):
                name_in_file,density,ions_in_solution,molar_mass,kappa = line.split()
                
                if 'kappa' in spec_modifications.keys():
                    kappa = spec_modifications['kappa']
                
                if 'density' in spec_modifications.keys():
                    density = spec_modifications['density']
                
                if 'surface_tension' in spec_modifications.keys():
                    surface_tension = spec_modifications['surface_tension']
                else:
                    surface_tension=0.072
    return AerosolSpecies(
        name=name,
        density=float(density),
        kappa=float(kappa),
        molar_mass=float(molar_mass.replace('d','e')),
        surface_tension=float(surface_tension))