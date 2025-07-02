#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
class and functions for aerosol species

@author: Laura Fierce
"""
from dataclasses import dataclass
from typing import Optional
from . import data_path


@dataclass
class AerosolSpecies:
    """AerosolSpecies: the definition of an aerosol species in terms of species-
    specific parameters (no state information)"""
    name: str
    density: Optional[float] = None
    kappa: Optional[float] = None
    molar_mass: Optional[float] = None
    surface_tension: float = 0.072

    def __post_init__(self):
        # Load defaults from file only if any key value is None
        if self.density is None or self.kappa is None or self.molar_mass is None:
            specdata_path = data_path / 'species_data'
            aero_datafile = specdata_path / 'aero_data.dat'
            found = False
            with open(aero_datafile) as data_file:
                for line in data_file:
                    if line.upper().startswith(self.name.upper()):
                        name_in_file, density, ions_in_solution, molar_mass, kappa = line.split()
                        if self.density is None:
                            self.density = float(density)
                        if self.kappa is None:
                            self.kappa = float(kappa)
                        if self.molar_mass is None:
                            self.molar_mass = float(molar_mass.replace('d','e'))
                        found = True
                        break
            if not found:
                raise ValueError(f"Species data for '{self.name}' not found in data file.")
            # Optionally: raise an error if still None (not found in data file)
            if self.density is None or self.kappa is None or self.molar_mass is None:
                raise ValueError(f"Could not set all required fields for '{self.name}'.")


def retrieve_one_species(name, specdata_path=data_path / 'species_data', spec_modifications={}):
    aero_datafile = specdata_path / 'aero_data.dat'
    with open(aero_datafile) as data_file:
        for line in data_file:
            if line.upper().startswith(name.upper()):
                name_in_file, density, ions_in_solution, molar_mass, kappa = line.split()

                if 'kappa' in spec_modifications.keys():
                    kappa = spec_modifications['kappa']

                if 'density' in spec_modifications.keys():
                    density = spec_modifications['density']

                if 'surface_tension' in spec_modifications.keys():
                    surface_tension = spec_modifications['surface_tension']
                else:
                    surface_tension = 0.072
                return AerosolSpecies(
                    name=name,
                    density=float(density),
                    kappa=float(kappa),
                    molar_mass=float(molar_mass.replace('d','e')),
                    surface_tension=float(surface_tension)
                )
    raise ValueError(f"Species data for '{name}' not found in data file.")