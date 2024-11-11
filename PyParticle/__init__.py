#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyParticle python package

@author: Laura Fierce
"""
import os
import numpy as np
data_path = os.getcwd() + '/datasets/'
from .utilities import get_number
# #from .utilities import Py3Wrapper
from .aerosol_particle import Particle, make_particle, make_particle_from_masses
from .aerosol_species import AerosolSpecies, retrieve_one_species
from .optics import make_optical_particle

from .particle_population import ParticlePopulation

from . import builder
from .builder import monodisperse
# from .builder import partmc

