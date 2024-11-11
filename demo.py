#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstration script for using PyParticle to compute particle properties 

@author: Laura Fierce
"""


import numpy as np
import PyParticle


D = 100e-9
aero_spec_names = ['BC','SO4','H2O']
aero_spec_fracs = np.hstack([0.7,0.2,0.1])
particle = PyParticle.make_particle(D, aero_spec_names, aero_spec_fracs)

rh_grid = np.hstack([0.,0.99])
wvl_grid = np.hstack([350e-9,550e-9,750e-9])

cs_particle = PyParticle.make_optical_particle(particle, rh_grid, wvl_grid)

print(cs_particle)
