#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some basic functions used by the other modules

@author: Laura Fierce
"""

import numpy as np

def get_number(string_val):
    if string_val.endswith('\n'):
        string_val = string_val[:-2]
        
    if '×' in string_val:
        idx = string_val.find('×')
        front_part = float(string_val[:idx])
        back_part = string_val[(idx+1):][2:]
        if back_part.startswith('−'):
            exponent = -float(back_part[1:])
        else:
            exponent = float(back_part)
        number = front_part*10.**exponent
    else:
        number = float(string_val)
    return number


def power_moments_from_lognormal(k,N,gmd,gsd):
    return N*np.exp(k*np.log(gmd) + k**2 * np.log(gsd)/2)

def power_moments_from_lognormals(k, Ns, GMDs, GSDs):
    return np.sum([power_moments_from_lognormal(k,N,gmd,gsd) for (N,gmd,gsd) in zip(Ns,GMDs,GSDs)])