#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small numeric utilities used by PyParticle.

This module provides helpers for parsing numeric strings and computing
power moments of lognormal size distributions used by population builders
and examples.
"""

import numpy as np


def get_number(string_val):
    """Parse a numeric string into a float.

    Accepts plain floats or strings using a multiplication sign and exponent
    notation (e.g. "1.23×10^3").

    Parameters
    ----------
    string_val : str
        Numeric string to parse.

    Returns
    -------
    float
        Parsed numeric value.
    """
    if string_val.endswith('\n'):
        string_val = string_val[:-2]

    if '\u00d7' in string_val:
        idx = string_val.find('\u00d7')
        front_part = float(string_val[:idx])
        back_part = string_val[(idx+1):][2:]
        if back_part.startswith('\u2212'):
            exponent = -float(back_part[1:])
        else:
            exponent = float(back_part)
        number = front_part * 10.0 ** exponent
    else:
        number = float(string_val)
    return number


def power_moments_from_lognormal(k, N, gmd, gsd):
    """Compute the k-th power moment for one lognormal mode.

    Parameters
    ----------
    k : int
        Moment order.
    N : float
        Number concentration prefactor.
    gmd : float
        Geometric mean diameter.
    gsd : float
        Geometric standard deviation.

    Returns
    -------
    float
        k-th power moment value.
    """
    return N * np.exp(k * np.log(gmd) + k ** 2 * np.log(gsd) / 2.0)


def power_moments_from_lognormals(k, Ns, GMDs, GSDs):
    """Sum k-th power moments across multiple lognormal modes.

    Parameters mirror `power_moments_from_lognormal` but accept sequences.
    """
    return np.sum([
        power_moments_from_lognormal(k, N, gmd, gsd)
        for (N, gmd, gsd) in zip(Ns, GMDs, GSDs)
    ])