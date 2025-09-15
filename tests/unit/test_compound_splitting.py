#!/usr/bin/env python3
import math

import numpy as np

from PyParticle.population.utils import expand_compounds_for_population


def close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_nacl_splits_into_na_and_cl():
    names_list = [["NaCl"]]
    fracs_list = [[1.0]]
    out_names, out_fracs = expand_compounds_for_population(names_list, fracs_list)
    names = out_names[0]
    fracs = out_fracs[0]
    assert names == ["Na", "Cl"]
    s = sum(fracs)
    assert close(s, 1.0, 1e-6)
    # expected approx ratios by molar mass 23 and 35.5
    expected_na = 23.0 / (23.0 + 35.5)
    expected_cl = 35.5 / (23.0 + 35.5)
    assert math.isclose(fracs[0], expected_na, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(fracs[1], expected_cl, rel_tol=1e-3, abs_tol=1e-3)


def test_ammonium_sulfate_parentheses_forms():
    # (NH4)2SO4
    names_list = [["(NH4)2SO4"]]
    fracs_list = [[1.0]]
    out_names, out_fracs = expand_compounds_for_population(names_list, fracs_list)
    names = out_names[0]
    fracs = out_fracs[0]
    assert names == ["NH4", "SO4"]
    s = sum(fracs)
    assert close(s, 1.0, 1e-6)
    # based on molar masses: NH4 ~ 18e-3, SO4 ~ 96e-3
    expected_nh4 = (2 * 18.0) / (2 * 18.0 + 96.0)
    expected_so4 = 96.0 / (2 * 18.0 + 96.0)
    assert math.isclose(fracs[0], expected_nh4, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(fracs[1], expected_so4, rel_tol=1e-3, abs_tol=1e-3)

    # Alternate grouping NH4(SO4)2
    names_list = [["NH4(SO4)2"]]
    fracs_list = [[1.0]]
    out_names, out_fracs = expand_compounds_for_population(names_list, fracs_list)
    names2 = out_names[0]
    fracs2 = out_fracs[0]
    assert names2 == ["NH4", "SO4"]
    expected_nh4_2 = 18.0 / (18.0 + 2 * 96.0)
    expected_so4_2 = (2 * 96.0) / (18.0 + 2 * 96.0)
    assert math.isclose(fracs2[0], expected_nh4_2, rel_tol=1e-3, abs_tol=1e-3)
    assert math.isclose(fracs2[1], expected_so4_2, rel_tol=1e-3, abs_tol=1e-3)


def test_mixed_compound_and_base_species():
    names_list = [["NaCl", "SO4"]]
    fracs_list = [[0.3, 0.7]]
    out_names, out_fracs = expand_compounds_for_population(names_list, fracs_list)
    names = out_names[0]
    fracs = out_fracs[0]
    # After expansion: Na, Cl, SO4
    assert names == ["Na", "Cl", "SO4"]
    assert math.isclose(sum(fracs), 1.0, rel_tol=1e-6, abs_tol=1e-6)