import numpy as np
from PyParticle.optics.builder import build_optical_population

def test_optical_population_shapes_and_units(population, wvl_grid_small, rh_grid_zero):
    o = build_optical_population(population, {"type": "homogeneous",
                                              "wvl_grid": wvl_grid_small,
                                              "rh_grid": rh_grid_zero})
    b_ext = o.get_optical_coeff("b_ext")  # [nRH, nWvl]
    assert b_ext.ndim == 2
    assert b_ext.shape == (rh_grid_zero.size, wvl_grid_small.size)
    assert (b_ext >= 0).all()

    g = o.get_optical_coeff("g")
    assert g.ndim == 2
    assert (g >= -1).all() and (g <= 1).all()
