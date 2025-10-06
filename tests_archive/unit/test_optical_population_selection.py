import numpy as np
from PyParticle.optics.builder import build_optical_population

def test_scalar_selection(population, wvl_grid_small, rh_grid_zero):
    o = build_optical_population(population, {"type": "homogeneous",
                                              "wvl_grid": wvl_grid_small,
                                              "rh_grid": rh_grid_zero})
    v = o.get_optical_coeff("b_scat", rh=float(rh_grid_zero[0]), wvl=float(wvl_grid_small[0]))
    assert np.isscalar(v)
