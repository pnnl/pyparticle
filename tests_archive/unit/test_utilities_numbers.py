import numpy as np

from PyParticle.utilities import get_number, power_moments_from_lognormal

def test_get_number_parses_unicode_times():
    assert np.isclose(get_number("1.2×10−3"), 1.2e-3)
    assert np.isclose(get_number("3×10^2"), 300.0)

def test_power_moment_reasonable():
    # For k=0, returns N (moment of order 0)
    N = 1e6
    gmd = 100e-9
    gsd = 1.6
    m0 = power_moments_from_lognormal(0, N, gmd, gsd)
    assert np.isclose(m0, N)
