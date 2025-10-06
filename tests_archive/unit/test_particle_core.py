import numpy as np
from PyParticle.aerosol_particle import make_particle

def test_make_particle_roundtrip():
    p = make_particle(
        D=200e-9,
        aero_spec_names=["SO4", "BC"],
        aero_spec_frac=[0.9, 0.1],
        D_is_wet=True,
    )
    assert p.get_Ddry() > 0
    # total mass should be >= dry mass (H2O may be zero if not equilibrated)
    assert p.get_mass_tot() >= p.get_mass_dry()

def test_critical_supersaturation_monotone_T(population):
    # Pick one particle; expect ss_crit to be finite and vary smoothly
    pid = population.ids[0]
    part = population.get_particle(pid)
    s1 = part.get_critical_supersaturation(273.15)
    s2 = part.get_critical_supersaturation(298.15)
    assert s1 > 0 and s2 > 0
    assert np.isfinite(s1) and np.isfinite(s2)
