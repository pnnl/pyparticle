import pytest
import numpy as np
import PyParticle

def test_make_particle_basic():
    """Test creation of a particle with basic species and fractions."""
    D = 100e-9
    aero_spec_names = ['BC', 'SO4', 'H2O']
    aero_spec_fracs = [0.7, 0.25, 0.05]
    particle = PyParticle.make_particle(D, aero_spec_names, aero_spec_fracs)
    assert particle is not None
    assert hasattr(particle, 'diameter')
    assert np.isclose(particle.get_diameter(), D)
    assert set(particle.aero_spec_names) == set(aero_spec_names)
    assert np.isclose(np.sum(particle.aero_spec_fracs), 1.0)

def test_invalid_fractions_sum():
    """Check error raised if species fractions do not sum to 1."""
    D = 100e-9
    aero_spec_names = ['BC', 'SO4']
    aero_spec_fracs = [0.8, 0.1]  # Only sums to 0.9
    with pytest.raises(Exception):
        PyParticle.make_particle(D, aero_spec_names, aero_spec_fracs)

def test_make_optical_particle():
    """Test basic optical properties calculation."""
    D = 100e-9
    names = ['BC', 'SO4', 'H2O']
    fracs = [0.7, 0.25, 0.05]
    particle = PyParticle.make_particle(D, names, fracs)
    rh_grid = np.array([0.])
    wvl_grid = np.array([550e-9])
    cs_particle = PyParticle.make_optical_particle(particle, rh_grid, wvl_grid)
    # Check some expected attributes
    assert hasattr(cs_particle, "Qabs")
    assert hasattr(cs_particle, "Qscat")
    assert hasattr(cs_particle, "Cabs")
    assert hasattr(cs_particle, "Csca")

def test_make_particle_edge_cases():
    """Test edge case: all H2O (should still make a particle)."""
    D = 200e-9
    names = ['H2O']
    fracs = [1.0]
    particle = PyParticle.make_particle(D, names, fracs)
    assert particle is not None
    assert np.isclose(particle.get_diameter, D)
    assert particle.aero_spec_names == ['H2O']
