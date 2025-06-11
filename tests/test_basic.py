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
    assert hasattr(particle, 'masses')
    assert hasattr(particle, 'species')
    assert np.isclose(particle.get_Dwet(), D)
    assert all([spec.name for (spec,spec_name) in zip(particle.species,aero_spec_names)])
    assert np.isclose(np.sum(particle.masses), particle.get_mass_tot())

def test_make_particle_dry():
    """Test creation of a particle with basic species and fractions."""
    D = 100e-9
    aero_spec_names = ['BC', 'SO4']
    aero_spec_fracs = [0.75, 0.25]
    particle = PyParticle.make_particle(D, aero_spec_names, aero_spec_fracs, D_is_wet=False)
    assert particle is not None
    assert hasattr(particle, 'masses')
    assert np.isclose(particle.get_Ddry(), D)
    assert particle.species[-1].name == 'H2O'
    assert np.isclose(np.sum(particle.masses), particle.get_mass_tot())
    assert np.isclose(np.sum(particle.masses), particle.get_mass_dry())
    
def test_invalid_fractions_sum():
    """Check error raised if species fractions do not sum to 1."""
    D = 100e-9
    aero_spec_names = ['BC', 'SO4']
    aero_spec_fracs = [0.8, 0.1]  # Only sums to 0.9
    with pytest.raises(Exception):
        PyParticle.make_particle(D, aero_spec_names, aero_spec_fracs)

def test_make_cs_particle():
    """Test basic optical properties calculation."""
    D = 100e-9
    names = ['BC', 'SO4', 'H2O']
    fracs = [0.7, 0.25, 0.05]
    particle = PyParticle.make_particle(D, names, fracs)
    rh_grid = np.array([0.])
    wvl_grid = np.array([550e-9])
    cs_particle = PyParticle.make_optical_particle(particle, rh_grid, wvl_grid, morphology='core-shell')
    # Check some expected attributes
    assert hasattr(cs_particle, 'shell_ris')
    assert hasattr(cs_particle, 'core_ris')
    

def test_make_particle_edge_cases():
    """Test edge case: all H2O (should still make a particle)."""
    D = 200e-9
    names = ['H2O']
    fracs = [1.0]
    particle = PyParticle.make_particle(D, names, fracs)
    assert particle is not None
    assert np.isclose(particle.get_Dwet(), D)
    assert [spec.name for spec in particle.species] == ['H2O']    
