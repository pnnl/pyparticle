import pytest
from PyParticle.population import build_population, ParticlePopulation

binned_lognormals_config = {
    'type': 'binned_lognormals',
    'aero_spec_names': ('SO4', 'BC'),
    'D_min': 0.01,
    'D_max': 1.0,
    'N_bins': 10,
    'N': [1000],
    'GMD': [0.1],
    'GSD': [1.5],
    'aero_spec_fracs': [[0.8, 0.2]],
}

monodispers_config = {
    'type': 'monodispers',
    'aero_spec_names': ('SO4', 'BC'),
    'N': [1000, 500],
    'D': [0.1, 0.2],
    'aero_spec_fracs': [[0.7, 0.3], [0.2, 0.8]],
}

partmc_config = {
    'type': 'partmc',
    'partmc_dir': "dummy_path",
    'timestep': 0,
    'repeat': 0,
}

def test_binned_lognormals_population():
    pop = build_population(binned_lognormals_config)
    assert isinstance(pop, ParticlePopulation)
    assert hasattr(pop, 'species')
    assert hasattr(pop, 'num_concs')

def test_monodispers_population():
    pop = build_population(monodispers_config)
    assert isinstance(pop, ParticlePopulation)
    assert hasattr(pop, 'species')
    assert hasattr(pop, 'num_concs')

@pytest.mark.skip(reason="Requires a real NetCDF file and path for PARTMC test")
def test_partmc_population():
    pop = build_population(partmc_config)
    assert isinstance(pop, ParticlePopulation)

def test_population_interface():
    pop = build_population(binned_lognormals_config)
    n_tot = pop.get_Ntot()
    assert n_tot > 0
    r_eff = pop.get_effective_radius()
    assert r_eff > 0