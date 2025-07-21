import pytest
from PyParticle.population import build_population, ParticlePopulation

def test_binned_lognormals_population():
    config = {
        'type': 'binned_lognormals',
        'aero_spec_names': ['SO4', 'BC'],
        'D_min': 0.01,
        'D_max': 1.0,
        'N_bins': 10,
        'N': [1000],
        'GMD': [0.1],
        'GSD': [1.5],
        'aero_spec_fracs': [[0.8, 0.2]],
    }
    pop = build_population(config)
    assert isinstance(pop, ParticlePopulation)
    assert hasattr(pop, 'species')
    assert hasattr(pop, 'num_concs')
    assert pop.get_Ntot() > 0
    assert pop.get_effective_radius() > 0

def test_binned_lognormals_population_with_spec_modifications():
    config = {
        'type': 'binned_lognormals',
        'aero_spec_names': ['SO4', 'BC'],
        'D_min': 0.01,
        'D_max': 1.0,
        'N_bins': 10,
        'N': [1000],
        'GMD': [0.1],
        'GSD': [1.5],
        'aero_spec_fracs': [[0.8, 0.2]],
        'species_modifications': {'SO4': {'density': 3333}, 'BC': {'kappa': 0.1}}
    }
    pop = build_population(config)
    assert isinstance(pop, ParticlePopulation)
    assert pop.species[0].density == 3333
    assert pop.species[1].kappa == 0.1

def test_monodispers_population():
    config = {
        'type': 'monodispers',
        'aero_spec_names': ['SO4', 'BC'],
        'N': [1000, 500],
        'D': [0.1, 0.2],
        'aero_spec_fracs': [[0.7, 0.3], [0.2, 0.8]],
    }
    pop = build_population(config)
    assert isinstance(pop, ParticlePopulation)
    assert hasattr(pop, 'species')
    assert hasattr(pop, 'num_concs')
    assert pop.get_Ntot() > 0
    assert pop.get_effective_radius() > 0

def test_monodispers_population_with_spec_modifications():
    config = {
        'type': 'monodispers',
        'aero_spec_names': ['SO4', 'BC'],
        'N': [1000, 500],
        'D': [0.1, 0.2],
        'aero_spec_fracs': [[0.7, 0.3], [0.2, 0.8]],
        'species_modifications': {'SO4': {'density': 4444, 'surface_tension': 0.09}, 'BC': {'kappa': 0.22}}
    }
    pop = build_population(config)
    assert isinstance(pop, ParticlePopulation)
    assert pop.species[0].density == 4444
    assert pop.species[0].surface_tension == 0.09
    assert pop.species[1].kappa == 0.22

@pytest.mark.skip(reason="Requires a real PARTMC NetCDF file and path.")
def test_partmc_population():
    config = {
        'type': 'partmc',
        'partmc_dir': "dummy_path",
        'timestep': 0,
        'repeat': 0,
    }
    pop = build_population(config)
    assert isinstance(pop, ParticlePopulation)