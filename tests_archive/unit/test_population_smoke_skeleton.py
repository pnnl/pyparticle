import pytest
from PyParticle.population.builder import build_population

@pytest.mark.smoke
def test_binned_lognormal_smoke():
    cfg = {
        "type": "binned_lognormals",
        "N": [1e6], "GMD": [100e-9], "GSD": [1.6],
        "aero_spec_names": [["SO4"]], "aero_spec_fracs": [[1.0]],
        "N_bins": 10,
        "species_modifications": {"SO4": {"n_550":1.45}}
    }
    pop = build_population(cfg)
    # Basic smoke assertions
    assert hasattr(pop, 'ids')
    assert len(pop.ids) == 10
    assert hasattr(pop, 'species_modifications')
