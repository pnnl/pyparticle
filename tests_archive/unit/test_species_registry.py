from PyParticle.species.registry import retrieve_one_species, get_species

def test_retrieve_default_species_SO4():
    so4 = retrieve_one_species("SO4")
    assert so4.name.upper() == "SO4"
    assert so4.density and so4.kappa is not None

def test_get_species_with_overrides():
    sp = get_species("SO4", density=2000.0)
    assert sp.density == 2000.0
