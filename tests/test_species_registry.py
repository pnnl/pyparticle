import pytest
from PyParticle.species.base import AerosolSpecies
from PyParticle.species.registry import (
    get_species,
    register_species,
    list_species,
    extend_species,
)

def test_get_species_from_dat(monkeypatch):
    # This test assumes "SO4" is in your aero_data.dat with known values.
    so4 = get_species("SO4")
    assert isinstance(so4, AerosolSpecies)
    assert so4.name.upper() == "SO4"
    assert so4.density is not None
    assert so4.kappa is not None
    assert so4.molar_mass is not None

def test_get_species_with_modification():
    so4_mod = get_species("SO4", density=2000, kappa=0.7)
    assert so4_mod.density == 2000
    assert so4_mod.kappa == 0.7

def test_register_and_get_species():
    new_spec = AerosolSpecies(name="TESTSPEC", density=1111, kappa=0.3, molar_mass=99.99)
    register_species(new_spec)
    got = get_species("TESTSPEC")
    assert got.name == "TESTSPEC"
    assert got.density == 1111
    assert got.kappa == 0.3
    assert got.molar_mass == 99.99
    assert "TESTSPEC" in list_species()

def test_extend_species_is_alias():
    spec2 = AerosolSpecies(name="ALIAS", density=2222, kappa=0.2, molar_mass=88.88)
    extend_species(spec2)
    got = get_species("ALIAS")
    assert got.name == "ALIAS"
    assert got.density == 2222

def test_file_not_found_raises():
    with pytest.raises(ValueError):
        get_species("DOES_NOT_EXIST")

def test_get_species_with_spec_modifications_dict():
    # Simulate the retrieval as would be done through population builders
    # via spec_modifications dictionary per species
    spec_modifications = {'SO4': {'density': 3000, 'surface_tension': 0.08}}
    so4 = get_species("SO4", **spec_modifications["SO4"])
    assert so4.density == 3000
    assert so4.surface_tension == 0.08