import os
import importlib
import sys
from pathlib import Path


def test_pyparticle_data_path_env_override(tmp_path, monkeypatch):
    """When PYPARTICLE_DATA_PATH is set to a custom datasets dir, retrieve_one_species should read from it."""
    # Create custom datasets path with species_data/aero_data.dat
    custom = tmp_path / "datasets" / "species_data"
    custom.mkdir(parents=True, exist_ok=True)
    aero = custom / "aero_data.dat"
    aero.write_text("H2O 5555 0 18d-3 0\n")

    # Point env var to the datasets root
    monkeypatch.setenv("PYPARTICLE_DATA_PATH", str(tmp_path / "datasets"))

    # Ensure fresh import of module to pick up env var
    if "PyParticle" in sys.modules:
        importlib.reload(sys.modules["PyParticle"])
    else:
        import PyParticle
        importlib.reload(PyParticle)

    from PyParticle.species.registry import retrieve_one_species
    h2o = retrieve_one_species("H2O")
    assert abs(float(h2o.density) - 5555.0) < 1e-6