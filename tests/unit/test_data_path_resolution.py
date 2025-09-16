import os
import sys
import tempfile
import importlib
from pathlib import Path

import pytest


def test_package_data_resolves_to_pyparticle(tmp_path, monkeypatch):
    """Ensure PyParticle reads its bundled datasets even when CWD has a datasets/ tree.

    Create a fake datasets tree in the current working directory with a bogus
    aero_data.dat, change CWD to that directory, import (or reload) PyParticle,
    and verify that `retrieve_one_species('H2O')` returns the packaged value.
    """
    # Create a fake datasets in tmp_path that would confuse naive CWD-based lookups
    fake_datasets = tmp_path / "datasets" / "species_data"
    fake_datasets.mkdir(parents=True, exist_ok=True)
    fake_file = fake_datasets / "aero_data.dat"
    fake_file.write_text("H2O 9999 0 99d-3 0\n")

    # Change cwd to the temp directory
    old_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)

        # Ensure fresh import: remove PyParticle if already imported
        # Some CI/test envs may not have heavy deps (numpy) installed. Provide
        # a lightweight stub for imports during package import to avoid
        # ModuleNotFoundError while we only test data resolution logic.
        import types
        if "numpy" not in sys.modules:
            sys.modules["numpy"] = types.ModuleType("numpy")
            # Minimal compatibility: provide array and float64 placeholder
            setattr(sys.modules["numpy"], "array", lambda *a, **k: list(a))

        if "PyParticle" in sys.modules:
            importlib.reload(sys.modules["PyParticle"])
        else:
            import PyParticle
            importlib.reload(PyParticle)

        from PyParticle.species.registry import retrieve_one_species

        # Retrieve H2O from the package data; packaged aero_data.dat has H2O density 1000
        h2o = retrieve_one_species("H2O")
        assert hasattr(h2o, "density")
        # Density in packaged data is 1000; ensure we didn't read the fake 9999 value
        assert abs(float(h2o.density) - 1000.0) < 1e-6
    finally:
        os.chdir(old_cwd)
        # Removed stray '*** End Patch' text to fix syntax error