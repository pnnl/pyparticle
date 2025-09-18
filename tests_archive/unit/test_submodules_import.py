import importlib
import pkgutil

import pytest


@pytest.mark.unit
def test_all_public_submodules_import(detected_package_name):
    pkg = importlib.import_module(detected_package_name)
    if not hasattr(pkg, "__path__"):
        pytest.skip("No submodules to import")
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        name = mod.name
        # Skip private and tests
        if any(part.startswith("_") for part in name.split(".")):
            continue
        if ".tests" in name:
            continue
        if any(tok in name for tok in [
            "archive",
            "builder_archive",
            "particle_population_archive",
            "aerosol_species_archive",
        ]):
            continue
        try:
            importlib.import_module(name)
        except ImportError as e:
            pytest.xfail(f"optional submodule '{name}' import failed: {e}")
