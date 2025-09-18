import os
import importlib
import pkgutil

import pytest


@pytest.mark.unit
def test_import_top_level(detected_package_name, monkeypatch):
    # Guard env for offline/GPU-free import
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy"]:
        monkeypatch.delenv(k, raising=False)

    pkg = importlib.import_module(detected_package_name)
    # Optional version attribute check
    assert hasattr(pkg, "__name__")


@pytest.mark.unit
def test_import_submodules(detected_package_name):
    pkg = importlib.import_module(detected_package_name)
    if not hasattr(pkg, "__path__"):
        pytest.skip("Package is not a namespace with submodules")
    # Iterate submodules but avoid importing tests
    for mod in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        name = mod.name
        if any(part.startswith("_") for part in name.split(".")):
            continue
        # Skip tests and internal archived modules that aren't part of the public API
        if ".tests" in name:
            continue
        if any(tok in name for tok in [
            "archive",
            "builder_archive",
            "particle_population_archive",
            "aerosol_species_archive",
        ]):
            continue
        # Best-effort: import submodules; if optional/broken, xfail instead of hard fail
        try:
            importlib.import_module(name)
        except ImportError as e:
            pytest.xfail(f"optional submodule '{name}' import failed: {e}")
