import importlib

import pytest


@pytest.mark.unit
def test_top_level_has_init(detected_package_name):
    pkg = importlib.import_module(detected_package_name)
    assert hasattr(pkg, "__doc__")


@pytest.mark.unit
def test_population_and_optics_modules_present(detected_package_name):
    # Minimal presence checks for common modules
    importlib.import_module(f"{detected_package_name}.population")
    importlib.import_module(f"{detected_package_name}.optics")
