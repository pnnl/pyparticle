import os
import sys
import json
import pathlib
import pytest
import numpy as np

# Headless matplotlib
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure local src/ on sys.path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
    # also add repo root so top-level folders like `examples/` are importable
    if str(ROOT) not in sys.path:
        sys.path.insert(1, str(ROOT))

# Import builder modules directly (avoid package-level side effects in __init__)
from PyParticle.population.builder import build_population
from PyParticle.optics.builder import build_optical_population

FIX = ROOT / "tests" / "fixtures"
DATASETS = SRC / "PyParticle" / "datasets"


@pytest.fixture(scope="session", autouse=True)
def data_path_env():
    # Point loaders to the checked-in datasets
    os.environ.setdefault("PYPARTICLE_DATA_PATH", str(DATASETS))
    yield


@pytest.fixture(scope="session")
def small_cfg():
    p = FIX / "binned_lognormal_small.json"
    assert p.exists(), f"Missing {p}. Commit the small fixture JSON."
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def population(small_cfg):
    return build_population(small_cfg["population"])


@pytest.fixture(scope="session")
def wvl_grid_small():
    return np.array([450e-9, 550e-9, 700e-9])


@pytest.fixture(scope="session")
def rh_grid_zero():
    return np.array([0.0])
