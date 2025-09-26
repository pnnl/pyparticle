import os
import sys
import json
import pathlib
import pytest

# Force headless matplotlib backend for plotting tests
os.environ.setdefault("MPLBACKEND", "Agg")

# Ensure local src/ is first on sys.path so tests import the workspace package
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# Import the builder module directly from the workspace to avoid importing
# src/PyParticle/__init__.py which has package-level relative imports that
# may fail in the test environment.
from PyParticle.population.builder import build_population

FIX = ROOT / "tests" / "fixtures"


@pytest.fixture(scope="session")
def small_cfg():
    p = FIX / "binned_lognormal_small.json"
    assert p.exists(), f"Missing {p}. Commit the small fixture JSON."
    return json.loads(p.read_text())


@pytest.fixture(scope="session")
def population(small_cfg):
    # Build and return a ParticlePopulation using the real builder
    return build_population(small_cfg["population"])
