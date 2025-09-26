import os
import numpy as np
import pytest

# Ensure headless backend for matplotlib during tests
os.environ.setdefault("MPLBACKEND", "Agg")


@pytest.fixture(scope="session")
def rng_seed():
    np.random.seed(1337)


@pytest.fixture
def small_binned_pop():
    """Return a small, deterministic binned_lognormals population for unit tests."""
    from PyParticle import build_population

    cfg = {
        "type": "binned_lognormals",
        # binned_lognormals expects parallel lists: N, GMD, GSD, aero_spec_names, aero_spec_fracs
        "N": [100.0],
        "GMD": [100e-9],
        "GSD": [1.6],
        "aero_spec_names": [["SO4"]],
        "aero_spec_fracs": [[1.0]],
        "aero_spec_names_list": None,
        "aero_spec_fracs_list": None,
        "species": {"SO4": {"density": 1770.0}},
        "N_bins": 20,
        "D_min": 1e-9,
        "D_max": 2e-6,
    }
    pop = build_population(cfg)
    return pop


@pytest.fixture
def disable_network(monkeypatch):
    monkeypatch.setenv("PYTEST_ALLOW_NETWORK", "0")
    return None
