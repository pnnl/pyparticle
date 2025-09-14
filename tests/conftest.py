import os
import sys
import json
import ast
import random
import socket
import types
from contextlib import contextmanager
from typing import Iterator

import pytest


@pytest.fixture(scope="session", autouse=True)
def _deterministic_session_env():
    """
    Session-wide deterministic environment:
    - Headless plotting
    - Disable GPU by default
    - Seed RNGs for random, numpy, and torch (if installed)
    - Block network by default (tests can opt-out by using allow_network context)
    """
    # Headless backend for matplotlib
    os.environ["MPLBACKEND"] = "Agg"
    # Disable GPU/CUDA by default
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # Seed RNGs
    seed = int(os.environ.get("PYTEST_SEED", "1337"))
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) if getattr(torch, "cuda", None) else None
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass

    # Clear proxy-related env to reduce chance of network egress
    for key in [
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "http_proxy",
        "https_proxy",
        "NO_PROXY",
        "no_proxy",
    ]:
        if key in os.environ:
            os.environ.pop(key, None)


@pytest.fixture(scope="session")
def small_random_state():
    try:
        import numpy as np

        return np.random.RandomState(123)
    except Exception:
        return None


@pytest.fixture()
def small_ndarray(small_random_state):
    try:
        import numpy as np

        rs = small_random_state or np.random.RandomState(123)
        return rs.randn(2, 3)
    except Exception:
        pytest.skip("numpy not available")


@pytest.fixture()
def small_dataframe(small_random_state):
    try:
        import pandas as pd  # type: ignore
        import numpy as np

        rs = small_random_state or np.random.RandomState(123)
        return pd.DataFrame({"a": rs.randn(5), "b": rs.randint(0, 3, size=5)})
    except Exception:
        pytest.skip("pandas not available")


@pytest.fixture()
def small_series(small_dataframe):
    try:
        import pandas as pd  # type: ignore

        return small_dataframe["a"] if hasattr(small_dataframe, "__getitem__") else pd.Series([])
    except Exception:
        pytest.skip("pandas not available")


@pytest.fixture()
def small_xarray(small_ndarray):
    try:
        import xarray as xr  # type: ignore

        return xr.DataArray(small_ndarray, dims=["x", "y"])
    except Exception:
        pytest.skip("xarray not available")


@pytest.fixture()
def synthetic_text_file(tmp_path) -> str:
    p = tmp_path / "synthetic.txt"
    p.write_text("hello world\n")
    return str(p)


@pytest.fixture()
def synthetic_binary_file(tmp_path) -> str:
    p = tmp_path / "synthetic.bin"
    p.write_bytes(b"\x00\x01\x02\x03")
    return str(p)


# Simple socket guard to block external network by default
class _DenySocket(socket.socket):
    def connect(self, *args, **kwargs):  # noqa: D401
        raise RuntimeError("Network access is disabled during tests.")


@contextmanager
def allow_network() -> Iterator[None]:
    original = socket.socket
    try:
        socket.socket = original  # type: ignore
        yield
    finally:
        socket.socket = original  # type: ignore


@pytest.fixture(autouse=True)
def _deny_network(monkeypatch):
    # Allow opt-out via env var
    if os.environ.get("PYTEST_ALLOW_NETWORK", "0") == "1":
        yield
        return
    monkeypatch.setattr(socket, "socket", _DenySocket)
    yield


def pytest_configure(config):
    # Register markers to avoid PytestUnknownMarkWarning
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "examples: tests that validate or run examples")
    config.addinivalue_line("markers", "integration: integration tests (may be slower)")


@pytest.fixture(scope="session")
def examples_gate():
    """Gate execution of examples via env var PYPARTICLE_RUN_EXAMPLES (default off)."""
    run = os.environ.get("PYPARTICLE_RUN_EXAMPLES", "0") == "1"
    if run:
        os.environ["PYPARTICLE_FAST"] = "1"
    return run


@pytest.fixture(scope="session")
def detected_package_name():
    """Detect top-level package name automatically."""
    # Heuristic: prefer importable PyParticle; else scan common locations.
    for name in ("PyParticle",):
        try:
            __import__(name)
            return name
        except Exception:
            continue
    # Try to put 'src' on sys.path and retry
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    src_dir = os.path.join(repo_root, "src")
    if os.path.isdir(src_dir) and src_dir not in sys.path:
        sys.path.insert(0, src_dir)
        try:
            __import__("PyParticle")
            return "PyParticle"
        except Exception:
            pass
    # As a fallback, try adding repo root (in case package at root)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        __import__("PyParticle")
        return "PyParticle"
    except Exception:
        pytest.skip("Unable to detect/import top-level package")
