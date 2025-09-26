import os
import importlib.util
import pytest
from pathlib import Path

pytestmark = pytest.mark.slow

def test_examples_importable():
    # Lightweight: just ensure files exist and are importable if they define a main guard.
    root = Path(__file__).resolve().parents[2] / "examples"
    assert root.exists()
    # Add specific example files here if you want stricter checks
    # e.g., "compare_bscat_vs_wvl.py"
    targets = []
    for rel in targets:
        p = root / rel
        spec = importlib.util.spec_from_file_location(p.stem, p)
        assert spec is not None and spec.loader is not None
