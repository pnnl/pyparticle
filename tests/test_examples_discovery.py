import ast
import json
from pathlib import Path

import pytest

from .utils import discover_examples, run_example


@pytest.mark.unit
def test_discover_and_validate_examples_syntax(tmp_path, examples_gate):
    root = Path(__file__).resolve().parents[1]
    files = discover_examples([root])
    py_files = [str(p) for p in files if Path(p).suffix == ".py"]
    nb_files = [str(p) for p in files if Path(p).suffix == ".ipynb"]
    # Validate .py syntax via AST
    for py in py_files:
        with open(py, "r", encoding="utf-8") as f:
            ast.parse(f.read(), filename=py)
    # Validate .ipynb structure (don't execute)
    for nb in nb_files:
        with open(nb, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "cells" in data


@pytest.mark.examples
def test_run_examples_fast_mode():
    import os
    if not (os.environ.get("PYPARTICLE_RUN_EXAMPLES", "0") == "1"):
        pytest.skip("Examples execution gated by PYPARTICLE_RUN_EXAMPLES=1")

    root = Path(__file__).resolve().parents[1]
    files = discover_examples([root])
    py_files = [str(p) for p in files if Path(p).suffix == ".py"]
    base_env = dict(os.environ)
    base_env.update({
        "MPLBACKEND": "Agg",
        "CUDA_VISIBLE_DEVICES": "",
        "PYPARTICLE_FAST": "1",
        "PYTEST_SEED": base_env.get("PYTEST_SEED", "1337"),
    })
    for py in py_files:
        rc, out, err = run_example(Path(py), timeout=60, env=base_env)
        assert rc == 0, f"Example failed: {py}\nSTDOUT:\n{out}\nSTDERR:\n{err}"
